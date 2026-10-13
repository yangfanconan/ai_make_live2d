"""Express4D 推理 —— CSV → 模型推理 → fit_to_model → motion3.json

管线：
  1. load_take(csv) → (features [T,N_IN], targets [T,N_OUT], times [T])
  2. 构建滑窗序列 → model.forward_seq → unit 参数 [T, N_OUT]
  3. unit_to_canonical → fit_to_model（模型量程缩放）
  4. 复用 motion_author 的 lowpass + resample_lin + motion3 写出

用法：
  python3 tools/agents/express4d/infer.py \
      --csv take.csv --model Haru \
      --ckpt tools/data/express4d/models/express4d_v1.pt \
      --out tools/data/out/express4d_take.motion3.json
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents import arkit_map as AM
from agents import motion_author as MA
from agents.common import TOOLS, log, write_json
from agents.express4d.dataset import load_take, unit_to_canonical, LIVE2D_PARAMS, N_OUT
from agents.express4d.model import load_model


def infer(csv_path, model_path, out_path, model_name='Haru',
          window=15, cutoff=12.0, out_fps=30, loop=False, device='cpu'):
    """CSV → 模型推理 → motion3.json

    返回 motion dict + report。
    """
    # 加载模型
    net, meta = load_model(model_path, device=device)
    net.eval()
    log(f'模型加载: {model_path} (window={net.window})')

    # 加载 CSV
    feats, l1_targets, times = load_take(csv_path)
    T = len(feats)
    log(f'CSV: {T} 帧 / {times[-1]:.1f}s')

    # 构建滑窗序列
    half = window // 2
    windows = []
    for fi in range(T):
        lo = max(0, fi - half)
        hi = min(T, fi + half + 1)
        chunk = feats[lo:hi]
        if len(chunk) < window:
            pad = window - len(chunk)
            if lo == 0:
                chunk = np.concatenate([np.repeat(feats[:1], pad, axis=0), chunk])
            else:
                chunk = np.concatenate([chunk, np.repeat(feats[-1:], pad, axis=0)])
        windows.append(chunk)
    windows = np.stack(windows)  # (T, W, N_IN)

    # 推理
    with torch.no_grad():
        x = torch.from_numpy(windows).to(device).float()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        # forward_seq 接受 (T, W, N_IN)
        unit_preds = net.forward_seq(x)  # (T, N_OUT)
    unit_preds = unit_preds.cpu().numpy()

    # unit → canonical → model-specific
    ranges, signs = AM.load_calib(model_name, TOOLS)
    model_params = {}  # pid -> [(t, v)]
    for ti in range(T):
        canon = unit_to_canonical(unit_preds[ti])
        fitted = AM.fit_to_model(canon, ranges)
        for pid, v in fitted.items():
            model_params.setdefault(pid, []).append((times[ti], v))

    # 低通 + 钳制 + 活跃度筛选（复用 motion_author 逻辑）
    curves_src, dropped = {}, []
    for pid, ser in model_params.items():
        ser = MA.lowpass(ser, cutoff)
        r = ranges[pid]
        clamped = [(t, max(r['min'], min(r['max'], v))) for t, v in ser]
        vs = [v for _, v in clamped]
        if max(vs) - min(vs) < 1e-3:
            dropped.append(pid)
            continue
        curves_src[pid] = clamped

    # 30fps 重采样 → motion3
    duration_src = times[-1]
    n_grid = int(round(duration_src * out_fps))
    grid = [i / out_fps for i in range(n_grid + 1)]
    curves = []
    for pid, ser in sorted(curves_src.items()):
        vals = [float(round(MA.resample_lin(ser, t), 5)) for t in grid]
        seg = [grid[0], vals[0]]
        for i in range(1, len(grid)):
            seg += [0, grid[i], vals[i]]
        curves.append({'Target': 'Parameter', 'Id': pid, 'Segments': seg})

    duration = round(n_grid / out_fps, 6)
    motion = {
        'Version': 3,
        'Meta': {
            'Duration': duration,
            'Fps': float(out_fps),
            'Loop': bool(loop),
            'AreBeziersRestricted': True,
            'CurveCount': len(curves),
            'TotalSegmentCount': len(curves) * max(len(grid) - 1, 0),
            'TotalPointCount': len(curves) * len(grid),
            'UserDataCount': 0,
            'TotalUserDataSize': 0,
        },
        'Curves': curves,
    }
    write_json(out_path, motion)

    report = {
        'csv': os.path.abspath(csv_path),
        'model': model_name,
        'ckpt': os.path.abspath(model_path),
        'out': os.path.abspath(out_path),
        'frames_in': T,
        'duration_src': round(duration_src, 3),
        'channels': sorted(curves_src.keys()),
        'channels_dropped_flat': sorted(dropped),
        'cutoff_hz': cutoff,
        'fps': out_fps,
    }
    return motion, report


def main():
    ap = argparse.ArgumentParser(description='Express4D 推理: CSV → motion3.json')
    ap.add_argument('--csv', required=True, help='ARKit CSV 输入')
    ap.add_argument('--model', required=True, help='Live2D 模型名（用于量程缩放）')
    ap.add_argument('--ckpt', required=True, help='模型权重 .pt')
    ap.add_argument('--out', required=True, help='输出 motion3.json 路径')
    ap.add_argument('--cutoff', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _, rep = infer(
        args.csv, args.ckpt, args.out,
        model_name=args.model, cutoff=args.cutoff,
        out_fps=args.fps, loop=args.loop, device=args.device,
    )
    log(f"motion3 已写出: {rep['out']}")
    log(f"  {rep['duration_src']}s / {rep['frames_in']} 帧 → "
        f"{len(rep['channels'])} 条曲线")
    log(f"  活跃通道: {rep['channels']}")


if __name__ == '__main__':
    main()
