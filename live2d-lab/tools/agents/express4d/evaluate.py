"""Express4D 评估 —— L1（规则映射）vs L2（学习模型）对比报告

逐参数计算 MAE / RMSE / R²，输出 JSON + 可选 contact sheet。

用法：
  python3 tools/agents/express4d/evaluate.py \
      --csv tools/data/synth/synth_blink.csv \
      --ckpt tools/data/express4d/models/express4d_v1.pt \
      --model Haru
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents import arkit_map as AM
from agents.common import TOOLS, log, write_json
from agents.express4d.dataset import load_take, unit_to_canonical, LIVE2D_PARAMS, N_OUT
from agents.express4d.model import load_model


def evaluate(csv_path, ckpt_path, model_name='Haru', device='cpu'):
    """对比 L1 vs L2 在给定 CSV 上的逐参数误差"""
    # 数据
    feats, l1_targets, times = load_take(csv_path)
    T = len(feats)

    # L1 教师标签（canonical 归一化空间，已经是 l1_targets）
    l1_unit = l1_targets  # (T, N_OUT)

    # L2 模型推理
    net, _ = load_model(ckpt_path, device=device)
    net.eval()
    window = net.window
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
    windows = np.stack(windows)

    with torch.no_grad():
        x = torch.from_numpy(windows).to(device).float()
        l2_unit = net.forward_seq(x).cpu().numpy()  # (T, N_OUT)

    # 逐参数统计
    per_param = {}
    for i, pid in enumerate(LIVE2D_PARAMS):
        l1_col = l1_unit[:, i]
        l2_col = l2_unit[:, i]
        mae = float(np.mean(np.abs(l2_col - l1_col)))
        rmse = float(np.sqrt(np.mean((l2_col - l1_col) ** 2)))
        # R²（以 L1 为 ground truth）；常量通道（std<1e-4）无意义，标 None
        l1_std = float(l1_col.std())
        if l1_std < 1e-4:
            r2 = None
        else:
            ss_res = float(np.sum((l2_col - l1_col) ** 2))
            ss_tot = float(np.sum((l1_col - l1_col.mean()) ** 2)) + 1e-9
            r2 = 1.0 - ss_res / ss_tot
        per_param[pid] = {
            'mae': round(mae, 6),
            'rmse': round(rmse, 6),
            'r2': round(r2, 4) if r2 is not None else None,
            'l1_mean': round(float(l1_col.mean()), 4),
            'l1_std': round(float(l1_col.std()), 4),
            'l2_mean': round(float(l2_col.mean()), 4),
            'l2_std': round(float(l2_col.std()), 4),
        }

    # 全局统计
    global_mae = float(np.mean(np.abs(l2_unit - l1_unit)))
    global_rmse = float(np.sqrt(np.mean((l2_unit - l1_unit) ** 2)))

    return {
        'csv': os.path.basename(csv_path),
        'model': model_name,
        'ckpt': os.path.basename(ckpt_path),
        'frames': T,
        'global': {
            'mae': round(global_mae, 6),
            'rmse': round(global_rmse, 6),
        },
        'per_param': per_param,
        'note': 'MAE/RMSE 在 [0,1] canonical 归一化空间计算；R² 以 L1 教师为基准',
    }


def main():
    ap = argparse.ArgumentParser(description='Express4D L1 vs L2 对比')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--model', default='Haru')
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    ap.add_argument('--out', default=None, help='报告输出路径')
    args = ap.parse_args()

    report = evaluate(args.csv, args.ckpt, args.model, args.device)

    log(f"评估: {report['csv']} ({report['frames']} 帧)")
    log(f"  全局 MAE={report['global']['mae']:.6f}  RMSE={report['global']['rmse']:.6f}")
    log(f"  逐参数:")
    for pid, s in report['per_param'].items():
        if s['mae'] > 0.001 or (s['r2'] is not None and s['r2'] < 0.99):
            r2_str = f"{s['r2']:.3f}" if s['r2'] is not None else "N/A"
            log(f"    {pid:20s}  MAE={s['mae']:.4f}  R²={r2_str}  "
                f"L1[{s['l1_mean']:.2f}±{s['l1_std']:.2f}] L2[{s['l2_mean']:.2f}±{s['l2_std']:.2f}]")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        write_json(args.out, report)
        log(f"报告 → {args.out}")


if __name__ == '__main__':
    main()
