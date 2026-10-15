#!/usr/bin/env python3
"""pose_ingest —— 视频 → DWPose(RTMW) 17 身体关键点 → 归一化 CSV（与 arkit_ingest 同构）

模型：rtmlib Wholebody（YOLOX-m 检测 + RTMW-DW 133 点，取前 17 COCO 身体点）
加速：onnxruntime CoreMLExecutionProvider（GPU/ANE，M5 Pro 实测 4.4×）；
      不可用时静默回退 CPU。
归一化：原点 = 双髋中点（低置信回退双肩中点），尺度 = 原点到双肩中点距离，
        y 轴翻转为向上为正 → 与 Live2D 语义一致（抬头 = y 增大）。
质量门控：conf < 阈值的关键点置 NaN → 逐列线性插值（端点保持）；
          全帧无检出则丢弃该帧并计入 summary。

CSV 列：t, c0..c16（置信度）, x0..x16, y0..y16（归一化坐标，y 向上）
用法：
  /Users/yangfan/miniconda3/bin/python tools/agents/pose/ingest.py \
      --video clip.mp4 --name mytake
产物：tools/data/poses/<name>.csv + <name>_summary.json（gitignore 目录）
"""
import argparse
import csv
import math
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log, write_json

DEFAULT_OUT = os.path.join(TOOLS, 'data', 'poses')

# COCO17 顺序（rtmlib Wholebody 输出前 17 维）
COCO17 = ['nose', 'leye', 'reye', 'lear', 'rear',
          'lsho', 'rsho', 'lelb', 'relb', 'lwri', 'rwri',
          'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
KP = {n: i for i, n in enumerate(COCO17)}


def _patch_coreml():
    """monkey-patch onnxruntime 会话注入 CoreML EP；失败返回 False（回退 CPU）"""
    try:
        import onnxruntime as ort
        if 'CoreMLExecutionProvider' not in ort.get_available_providers():
            return False
        orig = ort.InferenceSession.__init__

        def patched(self, *a, **kw):
            kw['providers'] = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
            orig(self, *a, **kw)

        ort.InferenceSession.__init__ = patched
        return True
    except Exception:
        return False


def pick_person(kps, scores):
    """多人时取 bbox 面积最大者；返回 (kp17(17,2) 像素, conf(17,)) 或 None"""
    if not len(kps):
        return None
    areas = []
    for k in kps:
        v = k[:17]
        areas.append((v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min()))
    i = int(np.argmax(areas))
    return kps[i][:17].copy(), scores[i][:17].copy()


def normalize_frame(kp, conf, conf_thr):
    """像素坐标 → 归一化（原点双髋中点/尺度躯干长，y 向上）。
    返回 (xn(17,), yn(17,), conf(17,))，低置信点为 NaN。"""
    good = conf >= conf_thr
    mh = good[KP['lhip']] and good[KP['rhip']]
    ms = good[KP['lsho']] and good[KP['rsho']]
    if mh:
        ox = (kp[KP['lhip'], 0] + kp[KP['rhip'], 0]) / 2
        oy = (kp[KP['lhip'], 1] + kp[KP['rhip'], 1]) / 2
        sx = (kp[KP['lsho'], 0] + kp[KP['rsho'], 0]) / 2
        sy = (kp[KP['lsho'], 1] + kp[KP['rsho'], 1]) / 2
    elif ms:  # 髋不可见（半身镜头）→ 用肩作原点、肩宽作尺度
        ox = (kp[KP['lsho'], 0] + kp[KP['rsho'], 0]) / 2
        oy = (kp[KP['lsho'], 1] + kp[KP['rsho'], 1]) / 2
        sx = kp[KP['lsho'], 0]
        sy = kp[KP['lsho'], 1]
    else:
        return None
    scale = math.hypot(sx - ox, sy - oy)
    if scale < 1e-6:
        return None
    xn = (kp[:, 0] - ox) / scale
    yn = -(kp[:, 1] - oy) / scale  # 图像 y 向下 → 翻转为向上
    xn = np.where(good, xn, np.nan)
    yn = np.where(good, yn, np.nan)
    return xn, yn, conf


def interp_nans(mat):
    """(T, K) 逐列线性插值 NaN（端点保持）"""
    out = mat.copy()
    T = out.shape[0]
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = np.arange(T)
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            col[:] = 0.0
        elif ok.sum() < T:
            col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        out[:, j] = col
    return out


def main():
    ap = argparse.ArgumentParser(description='视频 → DWPose 身体点位 CSV')
    ap.add_argument('--video', required=True)
    ap.add_argument('--name', default=None, help='输出名（默认视频文件名去扩展名）')
    ap.add_argument('--out-dir', default=DEFAULT_OUT)
    ap.add_argument('--conf', type=float, default=0.30, help='关键点置信度门控')
    ap.add_argument('--stride', type=int, default=1, help='帧间隔（>1 为降采样）')
    ap.add_argument('--max-frames', type=int, default=0, help='0 = 不限')
    args = ap.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.video))[0]
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f'{name}.csv')

    accel = _patch_coreml()
    from rtmlib import Wholebody  # noqa: E402（patch 后 import，会话才走 CoreML）
    wb = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')
    log(f'DWPose 就绪（{"CoreML GPU/ANE" if accel else "CPU"}）')

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rows_raw, t_idx, miss = [], [], 0
    t0 = time.time()
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.stride == 0:
            try:
                kps, scores = wb(frame)
            except Exception:
                kps, scores = np.empty(0), np.empty(0)
            sel = pick_person(kps, scores)
            if sel is not None:
                r = normalize_frame(*sel, args.conf)
                if r is not None:
                    rows_raw.append(r)
                    t_idx.append(fi / fps)
                else:
                    miss += 1
            else:
                miss += 1
        fi += 1
        if args.max_frames and len(rows_raw) >= args.max_frames:
            break
    cap.release()
    dt = time.time() - t0

    if len(rows_raw) < 5:
        log(f'有效帧不足（{len(rows_raw)}），检查视频/阈值')
        sys.exit(1)

    X = interp_nans(np.stack([r[0] for r in rows_raw]))  # (T,17)
    Y = interp_nans(np.stack([r[1] for r in rows_raw]))
    C = np.stack([r[2] for r in rows_raw])
    T = len(rows_raw)

    with open(csv_path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['t'] + [f'c{i}' for i in range(17)]
                   + [f'x{i}' for i in range(17)] + [f'y{i}' for i in range(17)])
        for k in range(T):
            w.writerow([f'{t_idx[k]:.4f}']
                       + [f'{v:.4f}' for v in C[k]]
                       + [f'{v:.4f}' for v in X[k]]
                       + [f'{v:.4f}' for v in Y[k]])

    gated = float((C < args.conf).mean())
    summary = {
        'video': os.path.abspath(args.video),
        'csv': os.path.abspath(csv_path),
        'res': f'{W}x{H}', 'src_fps': round(fps, 2), 'stride': args.stride,
        'frames_total': fi, 'frames_kept': T, 'frames_miss': miss,
        'duration_s': round(t_idx[-1], 2),
        'conf_thr': args.conf,
        'conf_mean_body17': round(float(np.nanmean(np.where(C >= args.conf, C, np.nan))), 3),
        'kpts_gated_ratio': round(gated, 3),
        'accel': 'coreml' if accel else 'cpu',
        'wall_s': round(dt, 1),
        'eff_fps': round(T / dt, 1),
    }
    write_json(os.path.join(args.out_dir, f'{name}_summary.json'), summary)
    log(f"{T}/{fi} 帧保留（miss {miss}）/ {summary['duration_s']}s → {csv_path}")
    log(f"吞吐 {summary['eff_fps']} fps / conf 门控率 {gated * 100:.0f}%")


if __name__ == '__main__':
    main()
