"""Script2Motion —— L3 全链路：文本关键帧序列 → Live2D motion3.json

链路：
  关键帧文本 ──(text2pose)──▶ 61 维 ARKit ──(bridge)──▶ 31 特征姿势
      ──(余弦缓动插值 + 中性引入/驻留)──▶ ARKit CSV(60fps)
      ──(express4d v5)──▶ Live2D canonical ──(fit_to_model)──▶ motion3.json

用法：
  # 剧本驱动（KeyframeFace Script.csv 的关键帧描述 + 帧号）
  /Users/yangfan/miniconda3/bin/python tools/agents/text2pose/script2motion.py \
      --script-id 7 --model Haru

  # 自由文本关键帧
  ... --kf "a warm gentle smile@1.0" --kf "brows furrow in thought@3.0"

  # 评估模式：预测姿势 vs raw take 对应帧（31 特征 MAE，含 mean 基线对照）
  ... --script-id 7 --eval
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log, write_json
from agents.express4d.dataset import ARKIT_FEATURES
from agents.express4d import infer as e4d_infer
from agents.express4d.arkit_ingest import PASCAL, MERGED
from agents.text2pose.bridge import pose61_to_arkit31, clip01_row
from agents.text2pose.dataset_t2p import KF_BASE
from agents.text2pose.infer_t2p import Text2Pose

MODELS_DIR = os.path.join(TOOLS, 'data', 'express4d', 'models')
OUT_DIR = os.path.join(TOOLS, 'data', 'out')
SCRIPT_CSV = os.path.join(KF_BASE, 'Script.csv')
RAW_BASE = os.path.join(TOOLS, 'data', 'express4d', 'raw', 'keyframeface',
                        'extracted', 'KeyframeFace Dataset', 'raw_data')
SRC_FPS = 60.0

NEUTRAL = {f: 0.0 for f in ARKIT_FEATURES}

KF_COLS = [(1, 'Keyframe One Description_ARKit', 'frameone_num'),
           (2, 'Keyframe Two Description_ARKit', 'frametwo_num'),
           (3, 'Keyframe Three Description_ARKit', 'framethree_num'),
           (4, 'Keyframe Four Description_ARKit', 'framefour_num'),
           (5, 'Keyframe Five Description_ARKit', 'framefive_num')]


def load_script_keyframes(script_id):
    """Script.csv → [(t_sec, text)]（60fps 帧号 → 秒）"""
    with open(SCRIPT_CSV, encoding='utf-8-sig') as fh:
        for r in csv.DictReader(fh):
            if int(r['Script Number']) == script_id:
                kfs = []
                for _, text_col, num_col in KF_COLS:
                    text, num = r.get(text_col, ''), r.get(num_col, '')
                    if text and text != 'None' and num and num != 'None':
                        kfs.append((int(num) / SRC_FPS, text))
                return kfs, r.get('Emotion Description', '')
    raise ValueError(f'Script Number {script_id} 不存在')


def raw_take_path(script_id):
    sub, scr = (script_id - 1) // 100, (script_id - 1) % 100 + 1
    return os.path.join(RAW_BASE, f'subject_{sub:02d}',
                        f'script_{scr:04d}', f'script_{scr:04d}.csv')


def build_timeline(poses, tail_hold=1.0, lead_min=0.5):
    """[(t, pose31)] → 完整时间轴（中性引入 + 尾驻留），返回 [(t, pose31)]"""
    tl = [(t, p) for t, p in poses]
    if tl[0][0] > lead_min:
        tl = [(0.0, NEUTRAL)] + tl
    tl.append((tl[-1][0] + tail_hold, tl[-1][1]))
    return tl


def sample_csv_rows(timeline, fps=SRC_FPS):
    """姿势时间轴 → 余弦缓动逐帧行 [{'t', feat...}]"""
    rows = []
    t_end = timeline[-1][0]
    n = int(round(t_end * fps)) + 1
    seg = 0
    for i in range(n):
        t = i / fps
        while seg < len(timeline) - 2 and t > timeline[seg + 1][0]:
            seg += 1
        t0, p0 = timeline[seg]
        t1, p1 = timeline[seg + 1]
        u = 0.0 if t1 <= t0 else min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        w = 0.5 * (1.0 - math.cos(math.pi * u))  # 余弦缓动
        row = {'t': round(t, 6)}
        for f in ARKIT_FEATURES:
            row[f] = f'{p0[f] + (p1[f] - p0[f]) * w:.6f}'
        rows.append(row)
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['t'] + ARKIT_FEATURES)
        w.writeheader()
        w.writerows(rows)


def _raw_take_to_31(path):
    """KeyframeFace raw CSV → (features [T,31], times[T])，不依赖 motion_author"""
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    times = np.arange(len(rows), dtype=np.float32) / SRC_FPS
    feats = np.zeros((len(rows), len(ARKIT_FEATURES)), dtype=np.float32)
    for i, r in enumerate(rows):
        for j, feat in enumerate(ARKIT_FEATURES):
            if feat in ('headYaw', 'headPitch', 'headRoll'):
                src = 'Head' + feat[4:].capitalize()
                feats[i, j] = float(r.get(src, 0.0))
            else:
                src = PASCAL.get(feat, feat)
                v = r.get(src)
                if v is None and feat in MERGED:
                    v = r.get(MERGED[feat])
                feats[i, j] = float(v or 0.0)
    return feats, times


def eval_script(t2p, script_id, kfs):
    """预测姿势 vs raw take 对应帧（31 特征 MAE）；mean 基线对照"""
    path = raw_take_path(script_id)
    if not os.path.exists(path):
        log(f'[eval] raw take 不存在: {path}')
        return None
    feats, times = _raw_take_to_31(path)  # (T,31) 已 camelCase/弧度
    mean31 = clip01_row(pose61_to_arkit31(t2p.predict('neutral expressionless face')))
    diffs, base = [], []
    for t, text in kfs:
        fi = int(np.argmin(np.abs(times - t)))
        gt = feats[fi]
        pred = t2p.predict_arkit31(text)
        pv = np.array([pred[f] for f in ARKIT_FEATURES], dtype=np.float32)
        mv = np.array([mean31[f] for f in ARKIT_FEATURES], dtype=np.float32)
        diffs.append(np.abs(pv - gt))
        base.append(np.abs(mv - gt))
    mae = float(np.concatenate(diffs).mean())
    mae_base = float(np.concatenate(base).mean())
    return {'mae': round(mae, 4), 'baseline_mae': round(mae_base, 4),
            'n_keyframes': len(kfs)}


def main():
    ap = argparse.ArgumentParser(description='L3 文本剧本 → motion3.json')
    ap.add_argument('--script-id', type=int, default=None)
    ap.add_argument('--kf', action='append', default=[],
                    help='"文本@秒" 自由关键帧（可多个）')
    ap.add_argument('--model', default='Haru', help='Live2D 模型名')
    ap.add_argument('--t2p', default=os.path.join(MODELS_DIR, 'text2pose_v1.pt'))
    ap.add_argument('--e4d', default=os.path.join(MODELS_DIR, 'express4d_v5.pt'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--name', default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--eval', action='store_true')
    args = ap.parse_args()

    if not args.script_id and not args.kf:
        ap.error('需 --script-id 或 --kf')

    t2p = Text2Pose(args.t2p, device=args.device)
    if args.script_id:
        kfs, emotion = load_script_keyframes(args.script_id)
        log(f'剧本 #{args.script_id}「{emotion}」: {len(kfs)} 关键帧')
        name = args.name or f'script_{args.script_id:04d}'
    else:
        kfs = []
        for spec in args.kf:
            text, _, ts = spec.rpartition('@')
            kfs.append((float(ts), text))
        kfs.sort()
        emotion = ''
        name = args.name or 'free'

    # 文本 → 姿势
    poses = [(t, t2p.predict_arkit31(text)) for t, text in kfs]
    for (t, text), (_, p) in zip(kfs, poses):
        active = {k: round(v, 2) for k, v in p.items() if abs(v) >= 0.08}
        log(f'  kf@{t:5.2f}s {active}  «{text[:60]}»')

    # 时间轴 → CSV
    rows = sample_csv_rows(build_timeline(poses))
    csv_path = os.path.join(OUT_DIR, f't2p_{name}.csv')
    write_csv(csv_path, rows)
    log(f'ARKit CSV: {csv_path}  ({len(rows)} 帧 / {rows[-1]["t"]:.1f}s)')

    # CSV → motion3
    out_path = args.out or os.path.join(OUT_DIR, f't2p_{name}.motion3.json')
    _, rep = e4d_infer.infer(csv_path, args.e4d, out_path,
                             model_name=args.model, device=args.device)
    log(f"motion3: {rep['out']}  ({rep['duration_src']}s, "
        f"{len(rep['channels'])} 通道: {rep['channels']})")

    if args.eval and args.script_id:
        res = eval_script(t2p, args.script_id, kfs)
        if res:
            log(f"[eval] 关键帧 31 特征 MAE={res['mae']} "
                f"(mean 基线 {res['baseline_mae']})")
            write_json(out_path + '.eval.json', res)


if __name__ == '__main__':
    main()
