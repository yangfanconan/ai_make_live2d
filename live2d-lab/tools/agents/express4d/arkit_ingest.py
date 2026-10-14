#!/usr/bin/env python3
"""KeyframeFace 开源 ARKit 数据接入 —— 外部数据集 → 本项目 L2 训练 CSV

数据源（CC-BY-NC-4.0，仅本地训练用、产物 gitignore、不再分发）：
  KeyframeFace Dataset（arXiv:2512.11321，ModelScope: chaochaooooooo/KeyframeFace_Dataset）
  raw_data/subject_XX/script_XXXX/script_XXXX.csv
    列：Timecode + 52 blendshape（PascalCase，0..1）+ 9 姿态（HeadYaw/Pitch/Roll
    与双眼 Yaw/Pitch/Roll，弧度量级，实测 p100 |yaw|≈0.36 rad，未削顶 → 弧度）

本项目 L2 管线消费格式（dataset.load_take → load_csv_rows + arkit_map.normalize_row）：
  t（float 秒，load_csv_rows 首选时间列）+ ARKIT_FEATURES 的 31 列
  （28 BS camelCase + headYaw/headPitch/headRoll 弧度）

转换要点：
  * 帧率 60fps（原始 Timecode 的帧字段逐行 +1），t = 行号 / 60 —— 显式重造时间轴，
    不解析其混合格式时间码
  * PascalCase → camelCase 由 arkit_map.normalize_row 的别名机制天然兼容；
    这里仍显式写出 camelCase，令下游与外部格式解耦
  * 质量过滤：最短帧数、有效数值列、BS 至少一处活动（防全零坏 take）
  * 眼 Look 通道沿用其 blendshape（EyeLookInLeft 等），不使用 9 维姿态里的眼睛欧拉角

用法：
  python3 tools/agents/express4d/arkit_ingest.py \
      --src "tools/data/express4d/raw/keyframeface/extracted/KeyframeFace Dataset/raw_data" \
      --subjects 8 --per-subject 12
  产物：tools/data/express4d/arkit/kf_<subject>_<script>.csv（gitignore 目录）
"""
import argparse
import csv
import glob
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log, write_json
from agents.express4d.dataset import ARKIT_FEATURES

DEFAULT_SRC = os.path.join(TOOLS, 'data', 'express4d', 'raw', 'keyframeface',
                           'extracted', 'KeyframeFace Dataset', 'raw_data')
DEFAULT_OUT = os.path.join(TOOLS, 'data', 'express4d', 'arkit')
FPS = 60.0

# KeyframeFace PascalCase → 本项目 camelCase（ARKIT_FEATURES 名）
PASCAL = {
    'eyeBlinkLeft': 'EyeBlinkLeft', 'eyeBlinkRight': 'EyeBlinkRight',
    'eyeSquintLeft': 'EyeSquintLeft', 'eyeSquintRight': 'EyeSquintRight',
    'eyeLookInLeft': 'EyeLookInLeft', 'eyeLookInRight': 'EyeLookInRight',
    'eyeLookOutLeft': 'EyeLookOutLeft', 'eyeLookOutRight': 'EyeLookOutRight',
    'eyeLookUpLeft': 'EyeLookUpLeft', 'eyeLookUpRight': 'EyeLookUpRight',
    'eyeLookDownLeft': 'EyeLookDownLeft', 'eyeLookDownRight': 'EyeLookDownRight',
    'cheekSquintLeft': 'CheekSquintLeft', 'cheekSquintRight': 'CheekSquintRight',
    'browInnerUp': 'BrowInnerUp', 'browOuterUpLeft': 'BrowOuterUpLeft',
    'browOuterUpRight': 'BrowOuterUpRight',
    'browDownLeft': 'BrowDownLeft', 'browDownRight': 'BrowDownRight',
    'jawOpen': 'JawOpen', 'mouthClose': 'MouthClose',
    'mouthSmileLeft': 'MouthSmileLeft', 'mouthSmileRight': 'MouthSmileRight',
    'mouthFrownLeft': 'MouthFrownLeft', 'mouthFrownRight': 'MouthFrownRight',
    'mouthFunnelLeft': 'MouthFunnelLeft', 'mouthFunnelRight': 'MouthFunnelRight',
    'mouthPuckerLeft': 'MouthPuckerLeft', 'mouthPuckerRight': 'MouthPuckerRight',
    'headYaw': 'HeadYaw', 'headPitch': 'HeadPitch', 'headRoll': 'HeadRoll',
}


# KeyframeFace 的 funnel/pucker 是合并单列（无左右分侧）→ 双侧共用
MERGED = {'mouthFunnelLeft': 'MouthFunnel', 'mouthFunnelRight': 'MouthFunnel',
          'mouthPuckerLeft': 'MouthPucker', 'mouthPuckerRight': 'MouthPucker'}


def find_takes(src):
    """扫描 subject_XX/script_XXXX/script_XXXX.csv"""
    takes = []
    for p in sorted(glob.glob(os.path.join(src, 'subject_*', 'script_*', '*.csv'))):
        script_dir = os.path.basename(os.path.dirname(p))
        subject = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if os.path.basename(p) == f'{script_dir}.csv':
            takes.append((subject, script_dir, p))
    return takes


def convert_take(path, out_path, min_rows):
    """单 take 转换；返回统计 dict 或 None（被过滤）"""
    with open(path, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) < min_rows:
        return None

    header = set(rows[0].keys())
    cols = {}
    for feat in ARKIT_FEATURES:
        if PASCAL[feat] in header:
            cols[feat] = PASCAL[feat]
        elif MERGED.get(feat) and MERGED[feat] in header:
            cols[feat] = MERGED[feat]
        else:
            return None  # 缺关键列，跳过该 take
    out_rows = []
    spans = {f: [float('inf'), float('-inf')] for f in
             ('eyeBlinkLeft', 'jawOpen', 'mouthSmileLeft', 'browInnerUp')}
    for i, r in enumerate(rows):
        rec = {'t': round(i / FPS, 6)}
        ok = True
        for feat in ARKIT_FEATURES:
            v = r.get(cols[feat], '')
            try:
                rec[feat] = f'{float(v):.6f}'
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        for f, (lo, hi) in spans.items():
            v = float(rec[f])
            spans[f] = (min(lo, v), max(hi, v))
        out_rows.append(rec)
    if len(out_rows) < min_rows:
        return None

    bs_span_max = max(hi - lo for lo, hi in spans.values())
    if bs_span_max < 0.05:  # 全程近乎静止的坏 take
        return None

    with open(out_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['t'] + ARKIT_FEATURES)
        w.writeheader()
        w.writerows(out_rows)

    return {
        'frames': len(out_rows),
        'duration_s': round(len(out_rows) / FPS, 2),
        'bs_span_max': round(bs_span_max, 4),
        'head_yaw_absmax': round(max(abs(float(r['headYaw'])) for r in out_rows), 4),
    }


def main():
    ap = argparse.ArgumentParser(description='KeyframeFace ARKit 数据接入 L2')
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--subjects', type=int, default=8, help='取前 N 个 subject')
    ap.add_argument('--per-subject', type=int, default=12, help='每 subject 随机取 M 条')
    ap.add_argument('--min-rows', type=int, default=120, help='最短帧数（2s@60fps）')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    takes = find_takes(args.src)
    if not takes:
        ap.error(f'未发现 take CSV：{args.src}')
    subjects = sorted({s for s, _, _ in takes})[:args.subjects]
    log(f'发现 {len(takes)} takes / {len({s for s, _, _ in takes})} subjects，'
        f'选用 {len(subjects)}: {subjects[0]}…{subjects[-1]}')

    os.makedirs(args.out, exist_ok=True)
    summary, skipped = [], 0
    for sub in subjects:
        pool = [t for t in takes if t[0] == sub]
        chosen = rng.sample(pool, min(args.per_subject, len(pool)))
        for subject, script, path in chosen:
            name = f'kf_{subject}_{script}.csv'
            st = convert_take(path, os.path.join(args.out, name), args.min_rows)
            if st is None:
                skipped += 1
                continue
            st.update(take=name, subject=subject)
            summary.append(st)

    total_frames = sum(s['frames'] for s in summary)
    yaw_absmax = max(s['head_yaw_absmax'] for s in summary)
    log(f'转换 {len(summary)} takes（过滤 {skipped}）/ {total_frames} 帧 / '
        f'{total_frames / FPS / 60:.1f} min；|headYaw|max={yaw_absmax:.3f} rad')
    write_json(os.path.join(args.out, 'ingest_summary.json'), {
        'source': 'KeyframeFace Dataset (arXiv:2512.11321, CC-BY-NC-4.0)',
        'src_dir': args.src, 'n_takes': len(summary), 'n_frames': total_frames,
        'fps': FPS, 'subjects': subjects, 'skipped': skipped, 'takes': summary,
    })
    log(f'摘要 → {os.path.join(args.out, "ingest_summary.json")}')


if __name__ == '__main__':
    main()
