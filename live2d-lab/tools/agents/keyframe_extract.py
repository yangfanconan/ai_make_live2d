#!/usr/bin/env python3
"""关键帧检测 + 关键姿势库生成 —— 阶段 3（自动绑骨）第一块

输入 Live Link Face CSV（ARKit BS），输出：
  1. keyframes.json —— 每参数的关键帧表（onset/peak/release，canonical 空间），
     供后续 keyform 顶点位移生成参考
  2. poses.json   —— 离散关键姿势库（模型量程空间），可直接喂
     qa_director.py --poses 做自动扫描验收

算法（canonical 标准语义空间）：
  - 逐帧 normalize_row → map_frame → 每参数时序 → 12Hz 低通
  - 事件段：|v-def| ≥ 0.12 连续区（<0.15s 间隙合并；段峰值 <0.15 丢弃）
  - 每段取峰值关键帧；时间相近(<0.25s)的跨参数峰值聚成一个姿势
  - 每姿势 = 簇中心时刻的全参数快照 → fit_to_model

用法：
  python3 tools/agents/keyframe_extract.py <csv> [--model Haru] [--out-dir DIR]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import arkit_map as AM
from agents import motion_author as MA
from agents.common import ROOT, TOOLS, log

ACT_THR = 0.12     # 活跃门限（|v-def|，canonical）
GAP_MERGE = 0.15   # 事件段间隙合并（s）
PEAK_MIN = 0.15    # 段峰值最小幅度
CLUSTER_GAP = 0.25 # 姿势聚类时间容差（s）

SHORT = {
    'ParamAngleX': 'yaw', 'ParamAngleY': 'pitch', 'ParamAngleZ': 'roll',
    'ParamEyeLOpen': 'blinkL', 'ParamEyeROpen': 'blinkR',
    'ParamEyeBallX': 'gazeX', 'ParamEyeBallY': 'gazeY',
    'ParamEyeLSmile': 'smileL', 'ParamEyeRSmile': 'smileR',
    'ParamBrowLY': 'browL', 'ParamBrowRY': 'browR',
    'ParamMouthOpenY': 'mouth', 'ParamMouthForm': 'form',
}


def canonical_series(rows):
    """[(t, raw_row)] → {pid: [(t, v)]}（canonical 空间，未滤波）"""
    series = {}
    for t, row in rows:
        bs = AM.normalize_row(row)
        c = AM.map_frame(bs)
        for pid, v in c.items():
            series.setdefault(pid, []).append((t, v))
    return series


def find_events(tv, defv, thr):
    """[(t,v)] → 事件段 [(t_on, t_off, t_peak, v_peak)]；thr 按通道量纲缩放"""
    segs, cur = [], []
    for t, v in tv:
        if abs(v - defv) >= thr:
            cur.append((t, v))
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    # 合并小间隙
    merged = []
    for s in segs:
        if merged and s[0][0] - merged[-1][-1][0] < GAP_MERGE:
            merged[-1].extend(s)
        else:
            merged.append(list(s))
    out = []
    for s in merged:
        peak = max(s, key=lambda x: abs(x[1] - defv))
        if abs(peak[1] - defv) < 1.25 * thr:
            continue
        out.append((s[0][0], s[-1][0], peak[0], peak[1]))
    return out


def extract(csv_path, model, out_dir):
    rows = MA.load_csv_rows(csv_path)
    series = canonical_series(rows)
    # 12Hz 低通（复用 motion_author）
    series = {pid: MA.lowpass(tv) for pid, tv in series.items()}
    ranges, _signs = AM.load_calib(model, TOOLS)
    dur = rows[-1][0]

    # ── 关键帧表 ──
    keyframes, peaks = {}, []  # peaks: (t, pid, v)
    for pid, tv in sorted(series.items()):
        smin, smax, defv = AM.STD_RANGES[pid]
        # 门限按通道量程缩放：0..1 面部通道 0.12；±30° 头部 7.2°（抗传感器噪声）
        thr = ACT_THR * (smax - smin)
        evs = find_events(tv, defv, thr)
        if not evs:
            continue
        keyframes[pid] = [
            {'t_on': round(a, 3), 't_off': round(b, 3),
             't_peak': round(tp, 3), 'v_peak': round(vp, 3), 'type': 'peak_seg'}
            for a, b, tp, vp in evs]
        for a, b, tp, vp in evs:
            peaks.append((tp, pid, vp))

    # ── 姿势聚类 ──
    peaks.sort()
    clusters, cur = [], [peaks[0]] if peaks else []
    for p in peaks[1:]:
        if p[0] - cur[-1][0] < CLUSTER_GAP:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    if cur:
        clusters.append(cur)

    def snap(t):
        """t 时刻全参数快照（模型量程）"""
        c = {pid: MA.resample_lin(tv, t) for pid, tv in series.items()}
        return AM.fit_to_model(c, ranges)

    poses = []
    for i, cl in enumerate(clusters, 1):
        t_mid = sum(p[0] for p in cl) / len(cl)
        # 主参数 = 偏离默认最大者
        main = max(cl, key=lambda p: abs(p[2] - AM.STD_RANGES[p[1]][2]))
        tag = SHORT.get(main[1], main[1])
        poses.append({'name': f'pose_{i:02d}_{tag}_{t_mid:.1f}s',
                      't': round(t_mid, 2), 'params': snap(t_mid),
                      'drivers': sorted({p[1] for p in cl})})

    os.makedirs(out_dir, exist_ok=True)
    kf_path = os.path.join(out_dir, 'keyframes.json')
    ps_path = os.path.join(out_dir, 'poses.json')
    json.dump({'source': os.path.basename(csv_path), 'model': model,
               'duration_s': round(dur, 2), 'params': keyframes},
              open(kf_path, 'w'), ensure_ascii=False, indent=1)
    json.dump(poses, open(ps_path, 'w'), ensure_ascii=False, indent=1)

    log(f'{csv_path} → {dur:.1f}s, {len(keyframes)} 活跃参数, '
        f'{sum(len(v) for v in keyframes.values())} 关键帧, {len(poses)} 个姿势')
    for p in poses:
        log(f"  {p['name']}  drivers: {','.join(SHORT.get(d, d) for d in p['drivers'])}")
    log(f'输出: {kf_path}\n      {ps_path}')
    return poses


def main():
    ap = argparse.ArgumentParser(description='关键帧检测 + 关键姿势库生成')
    ap.add_argument('csv', help='Live Link Face CSV')
    ap.add_argument('--model', default='Haru')
    ap.add_argument('--out-dir', default=None)
    args = ap.parse_args()
    out = args.out_dir or os.path.join(ROOT, 'tools', 'data', 'poses')
    extract(args.csv, args.model, out)


if __name__ == '__main__':
    main()
