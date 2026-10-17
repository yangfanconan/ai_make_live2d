#!/usr/bin/env python3
"""motion_author —— ARKit CSV（Live Link Face / Express4D）→ Cubism motion3.json

管线：CSV 表头自适应 → 逐行 arkit_map → 12Hz 一阶低通 → 量程钳制（附 clip 统计）
     → 30fps 线性段（segment type 0）→ motion3.json（Meta 自动统计）。

用法：
  python3 tools/agents/motion_author.py --csv take.csv --model Haru \
      --out tools/data/out/blink.motion3.json
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import arkit_map as AM
from agents.common import TOOLS, MODELS, write_json, log


def parse_timecode(tc, fps=60):
    """Live Link Face 形式 HH:MM:SS:FF（末段为帧）或 HH:MM:SS.mmm。返回秒。"""
    tc = tc.strip()
    if ':' not in tc:
        return float(tc)
    parts = tc.split(':')
    if len(parts) == 4:
        h, m, s, f = parts
        return int(h) * 3600 + int(m) * 60 + int(s) + int(f) / float(fps)
    h, m, rest = parts
    return int(h) * 3600 + int(m) * 60 + float(rest)


def load_csv_rows(path, tc_fps=60):
    """返回 [(t_seconds, {原始列名: 字符串值})]，时间列从 Timecode/t/time 自适应。"""
    with open(path, newline='', encoding='utf-8-sig') as fh:
        reader = csv.DictReader(fh)
        headers = reader.fieldnames or []
        tcol = None
        for cand in ('t', 'time', 'Time', 'timestamp'):
            if cand in headers:
                tcol = cand
                break
        if tcol is None and 'Timecode' in headers:
            tcol = 'Timecode'
        if tcol is None:
            raise ValueError(f'CSV 缺少时间列（Timecode/t/time）：{headers[:8]}')
        rows = []
        t0 = None
        for r in reader:
            tv = r.get(tcol, '')
            if tv in (None, ''):
                continue
            t = parse_timecode(tv, tc_fps) if tcol == 'Timecode' else float(tv)
            if t0 is None:
                t0 = t
            rows.append((t - t0, r))
    rows.sort(key=lambda x: x[0])
    return rows


def lowpass(series_tv, cutoff=12.0):
    """一阶 IIR（RC 低通），按真实采样间隔算 alpha。series_tv: [(t,v)]。"""
    if not series_tv:
        return []
    rc = 1.0 / (2 * 3.14159265358979 * cutoff)
    out = []
    y = series_tv[0][1]
    pt = series_tv[0][0]
    for t, x in series_tv:
        dt = max(t - pt, 1e-4)
        alpha = dt / (rc + dt)
        y += alpha * (x - y)
        out.append((t, y))
        pt = t
    return out


def resample_lin(series_tv, t):
    """分段线性采样（端点外保持）。"""
    if t <= series_tv[0][0]:
        return series_tv[0][1]
    if t >= series_tv[-1][0]:
        return series_tv[-1][1]
    for i in range(len(series_tv) - 1):
        t0, v0 = series_tv[i]
        t1, v1 = series_tv[i + 1]
        if t0 <= t <= t1:
            k = (t - t0) / max(t1 - t0, 1e-9)
            return v0 + k * (v1 - v0)
    return series_tv[-1][1]


def author(csv_path, model, out_path, cutoff=12.0, out_fps=30, loop=False,
           tools_dir=TOOLS, activity_eps=1e-3, tc_fps=60):
    ranges, signs = AM.load_calib(model, tools_dir)

    # 1) 逐行映射到模型量程
    rows = load_csv_rows(csv_path, tc_fps=tc_fps)
    if len(rows) < 2:
        raise ValueError(f'CSV 有效行不足：{len(rows)}')
    raw_series = {}  # pid -> [(t, v)]
    for t, row in rows:
        params = AM.map_row_to_model(row, ranges, signs)
        for pid, v in params.items():
            raw_series.setdefault(pid, []).append((t, v))

    duration_src = rows[-1][0]

    # 2) 低通 + 钳制/clip 统计 + 活跃度筛选
    curves_src, clip_stats, dropped = {}, {}, []
    for pid, ser in raw_series.items():
        ser = lowpass(ser, cutoff)
        r = ranges[pid]
        n_clip = 0
        clamped = []
        for t, v in ser:
            if v <= r['min'] + 1e-6 or v >= r['max'] - 1e-6:
                n_clip += 1
            clamped.append((t, max(r['min'], min(r['max'], v))))
        vs = [v for _, v in clamped]
        if max(vs) - min(vs) < activity_eps:
            dropped.append(pid)
            continue
        curves_src[pid] = clamped
        ratio = n_clip / max(len(clamped), 1)
        if ratio > 0.02:
            clip_stats[pid] = round(ratio, 3)

    # 3) 30fps 重采样 → motion3 线性段
    n_grid = int(round(duration_src * out_fps))
    grid = [i / out_fps for i in range(n_grid + 1)]
    curves = []
    total_segments = 0
    total_points = 0
    for pid, ser in sorted(curves_src.items()):
        vals = [round(resample_lin(ser, t), 5) for t in grid]
        seg = [grid[0], vals[0]]
        for i in range(1, len(grid)):
            seg += [0, grid[i], vals[i]]  # 0 = 线性段
        curves.append({'Target': 'Parameter', 'Id': pid, 'Segments': seg})
        total_segments += len(grid) - 1
        total_points += len(grid)

    duration = round(n_grid / out_fps, 6)
    motion = {
        'Version': 3,
        'Meta': {
            'Duration': duration,
            'Fps': float(out_fps),
            'Loop': bool(loop),
            'AreBeziersRestricted': True,
            'CurveCount': len(curves),
            'TotalSegmentCount': total_segments,
            'TotalPointCount': total_points,
            'UserDataCount': 0,
            'TotalUserDataSize': 0,
        },
        'Curves': curves,
    }
    write_json(out_path, motion)

    report = {
        'csv': os.path.abspath(csv_path),
        'model': model,
        'out': os.path.abspath(out_path),
        'duration_src': round(duration_src, 3),
        'frames_in': len(rows),
        'channels': sorted(curves_src.keys()),
        'channels_dropped_flat': sorted(dropped),
        'clip_ratio': clip_stats,
        'cutoff_hz': cutoff,
        'fps': out_fps,
    }
    return motion, report


def main():
    ap = argparse.ArgumentParser(description='ARKit CSV → Cubism motion3.json')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cutoff', type=float, default=12.0)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--loop', action='store_true')
    ap.add_argument('--tc-fps', type=float, default=60,
                    help='HH:MM:SS:FF 时间码的帧率（Live Link Face 导出设置，30fps 导出填 30）')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _, rep = author(args.csv, args.model, args.out,
                    cutoff=args.cutoff, out_fps=args.fps, loop=args.loop,
                    tc_fps=args.tc_fps)
    log(f'motion3 已写出：{rep["out"]}')
    log(f'  时长 {rep["duration_src"]}s / {rep["frames_in"]} 行 → '
        f'{len(rep["channels"])} 条曲线')
    log(f'  活跃通道：{rep["channels"]}')
    if rep['channels_dropped_flat']:
        log(f'  平坦通道已省略：{rep["channels_dropped_flat"]}')
    if rep['clip_ratio']:
        log(f'  ⚠ 贴量程比例 >2%（检查增益/校准）：{rep["clip_ratio"]}')


if __name__ == '__main__':
    main()
