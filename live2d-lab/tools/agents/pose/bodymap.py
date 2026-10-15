#!/usr/bin/env python3
"""bodymap —— 身体点位 CSV → Live2D 身体参数（L1 几何规则）→ motion3.json

通道（canonical 空间，fit_to_model 前）：
  ParamBodyAngleX  躯干侧倾：双髋中点→双肩中点向量 atan2(dx,dy)
  ParamBodyAngleZ  躯干侧转（roll）：肩线与水平线夹角
  ParamBodyAngleY  躯干扭转（透视）：鼻相对肩中点的水平偏移（头先行代理）
  ParamAngleX/Y/Z  头部朝向：鼻/眼/耳几何（与 ARKit 路线独立，供无表情流时使用）
  ParamBreath      呼吸代理：躯干长度低频波动 → 0..1（启发式，幅度受限）

管线：ingest CSV → 逐帧几何 → 8Hz 低通 → canonical 钳制 → fit_to_model
     → 30fps 线性段 motion3（复用 motion_author 的 lowpass/resample_lin 与 Meta 结构）
QA（数值）：每通道 min/max/clip%/速度 p95；超界即 report 标记。

用法：
  /Users/yangfan/miniconda3/bin/python tools/agents/pose/bodymap.py \
      --csv tools/data/poses/mma_demo.csv --model Haru \
      --out tools/data/out/mma_demo.body.motion3.json
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents import arkit_map as AM
from agents.common import TOOLS, log, write_json
from agents.motion_author import lowpass, resample_lin

# 几何符号（QA 目检方向反了只改这里）
BODY_SIGN = {'ParamBodyAngleX': 1.0, 'ParamBodyAngleY': 1.0, 'ParamBodyAngleZ': 1.0}
# Breath 启发式上限幅度（canonical 0..1 内的摆动半幅）
BREATH_AMP = 0.25
# 扭转（BodyAngleY）映射：鼻偏移（躯干尺度单位）→ 度（保守增益防饱和）
TWIST_GAIN = 30.0

I = {'nose': 0, 'leye': 1, 'reye': 2, 'lear': 3, 'rear': 4,
     'lsho': 5, 'rsho': 6, 'lhip': 11, 'rhip': 12}


def load_pose_csv(path):
    """ingest CSV → (times(T,), C(T,17), X(T,17), Y(T,17))"""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    t_col = head.index('t')
    c0, x0, y0 = head.index('c0'), head.index('x0'), head.index('y0')
    t, C, X, Y = [], [], [], []
    for r in rows[1:]:
        t.append(float(r[t_col]))
        C.append([float(v) for v in r[c0:c0 + 17]])
        X.append([float(v) for v in r[x0:x0 + 17]])
        Y.append([float(v) for v in r[y0:y0 + 17]])
    return np.array(t), np.array(C), np.array(X), np.array(Y)


def frame_params(x, y, c, conf_thr=0.3):
    """单帧归一化点位 → (canonical 参数 dict, 躯干长度或 None)"""
    p = {}
    g = c >= conf_thr
    lsho, rsho = I['lsho'], I['rsho']
    lhip, rhip = I['lhip'], I['rhip']

    has_sho = g[lsho] and g[rsho]
    sho_mid = ((x[lsho] + x[rsho]) / 2, (y[lsho] + y[rsho]) / 2) if has_sho else None
    hip_mid = ((x[lhip] + x[rhip]) / 2, (y[lhip] + y[rhip]) / 2) \
        if g[lhip] and g[rhip] else None

    # ── 躯干侧倾（BodyAngleX）：髋中点→肩中点向量的侧偏角
    torso_len = None
    if sho_mid and hip_mid:
        dx = sho_mid[0] - hip_mid[0]
        dy = max(sho_mid[1] - hip_mid[1], 1e-3)
        p['ParamBodyAngleX'] = math.degrees(math.atan2(dx, dy))
        torso_len = math.hypot(dx, dy)  # Breath 用

    # ── 躯干 roll（BodyAngleZ）：肩线角（屏幕系顺时针为正 → Live2D Z 取负）
    if has_sho:
        ang = math.degrees(math.atan2(y[rsho] - y[lsho], x[rsho] - x[lsho]))
        p['ParamBodyAngleZ'] = -ang

    # ── 躯干扭转（BodyAngleY）：鼻水平偏移代理
    if sho_mid and g[I['nose']]:
        p['ParamBodyAngleY'] = (x[I['nose']] - sho_mid[0]) * TWIST_GAIN

    # ── 头部朝向（body17 几何，无 ARKit 时的独立来源）
    nose, leye, reye, lear, rear = (I[k] for k in ('nose', 'leye', 'reye', 'lear', 'rear'))
    ear_w = abs(x[lear] - x[rear]) if g[lear] and g[rear] else 0.0
    if ear_w > 0.20 and g[nose] and g[lear] and g[rear]:  # 头部近正面才可信
        ear_mid = ((x[lear] + x[rear]) / 2, (y[lear] + y[rear]) / 2)
        # yaw：鼻在耳中线法向上的偏移 / 耳距（偏移钳 ±0.5 防侧脸发散）
        off = max(-0.5, min(0.5, (x[nose] - ear_mid[0]) / ear_w))
        p['ParamAngleX'] = -off * 60.0
    if g[leye] and g[reye]:
        # roll：眼线角
        p['ParamAngleZ'] = -math.degrees(math.atan2(y[reye] - y[leye],
                                                    x[reye] - x[leye] + 1e-9))
    if sho_mid and g[nose] and g[leye] and g[reye]:
        eye_mid = ((x[leye] + x[reye]) / 2, (y[leye] + y[reye]) / 2)
        head_len = abs(eye_mid[1] - sho_mid[1]) + 1e-3
        p['ParamAngleY'] = ((y[nose] - eye_mid[1]) / head_len) * 45.0 - 10.0

    return p, torso_len


def bodymap(csv_path, model, out_path, cutoff=8.0, out_fps=30, conf_thr=0.3):
    t, C, X, Y = load_pose_csv(csv_path)
    if len(t) < 5:
        raise ValueError(f'{csv_path}: 帧数不足（{len(t)}）')

    # 1) 逐帧几何 → 通道时间序列
    series = {}
    torso_len = []
    for k in range(len(t)):
        p, tl = frame_params(X[k], Y[k], C[k], conf_thr)
        if tl is not None:
            torso_len.append((t[k], tl))
        for pid, v in p.items():
            series.setdefault(pid, []).append((t[k], v))

    # 2) Breath：躯干长度去均值低通 → 0..1 摆动（幅度受限）
    if len(torso_len) >= 5:
        sm = lowpass(torso_len, cutoff=1.2)  # 呼吸频段 ~0.2-0.5Hz
        vals = np.array([v for _, v in sm])
        dev = vals - vals.mean()
        amp = max(np.abs(dev).std(), 1e-3)
        breath = [0.5 + BREATH_AMP * np.clip(d / (2 * amp), -1, 1) for _, d in sm]
        series['ParamBreath'] = [(tv[0], b) for tv, b in zip(sm, breath)]

    # 3) 低通 + canonical 钳制
    ranges, _ = AM.load_calib(model, TOOLS)
    canon_series = {}
    for pid, ser in series.items():
        ser = lowpass(ser, cutoff)
        std = AM.STD_RANGES.get(pid)
        if std:  # 角度通道：canonical 量程钳制
            ser = [(tt, max(std[0], min(std[1], v * BODY_SIGN.get(pid, 1.0)))) for tt, v in ser]
        else:    # Breath 等非 STD 通道：0..1
            ser = [(tt, max(0.0, min(1.0, v))) for tt, v in ser]
        canon_series[pid] = ser

    # 4) fit_to_model（仅 STD 通道；Breath 直通）→ 30fps motion3
    n_grid = int(round(t[-1] * out_fps))
    grid = [i / out_fps for i in range(n_grid + 1)]
    curves, qa = [], {}
    for pid, ser in sorted(canon_series.items()):
        raw = [resample_lin(ser, tt) for tt in grid]
        if pid in AM.STD_RANGES:
            vals = [AM.fit_to_model({pid: v}, ranges).get(pid, v) for v in raw]
        else:  # Breath 等非 STD 通道直通
            vals = raw
        if max(vals) - min(vals) < 1e-3:
            continue
        seg = [grid[0], vals[0]]
        for i in range(1, len(grid)):
            seg += [0, grid[i], vals[i]]
        curves.append({'Target': 'Parameter', 'Id': pid, 'Segments': seg})
        vs = np.array(vals)
        r = ranges.get(pid, {'min': 0, 'max': 1})
        speed = np.abs(np.diff(vs)) * out_fps
        qa[pid] = {'min': round(float(vs.min()), 2), 'max': round(float(vs.max()), 2),
                   'clip%': round(float(((vs <= r['min'] + 1e-6) |
                                         (vs >= r['max'] - 1e-6)).mean() * 100), 1),
                   'speed_p95': round(float(np.percentile(speed, 95)), 2)}

    motion = {
        'Version': 3,
        'Meta': {
            'Duration': round(n_grid / out_fps, 6), 'Fps': float(out_fps), 'Loop': False,
            'AreBeziersRestricted': True, 'CurveCount': len(curves),
            'TotalSegmentCount': max(len(grid) - 1, 0) * len(curves),
            'TotalPointCount': len(grid) * len(curves),
            'UserDataCount': 0, 'TotalUserDataSize': 0,
        },
        'Curves': curves,
    }
    write_json(out_path, motion)
    report = {'csv': os.path.abspath(csv_path), 'model': model,
              'out': os.path.abspath(out_path),
              'duration_s': round(float(t[-1]), 2), 'frames': len(t),
              'channels': sorted(canon_series.keys()), 'qa': qa,
              'cutoff_hz': cutoff, 'fps': out_fps}
    return motion, report


def main():
    ap = argparse.ArgumentParser(description='身体点位 → Live2D 身体参数 motion3')
    ap.add_argument('--csv', required=True)
    ap.add_argument('--model', default='Haru')
    ap.add_argument('--out', required=True)
    ap.add_argument('--cutoff', type=float, default=8.0)
    ap.add_argument('--fps', type=int, default=30)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    _, rep = bodymap(args.csv, args.model, args.out,
                     cutoff=args.cutoff, out_fps=args.fps)
    log(f"motion3: {rep['out']}  ({rep['duration_s']}s / {rep['frames']} 帧)")
    log(f"通道: {rep['channels']}")
    for pid, q in rep['qa'].items():
        flag = ' ⚠贴量程' if q['clip%'] > 20 else ''
        log(f"  {pid:18s} [{q['min']:7.2f}, {q['max']:7.2f}]  "
            f"clip={q['clip%']:4.1f}%  spd95={q['speed_p95']}{flag}")
    write_json(args.out + '.report.json', rep)


if __name__ == '__main__':
    main()
