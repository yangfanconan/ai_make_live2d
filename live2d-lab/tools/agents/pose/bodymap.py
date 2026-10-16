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

# 本模块扩展通道的 canonical 量程（不在 arkit_map.STDANGES 内，bodymap 自行钳制）
EXT_RANGES = {
    'ParamArmLA': (-10, 10, 0), 'ParamArmRA': (-10, 10, 0),
    'ParamArmLB': (-10, 10, 0), 'ParamArmRB': (-10, 10, 0),
    'ParamHandAngleL': (-30, 30, 0), 'ParamHandAngleR': (-30, 30, 0),
    'ParamBreath': (0, 1, 0),
}
ARM_GAIN = 9.0 / 90.0     # 侧举 0..90° → 0..9（canonical ±10 内）
FOREARM_GAIN = 10.0 / 180.0  # 肘弯曲 0..180° → 0..10
HAND_TILT_GAIN = 30.0 / 45.0  # 腕偏角 ±45° → ±30

# 通道级标定（kinetics-mini 7 片段日常动作实测 QA 的结论）：
# 2D 投影角（眼线/肩线）在人物侧对镜头时会瞬间贴满量程（clip 53-98%、spd95>1000）
# → 降增益 + 降截止；手部关键点在低分辨率视频上噪声大（spd95 160-460）→ 更低截止。
CH_CALIB = {
    'ParamAngleZ':     {'cutoff': 3.0, 'gain': 0.5},
    'ParamBodyAngleZ': {'cutoff': 4.0, 'gain': 0.5},
    'ParamBodyAngleX': {'cutoff': 4.0, 'gain': 0.7},
    'ParamHandAngleL': {'cutoff': 2.5, 'gain': 1.0},
    'ParamHandAngleR': {'cutoff': 2.5, 'gain': 1.0},
}

I = {'nose': 0, 'leye': 1, 'reye': 2, 'lear': 3, 'rear': 4,
     'lsho': 5, 'rsho': 6, 'lelb': 7, 'relb': 8, 'lwri': 9, 'rwri': 10,
     'lhip': 11, 'rhip': 12}


def load_pose_csv(path):
    """ingest CSV → (t, C, X, Y[, CL, XL, YL, CR, XR, YR])；手部列缺省时返回 6 元组"""
    with open(path, newline='') as fh:
        rows = list(csv.reader(fh))
    head = rows[0]
    t_col = head.index('t')
    c0, x0, y0 = head.index('c0'), head.index('x0'), head.index('y0')

    def block(side):
        """side='L'/'R' → (conf列, x列, y列) 索引列表"""
        return ([head.index(f'hc{side}{i}') for i in range(21)],
                [head.index(f'hx{side}{i}') for i in range(21)],
                [head.index(f'hy{side}{i}') for i in range(21)])

    has_hands = 'hcL0' in head
    t, C, X, Y = [], [], [], []
    CL, XL, YL, CR, XR, YR = ([] for _ in range(6))
    if has_hands:
        cl, xl, yl = block('L')
        cr, xr, yr = block('R')
    for r in rows[1:]:
        t.append(float(r[t_col]))
        C.append([float(v) for v in r[c0:c0 + 17]])
        X.append([float(v) for v in r[x0:x0 + 17]])
        Y.append([float(v) for v in r[y0:y0 + 17]])
        if has_hands:
            CL.append([float(r[i]) for i in cl]); XL.append([float(r[i]) for i in xl])
            YL.append([float(r[i]) for i in yl])
            CR.append([float(r[i]) for i in cr]); XR.append([float(r[i]) for i in xr])
            YR.append([float(r[i]) for i in yr])
    out = [np.array(t), np.array(C), np.array(X), np.array(Y)]
    if has_hands:
        out += [np.array(a) for a in (CL, XL, YL, CR, XR, YR)]
    return tuple(out)


def frame_params(x, y, c, conf_thr=0.3, hand=None):
    """单帧归一化点位 → (canonical 参数 dict, 躯干长度或 None)
    hand: None 或 {'L': (hx, hy, hc), 'R': (...)} 各 21 点（0=腕 9=中指根）"""
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
    # 门控：侧对镜头时肩线屏幕角是「朝向投影」非 roll，仅肩横向充分展开（近正面）才采信
    if has_sho and torso_len and abs(x[rsho] - x[lsho]) / torso_len > 0.30:
        ang = math.degrees(math.atan2(y[rsho] - y[lsho], x[rsho] - x[lsho]))
        if ang > 90:    # 180° 折叠：镜像朝向下水平线会得 ±180，规约回小倾角
            ang -= 180
        elif ang < -90:
            ang += 180
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
    if ear_w > 0.20 and g[leye] and g[reye]:
        # roll：眼线角（与 yaw 同门控：侧脸时眼线是朝向投影，非 roll）
        ang = math.degrees(math.atan2(y[reye] - y[leye], x[reye] - x[leye] + 1e-9))
        if ang > 90:    # 180° 折叠：镜像朝向下水平眼线会得 ±180，规约回小倾角
            ang -= 180
        elif ang < -90:
            ang += 180
        p['ParamAngleZ'] = -ang
    if sho_mid and g[nose] and g[leye] and g[reye]:
        eye_mid = ((x[leye] + x[reye]) / 2, (y[leye] + y[reye]) / 2)
        head_len = abs(eye_mid[1] - sho_mid[1]) + 1e-3
        p['ParamAngleY'] = ((y[nose] - eye_mid[1]) / head_len) * 45.0 - 10.0

    # ── 手臂（body17 的肩肘腕；A=上臂侧举，B=前臂弯曲）
    for side, sign, sho_i, elb_i, wri_i, a_id, b_id in (
            ('L', -1, I['lsho'], I['lelb'], I['lwri'], 'ParamArmLA', 'ParamArmLB'),
            ('R', 1, I['rsho'], I['relb'], I['rwri'], 'ParamArmRA', 'ParamArmRB')):
        if g[sho_i] and g[elb_i]:
            vx, vy = x[elb_i] - x[sho_i], y[elb_i] - y[sho_i]
            # 下垂为 0°，侧举（远离躯干方向）为正
            raise_deg = math.degrees(math.atan2(sign * vx, vy if vy > 0 else 0.001))
            p[a_id] = max(0.0, min(90.0, raise_deg if vy > 0 else 90.0)) * ARM_GAIN
            if g[wri_i]:
                ux, uy = x[wri_i] - x[elb_i], y[wri_i] - y[elb_i]
                su = math.hypot(vx, vy) * math.hypot(ux, uy)
                if su > 1e-6:
                    cosv = max(-1.0, min(1.0, (vx * ux + vy * uy) / su))
                    bend = math.degrees(math.acos(cosv))  # 180=伸直 → 0=全折
                    p[b_id] = (180.0 - bend) * FOREARM_GAIN - 5.0

    # ── 手腕（hand21: 0=腕, 9=中指根；腕偏角 vs 前臂方向）
    if hand:
        for side, elb_i, wri_i, pid in (
                ('L', I['lelb'], I['lwri'], 'ParamHandAngleL'),
                ('R', I['relb'], I['rwri'], 'ParamHandAngleR')):
            h = hand.get(side)
            if h is None or not (g[elb_i] and g[wri_i]):
                continue
            hx, hy, hc = h
            if hc[0] < conf_thr or hc[9] < conf_thr:
                continue
            fx, fy = x[wri_i] - x[elb_i], y[wri_i] - y[elb_i]  # 前臂
            px, py = hx[9] - hx[0], hy[9] - hy[0]              # 掌方向
            ang = math.degrees(math.atan2(fx * py - fy * px, fx * px + fy * py))
            p[pid] = max(-45.0, min(45.0, ang)) * HAND_TILT_GAIN

    return p, torso_len


def bodymap(csv_path, model, out_path, cutoff=8.0, out_fps=30, conf_thr=0.3):
    loaded = load_pose_csv(csv_path)
    t, C, X, Y = loaded[:4]
    hands = None
    if len(loaded) == 10:
        _, XL, YL, _, XR, YR = loaded[4:]
        CL, CR = loaded[4], loaded[7]
        hands = {'CL': CL, 'XL': XL, 'YL': YL, 'CR': CR, 'XR': XR, 'YR': YR}
    if len(t) < 5:
        raise ValueError(f'{csv_path}: 帧数不足（{len(t)}）')

    # 1) 逐帧几何 → 通道时间序列
    series = {}
    torso_len = []
    for k in range(len(t)):
        hand = None
        if hands:
            hand = {'L': (hands['XL'][k], hands['YL'][k], hands['CL'][k]),
                    'R': (hands['XR'][k], hands['YR'][k], hands['CR'][k])}
        p, tl = frame_params(X[k], Y[k], C[k], conf_thr, hand=hand)
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

    # 3) 低通 + canonical 钳制（STD 走 arkit_map；EXT 走本模块量程）
    ranges, _ = AM.load_calib(model, TOOLS)
    canon_series = {}
    for pid, ser in series.items():
        cal = CH_CALIB.get(pid, {})
        ch_cutoff = cal.get('cutoff', 4.0 if pid.startswith('ParamHand') else cutoff)
        ser = lowpass(ser, ch_cutoff)
        gain = cal.get('gain', 1.0) * BODY_SIGN.get(pid, 1.0)
        std = AM.STD_RANGES.get(pid) or EXT_RANGES.get(pid)
        if std:
            ser = [(tt, max(std[0], min(std[1], v * gain))) for tt, v in ser]
        else:  # 未知通道兜底 0..1
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
        else:  # EXT 通道：已按 EXT_RANGES 钳制，直通
            vals = raw
        if max(vals) - min(vals) < 1e-3:
            continue
        seg = [grid[0], vals[0]]
        for i in range(1, len(grid)):
            seg += [0, grid[i], vals[i]]
        curves.append({'Target': 'Parameter', 'Id': pid, 'Segments': seg})
        vs = np.array(vals)
        ext = EXT_RANGES.get(pid)
        r = ranges.get(pid) or (
            {'min': ext[0], 'max': ext[1]} if ext else {'min': 0, 'max': 1})
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
