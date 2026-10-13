#!/usr/bin/env python3
"""合成 ARKit 采集夹具（Live Link Face 表头 + Timecode），用于无真机/无 Express4D
时的管线回归。输出 tools/data/synth/<name>.csv，60fps。

三段标准验收序列：
  1. blink       连续眨眼（3 次）
  2. open_smile  张嘴 + 笑（jaw/smile/squint 组合）
  3. turn_brow   左右转头 + 点头 + 挑眉
"""
import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.common import ROOT, log

OUT_DIR = os.path.join(ROOT, 'tools', 'data', 'synth')
FPS = 60

COLS = ['Timecode',
        'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookInLeft', 'eyeLookInRight',
        'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight',
        'eyeLookDownLeft', 'eyeLookDownRight', 'eyeSquintLeft', 'eyeSquintRight',
        'cheekSquintLeft', 'cheekSquintRight', 'browInnerUp',
        'browOuterUpLeft', 'browOuterUpRight', 'browDownLeft', 'browDownRight',
        'jawOpen', 'mouthClose', 'mouthFunnelLeft', 'mouthFunnelRight',
        'mouthPuckerLeft', 'mouthPuckerRight', 'mouthSmileLeft', 'mouthSmileRight',
        'mouthFrownLeft', 'mouthFrownRight', 'headYaw', 'headPitch', 'headRoll']


def gate(t, start, end, attack=0.06, release=0.10):
    """梯形包络 0..1"""
    if t < start - attack or t > end + release:
        return 0.0
    if start - attack <= t < start:
        return (t - (start - attack)) / attack
    if start <= t <= end:
        return 1.0
    return max(0.0, 1.0 - (t - end) / release)


def blink_row(t, centers=0.7):
    """以 centers 时刻列表为中心的 0.18s 快速眨眼"""
    v = 0.0
    for c in centers:
        d = abs(t - c)
        if d < 0.09:
            v = max(v, math.cos(d / 0.09 * math.pi / 2) ** 1.5)
    return v


def row_zero():
    return {c: 0.0 for c in COLS if c != 'Timecode'}


def gen_blink(t):
    r = row_zero()
    b = blink_row(t, centers=[0.8, 2.2, 3.6])
    r['eyeBlinkLeft'] = r['eyeBlinkRight'] = b
    return r, 4.2


def gen_open_smile(t):
    r = row_zero()
    # 两轮张嘴（模拟说话），第二轮带笑
    o1 = gate(t, 0.6, 1.4)
    o2 = gate(t, 2.2, 3.2)
    r['jawOpen'] = max(0.85 * o1, 0.7 * o2)
    sm = gate(t, 2.2, 3.4, attack=0.2, release=0.25)
    r['mouthSmileLeft'] = r['mouthSmileRight'] = 0.9 * sm
    r['eyeSquintLeft'] = r['eyeSquintRight'] = 0.55 * sm
    r['cheekSquintLeft'] = r['cheekSquintRight'] = 0.4 * sm
    return r, 3.8


def gen_turn_brow(t):
    r = row_zero()
    # 左转（yaw+）→ 回正 → 右转（yaw-）→ 点头
    left = gate(t, 0.7, 1.6)
    right = gate(t, 2.2, 3.1)
    r['headYaw'] = math.radians(25) * left - math.radians(25) * right
    nod = gate(t, 3.5, 4.1)
    r['headPitch'] = math.radians(15) * nod
    # 转头时挑眉，惊讶感
    brow = max(left, right)
    r['browInnerUp'] = 0.8 * brow
    r['browOuterUpLeft'] = r['browOuterUpRight'] = 0.6 * brow
    return r, 4.5


SCENARIOS = {
    'synth_blink': gen_blink,
    'synth_open_smile': gen_open_smile,
    'synth_turn_brow': gen_turn_brow,
}


def write_csv(name, fn):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{name}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLS)
        i = 0
        while True:
            t = i / FPS
            row, dur = fn(t)
            if t > dur:
                break
            tc = f'00:00:{int(t):02d}:{int(round((t - int(t)) * FPS)):03d}'
            w.writerow([tc] + [f'{row[c]:.5f}' for c in COLS if c != 'Timecode'])
            i += 1
    log(f'{path}  ({i} 行 @ {FPS}fps)')
    return path


def main():
    only = sys.argv[1:] or list(SCENARIOS)
    for name in only:
        write_csv(name, SCENARIOS[name])


if __name__ == '__main__':
    main()
