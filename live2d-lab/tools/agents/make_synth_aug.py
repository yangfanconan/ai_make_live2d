#!/usr/bin/env python3
"""随机化增强合成采集 —— 扩充 L2 训练数据（零真机成本）

与 make_synth_takes.py 的分工：
  - synth_takes：3 段固定标准验收序列（QA 回归夹具，内容确定性）
  - synth_aug  ：随机化大量生成（L2 训练多样性），输出 tools/data/synth/aug/

随机化维度：
  1. 时长 3–6s；事件数 2–6 个，起始时刻随机不重叠
  2. 动作原语池：blink / jawOpen(说话) / smile+squint / frown / pucker /
     funnel / browUp / browDown / headYaw L,R / headPitch up,down /
     headRoll L,R / eyeLook 水平+垂直 / mouthClose
  3. 幅度 0.5–1.0 随机、时长/attack/release 随机抖动
  4. 左右通道不对称 ±12%（模拟真人）
  5. 传感器噪声：BS 通道加 N(0, 0.012)，头部角度加 N(0, 0.4°)
  6. 头部角度幅度随机（yaw 10–28°、pitch 8–18°、roll 5–12°）

用法：
  python3 tools/agents/make_synth_aug.py [N]      # 默认 24 takes
  python3 tools/agents/make_synth_aug.py 24 --seed 7
"""
import argparse
import csv
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.common import ROOT, log
from agents.make_synth_takes import COLS, FPS, gate, blink_row, row_zero

OUT_DIR = os.path.join(ROOT, 'tools', 'data', 'synth', 'aug')


def asym(rng, v, k=0.12):
    """左右不对称：v * (1 ± k)"""
    return v * (1.0 + rng.uniform(-k, k))


class Take:
    """一段随机 take：先排事件，再逐帧求值"""

    def __init__(self, rng):
        self.rng = rng
        self.dur = rng.uniform(3.0, 6.0)
        # 事件槽：不重叠地随机放 2–6 个原语
        n_events = rng.randint(2, 6)
        slots = sorted(rng.uniform(0.15, self.dur - 1.2) for _ in range(n_events))
        # 保证最小间隔 0.5s
        fixed = []
        for s in slots:
            if not fixed or s - fixed[-1] > 0.5:
                fixed.append(s)
        prim_names = list(PRIMS)
        self.events = [(rng.choice(prim_names), t0) for t0 in fixed]

    def eval(self, t):
        r = row_zero()
        for name, t0 in self.events:
            PRIMS[name](self.rng, t, t0, r)
        # 传感器噪声
        for c in r:
            if c.startswith('head'):
                r[c] += self.rng.gauss(0, math.radians(0.4))
            else:
                r[c] = max(0.0, min(1.0, r[c] + self.rng.gauss(0, 0.012)))
        return r


# ───────────────────── 动作原语池 ─────────────────────
# 每个: (rng, t, t0 事件起点, row) 原地写入

def p_blink(rng, t, t0, r):
    n = rng.randint(1, 4)
    for i in range(n):
        c = t0 + i * rng.uniform(0.35, 0.7)
        d = abs(t - c)
        if d < 0.09:
            v = math.cos(d / 0.09 * math.pi / 2) ** 1.5
            r['eyeBlinkLeft'] = max(r['eyeBlinkLeft'], asym(rng, v))
            r['eyeBlinkRight'] = max(r['eyeBlinkRight'], asym(rng, v))


def p_talk(rng, t, t0, r):
    """张嘴说话节奏：2–5 个音节包络"""
    n = rng.randint(2, 5)
    amp = rng.uniform(0.35, 0.9)
    for i in range(n):
        s = t0 + i * rng.uniform(0.25, 0.45)
        g = gate(t, s, s + rng.uniform(0.1, 0.2), attack=0.05, release=0.08)
        if g > 0:
            v = amp * g * rng.uniform(0.7, 1.0)
            r['jawOpen'] = max(r['jawOpen'], v)
            if rng.random() < 0.3:
                r['mouthClose'] = max(r['mouthClose'], 0.3 * g)


def p_smile(rng, t, t0, r):
    a = rng.uniform(0.2, 0.5)
    g = gate(t, t0, t0 + a, attack=rng.uniform(0.15, 0.3),
             release=rng.uniform(0.2, 0.4))
    if g > 0:
        r['mouthSmileLeft'] = max(r['mouthSmileLeft'], asym(rng, 0.9 * g))
        r['mouthSmileRight'] = max(r['mouthSmileRight'], asym(rng, 0.9 * g))
        r['eyeSquintLeft'] = max(r['eyeSquintLeft'], 0.55 * g)
        r['eyeSquintRight'] = max(r['eyeSquintRight'], 0.55 * g)
        r['cheekSquintLeft'] = max(r['cheekSquintLeft'], 0.4 * g)
        r['cheekSquintRight'] = max(r['cheekSquintRight'], 0.4 * g)


def p_frown(rng, t, t0, r):
    a = rng.uniform(0.3, 0.8)
    g = gate(t, t0, t0 + a, attack=0.2, release=0.25)
    if g > 0:
        r['mouthFrownLeft'] = max(r['mouthFrownLeft'], asym(rng, 0.7 * g))
        r['mouthFrownRight'] = max(r['mouthFrownRight'], asym(rng, 0.7 * g))
        r['browDownLeft'] = max(r['browDownLeft'], asym(rng, 0.6 * g))
        r['browDownRight'] = max(r['browDownRight'], asym(rng, 0.6 * g))


def p_pucker(rng, t, t0, r):
    a = rng.uniform(0.3, 0.7)
    g = gate(t, t0, t0 + a, attack=0.15, release=0.2)
    if g > 0:
        p = 0.8 * g
        r['mouthPuckerLeft'] = max(r['mouthPuckerLeft'], asym(rng, p))
        r['mouthPuckerRight'] = max(r['mouthPuckerRight'], asym(rng, p))
        f = 0.5 * g
        r['mouthFunnelLeft'] = max(r['mouthFunnelLeft'], asym(rng, f))
        r['mouthFunnelRight'] = max(r['mouthFunnelRight'], asym(rng, f))


def p_brow_up(rng, t, t0, r):
    a = rng.uniform(0.3, 0.8)
    g = gate(t, t0, t0 + a, attack=0.12, release=0.18)
    if g > 0:
        r['browInnerUp'] = max(r['browInnerUp'], asym(rng, 0.8 * g))
        r['browOuterUpLeft'] = max(r['browOuterUpLeft'], asym(rng, 0.6 * g))
        r['browOuterUpRight'] = max(r['browOuterUpRight'], asym(rng, 0.6 * g))


def p_brow_down(rng, t, t0, r):
    a = rng.uniform(0.3, 0.8)
    g = gate(t, t0, t0 + a, attack=0.15, release=0.2)
    if g > 0:
        r['browDownLeft'] = max(r['browDownLeft'], asym(rng, 0.8 * g))
        r['browDownRight'] = max(r['browDownRight'], asym(rng, 0.8 * g))


def _head(rng, t, t0, r, axis, sign, deg_lo, deg_hi):
    a = rng.uniform(0.7, 1.6)
    g = gate(t, t0, t0 + a, attack=0.25, release=0.3)
    if g > 0:
        r[axis] += sign * math.radians(rng.uniform(deg_lo, deg_hi)) * g


def p_yaw_l(rng, t, t0, r): _head(rng, t, t0, r, 'headYaw', +1, 10, 28)
def p_yaw_r(rng, t, t0, r): _head(rng, t, t0, r, 'headYaw', -1, 10, 28)
def p_pitch_up(rng, t, t0, r): _head(rng, t, t0, r, 'headPitch', +1, 8, 18)
def p_pitch_dn(rng, t, t0, r): _head(rng, t, t0, r, 'headPitch', -1, 8, 18)
def p_roll_l(rng, t, t0, r): _head(rng, t, t0, r, 'headRoll', +1, 5, 12)
def p_roll_r(rng, t, t0, r): _head(rng, t, t0, r, 'headRoll', -1, 5, 12)


def p_eye_look(rng, t, t0, r):
    """眼球扫视：水平或垂直，1–3 段"""
    horiz = rng.random() < 0.6
    n = rng.randint(1, 3)
    for i in range(n):
        s = t0 + i * rng.uniform(0.4, 0.8)
        a = rng.uniform(0.15, 0.4)
        g = gate(t, s, s + a, attack=0.06, release=0.1)
        if g <= 0:
            continue
        v = rng.uniform(0.4, 1.0) * g * rng.choice((-1, 1))
        if horiz:
            # v>0: 向右看（左眼 LookIn / 右眼 LookOut）
            r['eyeLookInLeft'] = max(r['eyeLookInLeft'], max(0, v))
            r['eyeLookOutRight'] = max(r['eyeLookOutRight'], max(0, v))
            r['eyeLookOutLeft'] = max(r['eyeLookOutLeft'], max(0, -v))
            r['eyeLookInRight'] = max(r['eyeLookInRight'], max(0, -v))
        else:
            r['eyeLookUpLeft'] = max(r['eyeLookUpLeft'], max(0, v))
            r['eyeLookUpRight'] = max(r['eyeLookUpRight'], max(0, v))
            r['eyeLookDownLeft'] = max(r['eyeLookDownLeft'], max(0, -v))
            r['eyeLookDownRight'] = max(r['eyeLookDownRight'], max(0, -v))


PRIMS = {
    'blink': p_blink, 'talk': p_talk, 'smile': p_smile, 'frown': p_frown,
    'pucker': p_pucker, 'brow_up': p_brow_up, 'brow_down': p_brow_down,
    'yaw_l': p_yaw_l, 'yaw_r': p_yaw_r,
    'pitch_up': p_pitch_up, 'pitch_dn': p_pitch_dn,
    'roll_l': p_roll_l, 'roll_r': p_roll_r,
    'eye_look': p_eye_look,
}


def write_take(idx, take):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'aug_{idx:03d}.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(COLS)
        i = 0
        while True:
            t = i / FPS
            if t > take.dur:
                break
            row = take.eval(t)
            tc = f'00:00:{int(t):02d}:{int(round((t - int(t)) * FPS)):03d}'
            w.writerow([tc] + [f'{row[c]:.5f}' for c in COLS if c != 'Timecode'])
            i += 1
    return path, i


def main():
    ap = argparse.ArgumentParser(description='随机化增强合成采集生成')
    ap.add_argument('n', nargs='?', type=int, default=24)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for idx in range(1, args.n + 1):
        take = Take(random.Random(rng.random()))
        _, rows = write_take(idx, take)
        total += rows
    log(f'{args.n} takes → {OUT_DIR}/  共 {total} 行')


if __name__ == '__main__':
    main()
