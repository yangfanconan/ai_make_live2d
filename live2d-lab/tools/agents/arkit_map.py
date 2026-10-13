#!/usr/bin/env python3
"""ARKit BlendShapes（Live Link Face / Express4D）→ Live2D 参数映射。

输入：一帧的 52/61 个 ARKit blendshape（0..1）+ 头部欧拉角（弧度）。
输出：Live2D 参数值（该模型的真实量程单位：角度为度、开合为量程内绝对值）。

设计要点：
- 先映射到「Live2D 标准语义空间」（Angle ±30°、EyeLOpen 0..1 rest=1、
  MouthForm -1..1 rest=0、Brow -1..1 rest=0 ……），再按 .motion-calib-<Model>.json
  的实测量程做**以默认值为锚点的分段线性缩放**——自动处理
  EyeLOpen max=2、MouthForm def=max=1 这类模型特异性。
- 角度符号是 ARKit↔Live2D 最易错处，集中在 ANGLE_SIGN，QA 校准时只改这里
  （或 calib 文件的 signs 段）。
"""
import json
import math
import os

# ───────────────────────── 标准语义量程（canonical space）─────────────────────────
# (min, max, default)
STD_RANGES = {
    'ParamAngleX': (-30, 30, 0),
    'ParamAngleY': (-30, 30, 0),
    'ParamAngleZ': (-30, 30, 0),
    'ParamBodyAngleX': (-10, 10, 0),
    'ParamBodyAngleY': (-10, 10, 0),
    'ParamBodyAngleZ': (-10, 10, 0),
    'ParamEyeLOpen': (0, 1, 1),
    'ParamEyeROpen': (0, 1, 1),
    'ParamEyeLSmile': (0, 1, 0),
    'ParamEyeRSmile': (0, 1, 0),
    'ParamEyeBallX': (-1, 1, 0),
    'ParamEyeBallY': (-1, 1, 0),
    'ParamBrowLY': (-1, 1, 0),
    'ParamBrowRY': (-1, 1, 0),
    'ParamBrowLX': (-1, 1, 0),
    'ParamBrowRX': (-1, 1, 0),
    'ParamMouthForm': (-1, 1, 0),
    'ParamMouthOpenY': (0, 1, 0),
}

# 角度符号（ARKit 弧度 → Live2D 度）。QA 静态姿势校准处，若转头方向反了只改这里。
# ARKit 约定（iPhone 前置摄像头，FaceAnchor）：
#   yaw>0  脸朝本人左侧转  → 屏幕上看起来向左 → Live2D ParamAngleX 负向（左为负）
#   pitch>0 抬头向上看     → Live2D ParamAngleY 负向（up pose 用的是负值）
#   roll>0  左耳向下（头向左肩倒）→ Live2D ParamAngleZ 负向（tilt_L 为负）
ANGLE_SIGN = {'ParamAngleX': -1, 'ParamAngleY': -1, 'ParamAngleZ': -1}

# blendshape 列名别名（Live Link Face 用驼峰；合成夹具允许下划线/小写）
ALIASES = {
    'eyeBlinkLeft': 'eyeBlinkLeft', 'eye_blink_left': 'eyeBlinkLeft',
    'eyeBlinkRight': 'eyeBlinkRight', 'eye_blink_right': 'eyeBlinkRight',
    'eyeSquintLeft': 'eyeSquintLeft', 'eye_squint_left': 'eyeSquintLeft',
    'eyeSquintRight': 'eyeSquintRight', 'eye_squint_right': 'eyeSquintRight',
    'eyeLookInLeft': 'eyeLookInLeft', 'eyeLookInRight': 'eyeLookInRight',
    'eyeLookOutLeft': 'eyeLookOutLeft', 'eyeLookOutRight': 'eyeLookOutRight',
    'eyeLookUpLeft': 'eyeLookUpLeft', 'eyeLookUpRight': 'eyeLookUpRight',
    'eyeLookDownLeft': 'eyeLookDownLeft', 'eyeLookDownRight': 'eyeLookDownRight',
    'cheekSquintLeft': 'cheekSquintLeft', 'cheekSquintRight': 'cheekSquintRight',
    'jawOpen': 'jawOpen', 'mouthClose': 'mouthClose',
    'mouthSmileLeft': 'mouthSmileLeft', 'mouthSmileRight': 'mouthSmileRight',
    'mouthFrownLeft': 'mouthFrownLeft', 'mouthFrownRight': 'mouthFrownRight',
    'mouthFunnelLeft': 'mouthFunnelLeft', 'mouthFunnelRight': 'mouthFunnelRight',
    'mouthPuckerLeft': 'mouthPuckerLeft', 'mouthPuckerRight': 'mouthPuckerRight',
    'browInnerUp': 'browInnerUp', 'browOuterUpLeft': 'browOuterUpLeft',
    'browOuterUpRight': 'browOuterUpRight',
    'browDownLeft': 'browDownLeft', 'browDownRight': 'browDownRight',
    'headYaw': 'headYaw', 'headPitch': 'headPitch', 'headRoll': 'headRoll',
    'yaw': 'headYaw', 'pitch': 'headPitch', 'roll': 'headRoll',
}


def normalize_row(raw):
    """把任意来源的一行 {列名: 值} 归一成 ARKit 标准名 -> float"""
    out = {}
    for k, v in raw.items():
        if v is None or v == '':
            continue
        key = k.strip()
        std = ALIASES.get(key) or ALIASES.get(key[0].lower() + key[1:])
        if std is None:
            continue
        try:
            out[std] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def _avg(bs, *keys):
    vals = [bs.get(k, 0.0) for k in keys]
    return sum(vals) / len(vals)


def _deg(rad):
    return math.degrees(float(rad))


def map_frame(bs, head=None):
    """单帧：bs 为归一化后的 ARKit 名->0..1，head 可选 {'yaw','pitch','roll'} 弧度。
    返回 canonical Live2D 参数（标准语义空间，未做模型量程缩放）。"""
    bs = dict(bs)
    head = head or {}
    c = {}

    # 头部三轴（度 + 符号）
    c['ParamAngleX'] = ANGLE_SIGN['ParamAngleX'] * _deg(head.get('yaw', bs.get('headYaw', 0)))
    c['ParamAngleY'] = ANGLE_SIGN['ParamAngleY'] * _deg(head.get('pitch', bs.get('headPitch', 0)))
    c['ParamAngleZ'] = ANGLE_SIGN['ParamAngleZ'] * _deg(head.get('roll', bs.get('headRoll', 0)))

    # 眼开合：眨眼主导；眯眼（笑/瞪眼眯）压一点
    for side, pid, blink, squint in (
            ('L', 'ParamEyeLOpen', 'eyeBlinkLeft', 'eyeSquintLeft'),
            ('R', 'ParamEyeROpen', 'eyeBlinkRight', 'eyeSquintRight')):
        blink_v = bs.get(blink, 0.0)
        squint_v = 0.4 * bs.get(squint, 0.0) + 0.3 * bs.get(f'cheekSquint{side}', 0.0)
        c[pid] = max(0.0, (1.0 - blink_v) * (1.0 - min(0.8, squint_v)))

    # 眼球：LookOut/In 为水平（受试者视角），Up/Down 为垂直
    # 受试者向右看（左眼 LookIn / 右眼 LookOut）→ 屏幕上瞳孔向右 → EyeBallX 正
    look_r = 0.5 * (_avg(bs, 'eyeLookInLeft', 'eyeLookOutRight'))
    look_l = 0.5 * (_avg(bs, 'eyeLookOutLeft', 'eyeLookInRight'))
    c['ParamEyeBallX'] = max(-1, min(1, look_r - look_l))
    look_up = _avg(bs, 'eyeLookUpLeft', 'eyeLookUpRight')
    look_dn = _avg(bs, 'eyeLookDownLeft', 'eyeLookDownRight')
    c['ParamEyeBallY'] = max(-1, min(1, look_up - look_dn))

    # 笑眯眼（Live2D 独立通道）
    c['ParamEyeLSmile'] = max(0.0, min(1.0, 0.7 * bs.get('eyeSquintLeft', 0.0)
                                       + 0.4 * bs.get('cheekSquintLeft', 0.0)))
    c['ParamEyeRSmile'] = max(0.0, min(1.0, 0.7 * bs.get('eyeSquintRight', 0.0)
                                       + 0.4 * bs.get('cheekSquintRight', 0.0)))

    # 眉：InnerUp + OuterUp - Down
    c['ParamBrowLY'] = max(-1, min(1, 0.7 * bs.get('browInnerUp', 0.0)
                                   + bs.get('browOuterUpLeft', 0.0)
                                   - bs.get('browDownLeft', 0.0)))
    c['ParamBrowRY'] = max(-1, min(1, 0.7 * bs.get('browInnerUp', 0.0)
                                   + bs.get('browOuterUpRight', 0.0)
                                   - bs.get('browDownRight', 0.0)))

    # 嘴：开合 / 唇形（笑正、噘嘴/ funnel / 皱眉负）
    opener = max(0.0, bs.get('jawOpen', 0.0) - 0.3 * bs.get('mouthClose', 0.0))
    c['ParamMouthOpenY'] = max(0.0, min(1.0, opener))
    smile = _avg(bs, 'mouthSmileLeft', 'mouthSmileRight')
    frown = _avg(bs, 'mouthFrownLeft', 'mouthFrownRight')
    pucker = max(_avg(bs, 'mouthPuckerLeft', 'mouthPuckerRight'),
                 _avg(bs, 'mouthFunnelLeft', 'mouthFunnelRight'))
    c['ParamMouthForm'] = max(-1, min(1, 0.9 * smile - 0.8 * frown - 0.7 * pucker))

    return c


def load_calib(model, tools_dir):
    """读 QA harness 实测的量程种子；不存在则回退标准量程。
    返回 (ranges, signs)。"""
    path = os.path.join(tools_dir, f'.motion-calib-{model}.json')
    ranges = {k: {'min': v[0], 'max': v[1], 'def': v[2]} for k, v in STD_RANGES.items()}
    signs = dict(ANGLE_SIGN)
    if os.path.exists(path):
        data = json.load(open(path))
        for pid, r in (data.get('ranges') or {}).items():
            if pid in STD_RANGES:  # 只采纳我们会产出的通道
                ranges[pid] = {'min': float(r['min']), 'max': float(r['max']),
                               'def': float(r['def'])}
        for pid, s in (data.get('signs') or {}).items():
            if pid in signs:
                signs[pid] = float(s)
    return ranges, signs


def fit_to_model(canonical, ranges):
    """标准语义值 → 模型实测量程（以 def 为锚点的两段缩放；自动 clamp）。"""
    out = {}
    for pid, v in canonical.items():
        r = ranges.get(pid)
        if r is None:
            continue
        smin, smax, sdef = STD_RANGES[pid]
        mn, mx, df = r['min'], r['max'], r['def']
        v = max(smin, min(smax, v))
        if v >= sdef:
            denom = (smax - sdef) or 1e-9
            val = df + (v - sdef) / denom * (mx - df) if smax > sdef else df
        else:
            denom = (sdef - smin) or 1e-9
            val = df - (sdef - v) / denom * (df - mn) if sdef > smin else df
        out[pid] = max(mn, min(mx, val))
    return out


def map_row_to_model(raw, ranges, signs=None):
    """给 motion_author 的一行式入口：原始行（blendshape + headYaw/Pitch/Roll 列）
    → 模型量程参数 dict。"""
    bs = normalize_row(raw)
    # ANGLE_SIGN 若被 calib 覆盖，在角度换算阶段生效：临时调整
    if signs:
        global ANGLE_SIGN
        saved = dict(ANGLE_SIGN)
        ANGLE_SIGN.update(signs)
        try:
            canon = map_frame(bs)
        finally:
            ANGLE_SIGN = saved
    else:
        canon = map_frame(bs)
    return fit_to_model(canon, ranges)
