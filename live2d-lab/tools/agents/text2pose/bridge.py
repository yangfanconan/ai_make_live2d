"""61 维 ARKit（jsonl 语义空间）→ 本项目 L2 管线 31 特征行 的桥接

要点：
  - 本项目消费 camelCase 的 28 BS + headYaw/Pitch/Roll（弧度）
  - MouthFunnel/MouthPucker 在数据集中为合并单列 → 双侧共用同一值
  - 头部姿态：jsonl 为 [-1,1] 归一化空间，raw CSV 为弧度；
    逐帧对齐不可行（标注量化到 0.1），采用分位分布匹配标定：
      HeadYaw ×1.81 / HeadPitch ×0.95 / HeadRoll ×1.14
    （标定脚本见本模块底部 main；数据：train_data.jsonl vs 前 12 个 raw take）
  - 眼球欧拉角（LeftEyeYaw 等）本项目不用（沿用 eyeLook* blendshape），丢弃
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.express4d.dataset import ARKIT_FEATURES

HEAD_SCALE = {'HeadYaw': 1.81, 'HeadPitch': 0.95, 'HeadRoll': 1.14}

# 本项目特征 ← jsonl 61 维键名（None 表示需特殊处理）
_DIRECT = {
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
    'mouthFunnelLeft': 'MouthFunnel', 'mouthFunnelRight': 'MouthFunnel',
    'mouthPuckerLeft': 'MouthPucker', 'mouthPuckerRight': 'MouthPucker',
}


def pose61_to_arkit31(params61):
    """dict(61 维, jsonl 空间) → dict(31 特征, camelCase, 头部弧度)"""
    row = {}
    for feat in ARKIT_FEATURES:
        if feat in ('headYaw', 'headPitch', 'headRoll'):
            src = 'Head' + feat[4:].capitalize()
            row[feat] = float(params61.get(src, 0.0)) * HEAD_SCALE[src]
        else:
            row[feat] = float(params61.get(_DIRECT[feat], 0.0))
    return row


def clip01_row(row):
    """BS 通道钳到 [0,1]（头部弧度不钳）"""
    for k, v in row.items():
        if not k.startswith('head'):
            row[k] = max(0.0, min(1.0, v))
    return row
