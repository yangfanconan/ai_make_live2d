"""Express4D 配对数据集 —— ARKit BS 序列 → Live2D canonical 参数（L1 教师标签）

数据来源（按优先级）：
  1. 真机配对：tools/data/express4d/raw/<take>/ 下 video + params.json（VTube Studio 双路录制）
  2. 合成夹具：tools/data/synth/*.csv（make_synth_takes.py 生成，L1 教师即标签源）
  3. 任意 ARKit CSV：--csv 参数指定

核心设计：
  - 模型预测 canonical 参数（标准语义空间，模型无关），推理时再 fit_to_model 做量程缩放
  - 时序滑窗：输入 (W, N_in) → 输出 (N_out,)，中心帧对齐
  - 按 take 划分 train/val（不按帧，防止泄漏）
"""
import csv
import math
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents import arkit_map as AM
from agents import motion_author as MA

# ───────────────────── 固定特征/输出列表 ─────────────────────
# ARKit 输入：31 维（28 BS + 3 头部欧拉角弧度）
ARKIT_FEATURES = [
    'eyeBlinkLeft', 'eyeBlinkRight',
    'eyeSquintLeft', 'eyeSquintRight',
    'eyeLookInLeft', 'eyeLookInRight',
    'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight',
    'eyeLookDownLeft', 'eyeLookDownRight',
    'cheekSquintLeft', 'cheekSquintRight',
    'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'browDownLeft', 'browDownRight',
    'jawOpen', 'mouthClose',
    'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnelLeft', 'mouthFunnelRight',
    'mouthPuckerLeft', 'mouthPuckerRight',
    'headYaw', 'headPitch', 'headRoll',
]
N_IN = len(ARKIT_FEATURES)

# Live2D 输出：18 维 canonical 参数（与 arkit_map.STD_RANGES 一致）
LIVE2D_PARAMS = list(AM.STD_RANGES.keys())
N_OUT = len(LIVE2D_PARAMS)

# canonical 参数的 (min, max) 用于归一化到 [0, 1]
PARAM_BOUNDS = np.array(
    [AM.STD_RANGES[p] for p in LIVE2D_PARAMS], dtype=np.float32
)  # (N_OUT, 3): min, max, def


def canonical_to_unit(canon_dict):
    """canonical 参数 dict → [0,1] 归一化向量（以 def 为锚点双向缩放）"""
    vec = np.zeros(N_OUT, dtype=np.float32)
    for i, pid in enumerate(LIVE2D_PARAMS):
        v = canon_dict.get(pid, 0.0)
        mn, mx, df = PARAM_BOUNDS[i]
        if v >= df:
            denom = (mx - df) or 1e-9
            vec[i] = (v - df) / denom * 0.5 + 0.5
        else:
            denom = (df - mn) or 1e-9
            vec[i] = 0.5 - (df - v) / denom * 0.5
    return np.clip(vec, 0.0, 1.0)


def unit_to_canonical(unit_vec):
    """[0,1] 归一化向量 → canonical 参数 dict"""
    out = {}
    for i, pid in enumerate(LIVE2D_PARAMS):
        u = float(np.clip(unit_vec[i], 0.0, 1.0))
        mn, mx, df = PARAM_BOUNDS[i]
        if u >= 0.5:
            out[pid] = df + (u - 0.5) * 2.0 * (mx - df)
        else:
            out[pid] = df - (0.5 - u) * 2.0 * (df - mn)
        out[pid] = max(mn, min(mx, out[pid]))
    return out


def row_to_features(raw_row):
    """CSV 原始行 → 31 维 ARKit 特征向量"""
    bs = AM.normalize_row(raw_row)
    return np.array([bs.get(f, 0.0) for f in ARKIT_FEATURES], dtype=np.float32)


def row_to_target(raw_row):
    """CSV 原始行 → 18 维 canonical 归一化目标（L1 教师标签）"""
    bs = AM.normalize_row(raw_row)
    canon = AM.map_frame(bs)
    return canonical_to_unit(canon)


def load_take(csv_path):
    """加载单个 take CSV → (features [T, N_IN], targets [T, N_OUT], times [T])"""
    rows = MA.load_csv_rows(csv_path)
    if len(rows) < 2:
        raise ValueError(f'{csv_path}: 有效行不足 ({len(rows)})')
    feats, tgts, times = [], [], []
    for t, row in rows:
        feats.append(row_to_features(row))
        tgts.append(row_to_target(row))
        times.append(t)
    return (
        np.stack(feats),
        np.stack(tgts),
        np.array(times, dtype=np.float32),
    )


class PairedTakeDataset(Dataset):
    """时序滑窗配对数据集

    输入: (window, N_IN)  以中心帧对齐的 W 帧窗口
    输出: (N_OUT,)         中心帧的 canonical 归一化参数

    边界处理: 首尾各 pad window//2 帧（复制边缘帧）
    """

    def __init__(self, csv_paths, window=15):
        self.window = window
        self.half = window // 2
        self.samples = []  # [(take_idx, frame_idx)]

        self.takes = []  # [(features [T,N_IN], targets [T,N_OUT], name)]
        for p in csv_paths:
            feat, tgt, t = load_take(p)
            self.takes.append((feat, tgt, os.path.basename(p)))
            for fi in range(len(feat)):
                self.samples.append((len(self.takes) - 1, fi))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ti, fi = self.samples[idx]
        feat, tgt, _ = self.takes[ti]
        T = len(feat)
        # 滑窗（边界复制）
        lo = max(0, fi - self.half)
        hi = min(T, fi + self.half + 1)
        chunk = feat[lo:hi]
        if len(chunk) < self.window:
            pad = self.window - len(chunk)
            if lo == 0:
                chunk = np.concatenate([np.repeat(feat[:1], pad, axis=0), chunk])
            else:
                chunk = np.concatenate([chunk, np.repeat(feat[-1:], pad, axis=0)])
        return torch.from_numpy(chunk.copy()), torch.from_numpy(tgt[fi].copy())


def split_takes(csv_paths, val_ratio=0.2, seed=42):
    """按 take 划分 train/val（保持时序连续性，防泄漏）"""
    rng = np.random.RandomState(seed)
    n = len(csv_paths)
    n_val = max(1, int(n * val_ratio))
    indices = rng.permutation(n)
    val_idx = set(indices[:n_val].tolist())
    train = [csv_paths[i] for i in range(n) if i not in val_idx]
    val = [csv_paths[i] for i in range(n) if i in val_idx]
    return train, val


def discover_csvs(synth_dir=None, express4d_dir=None):
    """自动发现合成夹具（含 aug/ 子目录）和真机 CSV"""
    paths = []
    if synth_dir and os.path.isdir(synth_dir):
        for root, _dirs, files in sorted(os.walk(synth_dir)):
            if '__pycache__' in root:
                continue
            for f in sorted(files):
                if f.endswith('.csv'):
                    paths.append(os.path.join(root, f))
    if express4d_dir and os.path.isdir(express4d_dir):
        for d in sorted(os.listdir(express4d_dir)):
            dp = os.path.join(express4d_dir, d)
            if os.path.isdir(dp):
                for f in sorted(os.listdir(dp)):
                    if f.endswith('.csv'):
                        paths.append(os.path.join(dp, f))
    return paths
