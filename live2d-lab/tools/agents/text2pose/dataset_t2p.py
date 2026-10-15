"""Text2Pose 数据集 —— KeyframeFace ARKit_semantic jsonl → (文本, 61 维 ARKit 目标)

数据源（CC-BY-NC-4.0，仅本地训练用）：
  processed_data/ARKit_semantic/{train,val}_data.jsonl
  每行: {messages:[system,user,assistant], script_id, keyframe}
    - user 含 "Keyframe Description Sequence:" 下多个 "- Keyframe Xxx: <描述>"
    - assistant 为 JSON：{Keyframe_xxx: {61 参数}}，值量化到 0.1
    - keyframe 字段指明该样本目标是哪个关键帧

输出空间：
  61 维固定键序（与原始数据一致），前 52 为 blendshape [0,1]，
  后 9 为头部/眼球姿态 [-1,1]（归一化空间；头部 → 弧度的换算见 bridge.py）
"""
import json
import os
import pickle
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS

KF_BASE = os.path.join(TOOLS, 'data', 'express4d', 'raw', 'keyframeface',
                       'extracted', 'KeyframeFace Dataset', 'processed_data')
TRAIN_JSONL = os.path.join(KF_BASE, 'ARKit_semantic', 'train_data.jsonl')
VAL_JSONL = os.path.join(KF_BASE, 'ARKit_semantic', 'val_data.jsonl')

# 61 维固定键序（实测 train/val 全量一致）
PARAM_ORDER = [
    'EyeBlinkLeft', 'EyeLookDownLeft', 'EyeLookInLeft', 'EyeLookOutLeft',
    'EyeLookUpLeft', 'EyeSquintLeft', 'EyeWideLeft',
    'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight', 'EyeLookOutRight',
    'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight',
    'JawForward', 'JawRight', 'JawLeft', 'JawOpen',
    'MouthClose', 'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
    'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft', 'MouthFrownRight',
    'MouthDimpleLeft', 'MouthDimpleRight', 'MouthStretchLeft', 'MouthStretchRight',
    'MouthRollLower', 'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
    'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft', 'MouthLowerDownRight',
    'MouthUpperUpLeft', 'MouthUpperUpRight',
    'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp',
    'BrowOuterUpLeft', 'BrowOuterUpRight',
    'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight',
    'NoseSneerLeft', 'NoseSneerRight', 'TongueOut',
    'HeadYaw', 'HeadPitch', 'HeadRoll',
    'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll',
    'RightEyeYaw', 'RightEyePitch', 'RightEyeRoll',
]
N_OUT = len(PARAM_ORDER)
N_BS = 52  # 前 52 维 blendshape [0,1]；后 9 维姿态 [-1,1]

KF_NAME = {f'Keyframe_{w}': f'Keyframe {w.title()}'
           for w in ('one', 'two', 'three', 'four', 'five')}


def parse_keyframe_texts(user_content):
    """user 消息 → {Keyframe_one: 描述文本, ...}"""
    out = {}
    for m in re.finditer(r'-\s*(Keyframe\s+\w+)\s*:\s*(.+?)(?=\n-\s*Keyframe|\Z)',
                         user_content, re.S):
        word = m.group(1).split()[1].lower()  # "Keyframe One" → "one"
        out[f'Keyframe_{word}'] = ' '.join(m.group(2).split())
    return out


def load_records(jsonl_path):
    """jsonl → [{'text','script_id','keyframe','target':(61,)float32}]"""
    recs = []
    for ln in open(jsonl_path, encoding='utf-8'):
        d = json.loads(ln)
        msgs = d['messages']
        user = next(m['content'] for m in msgs if m['role'] == 'user')
        asst = next(m['content'] for m in msgs if m['role'] == 'assistant')
        texts = parse_keyframe_texts(user)
        tgt_obj = json.loads(asst)
        kf = d['keyframe']
        if kf not in tgt_obj or kf not in texts:
            continue
        params = tgt_obj[kf]
        target = np.array([float(params[k]) for k in PARAM_ORDER], dtype=np.float32)
        recs.append({
            'text': texts[kf], 'script_id': d['script_id'], 'keyframe': kf,
            'target': target,
        })
    return recs


def group_indices():
    """按参数名前缀分组（评估报告用）"""
    groups = {}
    for i, k in enumerate(PARAM_ORDER):
        if k.startswith(('LeftEye', 'RightEye')):
            g = 'eye_pose'
        elif k.startswith('Head'):
            g = 'head_pose'
        elif k.startswith('Eye'):
            g = 'eye_bs'
        elif k.startswith('Jaw'):
            g = 'jaw'
        elif k.startswith('Mouth'):
            g = 'mouth'
        elif k.startswith('Brow'):
            g = 'brow'
        elif k.startswith(('Cheek', 'Nose', 'Tongue')):
            g = 'cheek_nose_tongue'
        else:
            g = 'other'
        groups.setdefault(g, []).append(i)
    return groups


class TfidfFeaturizer:
    """sklearn TF-IDF 包装（word 1-2gram），可 pickle 持久化"""

    def __init__(self, max_features=20000, min_df=2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), min_df=min_df,
            max_features=max_features, sublinear_tf=True,
            strip_accents='unicode',
        )

    def fit(self, texts):
        return self.vec.fit(texts)

    def transform(self, texts):
        return self.vec.transform(texts).toarray().astype(np.float32)

    @property
    def dim(self):
        return len(self.vec.vocabulary_)

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self.vec, fh)

    @staticmethod
    def load(path):
        with open(path, 'rb') as fh:
            vec = pickle.load(fh)
        f = TfidfFeaturizer.__new__(TfidfFeaturizer)
        f.vec = vec
        return f


class SentenceEncoder:
    """sentence-transformers 句向量（all-MiniLM-L6-v2，384 维 L2 归一化）"""

    MODEL_NAME = 'all-MiniLM-L6-v2'

    def __init__(self, model_name=None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name or self.MODEL_NAME)

    def fit(self, texts):
        """无拟合需求（预训练），返回 self 保持对称接口"""
        return self

    def transform(self, texts, batch_size=128):
        return self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ).astype(np.float32)

    @property
    def dim(self):
        return self.model.get_embedding_dimension()

    def save(self, path):
        """存 model_name（权重由 HF Hub 管理，不入库）"""
        with open(path + '.modelname', 'w') as fh:
            fh.write(self.MODEL_NAME)

    @staticmethod
    def load(path):
        p = path if os.path.exists(path) else path + '.modelname'
        if os.path.exists(p):
            name = open(p).read().strip()
            return SentenceEncoder(model_name=name)
        return SentenceEncoder()
