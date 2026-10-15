"""Text2Pose 推理 —— 文本 → 61 维 ARKit 参数 → 本项目 31 特征行

用法：
  /Users/yangfan/miniconda3/bin/python tools/agents/text2pose/infer_t2p.py \
      --ckpt tools/data/express4d/models/text2pose_v1.pt \
      --text "A restrained, ironic smile with furrowed brows, gaze lowered."
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log
from agents.text2pose.bridge import pose61_to_arkit31, clip01_row
from agents.text2pose.dataset_t2p import PARAM_ORDER, TfidfFeaturizer
from agents.text2pose.model_t2p import load_model

MODELS_DIR = os.path.join(TOOLS, 'data', 'express4d', 'models')


class Text2Pose:
    """加载 ckpt + tfidf 的推理器（供 script2motion 复用）"""

    def __init__(self, ckpt_path, device='cpu'):
        self.net, self.meta = load_model(ckpt_path, device=device)
        self.net.eval()
        self.device = device
        tfidf_path = os.path.join(
            os.path.dirname(ckpt_path),
            self.meta.get('tfidf', os.path.basename(ckpt_path).replace('.pt', '_tfidf.pkl')))
        self.feat = TfidfFeaturizer.load(tfidf_path)

    def predict(self, text):
        """文本 → dict(61 维, jsonl 空间)"""
        x = self.feat.transform([text])
        with torch.no_grad():
            y = self.net(torch.from_numpy(x).to(self.device)).cpu().numpy()[0]
        return {k: float(y[i]) for i, k in enumerate(PARAM_ORDER)}

    def predict_arkit31(self, text):
        """文本 → dict(31 特征, camelCase, 头部弧度, BS 已钳 [0,1])"""
        return clip01_row(pose61_to_arkit31(self.predict(text)))


def main():
    ap = argparse.ArgumentParser(description='Text2Pose 推理')
    ap.add_argument('--ckpt', default=os.path.join(MODELS_DIR, 'text2pose_v1.pt'))
    ap.add_argument('--text', required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    t2p = Text2Pose(args.ckpt, device=args.device)
    p61 = t2p.predict(args.text)
    active = {k: round(v, 3) for k, v in p61.items() if abs(v) >= 0.05}
    log(f'输入: {args.text[:80]}')
    log(f'61 维活跃参数 (|v|>=0.05): {active}')
    row = t2p.predict_arkit31(args.text)
    active31 = {k: round(v, 3) for k, v in row.items() if abs(v) >= 0.03}
    log(f'桥接 31 特征: {active31}')


if __name__ == '__main__':
    main()
