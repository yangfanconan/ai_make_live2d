"""Text2PoseNet —— L3 文本 → ARKit 61 维参数（MLP 骨干）

架构：
  TF-IDF 特征 (D) → MLP(D → 512 → 256 → 61) → 分段有界激活
    - 前 52 维 blendshape: Sigmoid → [0,1]
    - 后 9 维头部/眼球姿态: Tanh → [-1,1]

设计要点：
  - 与 L2 相同的小模型哲学：3448 样本下 MLP + 强文本特征比 transformer 更稳
  - 有界输出天然满足 ARKit 参数域，无需后处理裁剪
"""
import torch
import torch.nn as nn

from .dataset_t2p import N_BS, N_OUT


class Text2PoseNet(nn.Module):
    """输入 (B, D) TF-IDF → 输出 (B, 61) 有界 ARKit 参数"""

    def __init__(self, in_dim, hidden=(512, 256), dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, N_OUT))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        h = self.mlp(x)
        bs = torch.sigmoid(h[..., :N_BS])      # [0,1]
        pose = torch.tanh(h[..., N_BS:])       # [-1,1]
        return torch.cat([bs, pose], dim=-1)


def save_model(model, path, meta=None):
    torch.save({
        'state_dict': model.state_dict(),
        'in_dim': model.in_dim,
        'meta': meta or {},
    }, path)


def load_model(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = Text2PoseNet(in_dim=ckpt['in_dim']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return model, ckpt.get('meta', {})
