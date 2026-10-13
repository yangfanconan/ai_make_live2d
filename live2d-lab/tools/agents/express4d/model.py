"""Express4DNet —— L2 小模型：纯 MLP（逐帧非线性映射）

架构：
  Window (W, N_IN) → flatten → MLP(N_IN*W → 256 → 128 → N_OUT) → Sigmoid → [0,1]

设计要点：
  - MLP 做逐帧非线性映射（替代 arkit_map 的规则）
  - 时序平滑由 infer.py 的 lowpass(12Hz) 负责（复用 motion_author 设施）
  - Sigmoid 输出 [0,1]（canonical 归一化空间，推理时 unit_to_canonical 还原）
  - 滑窗输入给模型时序上下文（当前帧 ± half 帧），但不做序列卷积
"""
import torch
import torch.nn as nn

from .dataset import N_IN, N_OUT


class Express4DNet(nn.Module):
    """MLP 骨干的 Live2D 参数预测器

    输入: (B, W, N_IN)   W 帧滑窗
    输出: (B, N_OUT)     [0,1] 归一化 canonical 参数
    """

    def __init__(self, window=15, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.window = window
        flat_dim = N_IN * window

        layers = []
        d = flat_dim
        for h in hidden:
            layers += [
                nn.Linear(d, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            d = h
        layers.append(nn.Linear(d, N_OUT))
        self.mlp = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, W, N_IN) → flatten → MLP → sigmoid
        B = x.shape[0]
        h = self.mlp(x.reshape(B, -1))
        return self.sigmoid(h)

    def forward_seq(self, x_seq):
        """序列推理：逐帧 MLP（forward 的批量版），时序平滑由 infer.py lowpass 负责

        x_seq: (T, W, N_IN) → (T, N_OUT) [0,1]
        """
        with torch.no_grad():
            T = x_seq.shape[0]
            h = self.mlp(x_seq.reshape(T, -1))
            return self.sigmoid(h)


def save_model(model, path, meta=None):
    """保存模型权重 + 元信息"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # 从 Sequential 中提取 hidden 层维度
    hidden_dims = [m.out_features for m in model.mlp
                   if isinstance(m, nn.Linear) and m.out_features != N_OUT]
    ckpt = {
        'state_dict': model.state_dict(),
        'window': model.window,
        'config': {'hidden': hidden_dims},
        'meta': meta or {},
    }
    torch.save(ckpt, path)


def load_model(path, device='cpu'):
    """加载模型"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', {})
    model = Express4DNet(
        window=ckpt.get('window', 15),
        hidden=tuple(cfg.get('hidden', [256, 128])),
    )
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    return model, ckpt.get('meta', {})
