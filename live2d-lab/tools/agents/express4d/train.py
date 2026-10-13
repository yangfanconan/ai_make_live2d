"""Express4D 训练循环 —— MSE + 时序平滑正则

用法：
  python3 tools/agents/express4d/train.py \
      --csv tools/data/synth/synth_blink.csv tools/data/synth/synth_open_smile.csv \
      --model Haru --epochs 200 --window 15

  # 自动发现合成夹具
  python3 tools/agents/express4d/train.py --synth --model Haru

损失：
  L = MSE(pred, target) + λ_smooth * TV(pred)  # TV = 全变分（时序抖动惩罚）
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log, write_json
from agents.express4d.dataset import (
    PairedTakeDataset, split_takes, discover_csvs, N_IN, N_OUT, LIVE2D_PARAMS,
)
from agents.express4d.model import Express4DNet, save_model

SYNTH_DIR = os.path.join(TOOLS, 'data', 'synth')
EXPRESS4D_DIR = os.path.join(TOOLS, 'data', 'express4d', 'raw')
MODELS_DIR = os.path.join(TOOLS, 'data', 'express4d', 'models')


def smooth_loss(pred):
    """全变分正则：惩罚相邻帧预测值的跳变"""
    if pred.shape[0] < 2:
        return torch.tensor(0.0, device=pred.device)
    return torch.mean(torch.abs(pred[1:] - pred[:-1]))


def train_epoch(model, loader, optimizer, device, lam_smooth=0.01):
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        # 训练阶段纯 MSE；时序平滑由 forward_seq 的 1D 因果卷积在推理时实现
        loss = nn.functional.mse_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        total += loss.item() * len(y)
        n += len(y)
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description='Express4D L2 模型训练')
    ap.add_argument('--csv', nargs='*', default=[], help='ARKit CSV 文件列表')
    ap.add_argument('--synth', action='store_true', help='自动加载合成夹具')
    ap.add_argument('--express4d', action='store_true', help='加载真机配对数据')
    ap.add_argument('--window', type=int, default=15, help='时序窗口帧数')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--val-ratio', type=float, default=0.2)
    ap.add_argument('--out', default=None, help='模型输出路径')
    ap.add_argument('--tag', default='v1', help='模型标签')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = ap.parse_args()

    # 设备
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    log(f'设备: {device}')

    # 数据发现
    csv_paths = list(args.csv)
    if args.synth:
        csv_paths += discover_csvs(synth_dir=SYNTH_DIR)
    if args.express4d:
        csv_paths += discover_csvs(express4d_dir=EXPRESS4D_DIR)
    csv_paths = sorted(set(csv_paths))
    if not csv_paths:
        ap.error('未指定 CSV 数据源（--csv / --synth / --express4d）')
    log(f'数据: {len(csv_paths)} takes')
    for p in csv_paths:
        log(f'  {os.path.basename(p)}')

    # 划分
    train_csvs, val_csvs = split_takes(csv_paths, val_ratio=args.val_ratio)
    log(f'  train: {len(train_csvs)} takes / val: {len(val_csvs)} takes')

    # 数据集
    train_ds = PairedTakeDataset(train_csvs, window=args.window)
    val_ds = PairedTakeDataset(val_csvs, window=args.window) if val_csvs else None
    log(f'  train samples: {len(train_ds)} / val samples: {len(val_ds) if val_ds else 0}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    # 模型
    model = Express4DNet(window=args.window).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'模型: {n_params:,} 参数 / window={args.window} / in={N_IN} out={N_OUT}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    best_val = float('inf')
    out_path = args.out or os.path.join(MODELS_DIR, f'express4d_{args.tag}.pt')

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = eval_epoch(model, val_loader, device) if val_loader else 0.0
        scheduler.step()
        dt = time.time() - t0

        if val_loss < best_val:
            best_val = val_loss
            meta = {
                'epoch': ep,
                'train_loss': round(train_loss, 6),
                'val_loss': round(val_loss, 6),
                'best': True,
                'train_csvs': [os.path.basename(p) for p in train_csvs],
                'val_csvs': [os.path.basename(p) for p in val_csvs],
                'params': LIVE2D_PARAMS,
            }
            save_model(model, out_path, meta)

        if ep % 20 == 0 or ep == 1 or ep == args.epochs:
            log(f'  ep {ep:4d}/{args.epochs}  train={train_loss:.6f}  '
                f'val={val_loss:.6f}  best={best_val:.6f}  {dt:.1f}s')

    log(f'训练完成 → {out_path}  best_val={best_val:.6f}')

    # 报告
    report = {
        'model_path': os.path.abspath(out_path),
        'tag': args.tag,
        'window': args.window,
        'epochs': args.epochs,
        'best_val_loss': round(best_val, 6),
        'n_train_takes': len(train_csvs),
        'n_val_takes': len(val_csvs),
        'n_train_samples': len(train_ds),
        'n_params': n_params,
        'train_csvs': [os.path.basename(p) for p in train_csvs],
        'val_csvs': [os.path.basename(p) for p in val_csvs],
    }
    rep_path = os.path.join(os.path.dirname(out_path), f'express4d_{args.tag}_report.json')
    write_json(rep_path, report)
    log(f'报告 → {rep_path}')


if __name__ == '__main__':
    main()
