"""Text2Pose 训练 —— TF-IDF + MLP，通道方差加权 MSE

用法：
  /Users/yangfan/miniconda3/bin/python tools/agents/text2pose/train_t2p.py --tag v1

报告：
  - mean 基线（训练集均值预测）val MSE/MAE 作对照
  - 分组 MAE（eye/mouth/brow/jaw/head_pose/...）
产物（gitignored）：
  tools/data/express4d/models/text2pose_<tag>.pt + _tfidf.pkl + _report.json
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.common import TOOLS, log, write_json
from agents.text2pose.dataset_t2p import (
    TRAIN_JSONL, VAL_JSONL, PARAM_ORDER, N_OUT, TfidfFeaturizer,
    load_records, group_indices,
)
from agents.text2pose.model_t2p import Text2PoseNet, save_model

MODELS_DIR = os.path.join(TOOLS, 'data', 'express4d', 'models')


def evaluate(model, x, y, batch=512, device='cpu'):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).to(device)
            preds.append(model(xb).cpu().numpy())
    pred = np.concatenate(preds)
    mse = float(((pred - y) ** 2).mean())
    mae = float(np.abs(pred - y).mean())
    return mse, mae, pred


def main():
    ap = argparse.ArgumentParser(description='Text2Pose L3 训练')
    ap.add_argument('--train-jsonl', default=TRAIN_JSONL)
    ap.add_argument('--val-jsonl', default=VAL_JSONL)
    ap.add_argument('--max-features', type=int, default=20000)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--no-chan-weight', action='store_true')
    ap.add_argument('--tag', default='v1')
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = ap.parse_args()

    if args.device == 'auto':
        device = ('cuda' if torch.cuda.is_available()
                  else 'mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = args.device
    log(f'设备: {device}')

    # 数据
    train_recs = load_records(args.train_jsonl)
    val_recs = load_records(args.val_jsonl)
    log(f'数据: train={len(train_recs)} / val={len(val_recs)}')

    feat = TfidfFeaturizer(max_features=args.max_features)
    feat.fit([r['text'] for r in train_recs])
    x_train = feat.transform([r['text'] for r in train_recs])
    x_val = feat.transform([r['text'] for r in val_recs])
    y_train = np.stack([r['target'] for r in train_recs])
    y_val = np.stack([r['target'] for r in val_recs])
    log(f'TF-IDF 维度: {feat.dim}')

    # mean 基线
    mean_pred = y_train.mean(axis=0, keepdims=True)
    base_mse = float(((mean_pred - y_val) ** 2).mean())
    base_mae = float(np.abs(mean_pred - y_val).mean())
    log(f'mean 基线: val_mse={base_mse:.6f} val_mae={base_mae:.6f}')

    # 通道权重（按训练集目标标准差）
    weight = None
    if not args.no_chan_weight:
        std = y_train.std(axis=0) + 0.02
        w = std / std.mean()
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        top = sorted(zip(PARAM_ORDER, w.tolist()), key=lambda kv: -kv[1])[:5]
        log('通道权重 top5: ' + ', '.join(f'{p}={v:.2f}' for p, v in top))

    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = Text2PoseNet(in_dim=feat.dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f'模型: {n_params:,} 参数 / in={feat.dim} out={N_OUT}')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_path = os.path.join(MODELS_DIR, f'text2pose_{args.tag}.pt')
    tfidf_path = os.path.join(MODELS_DIR, f'text2pose_{args.tag}_tfidf.pkl')
    best_val = float('inf')
    best_state = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = ((((pred - yb) ** 2) * weight).mean()
                    if weight is not None
                    else nn.functional.mse_loss(pred, yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
        sched.step()
        val_mse, val_mae, _ = evaluate(model, x_val, y_val, device=device)
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            save_model(model, out_path, {
                'epoch': ep, 'train_loss': round(tot / n, 6),
                'val_mse': round(val_mse, 6), 'val_mae': round(val_mae, 6),
                'tfidf': os.path.basename(tfidf_path),
                'max_features': args.max_features,
                'param_order': PARAM_ORDER,
            })
            feat.save(tfidf_path)
        if ep % 25 == 0 or ep == 1 or ep == args.epochs:
            log(f'  ep {ep:4d}/{args.epochs}  train={tot / n:.6f}  '
                f'val_mse={val_mse:.6f}  val_mae={val_mae:.6f}  '
                f'best={best_val:.6f}  {time.time() - t0:.1f}s')

    # 最终报告（best 权重）
    model.load_state_dict(best_state)
    val_mse, val_mae, pred = evaluate(model, x_val, y_val, device=device)
    groups = {}
    for g, idxs in group_indices().items():
        idxs = np.array(idxs)
        groups[g] = {
            'mae': round(float(np.abs(pred[:, idxs] - y_val[:, idxs]).mean()), 6),
            'baseline_mae': round(float(np.abs(mean_pred[:, idxs] - y_val[:, idxs]).mean()), 6),
            'n_channels': len(idxs),
        }
    log(f'训练完成 → {out_path}  best_val_mse={best_val:.6f} val_mae={val_mae:.6f}')
    for g, m in sorted(groups.items()):
        imp = (1 - m['mae'] / max(m['baseline_mae'], 1e-9)) * 100
        log(f"  {g:18s} mae={m['mae']:.4f} baseline={m['baseline_mae']:.4f} "
            f"({imp:+.0f}%)")

    rep_path = os.path.join(MODELS_DIR, f'text2pose_{args.tag}_report.json')
    write_json(rep_path, {
        'model_path': os.path.abspath(out_path),
        'tfidf_path': os.path.abspath(tfidf_path),
        'tag': args.tag, 'epochs': args.epochs, 'n_params': n_params,
        'tfidf_dim': feat.dim, 'max_features': args.max_features,
        'n_train': len(train_recs), 'n_val': len(val_recs),
        'best_val_mse': round(best_val, 6), 'val_mae': round(val_mae, 6),
        'baseline': {'val_mse': round(base_mse, 6), 'val_mae': round(base_mae, 6)},
        'groups': groups,
    })
    log(f'报告 → {rep_path}')


if __name__ == '__main__':
    main()
