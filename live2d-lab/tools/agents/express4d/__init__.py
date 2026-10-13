"""Express4D 训练管线 —— L2 小模型（MLP + 时序平滑）

将 L1 规则映射（arkit_map.py）蒸馏为可学习模型，后续可用真机配对数据微调。

管线：
  1. dataset.py  — CSV → (ARKit BS 向量, L1 canonical 标签) 配对，时序滑窗
  2. model.py    — Express4DNet：MLP 骨干 + 1D 因果卷积时序平滑
  3. train.py    — MSE + 时序平滑正则训练循环
  4. infer.py    — CSV → 模型推理 → fit_to_model → motion3.json
  5. evaluate.py — L1 vs L2 对比报告
"""
