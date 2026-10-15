"""Pose（身体驱动）管线 —— 视频 → DWPose 点位 → Live2D 身体参数 motion3

ingest.py:  视频 → 17 身体关键点归一化 CSV（CoreML 加速，conf 门控 + 线性插值）
bodymap.py: 点位 CSV → BodyAngleX/Y/Z + AngleX/Y/Z + Breath → motion3.json
"""
