<div align="center">

# Live2D Lab

**程序驱动的 2D 角色动画实验台**

AI 换肤 · 视频驱动动作合成 · 自动化 QA 验收

</div>

---

## 项目定位

Live2D Lab 是一套围绕 [Cubism SDK](https://www.live2d.com/) 的工程化实验台，解决 Live2D 角色生产中的三个核心瓶颈：

1. **关键姿势生成**——从视频/合成数据自动映射面部动作到 Live2D 参数，生成可注册的 motion3
2. **AI 换肤**——从官方模型出发，通过 SAM 分割 + ComfyUI inpainting + 程序收尾实现全身素材替换
3. **自动化 QA**——离屏渲染 + 确定性参数注入 + heuristic 判定，替代人工目检

技术路线分三级递进：

| 级别 | 方案 | 周期 | 效果 |
|------|------|------|------|
| **L1 几何映射** | ARKit BlendShapes → Live2D 参数规则映射 | 1 周 | 60–70% 自然度 |
| **L2 小模型** | 自监督配对数据训练 MLP/Transformer | 2–3 周 | 80% 自然度 |
| **L3 多模态** | VLM 理解文本/视频语义直接生成关键姿势 | 1–2 月 | 商用级 |

> 当前实现已覆盖 **L1 几何映射** 的端到端管线，含 QA 验收闭环。

---

## 技术架构

### 总体数据流

```
真人视频 / 合成夹具
    │
    ▼ ARKit BlendShapes (52 BS + 头部欧拉角)
┌───────────────────────────┐
│  arkit_map.py             │  ← 标准语义空间 → 模型实测量程分段缩放
│  motion_author.py         │  ← 12Hz 低通 → 钳制 → 30fps motion3
└───────────┬───────────────┘
            │ motion3.json
            ▼
┌───────────────────────────┐
│  motion_deploy.py         │  ← 落盘 motions/auto + model3.json Auto 组注册
└───────────┬───────────────┘
            │
            ▼
┌───────────────────────────┐        ┌───────────────────────┐
│  qa_director.py           │───────▶│  capture_server.py     │
│  (plan → 截帧 → 判定)      │        │  (线程化 HTTPServer)   │
└───────────┬───────────────┘        └───────────┬───────────┘
            │                                     │ base64 截帧
            ▼                                     ▼
┌───────────────────────────┐        ┌───────────────────────┐
│  judges.py                │◀───────│  qa_harness.js         │
│  (heuristic + sheet +     │        │  (离屏 900×700 缓冲)    │
│   JUDGE_CMD 三级判定)      │        │  static/trajectory/motion│
└───────────────────────────┘        └───────────────────────┘
```

### 两条轨道

| 轨道 A：QA 验收 | 轨道 B：视频驱动动作合成 |
|---|---|
| 确定性参数注入 → 极值姿势截帧 | ARKit CSV → 参数映射 → motion3 落盘 |
| heuristic 判定 + contact sheet | 自监督配对（VTube Studio 双路录制） |
| 静态 / 轨迹 / 动作 三种模式 | 合成夹具回归 + 真机数据 gated |
| 替代人工目检 | 替代手 K 关键帧 |

### AI 换肤并行管线

```
官方模型贴图
    │
    ▼ SAM 分割（按需懒跑，缓存 masks/）
┌───────────────────────────┐
│  reskin_ai.py / .sh       │
│  ┌─ program 模式: HSV 迁移 │  ← 零结构风险，单色部件
│  └─ ai 模式: ComfyUI inpaint│  ← SD1.5 @768，中位质检 + 换 seed 重试
└───────────┬───────────────┘
            │ 羽化蒙版合成 → alpha 保留 → 颜色归一 → 残留岛迁移
            ▼
     新模型目录 + model3.json 缓存版本 bump
            │
            ▼ --qa 钩子
     qa_director.py 自动验收
```

---

## 目录结构

```
live2d-lab/
├── index.html                    # Demo 页面（参数滑块实时驱动）
├── css/panel.css                 # Demo 样式
├── js/
│   ├── app.js                    # Demo 主逻辑（模型加载/参数写入/缓存破坏）
│   └── qa_harness.js             # QA 离屏扫描引擎（三模式）
├── qa/harness.html               # QA Harness 入口页
│
├── assets/models/                # Live2D 模型资产
│   ├── Haru/                     # 官方示例（Cubism 4 SDK）
│   ├── HaruAI/                   # AI 换肤产物
│   ├── HaruGold/  HaruGoldAI/    # 金发变体
│   ├── HaruWineGold/             # 酒红西装 + 金发
│   ├── Hiyori/                   # 官方示例（Cubism 4 SDK）
│   ├── HiyoriAI/  HiyoriPinkGold/ HiyoriPurpleBlack/
│   └── Mark/                     # 官方示例（Cubism 4 SDK）
│
├── tools/
│   ├── agents/                   # 多 Agent 管线核心
│   │   ├── arkit_map.py          # ARKit BS → Live2D 参数映射（标准语义空间 → 模型量程）
│   │   ├── motion_author.py      # CSV → 低通/钳制/重采样 → motion3.json
│   │   ├── motion_deploy.py      # motion3 落盘 + model3.json Auto 组幂等注册
│   │   ├── make_synth_takes.py   # 合成 ARKit 夹具（blink/open_smile/turn_brow）
│   │   ├── qa_director.py        # QA Director（plan → 截帧 → 判定 → verdict）
│   │   ├── judges.py             # heuristic 判定 + contact sheet + JUDGE_CMD 协议
│   │   ├── common.py             # 共享基础设施（路径/服务/verdict IO）
│   │   └── rubrics/qa_rubric.md  # 多模态判官评分 rubric
│   │
│   ├── capture_server.py         # 线程化截帧服务器（/shot + /job 端点）
│   ├── reskin_ai.py              # AI 换色/全身素材替换管线（SAM → ComfyUI → 程序收尾）
│   ├── reskin_ai.sh              # 一键换肤脚本（封装 reskin_ai.py）
│   ├── reskin.sh                 # 即梦 i2i 重绘 + 程序保帧换色
│   │
│   ├── specs/                    # 全身素材替换 spec
│   │   ├── HaruWineGold.json
│   │   ├── HiyoriPinkGold.json
│   │   └── HiyoriPurpleBlack.json
│   │
│   ├── data/synth/               # 合成夹具 + QA 报告
│   │   ├── synth_blink.csv       #   60fps 合成 ARKit 数据
│   │   ├── synth_open_smile.csv
│   │   ├── synth_turn_brow.csv
│   │   └── b3_report.json        #   批量验证报告
│   │
│   └── .gitignore
│
└── vendor/                       # 前端依赖
    ├── pixi.min.js               #   PixiJS 渲染引擎
    ├── live2dcubismcore.min.js   #   Cubism Core
    └── pixi-live2d-display.min.js #   pixi-live2d-display 绑定层
```

---

## 快速开始

### 环境依赖

- Python 3.10+
- Node.js（可选，用于本地静态服务器）
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)（AI 换肤 inpainting）
- SAM 权重 `sam_vit_h_4b8939.pth`（放 `live2d-lab/` 根目录，2.4GB，`.gitignore` 已排除）

### 启动 Demo

```bash
cd live2d-lab
python3 -m http.server 8765
# 浏览器打开 http://127.0.0.1:8765/
# 拖动左侧滑块实时驱动 Live2D 参数
```

### AI 换肤

```bash
# 单部件换色（首次自动生成 SAM mask 对照表，人工确认后重跑）
./tools/reskin_ai.sh gold HaruGoldAI Haru
# 带 mask 编号重跑
MASKS=05,08,13,15 ./tools/reskin_ai.sh gold HaruGoldAI Haru

# 全身素材替换（spec 驱动）
SPEC=tools/specs/HaruWineGold.json ./tools/reskin_ai.sh
```

### 视频驱动动作合成

```bash
# 1. 生成合成夹具（无需真机）
python3 tools/agents/make_synth_takes.py

# 2. CSV → motion3.json
python3 tools/agents/motion_author.py \
    --csv tools/data/synth/synth_blink.csv \
    --model Haru \
    --out tools/data/out/synth_blink.motion3.json

# 3. 部署到模型 + 注册 Auto 组
python3 tools/agents/motion_deploy.py \
    --model Haru \
    --motion tools/data/out/synth_blink.motion3.json \
    --name synth_blink

# 4. QA 自动验收
python3 tools/agents/qa_director.py --model Haru --motion synth_blink
```

### QA 验收

```bash
# 启动截帧服务器
python3 tools/capture_server.py &

# 启动 Demo 静态服务器
python3 -m http.server 8765 &

# 静态极值姿势验收
python3 tools/agents/qa_director.py --model Haru

# 轨迹模式（多参数耦合穿帮检测）
python3 tools/agents/qa_director.py --model Haru --trajectory

# 动作模式（播放已注册 motion3）
python3 tools/agents/qa_director.py --model Haru --motion synth_blink

# 只重新判定已有 run（不重新截帧）
python3 tools/agents/qa_director.py --model Haru --judge-only
```

---

## 核心设计要点

### 参数注入（pixi-live2d-display 确定性）

QA Harness 在 `model.internalModel.on('beforeModelUpdate', applyHeld)` 注入参数——这是 framework 的正确挂钩点，保证参数在每帧更新前生效。**绝不能**手动调用 `coreModel.update()`，否则绕过 framework 的 save/load 会导致肢体异常。

```javascript
// qa_harness.js 核心注入逻辑
model.internalModel.on('beforeModelUpdate', () => {
    for (const [id, val] of Object.entries(heldParams))
        model.internalModel.coreModel.setParameterValueById(id, val);
});
```

### 确定性扫描

pixi-live2d-display 在每次更新后自动请求 Idle 组动作（含部件透明度切换），`stopAllMotions` 下一帧又重启。确定性扫描必须显式置空：

```javascript
model.internalModel.motionManager.groups.idle = null;
// 显式 startMotion 不受影响
```

### 五官局部动作 QA 判定

全身取景下五官只占几十~三百像素，RGB 均值差无法检测。改用**头部带 ROI**（角色 bbox 上部 32%）的变化像素计数：

| 判据 | 阈值 | 适用 |
|------|------|------|
| 面部变化像素 | ≥ 25px | 眨眼/嘴型 |
| 头身变化像素 | ≥ 200px | 转头/点头 |
| Alpha 占比 | ≥ 0.5% | 黑画布检测 |
| 帧间差突刺 | 6× 均值 | 轨迹异常 |

### ARKit → Live2D 参数映射

两段缩放策略，自动处理模型特异性（如 Haru `ParamMouthForm` 默认值=最大值、`ParamEyeLOpen` 量程 0..2）：

```
ARKit BS (0..1)
    │
    ▼ 第一段：映射到标准语义空间（Angle ±30°、EyeOpen 0..1 rest=1）
    │
    ▼ 第二段：以默认值为锚点，按 .motion-calib-<Model>.json 实测量程分段缩放
    │
    ▼ Live2D 参数（模型真实单位：角度为度、开合为量程内绝对值）
```

角度符号集中在 `ANGLE_SIGN` 常量，QA 校准时只改这里。

### AI 换肤颜色质检

SD1.5 inpaint 同 seed 下 mask 内容微差会导致采样路径完全改变，输出颜色不可控。质检流程：

1. **中位数判定**（非均值——高光/阴影会拖偏均值导致洗白擦线过关）
2. 判据：`V中位 > 目标+0.15` 或 `S中位 < 目标×0.6` → 换 seed 重试
3. 3 次全败 → 取 V 最接近者 → 整域 `hue_map` 颜色归一兜底

### 全身素材替换 spec 格式

```json
{
  "style": "整体风格语境，拼入所有 AI 组 prompt",
  "groups": [
    {
      "texture": 0,
      "masks": [1, 2, 5],
      "color": "gold",
      "mode": "ai",
      "prompt": "gold fabric texture, high quality"
    },
    {
      "texture": 1,
      "from": {"s": [0.145, 0.233]},
      "color": {"h": 0, "s": 0.85, "v": 0.15},
      "mode": "program"
    }
  ]
}
```

- `masks`：SAM 分割编号（首次运行自动生成对照表）
- `from`：源色选区（HSV 分位定界，零重叠精确认领）
- `mode`：`program` 纯 HSV 迁移 / `ai` ComfyUI inpainting

---

## 已落地模型

| 模型 | 来源 | 变更 |
|------|------|------|
| Haru | 官方 Cubism 4 SDK | 基准 |
| HaruGold / HaruGoldAI | AI 换肤 | 金发（程序 HSV / ComfyUI inpaint） |
| HaruWineGold | 全身 spec | 金发 + 酒红西装裙 + 金白条纹领巾 |
| Hiyori | 官方 Cubism 4 SDK | 基准 |
| HiyoriAI / HiyoriPinkGold / HiyoriPurpleBlack | AI 换肤 | 多色变体 |
| Mark | 官方 Cubism 4 SDK | 基准 |

Haru 已注册 3 个合成动作到 `Auto` 组：

| 动作 | 通道 | 时长 | QA 变化像素 |
|------|------|------|------------|
| `synth_blink` | EyeLOpen, EyeROpen | 4.2s | 302px（3 次眨眼） |
| `synth_open_smile` | EyeSmile + MouthOpenY | 3.8s | 222px（两轮张嘴） |
| `synth_turn_brow` | AngleX/Y + BrowLY/RY | 4.5s | 2130px（转头+挑眉） |

---

## 已知限制

- **自然度评分**：当前为 heuristic + 人工目检；`JUDGE_CMD` 多模态判官接入后替换为自动分数
- **动作截帧**：200ms 异步采样，单帧可能落在过渡中（参数精度验收走 trajectory 确定性模式）
- **覆盖范围**：合成夹具覆盖面部/头部；身体、手势、物理摆动不在视频驱动映射范围内
- **L2/L3 未实现**：当前仅 L1 几何映射，小模型训练和多模态生成是后续工作
- **网络依赖**：AI 换肤需 ComfyUI API 可达；SAM 首次分割需 2.4GB 权重

---

## 技术栈

| 层 | 技术 |
|----|------|
| 渲染 | PixiJS + pixi-live2d-display + Cubism Core |
| 动作映射 | ARKit BlendShapes / MediaPipe Face Landmarker |
| AI 换肤 | SAM (ViT-H) + ComfyUI (SD1.5 inpaint) + HSV 程序迁移 |
| QA 基础设施 | ThreadingHTTPServer + Canvas 离屏截帧 + numpy/PIL heuristic |
| 前端 | 原生 HTML/CSS/JS（无构建步骤） |

---

<div align="center">

**License**: MIT（代码）· 模型资产遵循 [Live2D Free Material License](https://www.live2d.com/en/download/free-material-license/)

</div>
