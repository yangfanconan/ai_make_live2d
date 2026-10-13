#!/usr/bin/env bash
# Live2D 一键 AI 换色/全身素材替换：SAM 分割 → ComfyUI inpainting → 程序收尾 → 部署
# 用法 1（单部件换色）: ./tools/reskin_ai.sh <颜色> [输出模型名] [源模型名]
#   颜色: gold silver white black brown pink red blue purple green
# 用法 2（全身素材替换）: SPEC=tools/specs/HaruNavyGold.json ./tools/reskin_ai.sh
# 环境变量:
#   SPEC=path.json      全身替换 spec（多部件组，见 tools/specs/ 示例）
#   MASKS=05,08,13,15   单部件模式：指定目标部件 mask（首次会生成对照表提示人工确认）
#   PROMPT="..."        覆盖 inpaint 英文提示词
#   PARTS=0             单部件模式：要 inpaint 的贴图序号（默认 0）
#   RESEG=1             强制重跑 SAM 分割
#   NO_DEPLOY=1         只生成贴图不部署模型
# 示例:
#   ./tools/reskin_ai.sh gold HaruGoldAI Haru          # 首次：生成对照表后退出
#   MASKS=05,08,13,15 ./tools/reskin_ai.sh gold HaruGoldAI Haru
#   SPEC=tools/specs/HaruNavyGold.json ./tools/reskin_ai.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ -n "${SPEC:-}" ]]; then
  ARGS=(--spec "$SPEC")
else
  COLOR="${1:?用法: reskin_ai.sh <颜色> [输出模型名] [源模型名] 或 SPEC=spec.json}"
  OUT="${2:-${SRC_NAME:-}AI}"
  SRC="${3:-Hiyori}"
  ARGS=(--model "$SRC" --color "$COLOR" --out "$OUT")
  [[ -n "${MASKS:-}" ]] && ARGS+=(--masks "$MASKS")
  [[ -n "${PROMPT:-}" ]] && ARGS+=(--prompt "$PROMPT")
  [[ -n "${PARTS:-}" ]] && ARGS+=(--parts "$PARTS")
fi
[[ -n "${RESEG:-}" ]] && ARGS+=(--reseg)
[[ -n "${NO_DEPLOY:-}" ]] && ARGS+=(--no-deploy)

exec python3 "$ROOT/tools/reskin_ai.py" "${ARGS[@]}"
