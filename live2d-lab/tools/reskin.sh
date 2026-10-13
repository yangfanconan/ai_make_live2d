#!/usr/bin/env bash
# Live2D 一键换肤：即梦 i2i 重绘大部件 + 本地程序保帧换色
# 用法: ./tools/reskin.sh "<风格提示词（外观描述）>" [输出模型名，默认 HiyoriAI] [源模型名，默认 Hiyori]
# 依赖: dreamina-canvas CLI（已登录）、python3 + Pillow
# 环境变量: STYLE=none 可跳过表情帧换色（默认 silver_red）
set -euo pipefail

PROMPT_HEAD="${1:?用法: ./tools/reskin.sh \"<风格描述>\" [输出模型名] [源模型名]}"
OUT_NAME="${2:-HiyoriAI}"
SRC_NAME="${3:-Hiyori}"
STYLE="${STYLE:-silver_red}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC_MODEL="$ROOT/assets/models/$SRC_NAME"
OUT_MODEL="$ROOT/assets/models/$OUT_NAME"
STATE="$ROOT/tools/.reskin-state.json"
TMP="$(mktemp -d /tmp/live2d-reskin.XXXXXX)"
CLI="${DREAMINA_CANVAS_CLI:-$HOME/.local/bin/dreamina-canvas}"
# 生成模型从 reskin.json 统一读取
GEN_MODEL=$(python3 -c "import json; print(json.load(open('$ROOT/tools/reskin.json'))['generator']['model'])")
GEN_RES=$(python3 -c "import json; print(json.load(open('$ROOT/tools/reskin.json'))['generator']['resolution'])")

J() { python3 -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1]))" "$1"; }

# 1. 画布复用（持久化 projectId，最小 job state，不含凭据）
if [[ -f "$STATE" ]]; then
  PROJECT_ID=$(J "d['projectId']" < "$STATE")
else
  CREATE=$("$CLI" canvas create "live2d-skin-lab" --use)
  PROJECT_ID=$(echo "$CREATE" | J "d['data']['project']['projectId']")
  echo "{\"projectId\": \"$PROJECT_ID\"}" > "$STATE"
fi
echo "画布: https://jimeng.jianying.com/ai-tool/ai-canvas/$PROJECT_ID"

# 2. 每张图集：白底化 -> 上传 -> i2i -> 等待 -> 下载
TEXDIR=$(basename "$(ls -d "$SRC_MODEL"/*.2048 2>/dev/null | head -1)")
GENS=()
for IDX in 00 01; do
  ORIG="$SRC_MODEL/$TEXDIR/texture_$IDX.png"
  FLAT="$TMP/flat_$IDX.png"
  python3 - "$ORIG" "$FLAT" <<'EOF'
import sys
from PIL import Image
img = Image.open(sys.argv[1]).convert("RGBA")
white = Image.new("RGBA", img.size, (255, 255, 255, 255))
Image.alpha_composite(white, img).convert("RGB").save(sys.argv[2])
EOF
  UP=$("$CLI" resource upload --file "$FLAT" --project-id "$PROJECT_ID" \
        --name "atlas-$IDX-$(date +%s)")
  RES_ID=$(echo "$UP" | J "d['data']['resourceId']")
  RUN=$("$CLI" node create image --mode i2i --model "$GEN_MODEL" \
        --ref "res:$RES_ID" \
        --prompt "这是动漫角色模型的纹理图集，白色区域是留白，绝对保持空白。图集中每个彩色部件必须出现在与输入图完全相同的位置、相同的形状和相同的大小，像贴纸一样原位重新绘制。禁止发明新的部件，禁止改变部件间距和整体布局。角色外观：$PROMPT_HEAD 平涂动漫风格。" \
        --ratio 1:1 --resolution "$GEN_RES" --title "reskin-atlas-$IDX" --run)
  SUBMIT=$(echo "$RUN" | J "d['data']['submission']['submitId']")
  "$CLI" operation wait "$SUBMIT" --timeout 300s > /dev/null
  RES=$(python3 -c "
import json, subprocess, sys
out = subprocess.run(['$CLI', 'operation', 'status', '$SUBMIT'], capture_output=True, text=True).stdout
d = json.loads(out)
print(d['data']['resources'][0]['resourceId'])")
  GEN="$TMP/gen_$IDX.jpg"
  "$CLI" resource download "$RES" --output "$GEN" > /dev/null
  GENS+=("$GEN")
  echo "atlas-$IDX 生成完成: $GEN"
done

# 3. 复制官方模型骨架（moc3/物理/动作 零改动），合成新贴图
rm -rf "$OUT_MODEL"
cp -R "$SRC_MODEL" "$OUT_MODEL"
python3 "$ROOT/tools/reskin.py" \
  --src "$SRC_MODEL" --out "$OUT_MODEL" \
  --gen-00 "${GENS[0]}" --gen-01 "${GENS[1]}" \
  --style "$STYLE"

echo "完成。demo 中加载: assets/models/$OUT_NAME/$(basename "$SRC_MODEL"/*.model3.json)（普通刷新即可）"
