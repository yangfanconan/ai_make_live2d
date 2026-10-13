#!/usr/bin/env python3
"""单区精修工具：以当前上线贴图为底，只重绘指定区域（羽化回贴 + 自动 bump 版本）

用法：
  python3 tools/reskin_patch.py --name chest                    # 用配置表命名区域及其默认 prompt
  python3 tools/reskin_patch.py --name chest --prompt "新指令"   # 命名区域 + 自定义 prompt
  python3 tools/reskin_patch.py --region 1156,844,2032,1584 --texture 0 --ratio 4:3 --prompt "..."

命名区域定义在 tools/reskin.json 的 patchRegions。生成模型同样取自 generator 配置。
"""
import argparse
import json
import os
import re
import subprocess
import time

from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(f"{ROOT}/tools/reskin.json"))
CLI = os.path.expanduser("~/.local/bin/dreamina-canvas")
PROJECT_ID = json.load(open(f"{ROOT}/tools/.reskin-state.json"))["projectId"]
MODEL_DIR = f"{ROOT}/assets/models/{CFG['model']}"
OUT_DIR = f"{ROOT}/assets/models/HiyoriAI"
TMP = "/tmp/live2d-reskin"


def run_cli(args, timeout=360):
    r = subprocess.run([CLI] + args, capture_output=True, text=True, timeout=timeout)
    return json.loads(r.stdout)


def main():
    ap = argparse.ArgumentParser(description="Live2D 单区精修")
    ap.add_argument("--name", help="reskin.json patchRegions 里的命名区域")
    ap.add_argument("--region", help="自定义区域 x0,y0,x1,y1（像素坐标）")
    ap.add_argument("--texture", type=int, choices=[0, 1], help="目标贴图序号（默认取命名区域配置或 0）")
    ap.add_argument("--ratio", help="生成比例（默认 1:1）")
    ap.add_argument("--prompt", help="修正指令（默认取命名区域配置）")
    args = ap.parse_args()

    region = cfg_tex = ratio = prompt = None
    if args.name:
        pr = CFG.get("patchRegions", {}).get(args.name)
        if not pr:
            print(f"未知命名区域: {args.name}，可用: {list(CFG.get('patchRegions', {}))}")
            return 1
        box = pr["box"]; cfg_tex = pr.get("texture", 0)
        ratio = pr.get("ratio", "1:1"); prompt = args.prompt or pr.get("prompt")
    elif args.region:
        box = [int(v) for v in args.region.split(",")]
        prompt = args.prompt
    else:
        print(__doc__)
        return 1
    tex = args.texture if args.texture is not None else (cfg_tex if cfg_tex is not None else 0)
    ratio = args.ratio or ratio or "1:1"
    if not prompt:
        print("缺少 prompt（命名区域无默认值时必须显式传入）")
        return 1

    tex_dir = [d for d in os.listdir(MODEL_DIR) if d.endswith(".2048")][0]
    cur_path = f"{OUT_DIR}/{tex_dir}/texture_{tex:02d}.png"
    orig_path = f"{MODEL_DIR}/{tex_dir}/texture_{tex:02d}.png"
    os.makedirs(TMP, exist_ok=True)

    # 1. 裁片（以当前上线贴图为底）
    cur = Image.open(cur_path).convert("RGBA")
    crop = cur.crop(tuple(box)).convert("RGB")
    crop_path = f"{TMP}/patch_{int(time.time())}.png"
    crop.save(crop_path)
    print(f"裁片 {crop.size} ratio={ratio} -> {crop_path}")

    # 2. 上传 + 生成 + 等待 + 下载
    up = run_cli(["resource", "upload", "--file", crop_path, "--project-id", PROJECT_ID,
                  "--name", f"patch-{int(time.time())}"])
    res_id = up["data"]["resourceId"]
    run = run_cli(["node", "create", "image", "--mode", "i2i",
                   "--model", CFG["generator"]["model"], "--ref", f"res:{res_id}",
                   "--prompt", prompt, "--ratio", ratio,
                   "--resolution", CFG["generator"]["resolution"],
                   "--title", f"patch-{int(time.time())}", "--run"])
    submit = run["data"]["submission"]["submitId"]
    gen_res = run["data"]["submission"]["resources"][0]["resourceId"]
    print(f"已受理（{run['data']['quote']['maxCredits']} 积分），等待生成…")
    run_cli(["operation", "wait", submit, "--timeout", "300s"])
    gen_path = f"{TMP}/patch_gen_{int(time.time())}.jpg"
    run_cli(["resource", "download", gen_res, "--output", gen_path])

    # 3. 羽化回贴（原 alpha 蒙版）
    gen = Image.open(gen_path).convert("RGB").resize((box[2]-box[0], box[3]-box[1]), Image.LANCZOS)
    orig = Image.open(orig_path).convert("RGBA")
    mask = orig.crop(tuple(box)).split()[3].filter(ImageFilter.GaussianBlur(2))
    cpx, gpx, mpx = cur.load(), gen.load(), mask.load()
    for y in range(box[3]-box[1]):
        for x in range(box[2]-box[0]):
            a = mpx[x, y]
            if a:
                wa = a / 255.0
                r, g, b = gpx[x, y]
                op = cpx[box[0]+x, box[1]+y]
                cpx[box[0]+x, box[1]+y] = (int(r*wa+op[0]*(1-wa)), int(g*wa+op[1]*(1-wa)),
                                           int(b*wa+op[2]*(1-wa)), max(a, op[3]))
    cur.save(cur_path)

    # 4. bump 贴图版本
    mp = f"{OUT_DIR}/Hiyori.model3.json"
    m = json.load(open(mp))
    v = str(int(time.time()))
    m["FileReferences"]["Textures"] = [re.sub(r"\?v=\d+", "", t) + f"?v={v}"
                                       for t in m["FileReferences"]["Textures"]]
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent="\t")
    print(f"精修完成并回贴 {cur_path}，版本 bump ?v={v}（普通刷新 demo 即可）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
