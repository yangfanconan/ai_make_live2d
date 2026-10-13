#!/usr/bin/env python3
"""按 plan.json 批量生成 i2i 裁片（断点续跑：已存在的结果自动跳过）"""
import json
import os
import subprocess
import sys
import time

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORK = f"{ROOT}/tools/.reskin-work"
CLI = os.path.expanduser("~/.local/bin/dreamina-canvas")
PROJECT_ID = json.load(open(f"{ROOT}/tools/.reskin-state.json"))["projectId"]
CFG = json.load(open(f"{ROOT}/tools/reskin.json"))
GEN_MODEL = CFG.get("generator", {}).get("model", "seedream_5.0_pro")
GEN_RES = CFG.get("generator", {}).get("resolution", "2K")
RATIOS = [("21:9", 21/9), ("16:9", 16/9), ("3:2", 1.5), ("4:3", 4/3), ("1:1", 1.0),
          ("3:4", 0.75), ("2:3", 2/3), ("9:16", 9/16)]


def run_cli(args, timeout=360):
    r = subprocess.run([CLI] + args, capture_output=True, text=True, timeout=timeout)
    return json.loads(r.stdout), r.returncode


def nearest_ratio(w, h):
    return min(RATIOS, key=lambda x: abs(x[1] - w / h))[0]


def main():
    plan = json.load(open(f"{WORK}/plan.json"))
    os.makedirs(f"{WORK}/gen", exist_ok=True)
    tasks = [(g, i) for g in plan["groups"] for i in g["islands"] if i["renderType"] == "i2i"]
    print(f"共 {len(tasks)} 张裁片待生成", flush=True)
    ok = fail = skip = 0
    for g, isl in tasks:
        name = os.path.basename(isl["crop"]).replace(".png", "")
        gen_path = f"{WORK}/gen/{name}.jpg"
        if os.path.exists(gen_path):
            skip += 1
            continue
        r = g["render"]
        w, h = Image.open(f"{WORK}/{isl['crop']}").size
        ratio = "1:1" if r.get("padSquare") else nearest_ratio(w, h)
        crop_path = f"{WORK}/{isl['crop']}"
        try:
            up, _ = run_cli(["resource", "upload", "--file", crop_path,
                             "--project-id", PROJECT_ID, "--name", f"crop-{name}"])
            res_id = up["data"]["resourceId"]
            run, _ = run_cli(["node", "create", "image", "--mode", "i2i",
                              "--model", GEN_MODEL, "--ref", f"res:{res_id}",
                              "--prompt", r["prompt"], "--ratio", ratio,
                              "--resolution", GEN_RES, "--title", f"v2-{name}", "--run"])
            submit = run["data"]["submission"]["submitId"]
            run_cli(["operation", "wait", submit, "--timeout", "300s"])
            st, _ = run_cli(["operation", "status", submit])
            rid = st["data"]["resources"][0]["resourceId"]
            run_cli(["resource", "download", rid, "--output", gen_path])
            print(f"[OK] {name} ratio={ratio} size={w}x{h}", flush=True)
            ok += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}", flush=True)
            fail += 1
        time.sleep(1)
    print(f"完成: ok={ok} skip={skip} fail={fail}", flush=True)
    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
