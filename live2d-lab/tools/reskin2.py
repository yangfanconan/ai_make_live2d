#!/usr/bin/env python3
"""Live2D 换肤管线 v2 —— 语义分组 plan/compose 两阶段合成器

plan:    python3 tools/reskin2.py plan   # 分析岛 -> 归组 -> 导出裁片 + plan.json
compose: python3 tools/reskin2.py compose  # 读 plan + 生成结果(gen/*.jpg) -> 蒙版回贴合成

分组规则见 tools/reskin.json；生成动作(即梦 CLI)在两阶段之间人工/脚本执行。
"""
import json
import os
import sys
from collections import deque

from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = f"{ROOT}/tools/reskin.json"
WORK = f"{ROOT}/tools/.reskin-work"


# ---------- 岛分析 ----------
def find_islands(alpha, scale_src=2048, thr=40, min_area=30):
    small = alpha.resize((512, 512), Image.NEAREST)
    px = small.load()
    visited = [[False] * 512 for _ in range(512)]
    isl = []
    for y in range(512):
        for x in range(512):
            if px[x, y] > thr and not visited[y][x]:
                q = deque([(x, y)]); visited[y][x] = True
                minx = maxx = x; miny = maxy = y; area = 0
                while q:
                    cx, cy = q.popleft(); area += 1
                    minx = min(minx, cx); maxx = max(maxx, cx)
                    miny = min(miny, cy); maxy = max(maxy, cy)
                    for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                        nx, ny = cx+dx, cy+dy
                        if 0 <= nx < 512 and 0 <= ny < 512 \
                                and not visited[ny][nx] and px[nx, ny] > thr:
                            visited[ny][nx] = True; q.append((nx, ny))
                if area > min_area:
                    k = scale_src / 512
                    isl.append({"box": [int(minx*k), int(miny*k), int(maxx*k), int(maxy*k)],
                                "area": int(area * k * k)})
    return isl


def skin_ratio(img, box):
    """岛内肤色像素占比（粗判：r>g>b 且偏暖）"""
    crop = img.crop(box).convert("RGBA")
    px = crop.load(); n = skin = 0
    for y in range(0, crop.size[1], 4):
        for x in range(0, crop.size[0], 4):
            r, g, b, a = px[x, y]
            if a > 60:
                n += 1
                if r > 120 and r >= g > b and 15 <= r - b <= 90:
                    skin += 1
    return skin / n if n else 0


def match_group(isl, tex, cfg, src_img):
    x0, y0, x1, y1 = isl["box"]
    w, h = x1 - x0, y1 - y0
    aspect = max(w, h) / max(1, min(w, h))
    orient = "vertical" if h >= w else "horizontal"
    for g in cfg["groups"]:
        m = g["match"]
        if "texture" in m and tex not in m["texture"]:
            continue
        a = m.get("area")
        if a and not (a[0] <= isl["area"] <= a[1]):
            continue
        asp = m.get("aspect")
        if asp and not (asp[0] <= aspect <= asp[1]):
            continue
        if "orient" in m and m["orient"] != orient:
            continue
        ty = m.get("topY")
        if ty and not (ty[0] <= y0 <= ty[1]):
            continue
        if "skinRatioMin" in m and skin_ratio(src_img, isl["box"]) < m["skinRatioMin"]:
            continue
        return g
    return cfg["defaults"]


def do_plan(cfg):
    src_dir = f"{ROOT}/assets/models/{cfg['model']}"
    tex_dir = [d for d in os.listdir(src_dir) if d.endswith(".2048")][0]
    os.makedirs(f"{WORK}/crops", exist_ok=True)
    plan = {"groups": []}
    grouped = {}
    for tex in (0, 1):
        orig = Image.open(f"{src_dir}/{tex_dir}/texture_{tex:02d}.png").convert("RGBA")
        white = Image.new("RGBA", orig.size, (255, 255, 255, 255))
        flat = Image.alpha_composite(white, orig).convert("RGB")
        for isl in find_islands(orig.split()[3]):
            g = match_group(isl, tex, cfg, flat)
            r = g["render"]
            entry = dict(isl)
            entry["texture"] = tex
            entry["groupId"] = g.get("id", "defaults")
            entry["renderType"] = r["type"]
            if r["type"] == "i2i":
                x0, y0, x1, y1 = isl["box"]
                pad = 24
                box = (max(0, x0-pad), max(0, y0-pad), min(2048, x1+pad), min(2048, y1+pad))
                crop = flat.crop(box)
                entry["rotated"] = bool(r.get("rotate")) and (y1-y0) > (x1-x0)
                if entry["rotated"]:
                    crop = crop.transpose(Image.ROTATE_90)
                cw, ch = crop.size
                if r.get("padSquare"):
                    side = max(cw, ch, 1024)
                    sq = Image.new("RGB", (side, side), (255, 255, 255))
                    cx, cy = (side-cw)//2, (side-ch)//2
                    sq.paste(crop, (cx, cy))
                    entry["contentBox"] = [cx, cy, cx+cw, cy+ch]
                    entry["square"] = side
                    crop = sq
                name = f"{entry['groupId']}_{tex}_{len(grouped.get(entry['groupId'], []))}"
                crop.save(f"{WORK}/crops/{name}.png")
                entry["crop"] = f"crops/{name}.png"
                entry["cropSize"] = [cw, ch]
                entry["pasteBox"] = list(box)
            grouped.setdefault(entry["groupId"], []).append(entry)
    for gid, items in grouped.items():
        g = next((x for x in cfg["groups"] if x.get("id") == gid), cfg["defaults"])
        plan["groups"].append({"id": gid, "label": g.get("label", gid),
                               "render": g["render"], "islands": items})
    json.dump(plan, open(f"{WORK}/plan.json", "w"), ensure_ascii=False, indent=1)
    for g in plan["groups"]:
        n_i2i = sum(1 for i in g["islands"] if i["renderType"] == "i2i")
        print(f"  {g['id']:14s} {g['label']:22s} 岛数={len(g['islands']):3d} "
              f"(i2i {n_i2i} / recolor {len(g['islands'])-n_i2i})")
    print(f"plan 已写入 {WORK}/plan.json；i2i 裁片在 {WORK}/crops/")


# ---------- 合成 ----------
def paste_island(out_px, gen_img, entry, alpha_src, feather=2):
    gen = gen_img
    if entry.get("padSquare"):
        side = entry["square"]
        if gen.size != (side, side):
            gen = gen.resize((side, side), Image.LANCZOS)
        gen = gen.crop(tuple(entry["contentBox"]))
    elif gen.size != tuple(entry["cropSize"]):
        gen = gen.resize(tuple(entry["cropSize"]), Image.LANCZOS)
    if entry.get("rotated"):
        gen = gen.transpose(Image.ROTATE_270)
    gp = gen.load()
    x0, y0, x1, y1 = entry["pasteBox"]
    mask = alpha_src.crop((x0, y0, x1, y1))
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    mp = mask.load()
    for y in range(y1 - y0):
        for x in range(x1 - x0):
            a = mp[x, y]
            if a:
                r, g, b = gp[x, y]
                op = out_px[x0 + x, y0 + y]
                wa = a / 255.0
                out_px[x0 + x, y0 + y] = (
                    int(r * wa + op[0] * (1 - wa)),
                    int(g * wa + op[1] * (1 - wa)),
                    int(b * wa + op[2] * (1 - wa)),
                    max(a, op[3]))


def do_compose(cfg):
    import colorsys
    sys.path.insert(0, ROOT)
    from tools.reskin import STYLES  # 复用 recolor 规则
    plan = json.load(open(f"{WORK}/plan.json"))
    src_dir = f"{ROOT}/assets/models/{cfg['model']}"
    tex_dir = [d for d in os.listdir(src_dir) if d.endswith(".2048")][0]
    out_dir = f"{ROOT}/assets/models/HiyoriAI/{tex_dir}"
    os.makedirs(out_dir, exist_ok=True)

    for tex in (0, 1):
        orig = Image.open(f"{src_dir}/{tex_dir}/texture_{tex:02d}.png").convert("RGBA")
        alpha = orig.split()[3]
        out = Image.new("RGBA", orig.size)
        opx = out.load()
        o = orig.load()
        # 底：未处理的区域保持 recolor（保守默认）
        rules = STYLES.get(cfg["defaults"]["render"].get("rules", "none"), [])
        for g in plan["groups"]:
            for isl in g["islands"]:
                if isl["texture"] != tex:
                    continue
                if isl["renderType"] == "i2i":
                    gen_path = f"{WORK}/gen/{os.path.basename(isl['crop']).replace('.png', '.jpg')}"
                    if not os.path.exists(gen_path):
                        print(f"  跳过（无生成结果）: {gen_path}")
                        continue
                    paste_island(opx, Image.open(gen_path).convert("RGB"), isl, alpha)
                else:
                    x0, y0, x1, y1 = isl["box"]
                    for y in range(y0, y1):
                        for x in range(x0, x1):
                            r, gg, b, a = o[x, y]
                            if a and rules:
                                from tools.reskin import apply_rules
                                r, gg, b = apply_rules((r, gg, b), rules)
                            opx[x, y] = (r, gg, b, a)
        out.putalpha(alpha.filter(ImageFilter.MinFilter(3)))
        out.save(f"{out_dir}/texture_{tex:02d}.png")
        print(f"  合成 texture_{tex:02d}.png -> {out_dir}")


if __name__ == "__main__":
    cfg = json.load(open(CFG_PATH))
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if cmd == "plan":
        do_plan(cfg)
    elif cmd == "compose":
        do_compose(cfg)
    else:
        print(__doc__)
