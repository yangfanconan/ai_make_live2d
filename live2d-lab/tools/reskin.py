#!/usr/bin/env python3
"""Live2D 贴图换肤合成器（结构感知分区混合）

管线原理：
  官方模型图集 = 表情帧阵列（小碎片岛）+ 大部件岛（脸/发/身/裙）。
  - 表情帧区：AI 整图重绘会破坏多姿态帧之间的对应关系（眨眼/口型穿帮），
    改用程序化 recolor（HSV 色相迁移），像素级保帧。
  - 大部件区：使用 AI i2i 生成图，获得重新设计的自由度与质感。
  最后统一套回原 alpha（边缘收缩 1px 去色晕）。

用法：
  python3 tools/reskin.py --src <官方模型目录> --out <输出目录> \
      --gen-00 <AI生成的图集0.jpg> --gen-01 <AI生成的图集1.jpg> \
      --style silver_red [--small-area 25000] [--dilate 8]
"""
import argparse
import colorsys
from collections import deque

from PIL import Image, ImageFilter

# 预置 recolor 规则：表情帧区的针对性换色（HSV 条件 → 目标色）
# 每条规则: (条件, 替换函数)。条件按顺序匹配，命中即停。
STYLES = {
    # 银白双马尾 + 红瞳 + 深蓝金水手服（配套 prompt 见 tools/reskin.sh）
    "silver_red": [
        # 蓝色虹膜 → 红色
        (lambda h, s, v: s > 0.25 and v > 0.2 and 180 <= h <= 260,
         lambda h, s, v: (352, min(0.85, s + 0.15), v * 0.9)),
        # 棕色眉/发丝帧 → 银灰
        (lambda h, s, v: 0.18 <= s <= 0.75 and 0.15 <= v <= 0.6 and 15 <= h <= 50,
         lambda h, s, v: (30, 0.06, 0.85)),
    ],
    # 不换色（表情帧区保持原样，仅主体区用 AI 图）
    "none": [],
}


def find_islands(alpha, thr=40, min_area=30, scale=4):
    """在 512 降采样 alpha 上找连通域，返回原尺寸坐标的 (x0,y0,x1,y1,area)。"""
    small = alpha.resize((512, 512), Image.NEAREST)
    px = small.load()
    visited = [[False] * 512 for _ in range(512)]
    islands = []
    for y in range(512):
        for x in range(512):
            if px[x, y] > thr and not visited[y][x]:
                q = deque([(x, y)])
                visited[y][x] = True
                minx = maxx = x
                miny = maxy = y
                area = 0
                while q:
                    cx, cy = q.popleft()
                    area += 1
                    minx = min(minx, cx); maxx = max(maxx, cx)
                    miny = min(miny, cy); maxy = max(maxy, cy)
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                                   (1, 1), (1, -1), (-1, 1), (-1, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < 512 and 0 <= ny < 512 \
                                and not visited[ny][nx] and px[nx, ny] > thr:
                            visited[ny][nx] = True
                            q.append((nx, ny))
                if area > min_area:
                    islands.append((minx * scale, miny * scale,
                                    maxx * scale, maxy * scale, area * scale * scale))
    return islands


def apply_rules(rgb, rules):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    hd = h * 360
    for cond, fn in rules:
        if cond(hd, s, v):
            nh, ns, nv = fn(hd, s, v)
            nr, ng, nb = colorsys.hsv_to_rgb(nh / 360, ns, nv)
            return int(nr * 255), int(ng * 255), int(nb * 255)
    return r, g, b


def process_atlas(orig_path, gen_path, out_path, rules, small_area, dilate):
    orig = Image.open(orig_path).convert("RGBA")
    gen = Image.open(gen_path).convert("RGB")
    if gen.size != orig.size:
        gen = gen.resize(orig.size, Image.LANCZOS)

    islands = find_islands(orig.split()[3])
    size = orig.size
    mask = Image.new("L", size, 0)
    mp = mask.load()
    n_small = 0
    for (x0, y0, x1, y1, area) in islands:
        if area < small_area:
            n_small += 1
            for yy in range(max(0, y0 - dilate), min(size[1], y1 + dilate)):
                for xx in range(max(0, x0 - dilate), min(size[0], x1 + dilate)):
                    mp[xx, yy] = 255

    op, gp = orig.load(), gen.load()
    out = Image.new("RGBA", size)
    opx = out.load()
    for y in range(size[1]):
        for x in range(size[0]):
            if mp[x, y]:
                r, g, b, a = op[x, y]
                nr, ng, nb = apply_rules((r, g, b), rules)
                opx[x, y] = (nr, ng, nb, a)
            else:
                r, g, b = gp[x, y]
                _, _, _, a = op[x, y]
                opx[x, y] = (r, g, b, a)

    alpha = orig.split()[3].filter(ImageFilter.MinFilter(3))
    out.putalpha(alpha)
    out.save(out_path)
    print(f"{out_path.split('/')[-1]}: 表情帧岛(程序recolor)={n_small} "
          f"大部件岛(AI)={len(islands) - n_small}")


def main():
    ap = argparse.ArgumentParser(description="Live2D 结构感知换肤合成")
    ap.add_argument("--src", required=True, help="官方模型目录（含 *.2048/ 贴图）")
    ap.add_argument("--out", required=True, help="输出模型目录（贴图写入其贴图子目录）")
    ap.add_argument("--gen-00", required=True, help="AI 生成的 texture_00 图")
    ap.add_argument("--gen-01", required=True, help="AI 生成的 texture_01 图")
    ap.add_argument("--style", default="silver_red", choices=list(STYLES.keys()))
    ap.add_argument("--small-area", type=int, default=25000,
                    help="小于该像素面积的岛视为表情帧区（默认 25000）")
    ap.add_argument("--dilate", type=int, default=8, help="表情帧 mask 外扩像素")
    args = ap.parse_args()

    import os
    tex_dirs = [d for d in os.listdir(args.src)
                if d.endswith(".2048") and os.path.isdir(os.path.join(args.src, d))]
    assert len(tex_dirs) == 1, f"期望恰好一个贴图目录，实际: {tex_dirs}"
    tex = tex_dirs[0]
    src_dir = os.path.join(args.src, tex)
    out_dir = os.path.join(args.out, tex)
    os.makedirs(out_dir, exist_ok=True)

    rules = STYLES[args.style]
    process_atlas(f"{src_dir}/texture_00.png", args.gen_00,
                  f"{out_dir}/texture_00.png", rules, args.small_area, args.dilate)
    process_atlas(f"{src_dir}/texture_01.png", args.gen_01,
                  f"{out_dir}/texture_01.png", rules, args.small_area, args.dilate)

    # 自动 bump model3.json 贴图版本（击穿浏览器缓存，普通刷新即可看到新版）
    import re, time, json as _json, glob as _glob
    mps = _glob.glob(os.path.join(args.out, "*.model3.json"))
    if mps:
        m = _json.load(open(mps[0]))
        v = str(int(time.time()))
        m["FileReferences"]["Textures"] = [
            re.sub(r"\?v=\d+", "", t) + f"?v={v}" for t in m["FileReferences"]["Textures"]]
        _json.dump(m, open(mps[0], "w"), ensure_ascii=False, indent="\t")
        print(f"贴图缓存版本已更新: ?v={v}")
    print(f"完成：{out_dir}/texture_0*.png（硬刷新 demo 即可查看）")


if __name__ == "__main__":
    main()
