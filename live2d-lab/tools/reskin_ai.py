#!/usr/bin/env python3
"""Live2D 一键 AI 换色/全身素材替换管线（SAM 分割 → ComfyUI inpainting → 程序收尾）

两种用法：
  A. 单部件换色（向后兼容）：
     python3 tools/reskin_ai.py --model Haru --out HaruGoldAI --color gold \
         [--masks 05,08,13,15] [--prompt "..."] [--reseg] [--small-comp 3000]
     首次运行若未指定 --masks，会生成对照表 tools/<Model>_masks_sheet.png
     并退出，人工确认后带 --masks 重跑。
  B. 全身素材替换（spec 驱动，多部件组）：
     python3 tools/reskin_ai.py --spec tools/specs/HaruNavyGold.json
     spec 每组声明 texture + masks（SAM 编号）或 from（源色选区）+ color（预设名
     或 {h,s,v[,prompt]}）+ mode（program 纯HSV迁移 / ai ComfyUI inpaint）。
     所有组的原色签名在任何像素改写前统一采样；残留岛迁移按组签名认领。

管线步骤（每组）：
  1. SAM 分割（按需懒跑，缓存 tools/masks/<Model>_t<idx>/）
  2. 解析组 mask（SAM 并集或颜色连通域），剔除亮红腮红
  3. program: 岛体+邻近深描边整域 hue_map（保相对明暗）
     ai: ComfyUI + SD1.5 inpainting @768（中位质检，洗白换 seed 重试 3 次）
         → 羽化蒙版合成回 2048、保留原 alpha → 颜色归一 → 亮度协调
  4. 残留岛色相迁移：全贴图扫描与该组原色签名同色的大连通域，整域迁移
  5. 统一落盘 + 部署到新模型目录 + bump model3.json 缓存版本
"""
import argparse
import colorsys
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import deque

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMFY = os.path.join(os.path.dirname(ROOT), 'ComfyUI')
COMFY_URL = 'http://127.0.0.1:8188'
SAM_CKPT = os.path.join(ROOT, 'sam_vit_h_4b8939.pth')

# 目标色预设：(H°, S, V基准, 英文prompt片段)
COLORS = {
    'gold':   (22, 0.43, 0.61, 'golden blonde anime hair'),
    'silver': (0, 0.05, 0.85, 'silver white anime hair'),
    'white':  (0, 0.03, 0.92, 'white anime hair'),
    'black':  (0, 0.08, 0.18, 'black anime hair'),
    'brown':  (25, 0.45, 0.45, 'brown anime hair'),
    'pink':   (340, 0.50, 0.80, 'pink anime hair'),
    'red':    (355, 0.65, 0.55, 'red anime hair'),
    'blue':   (220, 0.50, 0.55, 'blue anime hair'),
    'purple': (280, 0.45, 0.55, 'purple anime hair'),
    'green':  (130, 0.45, 0.55, 'green anime hair'),
}
NEG = 'lowres, blurry, artifacts, text, watermark, realistic, 3d'


def log(msg):
    print(f'[reskin-ai] {msg}', flush=True)


def rgb_to_hsv_arrays(rgb):
    """rgb float32 (H,W,3) 0-1 → h(0-360),s,v 数组"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = rgb.max(-1); mn = rgb.min(-1); d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-6), 0)
    v = mx
    h = np.zeros_like(mx)
    m = (mx == r) & (d > 0); h[m] = ((g - b)[m] / d[m]) % 6
    m = (mx == g) & (d > 0); h[m] = (b - r)[m] / d[m] + 2
    m = (mx == b) & (d > 0); h[m] = (r - g)[m] / d[m] + 4
    return h * 60, s, v


def hue_map_pixels(arr, mask, th, ts, tv_base, v_floor=None):
    """把 arr[mask] 像素整束映射到目标色相（保持相对明暗），原地修改"""
    sel = arr[mask]
    if len(sel) == 0:
        return
    vv = sel[..., :3].astype(np.float32).max(-1) / 255
    k = tv_base / max(vv.mean(), 1e-3)
    out = np.empty_like(sel)
    for i, px in enumerate(sel):
        hh, ss, v2 = colorsys.rgb_to_hsv(px[0] / 255, px[1] / 255, px[2] / 255)
        nv = min(0.95, v2 * k) if v_floor is None else float(np.clip(v2 * k, v_floor, 0.95))
        nr, ng, nb = colorsys.hsv_to_rgb(th / 360, ts, nv)
        out[i] = [int(nr * 255), int(ng * 255), int(nb * 255), px[3]]
    arr[mask] = out


def label_components(sel):
    """连通域标记，优先 scipy，退化到 BFS"""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(sel)
        return lab, n
    except ImportError:
        lab = np.zeros(sel.shape, np.int32)
        n = 0
        H, W = sel.shape
        for y in range(H):
            for x in range(W):
                if sel[y, x] and lab[y, x] == 0:
                    n += 1
                    q = deque([(y, x)]); lab[y, x] = n
                    while q:
                        cy, cx = q.popleft()
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < H and 0 <= nx < W and sel[ny, nx] and lab[ny, nx] == 0:
                                    lab[ny, nx] = n
                                    q.append((ny, nx))
        return lab, n


# ---------------- ComfyUI ----------------

def ensure_comfyui():
    try:
        urllib.request.urlopen(f'{COMFY_URL}/', timeout=2)
        log('ComfyUI 已在运行')
        return
    except Exception:
        pass
    log('启动 ComfyUI (--cpu) ...')
    subprocess.Popen(
        [sys.executable, 'main.py', '--cpu', '--listen', '127.0.0.1', '--port', '8188'],
        cwd=COMFY, stdout=open('/tmp/comfyui.log', 'ab'), stderr=subprocess.STDOUT)
    for _ in range(60):
        time.sleep(2)
        try:
            urllib.request.urlopen(f'{COMFY_URL}/', timeout=2)
            log('ComfyUI 就绪')
            return
        except Exception:
            continue
    raise RuntimeError('ComfyUI 启动超时，查看 /tmp/comfyui.log')


def comfy_inpaint(tex768_path, mask768_path, prompt, tag, seed=42):
    wf = json.load(open(os.path.join(ROOT, 'tools', 'comfyui_inpaint_api.json')))
    wf['3']['inputs']['seed'] = seed
    wf['10']['inputs']['image'] = os.path.basename(tex768_path)
    wf['11']['inputs']['image'] = os.path.basename(mask768_path)
    wf['6']['inputs']['text'] = prompt
    wf['9']['inputs']['filename_prefix'] = f'reskin_ai_{tag}'
    import uuid
    payload = json.dumps({'prompt': wf, 'client_id': str(uuid.uuid4())}).encode()
    req = urllib.request.Request(f'{COMFY_URL}/prompt', data=payload,
                                 headers={'Content-Type': 'application/json'})
    resp = json.load(urllib.request.urlopen(req))
    if resp.get('node_errors'):
        raise RuntimeError(f'工作流节点错误: {resp["node_errors"]}')
    pid = resp['prompt_id']
    t0 = time.time()
    while time.time() - t0 < 900:
        time.sleep(10)
        d = json.load(urllib.request.urlopen(f'{COMFY_URL}/history/{pid}'))
        if pid in d:
            st = d[pid].get('status', {})
            if st.get('status_str') == 'success':
                for o in d[pid].get('outputs', {}).values():
                    for im in o.get('images', []):
                        p = os.path.join(COMFY, 'output', im['filename'])
                        log(f'inpaint 完成: {im["filename"]} ({int(time.time()-t0)}s)')
                        return p
                raise RuntimeError('成功但无输出图')
            if st.get('status_str') == 'error':
                raise RuntimeError('ComfyUI 执行错误，查看 /tmp/comfyui.log')
        if int(time.time() - t0) % 60 == 0:
            log(f'inpaint 运行中... {int(time.time()-t0)}s')
    raise RuntimeError('inpaint 超时')


# ---------------- SAM ----------------

def segment_texture(tex_path, cache_dir, reseg=False):
    if os.path.isdir(cache_dir) and os.listdir(cache_dir) and not reseg:
        return
    os.makedirs(cache_dir, exist_ok=True)
    log(f'SAM 分割 {os.path.basename(tex_path)} ...')
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    sam = sam_model_registry['vit_h'](checkpoint=SAM_CKPT)
    sam.to('cpu').eval()
    gen = SamAutomaticMaskGenerator(sam, points_per_side=32, pred_iou_thresh=0.86,
                                    stability_score_thresh=0.9,
                                    min_mask_region_area=500)
    img = np.array(Image.open(tex_path).convert('RGB'))
    anns = gen.generate(img)
    anns.sort(key=lambda a: -a['area'])
    for i, ann in enumerate(anns):
        m = (ann['segmentation'] * 255).astype(np.uint8)
        Image.fromarray(m).save(os.path.join(cache_dir, f'mask_{i:02d}.png'))
    del sam, gen
    log(f'  共 {len(anns)} 区域 → {cache_dir}')


def masks_sheet(tex_path, cache_dir, out_png, top=20):
    tex = np.array(Image.open(tex_path).convert('RGB'))
    paths = sorted(os.listdir(cache_dir))[:top]
    cell = 224; cols = 5; rows = (len(paths) + cols - 1) // cols
    sheet = Image.new('RGB', (cols * cell, rows * (cell + 18)), (30, 30, 30))
    d = ImageDraw.Draw(sheet)
    for i, name in enumerate(paths):
        m = np.array(Image.open(os.path.join(cache_dir, name)).convert('L')) > 127
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        crop = tex[y0:y1 + 1, x0:x1 + 1].copy()
        mc = m[y0:y1 + 1, x0:x1 + 1]
        crop[mc] = (crop[mc] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(np.uint8)
        im = Image.fromarray(crop); im.thumbnail((cell - 8, cell - 8))
        cx, cy = (i % cols) * cell, (i // cols) * (cell + 18)
        sheet.paste(im, (cx + 4, cy + 4))
        d.text((cx + 6, cy + cell - 2),
               f"{name[:-4]} {m.sum()//1000}k ({x0},{y0})", fill=(255, 255, 120))
    sheet.save(out_png)


# ---------------- spec 与组处理 ----------------

def resolve_color(c):
    """颜色预设名或 {h,s,v[,prompt]} → (h, s, v, prompt片段)"""
    if isinstance(c, str):
        th, ts, tv, cp = COLORS[c]
        return th, ts, tv, cp
    return float(c['h']), float(c['s']), float(c['v']), \
        c.get('prompt') or 'flat anime colors'


def load_spec(args):
    """--spec 文件优先；否则由单色参数合成单组 spec（向后兼容）"""
    if args.spec:
        spec = json.load(open(args.spec))
        return (spec.get('model', args.model), spec.get('out', args.out),
                spec.get('groups', []), spec.get('defaults', {}),
                spec.get('style', ''))
    conf_path = os.path.join(ROOT, 'tools', f'.reskin-ai-{args.model}.json')
    conf = json.load(open(conf_path)) if os.path.exists(conf_path) else {}
    masks_ids = args.masks or conf.get('masks')
    ids = None
    if masks_ids:
        ids = [int(m) for m in masks_ids.split(',')]
        conf['masks'] = masks_ids
        json.dump(conf, open(conf_path, 'w'))
    groups = [{'id': 'part0', 'texture': int(args.parts.split(',')[0]),
               'masks': ids, 'color': args.color, 'mode': 'ai',
               'prompt': args.prompt}]
    return args.model, args.out, groups, {}, ''


def group_masks(g, masks_dir, arr, blush):
    """解析组的 mask：SAM 编号并集，或 from 源色连通域选区；随后剔除腮红"""
    merged = None
    if g.get('masks'):
        for mid in g['masks']:
            p = os.path.join(masks_dir, f'mask_{int(mid):02d}.png')
            m = np.array(Image.open(p).convert('L'))
            merged = m if merged is None else np.maximum(merged, m)
        merged = merged > 127
    elif g.get('from'):
        f = g['from']
        h0, s0, v0 = rgb_to_hsv_arrays(arr[..., :3].astype(np.float32) / 255)
        cand = (np.abs(((h0 - f['h'] + 180) % 360) - 180) <= f.get('tol_h', 30)) & \
               (np.abs(s0 - f['s']) <= f.get('tol_s', 0.22)) & \
               (np.abs(v0 - f['v']) <= f.get('tol_v', 0.28)) & (arr[..., 3] > 0)
        merged = np.zeros(arr.shape[:2], bool)
        lab, n = label_components(cand)
        for i in range(1, n + 1):
            comp = lab == i
            if comp.sum() >= f.get('min_area', 2000):
                merged |= comp
        if f.get('whole_island') and merged.any():
            # 整岛纳入：以深描边(v<=0.35)为天然边界的连通色块，
            # 种子命中的岛整体入选（光照阴影块随主体一起换色，避免染色染一半）
            body = (arr[..., 3] > 0) & (v0 > 0.35)
            lab2, n2 = label_components(body)
            hit = set(np.unique(lab2[merged])) - {0}
            merged = np.isin(lab2, list(hit))
    else:
        return None
    if blush is not None and not g.get('keep_blush'):
        merged &= ~blush
    return merged


def sample_signature(arr, merged):
    """组原色签名（中位 h/s/v），必须在任何像素改写前采样"""
    h0, s0, v0 = rgb_to_hsv_arrays(arr[..., :3].astype(np.float32) / 255)
    sel = merged & (s0 > 0.06) & (v0 > 0.1) & (v0 < 0.92) & (arr[..., 3] > 0)
    if sel.sum() == 0:
        return None
    return (float(np.median(h0[sel])), float(np.median(s0[sel])),
            float(np.median(v0[sel])))


def recolor_program(arr, merged, th, ts, tv):
    """program 换色：岛体 + 2px 邻近深色描边整域 hue_map（保相对明暗）"""
    vv = arr[..., :3].astype(np.float32).max(-1) / 255
    dil = np.array(Image.fromarray((merged * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(5))) > 127
    sel = merged | (dil & (vv < 0.5))
    hue_map_pixels(arr, sel, th, ts, tv)
    return int(sel.sum())


def inpaint_group(arr, merged, prompt, th, ts, tv, tag):
    """ai 换色：768 inpaint（中位质检重试）→ 羽化合成（保 alpha）→ 颜色归一 → 亮度协调"""
    H, W = arr.shape[:2]
    flat = Image.new('RGBA', (W, H), (255, 255, 255, 255))
    flat.alpha_composite(Image.fromarray(arr, 'RGBA'))
    Image.fromarray(np.array(flat.convert('RGB'))).resize((768, 768), Image.LANCZOS).save(
        f'{COMFY}/input/{tag}_tex.png')
    mask768_path = f'{COMFY}/input/{tag}_mask.png'
    Image.fromarray((merged * 255).astype(np.uint8)).resize((768, 768), Image.NEAREST).save(
        mask768_path)
    m768 = np.array(Image.open(mask768_path).convert('L')) > 127
    # 结构质检：SD 颜色靠不住（归一兜底），但纹理必须有——V 标准差过低即白板重试
    gen, best, seed = None, None, 42
    for attempt in range(1, 4):
        out768_path = comfy_inpaint(f'{COMFY}/input/{tag}_tex.png', mask768_path,
                                    prompt, tag, seed=seed)
        g768 = np.array(Image.open(out768_path).convert('RGB'))
        _, gs_, gv = rgb_to_hsv_arrays(g768.astype(np.float32) / 255)
        vstd = float(np.std(gv[m768]))
        ok = vstd >= 0.06
        log(f'  [{tag}] gen 质检 #{attempt} seed={seed}: V纹理std={vstd:.3f} '
            f'→ {"通过" if ok else "白板，换 seed 重试"}')
        cand = (vstd, np.array(Image.open(out768_path).convert('RGB')
                                 .resize((W, H), Image.LANCZOS)))
        if best is None or cand[0] > best[0]:
            best = cand
        if ok:
            gen = cand[1]
            break
        seed += 1
    if gen is None:
        log(f'  [{tag}] 3 次均白板，取纹理最丰富的输出兜底')
        gen = best[1]
    # 合成：羽化蒙版混合 RGB，保留原 alpha
    mf = Image.fromarray((merged * 255).astype(np.uint8))
    mf = mf.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.2))
    m3 = (np.array(mf, dtype=np.float32) / 255)[..., None]
    rgb = arr[..., :3].astype(np.float32) * (1 - m3) + gen.astype(np.float32) * m3
    arr[..., :3] = rgb.astype(np.uint8)
    # 颜色归一：inpaint 区显著偏离目标色时整域 hue_map 拉回（颜色靠程序兜底）
    _, ss_n, vv_n = rgb_to_hsv_arrays(arr[..., :3].astype(np.float32) / 255)
    vm2, sm2 = float(np.median(vv_n[merged])), float(np.median(ss_n[merged]))
    if sm2 < ts * 0.75 or abs(vm2 - tv) > 0.12:
        hue_map_pixels(arr, merged, th, ts, tv, v_floor=0.2)
        log(f'  [{tag}] 颜色归一: v中位={vm2:.2f} s中位={sm2:.2f} → 目标(tv={tv} ts={ts})')
    # 亮度协调：大岛过暗时提亮暗部（防深色帽圈）
    lab, n = label_components(merged)
    v_all = arr[..., :3].astype(np.float32).max(-1) / 255
    for i in range(1, n + 1):
        comp = lab == i
        if comp.sum() < 20000:
            continue
        vv = v_all[comp]
        med = float(np.median(vv))
        if med < tv * 0.78:
            dark = comp & (v_all < 0.45)
            vs_ = v_all[dark]
            nv = 0.42 + (vs_ - vs_.min()) / max(vs_.max() - vs_.min(), 1e-3) * 0.14
            sel_px = arr[dark]
            outp = np.empty_like(sel_px)
            for j, px in enumerate(sel_px):
                hh, ss, v2 = colorsys.rgb_to_hsv(px[0]/255, px[1]/255, px[2]/255)
                nr, ng, nb = colorsys.hsv_to_rgb(th/360, min(0.5, ss*1.1), float(nv[j]))
                outp[j] = [int(nr*255), int(ng*255), int(nb*255), px[3]]
            arr[dark] = outp
            log(f'  [{tag}] 亮度协调: 岛{i} V中位 {med:.2f} 过暗，已提亮 {dark.sum()}px')
    return np.array(Image.fromarray((merged * 255).astype(np.uint8)).filter(
        ImageFilter.MaxFilter(9))) > 127


def migrate_stray(arrs, texs, sig, th, ts, tv, excl, small_comp, where=''):
    """残留岛迁移：全贴图扫描与组原色签名同色的大连通域，整域迁移（excl 为所有组区域并集）"""
    if sig is None:
        log(f'{where}未采样到原色签名，跳过残留扫描')
        return
    sig_h, sig_s, sig_v = sig
    for t in texs:
        idx = t[8:10]
        arr = arrs[idx]
        hh, ss, vv = rgb_to_hsv_arrays(arr[..., :3].astype(np.float32) / 255)
        cand = (np.abs(((hh - sig_h + 180) % 360) - 180) < 30) & (ss > 0.03) & \
               (vv > 0.1) & (vv < 0.92) & (arr[..., 3] > 0)
        if idx in excl:
            cand &= ~excl[idx]
        cand_lab, cn = label_components(cand)
        for i in range(1, cn + 1):
            comp = cand_lab == i
            if comp.sum() < small_comp:
                continue
            # 组件级签名校验：中位 h/s/v 都要贴近组签名（防止误染其他部件）
            cm_h = float(np.median(hh[comp])); cm_s = float(np.median(ss[comp]))
            cm_v = float(np.median(vv[comp]))
            if abs(((cm_h - sig_h + 180) % 360) - 180) > 20 or \
               abs(cm_s - sig_s) > 0.13 or abs(cm_v - sig_v) > 0.20:
                log(f'  {where}跳过 t{idx} 岛 {comp.sum()}px（色签不符 h{cm_h:.0f} s{cm_s:.2f} v{cm_v:.2f}）')
                continue
            # 岛体 + 2px 邻近深色描边一并迁移
            comp_d = np.array(Image.fromarray((comp * 255).astype(np.uint8)).filter(
                ImageFilter.MaxFilter(5))) > 127
            sel = comp | (comp_d & (vv < 0.5))
            hue_map_pixels(arr, sel, th, ts, tv)
            ys_, xs_ = np.where(comp)
            log(f'  {where}残留岛迁移 t{idx}: {comp.sum()}px bbox x{xs_.min()}-{xs_.max()} y{ys_.min()}-{ys_.max()}')


# ---------------- 主流程 ----------------

def run_qa_after_deploy(out, no_browser=False):
    """部署后调 QA Director。任何 hard finding 都写 qa_report.json 并返回 False；
    贴图永不回滚（换色失败与绑骨失效要分开处理）。"""
    agents_dir = os.path.join(ROOT, 'tools', 'agents')
    runs_dir = os.path.join(ROOT, 'tools', '.runs')
    before = set(os.listdir(runs_dir)) if os.path.isdir(runs_dir) else set()
    cmd = [sys.executable, os.path.join(agents_dir, 'qa_director.py'),
           '--model', out, '--tag', 'reskin-qa', '--timeout', '300']
    if no_browser:
        cmd.append('--no-browser')
    log('QA 验收: ' + ' '.join(cmd))
    rc = subprocess.run(cmd).returncode

    candidates = [d for d in os.listdir(runs_dir)
                  if d not in before and d.endswith(f'-{out}-reskin-qa')] if os.path.isdir(runs_dir) else []
    job_dir = os.path.join(runs_dir, sorted(candidates)[-1]) if candidates else None
    verdict = None
    if job_dir and os.path.exists(os.path.join(job_dir, 'verdict.json')):
        verdict = json.load(open(os.path.join(job_dir, 'verdict.json')))

    report = {
        'model': out,
        'qa_passed': rc == 0,
        'qa_exit_code': rc,
        'verdict': verdict,
        'note': '贴图未回滚；hard finding 需人工看 contact_sheet 决定是否重做换色',
    }
    rp = os.path.join(ROOT, 'assets', 'models', out, 'qa_report.json')
    json.dump(report, open(rp, 'w'), ensure_ascii=False, indent=2)
    if verdict:
        log(f"QA 报告: {rp}  sheet: {verdict.get('artifacts', {}).get('contact_sheet')}")
    else:
        log(f'QA 未产出 verdict（harness 失败/超时），报告: {rp}')
    return rc == 0


def main():
    ap = argparse.ArgumentParser(description='Live2D 一键换色/全身素材替换（SAM + ComfyUI + 程序迁移）')
    ap.add_argument('--spec', default=None, help='全身替换 spec JSON（内含 model/out/groups）')
    ap.add_argument('--model', default=None, help='源模型名（assets/models 下，spec 内可覆盖）')
    ap.add_argument('--out', default=None, help='输出模型名（spec 内可覆盖）')
    ap.add_argument('--color', default='gold', choices=list(COLORS), help='单部件模式的颜色预设')
    ap.add_argument('--prompt', default=None, help='覆盖 inpaint 英文提示词（部件描述）')
    ap.add_argument('--masks', default=None, help='单部件模式：目标部件 SAM mask 编号，逗号分隔')
    ap.add_argument('--parts', default='0', help='单部件模式：目标贴图序号（默认 0）')
    ap.add_argument('--reseg', action='store_true', help='强制重跑 SAM')
    ap.add_argument('--small-comp', type=int, default=3000, help='残留岛迁移的最小连通域像素')
    ap.add_argument('--no-deploy', action='store_true', help='只生成贴图不部署模型')
    ap.add_argument('--qa', action='store_true',
                    help='部署后自动跑 QA Director 静态极值验收（hard finding → 退出码 2，不回滚贴图）')
    ap.add_argument('--qa-no-browser', action='store_true', help='QA 不自动开浏览器（CI/接管模式）')
    args = ap.parse_args()

    model, out, groups, defaults, style = load_spec(args)
    assert model and out and groups, '必须提供 model/out/groups（参数或 spec）'
    small_comp = defaults.get('small_comp', args.small_comp)
    min_ai_area = defaults.get('min_ai_area', 8000)

    src = os.path.join(ROOT, 'assets', 'models', model)
    tex_dirs = sorted(d for d in os.listdir(src)
                      if d.endswith('.2048') and os.path.isdir(os.path.join(src, d)))
    assert len(tex_dirs) == 1, f'期望恰好一个贴图目录: {tex_dirs}'
    tex_dir = os.path.join(src, tex_dirs[0])
    texs = sorted(f for f in os.listdir(tex_dir) if re.match(r'texture_\d+\.png$', f))
    log(f'源模型 {model}: {tex_dirs[0]}/{texs}  组: {[g["id"] for g in groups]}')

    # 1. 贴图入内存 + 引用贴图懒跑 SAM
    arrs = {t[8:10]: np.array(Image.open(os.path.join(tex_dir, t)).convert('RGBA'))
            for t in texs}
    ref_idx = sorted({str(g['texture']).zfill(2) for g in groups})
    for idx in ref_idx:
        assert f'texture_{idx}.png' in texs, f'spec 引用了不存在的 texture_{idx}'
        segment_texture(os.path.join(tex_dir, f'texture_{idx}.png'),
                        os.path.join(ROOT, 'tools', 'masks', f'{model}_t{idx}'), args.reseg)

    # 2. 腮红区（每贴图一次，供各组剔除）
    blushes = {}
    for idx, arr in arrs.items():
        t3 = arr[..., :3].astype(np.int16)
        b = (t3[..., 0] > 170) & (t3[..., 1] < 110) & (t3[..., 2] < 110)
        blushes[idx] = np.array(Image.fromarray((b * 255).astype(np.uint8)).filter(
            ImageFilter.MaxFilter(13))) > 127

    # 3. 解析各组 mask（缺定义则出对照表退出）
    resolved = {}
    for g in groups:
        idx = str(g['texture']).zfill(2)
        merged = group_masks(g, os.path.join(ROOT, 'tools', 'masks', f'{model}_t{idx}'),
                             arrs[idx], blushes.get(idx))
        if merged is None:
            sheet_png = os.path.join(ROOT, 'tools', f'{model}_masks_sheet.png')
            masks_sheet(os.path.join(tex_dir, f'texture_{idx}.png'),
                        os.path.join(ROOT, 'tools', 'masks', f'{model}_t{idx}'), sheet_png)
            log(f'组 {g["id"]} 未指定 masks/from，请查看对照表后在 spec 填写：{sheet_png}')
            sys.exit(2)
        resolved[g['id']] = merged
        log(f'组 {g["id"]}(t{idx}, {g.get("mode", "program")}): mask {int(merged.sum())}px')

    # 组间重叠防呆
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            gi, gj = groups[i], groups[j]
            if gi['texture'] != gj['texture']:
                continue
            inter = int((resolved[gi['id']] & resolved[gj['id']]).sum())
            if inter > 0 and not (gi.get('allow_overlap') or gj.get('allow_overlap')):
                raise SystemExit(f'组 {gi["id"]} 与 {gj["id"]} mask 重叠 {inter}px；'
                                 '如属故意请在组上加 "allow_overlap": true')

    # 4. 各组原色签名（必须在任何像素改写前统一采样）
    sigs = {}
    for g in groups:
        sigs[g['id']] = sample_signature(arrs[str(g['texture']).zfill(2)], resolved[g['id']])
        s = sigs[g['id']]
        if s:
            log(f'组 {g["id"]} 原色签名 h={s[0]:.0f}° s={s[1]:.2f} v={s[2]:.2f}')
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            si, sj = sigs[groups[i]['id']], sigs[groups[j]['id']]
            if si and sj and abs(((si[0] - sj[0] + 180) % 360) - 180) < 20 and \
               abs(si[1] - sj[1]) < 0.13 and abs(si[2] - sj[2]) < 0.20:
                log(f'警告: 组 {groups[i]["id"]} 与 {groups[j]["id"]} 原色签名相近，'
                    '残留岛迁移可能互相误认，建议错开颜色或缩小 stray 范围')

    # 5. 逐组执行（program 瞬时；ai 串行 inpaint，约 4 分钟/组）
    for g in groups:
        if g.get('mode', 'program') == 'ai' and not g.get('force') and \
                resolved[g['id']].sum() < min_ai_area:
            log(f'组 {g["id"]} 面积 {int(resolved[g["id"]].sum())}px < min_ai_area '
                f'{min_ai_area}，降级 program（防止误伤表情帧等小组）')
            g['mode'] = 'program'
    ai_groups = [g for g in groups if g.get('mode') == 'ai']
    if ai_groups:
        ensure_comfyui()
        log(f'{len(ai_groups)} 个 ai 组，预计 ~{len(ai_groups) * 4} 分钟（CPU 推理）')
    zones = {}
    for g in groups:
        idx = str(g['texture']).zfill(2)
        arr, merged = arrs[idx], resolved[g['id']]
        th, ts, tv, cp = resolve_color(g['color'])
        # AI prompt 拼接全身语境：SD inpaint 能看到整张贴图，告知整体角色与配色
        # 可让生成色调与服装协调（冷暖/明暗节奏统一），而非孤立色块
        if g.get('prompt'):
            prompt = g['prompt']
        else:
            ctx = f', {style}' if style else ''
            prompt = (f'{cp}{ctx}, harmonious color scheme, cel shading, '
                      f'flat anime style, clean lineart, high quality')
        if g.get('mode') == 'ai':
            tag = f'{out}_t{idx}_{g["id"]}'
            try:
                zones[g['id']] = (idx, inpaint_group(arr, merged, prompt, th, ts, tv, tag))
                continue
            except Exception as e:
                log(f'组 {g["id"]} ai 失败（{e}），降级 program 重着色')
        npx = recolor_program(arr, merged, th, ts, tv)
        zones[g['id']] = (idx, np.array(Image.fromarray(
            (merged * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(9))) > 127)
        log(f'组 {g["id"]} program 换色 {npx}px → h{th:.0f} s{ts} v{tv}')

    # 6. 逐组残留岛迁移（排除所有组区域，防止重复认领）
    excl = {}
    for gid, (idx, zone) in zones.items():
        excl[idx] = zone if idx not in excl else (excl[idx] | zone)
    for g in groups:
        th, ts, tv, _ = resolve_color(g['color'])
        migrate_stray(arrs, texs, sigs[g['id']], th, ts, tv, excl, small_comp,
                      where=f'[{g["id"]}] ')

    # 7. 落盘 + 部署
    outdir = os.path.join(ROOT, 'tools', '.reskin-ai-out', out)
    os.makedirs(outdir, exist_ok=True)
    for idx, arr in arrs.items():
        Image.fromarray(arr, 'RGBA').save(os.path.join(outdir, f'texture_{idx}.png'))
    if not args.no_deploy:
        out_model = os.path.join(ROOT, 'assets', 'models', out)
        if os.path.exists(out_model):
            shutil.rmtree(out_model)
        shutil.copytree(src, out_model)
        for t in texs:
            idx = t[8:10]
            shutil.copy(os.path.join(outdir, f'texture_{idx}.png'),
                        os.path.join(out_model, tex_dirs[0], t))
        mp = None
        for f in os.listdir(out_model):
            if f.endswith('.model3.json'):
                mp = os.path.join(out_model, f)
        if mp:
            mj = json.load(open(mp))
            v = str(int(time.time()))
            mj['FileReferences']['Textures'] = [re.sub(r'\?v=\d+', '', x) + f'?v={v}'
                                                for x in mj['FileReferences']['Textures']]
            json.dump(mj, open(mp, 'w'), ensure_ascii=False, indent='\t')
            log(f'贴图缓存版本 ?v={v}')
        log(f'完成！demo 加载: {out}（http://127.0.0.1:8765/ 已加载过该名需先切其他模型再切回）')
        if args.qa:
            if not run_qa_after_deploy(out, args.qa_no_browser):
                sys.exit(2)
    else:
        log(f'贴图已生成（未部署）: {outdir}/')


if __name__ == '__main__':
    main()
