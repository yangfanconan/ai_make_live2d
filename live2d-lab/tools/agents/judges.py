"""QA heuristic 判官 + contact sheet + JUDGE_CMD 外部多模态判官协议

所有 heuristic 只依赖 numpy/PIL，硬故障（黑画布、无响应、截断）本地必检出；
视觉穿帮（缝隙/撕裂/腔洞）交给 JUDGE_CMD 或人工看 sheet。
"""
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw

from .common import AGENTS, log, write_json

# 帧差超过该均值（0..1）才算「模型真的动了」（透明背景上的角色像素）
MOTION_MIN_DIFF = 0.004
# 非透明像素占比低于该值 → 黑画布
ALPHA_MIN_RATIO = 0.005
# 轨迹帧间差突刺倍数
SPIKE_RATIO = 6.0
# 面部局部动作在 900x700 全身取景下的最小变化像素（实测五官极值 42-320px）
FACE_MIN_PX = 25


def load_frame(path):
    return np.asarray(Image.open(path).convert('RGBA')).astype(np.float32) / 255.0


def alpha_mask(img):
    return img[..., 3] > 0.05


def frame_diff(a, b, region=None):
    """角色区域 RGB 均值绝对差（取两帧 alpha 并集），归一 0..1。
    region 可选 (y0,y1,x0,x1) 局部评估。"""
    m = alpha_mask(a) | alpha_mask(b)
    if region is not None:
        y0, y1, x0, x1 = region
        rr = np.zeros_like(m)
        rr[y0:y1, x0:x1] = True
        m &= rr
    if m.sum() < 100:
        return 0.0
    return float(np.abs(a[..., :3] - b[..., :3])[m].mean())


def changed_pixels(a, b, thresh=0.08, region=None):
    """RGB 单像素差 > thresh 的变化像素数（alpha 并集内）。局部五官动作在全身取景下
    均值差会被稀释，必须用绝对像素计数。"""
    m = alpha_mask(a) | alpha_mask(b)
    d = np.abs(a[..., :3] - b[..., :3]).mean(-1) > thresh
    m &= d
    if region is not None:
        y0, y1, x0, x1 = region
        rr = np.zeros_like(m)
        rr[y0:y1, x0:x1] = True
        m &= rr
    return int(m.sum())


def bbox(img):
    ys, xs = np.where(alpha_mask(img))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _bbox_overlap_ratio(b1, b2):
    if not b1 or not b2:
        return 0.0
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def scan_black_frames(job_dir):
    """仅做黑帧/空帧扫描（motion-only 作业无 rest 基线时使用）"""
    findings = []
    fdir = os.path.join(job_dir, 'frames')
    for f in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if not f.endswith('.png'):
            continue
        img = load_frame(os.path.join(fdir, f))
        r = float(alpha_mask(img).mean())
        if r < ALPHA_MIN_RATIO:
            findings.append({'code': 'BLACK_FRAME', 'severity': 'hard', 'pose': f[:-4],
                             'desc': f'非透明像素占比 {r:.4f}，疑似黑画布/加载失败'})
    return findings


def judge_static(job_dir, plan, manifest):
    """静态极值 pose 扫描的 heuristic 判定"""
    fdir = os.path.join(job_dir, 'frames')
    findings, scores = [], {}
    frames = {}
    for f in sorted(os.listdir(fdir)) if os.path.isdir(fdir) else []:
        if f.endswith('.png'):
            frames[f[:-4]] = load_frame(os.path.join(fdir, f))

    def F(code, sev, pose, desc):
        findings.append({'code': code, 'severity': sev, 'pose': pose, 'desc': desc})

    # 1) 黑画布 / 资源缺失
    cover = {}
    for name, img in frames.items():
        r = float(alpha_mask(img).mean())
        cover[name] = r
        if r < ALPHA_MIN_RATIO:
            F('BLACK_FRAME', 'hard', name, f'非透明像素占比 {r:.4f}，疑似黑画布/加载失败')
    scores['alpha_coverage'] = cover

    # 基线选择：rest 与热帧 rest2 中，覆盖率最接近全体中位的一张
    rest = frames.get('rest')
    if rest is None:
        F('BLACK_FRAME', 'hard', 'rest', '缺少 rest 基线帧')
        return findings, scores
    rest2 = frames.get('rest2')
    others_cov = [r for n, r in cover.items() if n not in ('rest', 'rest2')]
    med_cov = float(np.median(others_cov)) if others_cov else cover['rest']
    base = rest
    base_name = 'rest'
    if rest2 is not None and abs(cover.get('rest2', 0) - med_cov) < abs(cover['rest'] - med_cov):
        base, base_name = rest2, 'rest2'
    if rest2 is not None:
        drift = frame_diff(rest, rest2)
        drift_px = changed_pixels(rest, rest2)
        if drift > 0.006 or drift_px > 0.01 * (alpha_mask(rest).sum()):
            F('REST_BASELINE_DRIFT', 'soft', 'rest/rest2',
              f'首帧 rest 与热帧 rest2 不一致（均值差 {drift:.4f}，{drift_px}px），已采用 {base_name}')

    rest_bbox = bbox(base)
    # 截断：角色 bbox 紧贴画布四边
    H, W = base.shape[:2]
    if rest_bbox:
        x1, y1, x2, y2 = rest_bbox
        edges = sum([x1 <= 1, y1 <= 1, x2 >= W - 2, y2 >= H - 2])
        if edges >= 2:
            F('ART_TRUNCATED', 'soft', 'rest',
              f'角色 bbox 触碰画布边 {edges} 侧（可能截断；若模型本身满幅可忽略）')

    # 头部带（角色 bbox 上部 30%）：五官动作的局部评估区
    head_region = None
    if rest_bbox:
        bx1, by1, bx2, by2 = rest_bbox
        head_region = (by1, by1 + int((by2 - by1) * 0.32), bx1, bx2)

    FACE_POSES = {'eye_close_L', 'eye_close_R', 'eye_close_both', 'eye_close_both',
                  'mouth_open', 'mouth_smile', 'mouth_frown',
                  'brow_up', 'brow_down', 'gaze_tl', 'gaze_tr', 'gaze_bl', 'gaze_br'}
    GLOBAL_POSES = {'head_L', 'head_R', 'head_up', 'head_down', 'tilt_L', 'tilt_R',
                    'body_L', 'body_R', 'extreme_combo_L', 'extreme_combo_R'}
    # 运行时默认值（来自 harness manifest）：注入值落在默认位的参数不该要求画面变化
    defaults = {p['id']: p.get('def') for p in (manifest or {}).get('runtime_params', [])}

    def pose_at_default(name):
        """读 sidecar 的写后读回：若该 pose 实际生效的参数值全部等于默认值，
        说明这个极值方向在该模型上无空间（如 MouthForm def==max），非绑骨失效。"""
        sf = os.path.join(fdir, name + '.json')
        if not os.path.exists(sf):
            return False
        try:
            with open(sf) as fh:
                rb = (json.load(fh) or {}).get('readback') or {}
        except Exception:
            return False
        if not rb:
            return False
        for pid, v in rb.items():
            d = defaults.get(pid)
            if d is None or abs(float(v) - float(d)) > 1e-6:
                return False
        return True

    # 2) 逐 pose 能量（均值差用于评分；变化像素计数用于硬判定）
    energy, changed = {}, {}
    plan_pose = {p['name']: p for p in plan.get('poses', [])}
    for name, img in frames.items():
        if name in ('rest', 'rest2') or name.startswith('exp_'):
            continue
        d = frame_diff(base, img)
        energy[name] = d
        has_params = bool(plan_pose.get(name, {}).get('params'))
        if not has_params:
            continue
        if name in FACE_POSES:
            nchg = changed_pixels(base, img, region=head_region)
            changed[name] = nchg
            if nchg < FACE_MIN_PX:
                if pose_at_default(name):
                    F('POSE_AT_DEFAULT', 'soft', name,
                      f'注入值全部落在默认位（该方向无参数空间），无法驱动五官，跳过硬判定')
                else:
                    F('NO_MOTION', 'hard', name,
                      f'头部带变化像素仅 {nchg}（<{FACE_MIN_PX}），参数写入但五官未响应')
        else:
            nchg = changed_pixels(base, img)
            changed[name] = nchg
            ratio = nchg / max(int((alpha_mask(base) | alpha_mask(img)).sum()), 1)
            if name in GLOBAL_POSES and ratio < 0.004 and d < MOTION_MIN_DIFF:
                if pose_at_default(name):
                    F('POSE_AT_DEFAULT', 'soft', name,
                      '注入值全部落在默认位（该方向无参数空间），跳过硬判定')
                else:
                    F('NO_MOTION', 'hard', name,
                        f'头身姿态变化不足（{nchg}px / 均值差 {d:.5f}）')

    # 3) 面部 pose 不应造成整体 bbox 漂移
    for name in FACE_POSES:
        if name in frames and name in energy:
            bb = bbox(frames[name])
            ov = _bbox_overlap_ratio(rest_bbox, bb)
            if ov < 0.85:
                F('RIG_SHIFT_SUSPECT', 'soft', name,
                  f'面部 pose 整体 bbox 漂移过大（重叠 {ov:.2f}），疑似绑骨联动错误')

    # 4) 左右对称（用变化像素计数，避免均值稀释）
    def pair_e(l, r):
        if l in changed and r in changed:
            lo, hi = sorted((changed[l], changed[r]))
            if hi >= FACE_MIN_PX and lo / max(hi, 1) < 0.35:
                F('MOTION_ASYMMETRY', 'soft', f'{l}/{r}',
                  f'左右动作变化像素失衡 {changed[l]} vs {changed[r]}')
    pair_e('head_L', 'head_R')
    pair_e('eye_close_L', 'eye_close_R')
    pair_e('body_L', 'body_R')

    scores['baseline'] = base_name
    scores['motion_energy'] = {k: round(v, 5) for k, v in energy.items()}
    scores['changed_pixels'] = changed
    scores['n_frames'] = len(frames)
    return findings, scores


def judge_trajectory(job_dir, prefix=None):
    """轨迹模式：帧间能量连续性（突刺=插值/抽稀 bug；全程无变化=死轨迹）。
    prefix 非空时只评估以其开头的帧（多轨迹混存时互不干扰）。"""
    fdir = os.path.join(job_dir, 'frames')
    files = sorted(f for f in os.listdir(fdir)
                   if f.endswith('.png') and (prefix is None or f.startswith(prefix))) \
        if os.path.isdir(fdir) else []
    findings = []
    if len(files) < 3:
        return findings, {}

    def F(code, sev, pose, desc):
        findings.append({'code': code, 'severity': sev, 'pose': pose, 'desc': desc})

    imgs = [load_frame(os.path.join(fdir, f)) for f in files]
    # 头部带来自全体 alpha 并集 bbox（面部轨迹只有五官局部动，全身均值差会被稀释）
    union_mask = np.zeros(imgs[0].shape[:2], dtype=bool)
    for im in imgs:
        union_mask |= alpha_mask(im)
    ys, xs = np.where(union_mask)
    head_region = None
    if len(xs):
        head_region = (int(ys.min()), int(ys.min()) + int((ys.max() - ys.min()) * 0.32),
                       int(xs.min()), int(xs.max()))

    pairs = []
    for i in range(1, len(imgs)):
        g = changed_pixels(imgs[i - 1], imgs[i])
        h = changed_pixels(imgs[i - 1], imgs[i], region=head_region) if head_region else 0
        pairs.append((g, h))
    end_g = changed_pixels(imgs[0], imgs[-1])
    end_h = changed_pixels(imgs[0], imgs[-1], region=head_region) if head_region else 0
    max_g = max([g for g, _ in pairs] + [end_g])
    max_h = max([h for _, h in pairs] + [end_h])
    # 死轨迹：全身与头部带都没有超过静态判定同款阈值的变化
    if max_g < 200 and max_h < FACE_MIN_PX:
        F('MOTION_DEAD_CHANNEL', 'hard', files[0][:-4],
          f'轨迹全程无可见变化（全身 {max_g}px / 头部带 {max_h}px，源通道活跃但目标无响应）')
    act = [max(g, h) for g, h in pairs]
    med = float(np.median(act)) if act else 0.0
    for i, a in enumerate(act):
        if med > 5 and a > med * SPIKE_RATIO and a > 300:
            F('TRAJ_SPIKE', 'soft', files[i + 1][:-4],
              f'帧间变化像素突刺 {a}（中位 {med:.0f}），检查插值/抽稀')
            break
    return findings, {'traj_max_changed_px': max_g, 'traj_max_head_px': max_h,
                      'traj_endpoint_px': end_g, 'traj_median_act_px': round(med, 1)}


def make_contact_sheet(job_dir, plan, findings, out_name='contact_sheet.png'):
    fdir = os.path.join(job_dir, 'frames')
    files = sorted(f for f in os.listdir(fdir) if f.endswith('.png')) if os.path.isdir(fdir) else []
    if not files:
        return None
    cell_w, label_h = 320, 40
    cols = 4
    rows = (len(files) + cols - 1) // cols
    bad = {}
    for fnd in findings:
        pose = fnd.get('pose')
        if pose:
            bad.setdefault(pose, []).append(fnd['code'] + ('' if fnd['severity'] == 'hard' else '*'))

    thumbs = []
    for fn in files:
        im = Image.open(os.path.join(fdir, fn)).convert('RGBA')
        bg = Image.new('RGBA', im.size, (245, 245, 247, 255))
        bg.alpha_composite(im)
        bg = bg.convert('RGB')
        bg.thumbnail((cell_w - 12, 360))
        thumbs.append((fn[:-4], bg))

    row_h = label_h + max(t.height for _, t in thumbs) + 8
    sheet = Image.new('RGB', (cols * cell_w, rows * row_h + 10), (235, 236, 240))
    d = ImageDraw.Draw(sheet)
    for i, (name, im) in enumerate(thumbs):
        cx, cy = (i % cols) * cell_w, (i // cols) * row_h
        sheet.paste(im, (cx + 6, cy + label_h))
        color = (200, 40, 40) if name in bad else (30, 30, 30)
        d.text((cx + 6, cy + 4), name, fill=color)
        if name in bad:
            d.text((cx + 6, cy + 20), ','.join(bad[name]), fill=(200, 40, 40))
    out = os.path.join(job_dir, out_name)
    sheet.save(out)
    return out


RUBRIC = os.path.join(AGENTS, 'rubrics', 'qa_rubric.md')


def run_external_judge(job_dir, plan, frames_note):
    """JUDGE_CMD 外部判官协议：argv[1]=JSON 入参，程序回填 verdict 片段 JSON 到 out_path"""
    cmd = os.environ.get('JUDGE_CMD')
    if not cmd:
        return None
    out_path = os.path.join(job_dir, 'external_verdict.json')
    payload = {
        'job_dir': job_dir,
        'sheet': os.path.join(job_dir, 'contact_sheet.png'),
        'frames_dir': os.path.join(job_dir, 'frames'),
        'rubric': RUBRIC if os.path.isfile(RUBRIC) else None,
        'plan': plan,
        'note': frames_note,
        'out_path': out_path,
    }
    log('调用外部判官 JUDGE_CMD ...')
    # 超时按帧数动态放大：逐帧判定单帧最坏 ~40s（大模型/高分辨率图）
    fdir = os.path.join(job_dir, 'frames')
    n_frames = len([f for f in os.listdir(fdir) if f.endswith('.png')]) \
        if os.path.isdir(fdir) else 0
    timeout = max(600, min(n_frames, 60) * 40)
    try:
        r = subprocess.run([cmd, json.dumps(payload)], capture_output=True, text=True,
                           timeout=timeout)
        if r.returncode != 0:
            log(f'外部判官退出码 {r.returncode}: {r.stderr[:500]}')
            return None
        with open(out_path, encoding='utf-8') as fh:
            ext = json.load(fh)
        write_json(out_path, ext)
        return ext
    except Exception as e:
        log(f'外部判官调用失败: {e}')
        return None
