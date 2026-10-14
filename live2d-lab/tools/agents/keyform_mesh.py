#!/usr/bin/env python3
"""keyform 顶点位移生成 —— 阶段 3 第二块（几何层，Bunraku 对应物）

思路：不自己重算 Cubism 的形变数学，而是让 Cubism Core 自己求值，把变形后的顶点读回来。

  keyframes.json（哪些参数活跃、峰值多大）
    → 按参数线性扫掠姿势（t = 0.25 / 0.5 / 0.75 / 1.0，其余参数保持默认）
    → harness mesh 模式（写参数 → framework core.update 求值 → 读 Drawables 变形顶点）
    → 位移场 D(t) = pos(pose) - pos(rest)
    → 线性度检验 D(t) ≈ t·D(1)（验证 Live2D「keyform 间线性插值」模型）
    → 峰值处的完整顶点位移场 = 该参数的 keyform 位移数据

输出 tools/data/keyforms/keyforms_mesh.json：
  { 参数: { def, peak, median_r2, drawables: [{id, verts, max_disp_px,
                                               linearity_r2, keyform_peak:[...]}] } }

用法：
  python3 tools/agents/keyform_mesh.py --model Haru \
      --keyframes tools/data/poses/keyframes.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import arkit_map as AM
from agents import common as C
from agents import qa_director as QD

SWEEP_T = [0.25, 0.5, 0.75, 1.0]
MIN_DISP_PX = 1.0     # drawable 有效位移下限（位移场只保留真被驱动的网格）


def build_poses(kf_path, model):
    """每个活跃参数生成一条线性扫掠：canonical t → 模型量程实值（避免 harness 静默钳制）"""
    kfs = C.read_json(kf_path)
    assert kfs and 'params' in kfs, f'keyframes 格式错: {kf_path}'
    ranges, _signs = AM.load_calib(model, C.TOOLS)
    poses, meta, canon_val = [], {}, {}
    for pid, events in sorted(kfs['params'].items()):
        smin, smax, sdef = AM.STD_RANGES[pid]
        peak = max((e['v_peak'] for e in events), key=lambda v: abs(v - sdef))
        meta[pid] = {'def': sdef, 'peak': peak}
        for t in SWEEP_T:
            canon = sdef + t * (peak - sdef)
            canon = max(smin, min(smax, canon))
            name = f'swp_{pid}_t{int(round(t * 100)):03d}'
            mv = AM.fit_to_model({pid: canon}, ranges).get(pid)
            assert mv is not None, f'{pid} 不在模型量程表内'
            poses.append({'name': name, 'params': {pid: mv}})
            canon_val[name] = canon
    return poses, meta, canon_val


def run(model, poses):
    job, job_dir = C.new_job(model, 'mesh')
    _, rel_url = C.find_model_json(model)
    rel_url = '/' + rel_url     # 根绝对路径（harness 位于 /qa/ 子目录）
    plan = {
        'job': job, 'model': model, 'modelUrl': rel_url,
        'width': 900, 'height': 700, 'physics': False,
        'load_wait_ms': 5000, 'settle_ms': 600,
        'mode': 'mesh', 'poses': poses, 'expressions': False,
        'trajectories': [], 'motions': [],
    }
    C.write_json(os.path.join(job_dir, 'plan.json'), plan)
    C.ensure_capture()
    C.ensure_web()
    C.log(f'job={job}，{len(poses)} 个扫掠姿势，等待 harness 采集…')
    # 前台标签 + bust：既要绕开浏览器缓存，也要避免后台标签 rAF 节流导致 ticker 停摆
    C.open_harness(f'{C.WEB_URL}/qa/harness.html?job={job}&bust={int(time.time())}')
    done = QD.wait_done(job_dir, 420)
    return job, job_dir, done


def _median(xs):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[len(xs) // 2]


def analyze(job_dir, meta):
    mdir = os.path.join(job_dir, 'mesh')
    recs = {}
    for fn in sorted(os.listdir(mdir)):
        if not fn.endswith('.json'):
            continue
        p = C.read_json(os.path.join(mdir, fn))
        recs[p['name']] = p

    unit_px = 1.0
    for p in recs.values():
        if p.get('unit_px'):
            unit_px = float(p['unit_px'])
            break

    out = {}
    for pid, m in meta.items():
        # (drawable, 分量) → [(t, d)]
        samples, ids = {}, {}
        for t in SWEEP_T:
            p = recs.get(f'swp_{pid}_t{int(round(t * 100)):03d}')
            if not p:
                continue
            for dw in p['drawables']:
                ids[dw['i']] = dw['id']
                for j, dv in enumerate(dw['d']):
                    samples.setdefault((dw['i'], j), []).append((t, dv))

        peak_p = recs.get(f'swp_{pid}_t100')
        peak_vec = {dw['i']: dw['d'] for dw in (peak_p['drawables'] if peak_p else [])}

        # 逐 drawable 聚合：幅度 = 各分量峰值最大；线性度 = 各分量 R² 中位
        agg = {}
        for (i, j), s in samples.items():
            xs = [a for a, _ in s]
            ys = [b for _, b in s]
            sxx = sum(a * a for a in xs)
            if sxx < 1e-9:
                continue
            slope = sum(a * b for a, b in s) / sxx
            my = sum(ys) / len(ys)
            ss_tot = sum((b - my) ** 2 for b in ys)
            ss_res = sum((b - a * slope) ** 2 for a, b in s)
            r2 = 1.0 if ss_tot < 1e-12 else max(0.0, 1 - ss_res / ss_tot)
            row = agg.setdefault(i, {'amps': [], 'r2s': []})
            row['amps'].append(max(abs(b) for b in ys))
            row['r2s'].append(r2)

        rows, r2_all = [], []
        for i, row in agg.items():
            amp = max(row['amps']) * unit_px   # 原生单位 → 屏幕像素
            if amp < MIN_DISP_PX:
                continue        # 未被该参数驱动的网格不入库
            r2m = _median(row['r2s'])
            vec = peak_vec.get(i, [])
            rows.append({'id': agg_id(ids, i), 'verts': len(vec) // 2,
                         'max_disp_px': round(amp, 2), 'linearity_r2': round(r2m, 4),
                         'keyform_peak': vec})
            r2_all.append(r2m)
        rows.sort(key=lambda r: -r['max_disp_px'])
        out[pid] = {'def': m['def'], 'peak': m['peak'],
                    'median_r2': round(_median(r2_all), 4) if r2_all else None,
                    'drawables': rows}
    return out, recs, unit_px


def agg_id(ids, i):
    return ids.get(i, f'drawable_{i}')


def main():
    ap = argparse.ArgumentParser(description='keyform 顶点位移生成（Cubism 运行时读回）')
    ap.add_argument('--model', default='Haru')
    ap.add_argument('--keyframes', default=None,
                    help='keyframe_extract 产出的 keyframes.json（默认 tools/data/poses/keyframes.json）')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    kf = args.keyframes or os.path.join(C.ROOT, 'tools', 'data', 'poses', 'keyframes.json')
    out_path = args.out or os.path.join(C.ROOT, 'tools', 'data', 'keyforms', 'keyforms_mesh.json')

    poses, meta, _cv = build_poses(kf, args.model)
    job, job_dir, done = run(args.model, poses)
    C.log(f'收到 {len(done.get("meshes", []))} 份 mesh payload')

    result, recs, unit_px = analyze(job_dir, meta)
    payload = {'model': args.model, 'source': os.path.basename(kf), 'job': job,
               'unit_px': unit_px, 'sweep_t': SWEEP_T, 'parameters': result}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(payload, open(out_path, 'w'), ensure_ascii=False)

    C.log('── 关键参数位移场 ──')
    r2s = []
    for pid, r in result.items():
        if not r['drawables']:
            C.log(f'  {pid}: 无有效位移（峰值处该参数不驱动任何网格）')
            continue
        top = r['drawables'][0]
        C.log(f'  {pid}: {len(r["drawables"])} 网格, 最大位移 {top["max_disp_px"]}px'
              f' ({top["id"]}), 线性度 R²={r["median_r2"]}')
        if r['median_r2'] is not None:
            r2s.append(r['median_r2'])
    C.log(f'全局线性度中位 R² = {_median(r2s)}')
    C.log(f'输出: {out_path}')
    C.log(f'run dir: {job_dir}')


if __name__ == '__main__':
    main()
