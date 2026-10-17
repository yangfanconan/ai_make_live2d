#!/usr/bin/env python3
"""QA Director —— Live2D 模型自动验收（轨道 A）

用法：
  python3 tools/agents/qa_director.py --model Haru
  python3 tools/agents/qa_director.py --model Haru --no-expr
  python3 tools/agents/qa_director.py --model Haru --no-static \
      --trajectory path/to/traj.json
  python3 tools/agents/qa_director.py --model Haru --motion Auto:0:smile:4
  python3 tools/agents/qa_director.py --model Haru --no-browser   # 手动开页/CI

流程：ensure 服务 → 写 plan.json → 开 harness 页 → 等 done/error →
heuristic 判定 → contact sheet → 可选 JUDGE_CMD → verdict.json（hard 退出码 2）
"""
import argparse
import json
import os
import sys
import time
import urllib.parse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import common as C
from agents import judges as J

# 默认静态极值 pose（标准参数量程；不存在/不同量程由 harness 端过滤钳制）
DEFAULT_POSES = [
    ('rest', {}),
    ('head_L', {'ParamAngleX': -30}),
    ('head_R', {'ParamAngleX': 30}),
    ('head_up', {'ParamAngleY': -30}),
    ('head_down', {'ParamAngleY': 30}),
    ('tilt_L', {'ParamAngleZ': -30}),
    ('tilt_R', {'ParamAngleZ': 30}),
    ('body_L', {'ParamBodyAngleX': -10}),
    ('body_R', {'ParamBodyAngleX': 10}),
    ('eye_close_L', {'ParamEyeLOpen': 0}),
    ('eye_close_R', {'ParamEyeROpen': 0}),
    ('eye_close_both', {'ParamEyeLOpen': 0, 'ParamEyeROpen': 0}),
    ('gaze_tl', {'ParamEyeBallX': -1, 'ParamEyeBallY': 1}),
    ('gaze_br', {'ParamEyeBallX': 1, 'ParamEyeBallY': -1}),
    ('brow_up', {'ParamBrowLY': 1, 'ParamBrowRY': 1}),
    ('brow_down', {'ParamBrowLY': -1, 'ParamBrowRY': -1}),
    ('mouth_open', {'ParamMouthOpenY': 1}),
    ('mouth_smile', {'ParamMouthForm': 1}),
    ('mouth_frown', {'ParamMouthForm': -1}),
    ('extreme_combo_L', {'ParamAngleX': -30, 'ParamAngleY': 15, 'ParamBodyAngleX': -10,
                          'ParamEyeBallX': -1, 'ParamMouthForm': 1}),
]


def build_plan(model, args, rel_model_url):
    plan = {
        'job': None,  # 调用方填
        'model': model,
        'modelUrl': rel_model_url,
        'width': 900, 'height': 700,
        'physics': bool(args.physics),
        'load_wait_ms': args.load_wait_ms,
        'settle_ms': args.settle_ms,
        'mode': 'static',
        'poses': [],
        'expressions': False,
        'trajectories': [],
        'motions': [],
    }
    if not args.no_static:
        plan['poses'] = [{'name': n, 'params': p} for n, p in DEFAULT_POSES]
    if args.poses:
        poses = C.read_json(args.poses)
        assert isinstance(poses, list) and poses and 'params' in poses[0], \
            '姿势库格式应为 [{name, params:{ParamX: v}}]'
        # 判定器以 rest 帧为黑帧检测基线，姿势库没有时自动补
        if not any(p['name'] == 'rest' for p in poses):
            rest = next(p for n, p in DEFAULT_POSES if n == 'rest')
            poses = [{'name': 'rest', 'params': rest}] + poses
        plan['poses'] = poses
        C.log(f'关键姿势库: {len(poses)} 个姿势 ← {args.poses}')
    if not args.no_expr and not args.no_static:
        plan['expressions'] = True
    for tp in args.trajectory or []:
        tr = C.read_json(tp)
        assert tr and 'frames' in tr and 'name' in tr, f'轨迹文件格式错: {tp}'
        plan['trajectories'].append(tr)
    for ms in args.motion or []:
        parts = ms.split(':')
        assert len(parts) >= 2, '--motion 格式 Group:index[:name[:dur]]'
        plan['motions'].append({
            'group': parts[0], 'index': int(parts[1]),
            'name': parts[2] if len(parts) > 2 else f'{parts[0]}_{parts[1]}',
            'dur': float(parts[3]) if len(parts) > 3 else 4.0,
        })
    return plan


def wait_done(job_dir, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        err = os.path.join(job_dir, 'error.json')
        if os.path.isfile(err):
            raise RuntimeError('harness 报错: ' + json.dumps(C.read_json(err), ensure_ascii=False))
        done = os.path.join(job_dir, 'done.json')
        if os.path.isfile(done):
            return C.read_json(done)
        time.sleep(1)
    raise RuntimeError(f'等待 harness done 超时（{timeout}s）')


def save_calib_seed(model, manifest):
    """从运行时实测量程生成校准文件种子（轨道 B 使用）"""
    rp = manifest.get('manifest', {}).get('runtime_params')
    if not rp:
        return
    path = os.path.join(C.TOOLS, f'.motion-calib-{model}.json')
    if os.path.exists(path):
        return
    seed = {'_note': 'QA harness 从 coreModel 实测生成；gain/sign 可按模型微调',
            'ranges': {p['id']: {'min': p['min'], 'max': p['max'], 'def': p['def']}
                       for p in rp}}
    C.write_json(path, seed)
    C.log(f'量程种子: {path}')


def adjudicate(job_dir, plan, done, run_external=False, model=None):
    """对已落盘的 run 做 heuristic 判定 + sheet（+可选外部判官），返回 verdict"""
    model = model or plan.get('model')
    findings, scores = [], {}
    if plan['poses']:
        f1, s1 = J.judge_static(job_dir, plan, done.get('manifest', {}))
        findings += f1
        scores.update(s1)
    for tr in plan['trajectories']:
        f2, s2 = J.judge_trajectory(job_dir, prefix=tr['name'])
        for f in f2:
            f['trajectory'] = tr['name']
        findings += f2
        scores[f'traj_{tr["name"]}'] = s2
    # 无静态扫描时（trajectory/motion-only），judge_static 不会跑，黑帧在此兜底
    if not plan['poses']:
        findings += J.scan_black_frames(job_dir)

    sheet = J.make_contact_sheet(job_dir, plan, findings)
    if run_external:
        ext = J.run_external_judge(job_dir, plan,
                                   f"{len(done.get('frames', []))} frames, model={model}")
        if ext:
            findings += ext.get('findings', [])
            if 'natural_score' in ext:
                scores['natural_score'] = ext['natural_score']

    verdict = {
        'agent': 'qa_director', 'job': os.path.basename(job_dir.rstrip('/')), 'model': model,
        'pass': not [f for f in findings if f['severity'] == 'hard'],
        'scores': scores, 'findings': findings,
        'artifacts': {'contact_sheet': sheet, 'frames_dir': os.path.join(job_dir, 'frames'),
                      'plan': os.path.join(job_dir, 'plan.json')},
    }
    C.save_verdict(job_dir, verdict)
    return verdict


def _judge_only(job_dir, args):
    job_dir = os.path.abspath(job_dir)
    plan = C.read_json(os.path.join(job_dir, 'plan.json'))
    done = C.read_json(os.path.join(job_dir, 'done.json')) or {}
    verdict = adjudicate(job_dir, plan, done, run_external=not args.no_judge)
    hard = [f for f in verdict['findings'] if f['severity'] == 'hard']
    C.log(f'verdict: {"PASS" if verdict["pass"] else "FAIL"}  '
          f'hard={len(hard)} soft={len(verdict["findings"]) - len(hard)}')
    for f in verdict['findings']:
        C.log(f'  [{f["severity"]}] {f["code"]} @ {f.get("pose", f.get("trajectory", "?"))}: '
              f'{f["desc"]}')
    return 0 if verdict['pass'] else 2


def main():
    ap = argparse.ArgumentParser(description='Live2D QA Director 自动验收')
    ap.add_argument('--model', required=True)
    ap.add_argument('--tag', default='qa')
    ap.add_argument('--no-static', action='store_true')
    ap.add_argument('--no-expr', action='store_true')
    ap.add_argument('--physics', action='store_true', help='保留物理（默认关，保证确定性）')
    ap.add_argument('--trajectory', action='append', help='trajectory.json，可重复')
    ap.add_argument('--motion', action='append', help='Group:index[:name[:dur]]，可重复')
    ap.add_argument('--poses', metavar='POSES_JSON',
                    help='keyframe_extract 产出的关键姿势库，替代默认静态姿势表')
    ap.add_argument('--load-wait-ms', type=int, default=5000)
    ap.add_argument('--settle-ms', type=int, default=600)
    ap.add_argument('--timeout', type=int, default=240)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--no-judge', action='store_true', help='跳过 JUDGE_CMD')
    ap.add_argument('--judge-only', metavar='JOB_DIR',
                    help='对已完成的 run 目录本地重判（不起服务/不开浏览器，调阈值用）')
    args = ap.parse_args()

    if args.judge_only:
        sys.exit(_judge_only(args.judge_only, args))

    job, job_dir = C.new_job(args.model, args.tag)
    _, rel_url = C.find_model_json(args.model)
    rel_url = '/' + rel_url  # 根绝对路径（harness 位于 /qa/ 子目录，不能用文档相对路径）

    C.ensure_capture()
    C.ensure_web()

    plan = build_plan(args.model, args, rel_url)
    plan['job'] = job
    C.write_json(os.path.join(job_dir, 'plan.json'), plan)

    if not args.no_browser:
        url = f'{C.WEB_URL}/qa/harness.html?job={urllib.parse.quote(job)}'
        C.open_harness(url)  # Chrome：pixi-live2d-display 兼容性最佳
    else:
        C.log('no-browser：请手动打开 '
              f'{C.WEB_URL}/qa/harness.html?job={urllib.parse.quote(job)}')

    C.log(f'job={job}，等待 harness 扫描…')
    done = wait_done(job_dir, args.timeout)
    C.log(f'收到 {len(done.get("frames", []))} 帧，开始判定')
    save_calib_seed(args.model, done)

    verdict = adjudicate(job_dir, plan, done, run_external=not args.no_judge,
                         model=args.model)
    findings = verdict['findings']
    hard = [f for f in findings if f['severity'] == 'hard']
    sheet = verdict['artifacts']['contact_sheet']

    C.log('─' * 60)
    C.log(f'verdict: {"PASS" if verdict["pass"] else "FAIL"}  '
          f'hard={len(hard)} soft={len(findings) - len(hard)}')
    for f in findings:
        C.log(f'  [{f["severity"]}] {f["code"]} @ {f.get("pose", f.get("trajectory", "?"))}: '
              f'{f["desc"]}')
    C.log(f'sheet: {sheet}')
    C.log(f'run dir: {job_dir}')
    sys.exit(0 if verdict['pass'] else 2)


if __name__ == '__main__':
    main()
