#!/usr/bin/env python3
"""motion_deploy —— 把 motion_author 产出的 motion3.json 部署到模型并注册 Auto 组。

- 拷贝到 assets/models/<M>/motions/auto/<name>.motion3.json
- 在 <M>.model3.json 的 FileReferences.Motions.Auto 注册（幂等：按 File 去重）
- 首次改动前备份 <M>.model3.json.preauto.bak（只备一次，保留原始状态）
- --qa：部署后直接调 qa_director 做 motion 实放验收（subprocess 透传）

零回归边界：只改模型资源文件，不碰 index.html / js/app.js。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents import common as C

AUTO_GROUP = 'Auto'
AUTO_DIR = os.path.join('motions', 'auto')


def deploy(motion_path, model, name=None, group=AUTO_GROUP):
    mpath, _ = C.find_model_json(model)
    mdir = C.model_dir(model)
    name = name or os.path.splitext(os.path.basename(motion_path))[0]
    auto_abs = os.path.join(mdir, AUTO_DIR)
    os.makedirs(auto_abs, exist_ok=True)
    dst_name = f'{name}.motion3.json'
    dst_abs = os.path.join(auto_abs, dst_name)
    shutil.copyfile(motion_path, dst_abs)
    rel_file = f'{AUTO_DIR}/{dst_name}'.replace(os.sep, '/')

    data = json.load(open(mpath))
    fr = data.setdefault('FileReferences', {})
    motions = fr.setdefault('Motions', {})
    entries = motions.setdefault(group, [])
    entries[:] = [e for e in entries if e.get('File') != rel_file]  # 幂等
    entries.append({'File': rel_file, 'FadeInTime': 0.5, 'FadeOutTime': 0.5})

    bak = mpath + '.preauto.bak'
    if not os.path.exists(bak):
        shutil.copyfile(mpath, bak)
        C.log(f'已备份原始 model3: {os.path.basename(bak)}')
    C.write_json(mpath, data)
    index = len(entries) - 1
    C.log(f'已注册 {model}: Motions.{group}[{index}] = {rel_file}')
    return {'file': rel_file, 'group': group, 'index': index, 'name': name}


def run_qa(model, group, index, name, dur, extra=None):
    cmd = [sys.executable, os.path.join(C.AGENTS, 'qa_director.py'),
           '--model', model, '--tag', f'motion-{name}',
           '--no-static', '--no-expr',
           '--motion', f'{group}:{index}:{name}:{dur}']
    cmd += (extra or [])
    C.log('QA 实放验收: ' + ' '.join(cmd))
    return subprocess.run(cmd).returncode


def main():
    ap = argparse.ArgumentParser(description='motion3 部署 + Auto 组注册')
    ap.add_argument('--motion', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--name', help='注册名（默认取文件名）')
    ap.add_argument('--group', default=AUTO_GROUP)
    ap.add_argument('--qa', action='store_true', help='部署后立即跑 qa_director 实放验收')
    ap.add_argument('--dur', type=float, default=None, help='QA 截帧时长（秒，默认=动作时长）')
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    info = deploy(args.motion, args.model, args.name, args.group)

    if args.qa:
        dur = args.dur
        if dur is None:
            m = json.load(open(args.motion))
            dur = float(m['Meta']['Duration']) + 0.5
        extra = ['--no-browser'] if args.no_browser else []
        rc = run_qa(args.model, info['group'], info['index'], info['name'], dur, extra)
        sys.exit(rc)


if __name__ == '__main__':
    main()
