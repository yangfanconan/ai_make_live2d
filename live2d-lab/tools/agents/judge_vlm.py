#!/usr/bin/env python3
"""JUDGE_CMD 外部多模态判官 shim —— 把 QA 渲染帧发给远端 4090 上的 VLM 服务打分

协议（见 judges.run_external_judge）：argv[1] = payload JSON；把 verdict 片段写到
payload['out_path']。verdict 片段结构：{"natural_score": int, "findings": [...], "summary": str}

链路：Mac :8300 --ssh -N -L--> 192.168.2.2:8300（Qwen2.5-VL，OpenAI 兼容）。
远端 8300 对外不可达，全部流量走 SSH 本地转发；隧道由 ensure_tunnel() 自愈。

两种判定模式（VLM_MODE）：
  frames（默认）：逐帧送图，每 pose 单独请求。分辨率高、无示例可抄，判定最可靠。
  sheet：整张 contact sheet 一次请求。快，但小模型易复读 rubric 示例，仅作回退。

环境变量：
  VLM_MODE        frames | sheet（默认 frames）
  VLM_ENDPOINT    默认 http://127.0.0.1:8300/v1/chat/completions（隧道本地口）
  VLM_MODEL       默认 Qwen2.5-VL-3B-Instruct
  VLM_TIMEOUT     默认 600 秒（单次请求）
  VLM_SSH_KEY     默认 ~/.ssh/id_ed25519_trae4090
  VLM_SSH_HOST    默认 192.168.2.2   VLM_SSH_USER 默认 yangfan
  VLM_TUNNEL_PORT 默认 8300
"""
import base64
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

MODE = os.environ.get('VLM_MODE', 'frames')
ENDPOINT = os.environ.get('VLM_ENDPOINT', 'http://127.0.0.1:8300/v1/chat/completions')
MODEL = os.environ.get('VLM_MODEL', 'Qwen2.5-VL-3B-Instruct')
TIMEOUT = float(os.environ.get('VLM_TIMEOUT', '600'))
HEALTHZ = os.environ.get('VLM_HEALTHZ', 'http://127.0.0.1:8300/healthz')

CODES = ('TEAR/HOLE、LAYER_TEAR、FEATURE_STRETCH、ASYMMETRY、'
         'MOUTH_HOLE、CLIP_THROUGH、COMBO_BREAK')

# 逐帧模式：不带任何示例句子，杜绝小模型复读
PER_FRAME_PROMPT = (
    '你是 Live2D 模型验收专家。这是一张 Live2D 模型渲染截图（灰底画布中央为模型），'
    '当前参数姿势：{pose}（参数注入：{params}）。\n'
    '仔细检查画面中模型是否出现以下缺陷：透明缝隙或空洞、部件层间撕裂/分离、'
    '五官错位或异常拉伸、左右明显不对称、口腔内部空洞或黑块、'
    '头发与身体穿插、部件崩形。\n'
    '只输出 JSON，不要输出任何其他文字：\n'
    '{{"issues": [{{"code": "{codes} 中的一个", "severity": "hard 或 soft", '
    '"desc": "具体描述你实际看到的现象与位置"}}], "natural": 1到5的整数}}\n'
    'natural 为该姿势下模型观感自然度（5=完全正常）。没有可见缺陷时 issues 为空数组。'
    '注意：姿势本身（转头/仰俯/侧倾/单眼闭/张嘴）引起的视觉不对称是正常表现，不算缺陷；'
    '禁止编造：每条 desc 必须对应你真实看到的画面证据。'
)

SHEET_RUBRIC = (
    '你是 Live2D 绑骨/贴图验收专家。逐格检查截图中是否出现：透明缝隙/洞、层间撕裂、'
    '五官错位拉伸、左右不对称、口型腔洞、发身穿插、多参数耦合崩形。'
    '严格要求：只报告确实在图中看到的问题。只输出 JSON：'
    '{"natural_score": 1-5, "findings":[{"code":"TEAR_SUSPECT","severity":"hard|soft",'
    '"pose":"<pose 名>","desc":"..."}], "summary":"一句话总评"}\n'
    '重要：示例 JSON 仅供格式参考，禁止逐字复述示例中的 desc 文案；'
    '每条 finding 必须描述本批截图中实际看到的现象；rest 帧仅作对照，不报缺陷。'
)


def _b64(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()


def _http_ok(url, timeout=2):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def _ssh_args():
    key = os.path.expanduser(os.environ.get('VLM_SSH_KEY', '~/.ssh/id_ed25519_trae4090'))
    host = os.environ.get('VLM_SSH_HOST', '192.168.2.2')
    user = os.environ.get('VLM_SSH_USER', 'yangfan')
    return key, f'{user}@{host}'


def _fetch_model():
    """从 /v1/models 动态取远端实际模型 id（切 7B 后 Mac 侧无需改配置）"""
    global MODEL
    try:
        with urllib.request.urlopen(
                HEALTHZ.replace('/healthz', '/v1/models'), timeout=3) as r:
            data = json.loads(r.read().decode())
        ids = [m.get('id') for m in data.get('data', []) if m.get('id')]
        if ids:
            MODEL = ids[0]
    except Exception:
        pass


def ensure_tunnel():
    """三级自愈：① healthz 通直接用 → ② 起本地隧道 → ③ SSH 到远端拉起服务"""
    if '127.0.0.1' not in ENDPOINT and 'localhost' not in ENDPOINT:
        return  # 直连远端模式（防火墙放行时），无需隧道
    if _http_ok(HEALTHZ):
        _fetch_model()
        return
    key, dest = _ssh_args()
    port = os.environ.get('VLM_TUNNEL_PORT', '8300')
    subprocess.Popen(
        ['ssh', '-i', key,
         '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
         '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=15',
         '-o', 'ServerAliveCountMax=3', '-o', 'LogLevel=ERROR',
         '-f', '-N', '-L', f'{port}:127.0.0.1:{port}', dest],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(20):  # 隧道就绪最多 10s
        time.sleep(0.5)
        if _http_ok(HEALTHZ):
            print('judge_vlm: ssh tunnel ready', flush=True)
            _fetch_model()
            return
    # 隧道就绪但远端服务没响应 → SSH 过去幂等拉起（ensure_serve 自带加载等待）
    print('judge_vlm: tunnel up but service dead, relaunching remote serve ...',
          file=sys.stderr, flush=True)
    try:
        subprocess.run(
            ['ssh', '-i', key,
             '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
             '-o', 'LogLevel=ERROR', dest,
             '"D:\\live2d-svc\\venv\\Scripts\\python.exe" '
             '"D:\\live2d-svc\\run\\ensure_serve.py"'],
            capture_output=True, timeout=180)
    except Exception as e:
        print(f'judge_vlm: remote relaunch failed: {e}', file=sys.stderr)
        return
    for _ in range(20):
        time.sleep(0.5)
        if _http_ok(HEALTHZ):
            print('judge_vlm: remote serve relaunched', flush=True)
            _fetch_model()
            return
    print('judge_vlm: service still dead after relaunch, request will fail loudly',
          file=sys.stderr)


def _extract_json(text):
    text = re.sub(r'```(?:json)?', '', text)
    i, j = text.find('{'), text.rfind('}')
    if i < 0 or j <= i:
        raise ValueError('回复中未找到 JSON')
    return json.loads(text[i:j + 1])


def _ask(content, max_tokens=512):
    body = json.dumps({'model': MODEL,
                       'messages': [{'role': 'user', 'content': content}],
                       'max_tokens': max_tokens, 'temperature': 0.0}).encode()
    req = urllib.request.Request(ENDPOINT, data=body,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        resp = json.loads(r.read().decode())
    return resp['choices'][0]['message']['content'], resp.get('usage')


def _pick_pose_frames(payload):
    """按 plan.poses 顺序取 frames/<name>.png（rest* 帧仅对照，不送判）"""
    fdir = payload.get('frames_dir')
    if not fdir or not os.path.isdir(fdir):
        return []
    plan = payload.get('plan') or {}
    out = []
    for p in plan.get('poses', []):
        name = p.get('name', '')
        fp = os.path.join(fdir, name + '.png')
        if name.startswith('rest') or not os.path.isfile(fp):
            continue
        out.append((name, fp, p.get('params', {})))
    # plan 之外但存在的静态极值帧（如旧 run 的 rest2/extras）不追加以保持确定性
    return out


def _pick_sheet(payload):
    sheet = payload.get('sheet')
    if sheet and os.path.isfile(sheet):
        return [('contact_sheet.png', sheet, {})]
    return []


def judge_frames(frames):
    """逐帧判定：每帧独立请求，返回 (findings, natural_score, summary)"""
    findings, naturals, usage_sum = [], [], 0
    for name, path, params in frames:
        text = PER_FRAME_PROMPT.format(
            pose=name, params=json.dumps(params, ensure_ascii=False), codes=CODES)
        content = [
            {'type': 'text', 'text': text},
            {'type': 'image_url',
             'image_url': {'url': 'data:image/png;base64,' + _b64(path)}},
        ]
        try:
            ans, usage = _ask(content)
            usage_sum += (usage or {}).get('total_tokens', 0)
            v = _extract_json(ans)
        except urllib.error.URLError:
            raise  # 服务级故障：整体退出码 3
        except Exception as e:
            findings.append({'code': 'JUDGE_FRAME_ERR', 'severity': 'soft',
                             'pose': name, 'desc': f'该帧判定失败: {e}'})
            continue
        for it in (v.get('issues') or []):
            if not isinstance(it, dict) or 'desc' not in it:
                continue
            it.setdefault('pose', name)
            findings.append(it)
        if isinstance(v.get('natural'), (int, float)):
            naturals.append(int(v['natural']))
        print(f'  [{name}] natural={v.get("natural")} issues={len(v.get("issues") or [])}',
              flush=True)

    score = int(round(statistics.mean(naturals))) if naturals else None
    hard = sum(1 for f in findings if f.get('severity') == 'hard')
    summary = (f'逐帧判定 {len(frames)} 帧：hard={hard} '
               f'soft={len(findings) - hard}，natural 均值 '
               f'{round(statistics.mean(naturals), 2) if naturals else "n/a"}')
    return findings, score, summary, usage_sum


def judge_sheet(imgs):
    """整张 contact sheet 一次请求（回退模式）"""
    name, path, _ = imgs[0]
    content = [{'type': 'text', 'text': SHEET_RUBRIC}]
    content.append({'type': 'text', 'text': f'[image: {name}]'})
    content.append({'type': 'image_url',
                    'image_url': {'url': 'data:image/png;base64,' + _b64(path)}})
    ans, usage = _ask(content, max_tokens=768)
    v = _extract_json(ans)
    return (v.get('findings') or [], v.get('natural_score'), v.get('summary'),
            (usage or {}).get('total_tokens', 0))


def main():
    ensure_tunnel()
    payload = json.loads(sys.argv[1])
    out_path = payload['out_path']

    frames = _pick_pose_frames(payload) if MODE == 'frames' else []
    imgs = frames or _pick_sheet(payload)
    if not imgs:
        print('judge_vlm: 无可评图像（frames/plan 或 sheet 均缺失）', file=sys.stderr)
        sys.exit(2)

    try:
        if MODE == 'frames' and frames:
            findings, score, summary, usage = judge_frames(frames)
        else:
            findings, score, summary, usage = judge_sheet(imgs)
    except urllib.error.URLError as e:
        print(f'judge_vlm: 无法连接 VLM 服务 {ENDPOINT} -> {e}', file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f'judge_vlm: 判定失败 {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(4)

    verdict = {'findings': findings, 'summary': summary,
               '_meta': {'endpoint': ENDPOINT, 'model': MODEL, 'mode': MODE,
                         'images': [n for n, _, _ in imgs],
                         'total_tokens': usage}}
    if score is not None:
        verdict['natural_score'] = score
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2)
    print(f'judge_vlm: natural_score={score} findings={len(findings)} -> {out_path}')
    sys.exit(0)


if __name__ == '__main__':
    main()
