"""多 Agent 管线共享基础设施：路径、run 目录、服务 ensure、浏览器唤起、verdict IO"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, 'tools')
AGENTS = os.path.join(TOOLS, 'agents')
RUNS = os.path.join(TOOLS, '.runs')
MODELS = os.path.join(ROOT, 'assets', 'models')

CAP_URL = 'http://127.0.0.1:8900'
WEB_URL = 'http://127.0.0.1:8765'
CAP_PORT = 8900
WEB_PORT = 8765


def log(msg):
    print(f'[agents] {msg}', flush=True)


def new_job(model, tag=''):
    os.makedirs(RUNS, exist_ok=True)
    name = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{model}" + (f'-{tag}' if tag else '')
    jd = os.path.join(RUNS, name)
    os.makedirs(jd, exist_ok=True)
    return name, jd


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path, default=None):
    if not os.path.isfile(path):
        return default
    with open(path) as f:
        return json.load(f)


def model_dir(model):
    d = os.path.join(MODELS, model)
    assert os.path.isdir(d), f'模型目录不存在: {d}'
    return d


def find_model_json(model):
    """assets/models/<Model>/<Name>.model3.json 的 (绝对路径, 相对 web 根路径)"""
    d = model_dir(model)
    cands = [f for f in os.listdir(d) if f.endswith('.model3.json')]
    assert cands, f'{d} 下没有 .model3.json'
    fn = sorted(cands)[0]
    rel = f'assets/models/{model}/{fn}'
    return os.path.join(d, fn), rel


def _http_ok(url, timeout=2):
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except Exception:
        return False


def ensure_capture():
    if _http_ok(CAP_URL + '/health'):
        return
    log('启动 capture_server (:8900) ...')
    subprocess.Popen(
        [sys.executable, os.path.join(TOOLS, 'capture_server.py')],
        stdout=open('/tmp/qa_capture.log', 'ab'), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(0.5)
        if _http_ok(CAP_URL + '/health'):
            log('capture server 就绪')
            return
    raise RuntimeError('capture server 启动超时，见 /tmp/qa_capture.log')


def ensure_web():
    # 必须探测到本仓库的专属页面，防止占用 8765 的是其他根目录的服务器
    if _http_ok(WEB_URL + '/qa/harness.html'):
        return
    log('启动静态服务 (:8765, cwd=live2d-lab) ...')
    subprocess.Popen(
        [sys.executable, '-m', 'http.server', str(WEB_PORT), '--bind', '127.0.0.1'],
        cwd=ROOT, stdout=open('/tmp/qa_web.log', 'ab'), stderr=subprocess.STDOUT)
    for _ in range(30):
        time.sleep(0.5)
        if _http_ok(WEB_URL + '/qa/harness.html'):
            log('静态服务就绪')
            return
    raise RuntimeError('静态服务启动超时或 8765 被其他根目录服务占用，见 /tmp/qa_web.log')


def open_browser(url):
    log('唤起浏览器: ' + url)
    subprocess.Popen(['open', url])


def open_harness(url):
    """在 Chrome 打开 harness 页（非阻塞）。

    用 `open -a "Google Chrome"`：既让 Chrome 上前台（避免手动切页），又不像
    osascript 那样阻塞等待 Automation 授权（实测会挂住整个采集脚本）。
    mesh 采集走同步 core 级求值、不依赖 rAF ticker，标签即便在后台也能完成，
    因此这里不做旧标签清理（多开无害，各自拉自己的 job）。
    """
    subprocess.Popen(['open', '-a', 'Google Chrome', url])
    log('唤起浏览器(Chrome): ' + url)


def verdict_path(job_dir):
    return os.path.join(job_dir, 'verdict.json')


def save_verdict(job_dir, verdict):
    write_json(verdict_path(job_dir), verdict)
    return verdict


def hard_findings(verdict):
    return [f for f in verdict.get('findings', []) if f.get('severity') == 'hard']
