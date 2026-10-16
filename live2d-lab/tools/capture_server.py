#!/usr/bin/env python3
"""Live2D 截帧收集服务器（QA Director 基础设施）

端点：
  POST /shot                 旧接口（人工换肤验证）：{name, data(dataURL)} → tools/<name>_<HHMMSS>.png
  GET  /health               健康检查
  GET  /job/<job>/plan.json  取该 job 的扫描计划
  POST /job/<job>/frame      回传截帧：{name, data(dataURL), params?} → .runs/<job>/frames/<name>.png
  POST /job/<job>/done       扫描完成 sentinel：body 即 manifest（任意 JSON）→ .runs/<job>/done.json
  POST /job/<job>/error      扫描失败：body 即错误信息 → .runs/<job>/error.json

多线程实现（ThreadingHTTPServer + daemon_threads），解决旧版单线程服务器
连续收帧时偶发卡死的问题。
"""
import base64
import json
import re
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = '/Users/yangfan/MyGames/live2d-lab'
TOOLS = ROOT + '/tools'
RUNS = TOOLS + '/.runs'

JOB_RE = re.compile(r'^[A-Za-z0-9_.-]+$')
NAME_RE = re.compile(r'^[A-Za-z0-9_.-]+$')


def safe_join(base, *parts):
    """路径穿越防护：所有片段必须是安全名，且最终路径在 base 内"""
    for p in parts:
        if not NAME_RE.match(p):
            return None
    import os
    path = os.path.normpath(os.path.join(base, *parts))
    if not path.startswith(os.path.normpath(base) + os.sep) and path != os.path.normpath(base):
        return None
    return path


class H(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n))

    @staticmethod
    def _decode_dataurl(data):
        return base64.b64decode(data.split(',', 1)[1] if ',' in data else data)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self._json(200, {'ok': True})
            return
        m = re.match(r'^/job/([^/]+)/plan\.json$', self.path)
        if m:
            job = m.group(1)
            if not JOB_RE.match(job):
                self._json(400, {'ok': False, 'error': 'bad job id'})
                return
            import os
            p = safe_join(RUNS, job, 'plan.json')
            if p and os.path.isfile(p):
                data = open(p, 'rb').read()
                self.send_response(200)
                self._cors()
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {'ok': False, 'error': 'plan not found'})
            return
        self._json(404, {'ok': False, 'error': 'not found'})

    def do_POST(self):
        try:
            body = self._read_json()
        except Exception as e:
            self._json(400, {'ok': False, 'error': f'bad json: {e}'})
            return

        # 旧接口：/shot
        if self.path == '/shot':
            try:
                data = self._decode_dataurl(body['data'])
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(body.get('name', 'shot')))
                path = f"{TOOLS}/{name}_{datetime.now().strftime('%H%M%S')}.png"
                with open(path, 'wb') as f:
                    f.write(data)
                print('saved', path, flush=True)
                self._json(200, {'ok': True, 'path': path})
            except Exception as e:
                self._json(400, {'ok': False, 'error': str(e)})
            return

        # job 接口
        m = re.match(r'^/job/([^/]+)/(frame|done|error|mesh)$', self.path)
        if not m:
            self._json(404, {'ok': False, 'error': 'not found'})
            return
        job, kind = m.group(1), m.group(2)
        if not JOB_RE.match(job):
            self._json(400, {'ok': False, 'error': 'bad job id'})
            return

        import os
        job_dir = os.path.join(RUNS, job)

        if kind == 'frame':
            try:
                name = body.get('name', 'frame')
                if not NAME_RE.match(name):
                    # 任意 pose 名收敛为安全文件名
                    name = re.sub(r'[^A-Za-z0-9_.-]', '_', name)
                if not name.endswith('.png'):
                    name += '.png'
                fdir = os.path.join(job_dir, 'frames')
                os.makedirs(fdir, exist_ok=True)
                path = os.path.join(fdir, name)
                raw = self._decode_dataurl(body.get('data') or '')
                if len(raw) < 128 or not raw.startswith(b'\x89PNG'):
                    self._json(400, {'ok': False,
                                     'error': f'empty/invalid frame payload ({len(raw)} bytes)'})
                    return
                with open(path, 'wb') as f:
                    f.write(raw)
                if body.get('params') is not None or body.get('readback') is not None:
                    import json as _json
                    with open(os.path.join(fdir, name[:-4] + '.json'), 'w') as mf:
                        _json.dump({'params': body.get('params'),
                                    'readback': body.get('readback')}, mf, ensure_ascii=False)
                print(f'[job {job}] frame {name} ({len(body["data"])//1024}KB b64)', flush=True)
                self._json(200, {'ok': True, 'path': path})
            except Exception as e:
                self._json(400, {'ok': False, 'error': str(e)})
            return

        if kind == 'mesh':
            # keyform 顶点位移 payload（已由前端算好差值并裁掉静止 drawable）
            try:
                name = re.sub(r'[^A-Za-z0-9_.-]', '_', body.get('name', 'mesh'))
                mdir = os.path.join(job_dir, 'mesh')
                os.makedirs(mdir, exist_ok=True)
                path = os.path.join(mdir, name + '.json')
                with open(path, 'w') as f:
                    json.dump(body, f, ensure_ascii=False)
                print(f'[job {job}] mesh {name} ({body.get("mesh_total")}/{body.get("drawable_total")} '
                      f'drawables, max {body.get("max_abs")}px)', flush=True)
                self._json(200, {'ok': True, 'path': path})
            except Exception as e:
                self._json(400, {'ok': False, 'error': str(e)})
            return

        # done / error：整个 body 落盘为 sentinel
        try:
            os.makedirs(job_dir, exist_ok=True)
            fname = 'done.json' if kind == 'done' else 'error.json'
            with open(os.path.join(job_dir, fname), 'w') as f:
                json.dump(body, f, ensure_ascii=False, indent=2)
            print(f'[job {job}] {fname}: '
                  f'{len(body.get("frames", body.get("errors", [])))} entries', flush=True)
            self._json(200, {'ok': True})
        except Exception as e:
            self._json(400, {'ok': False, 'error': str(e)})

    def log_message(self, *a):
        pass


if __name__ == '__main__':
    import os
    os.makedirs(RUNS, exist_ok=True)
    srv = ThreadingHTTPServer(('127.0.0.1', 8900), H)
    srv.daemon_threads = True
    print('capture server on http://127.0.0.1:8900 (threads on)', flush=True)
    srv.serve_forever()
