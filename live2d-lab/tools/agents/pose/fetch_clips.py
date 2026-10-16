#!/usr/bin/env python3
"""从 hf-mirror 抓取 kinetics-mini 的普通直立人体片段，用于身体通道增益标定。

来源：nateraw/kinetics-mini（HF 上的 Kinetics-400 微型测试集，含若干 10s mp4）
用途：为 pose 管线提供「日常直立/行走」素材，替代此前只有仰卧起坐的极端素材。
"""
import json
import os
import time
import urllib.request

BASE = 'https://hf-mirror.com'
REPO = 'nateraw/kinetics-mini'
CLASSES = {'marching': 3, 'archery': 2, 'bowling': 2}
_HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(os.path.dirname(_HERE))  # tools/
OUT = os.path.join(TOOLS, 'data', 'poses', 'raw')


def api(path):
    url = f'{BASE}/api/datasets/{REPO}/tree/main/{path}'
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read().decode())


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 10000:
        print(f'  skip {os.path.basename(dst)}')
        return
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) < 10000:
                raise ValueError(f'body too small: {len(data)}')
            with open(dst, 'wb') as f:
                f.write(data)
            print(f'  ok   {os.path.basename(dst)} {len(data) / 1e6:.2f} MB')
            return True
        except Exception as e:
            print(f'  retry {attempt + 1}/4 {os.path.basename(dst)}: {e}')
            time.sleep(3 * (attempt + 1))
    return False


def main():
    os.makedirs(OUT, exist_ok=True)
    for cls, n in CLASSES.items():
        print(f'=== {cls} ===')
        try:
            files = [x for x in api(f'train/{cls}') if x['path'].endswith('.mp4')]
        except Exception as e:
            print(f'  list failed: {e}')
            continue
        for x in files[:n]:
            name = f'{cls}__{os.path.basename(x["path"])}'
            fetch(f'{BASE}/datasets/{REPO}/resolve/main/{x["path"]}',
                  os.path.join(OUT, name))
    print('raw dir:', OUT)
    print('files:', sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()
