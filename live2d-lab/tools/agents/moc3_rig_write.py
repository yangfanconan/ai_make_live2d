#!/usr/bin/env python3
"""moc3 绑骨写回 —— 阶段 3 第三块（自动绑骨落地）

把「参数 → keyform 位移」的修改直接写回 .moc3 二进制，运行时（Cubism Core）
加载后即生效。位移分析（keyform_mesh.py）→ 定位驱动 keyform → 写回 → QA 验证
构成完整闭环。

moc3 v1 结构要点（py-moc3，tools/third_party，MIT）：
  对象（warp/rotation deformer、artmesh）持有 keyform_binding_band →
  keyform_binding_index → keyform_binding（一组 keys，即某参数的关键值表）。
  参数→binding 是**直接区间**：parameter.keyform_binding_begin/count。
  多 binding 对象的 keyform 枚举顺序：band 列表**最后一个 binding 是主轴**
  （stride 最大）。已用 B_EYE_001_01 质心网格 + AngleX=-25 实测双重验证。

写回策略（保持 rest 姿态不变）：
  对目标参数的每个驱动对象、每个 keyform 组合，把各轴向「相对该轴默认 key
  组合的偏差」缩放 f 倍：
    v'(c) = v(c) + (f-1) * (v(c) - v(c_axis_default))
  默认 key 处偏差为 0 → rest 完全不动；线性插值下任意中间值同步缩放。

用法：
  python3 tools/agents/moc3_rig_write.py assets/models/Haru/Haru.moc3 \
      --scale ParamAngleX=1.6 --scale ParamBrowLY=2.0 --out HaruRigAmp.moc3
  python3 tools/agents/moc3_rig_write.py <moc3> --info
"""
import argparse
import os
import sys

_TP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'third_party', 'py-moc3')
sys.path.insert(0, _TP)
try:
    from moc3 import Moc3  # noqa: E402
except ImportError:
    # third_party 不入库（.gitignore），首次运行自动从 GitHub 拉取 py-moc3（MIT）
    # 走 api.github.com tarball：github.com 直连在部分网络下不可用
    import tarfile
    import tempfile
    import urllib.request
    url = 'https://api.github.com/repos/Ludentes/py-moc3/tarball'
    print(f'[boot] 拉取 py-moc3 → {_TP} …')
    with tempfile.TemporaryDirectory() as td:
        tgz = os.path.join(td, 'm.tgz')
        urllib.request.urlretrieve(url, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(td)
        src = None
        for root, dirs, _files in os.walk(td):
            if root.endswith('src/moc3'):
                src = root
                break
        assert src, 'py-moc3 tarball 结构异常'
        os.makedirs(_TP, exist_ok=True)
        os.rename(src, os.path.join(_TP, 'moc3'))
    from moc3 import Moc3  # noqa: E402


class Rig:
    """moc3 绑骨视图：binding 语义封装 + keyform 寻址"""

    def __init__(self, path):
        self.path = path
        self.m = Moc3.from_file(path)
        self._build_binding_maps()

    def g(self, name):
        return self.m[name]

    def _build_binding_maps(self):
        m = self.m
        self.pids = m['parameter.ids']
        self.pb_beg = m['parameter.keyform_binding_begin_indices']
        self.pb_cnt = m['parameter.keyform_binding_counts']
        self.kb_beg = m['keyform_binding_band.begin_indices']
        self.kb_cnt = m['keyform_binding_band.counts']
        self.kbi = m['keyform_binding_index.indices']
        self.bind_kb = m['keyform_binding.keys_begin_indices']
        self.bind_kc = m['keyform_binding.keys_counts']
        self.keys = m['keys.values']
        # binding -> 参数 id（参数直接区间反查，一 binding 一参数）
        self.binding_param = {}
        for p in range(len(self.pids)):
            for b in range(self.pb_beg[p], self.pb_beg[p] + self.pb_cnt[p]):
                self.binding_param[b] = self.pids[p]

    def param_bindings(self, pid):
        p = self.pids.index(pid)
        return list(range(self.pb_beg[p], self.pb_beg[p] + self.pb_cnt[p]))

    def band_bindings(self, band):
        """band 列表顺序保持原样（最后一个为主轴）"""
        if band < 0:
            return []
        b, e = self.kb_beg[band], self.kb_beg[band] + self.kb_cnt[band]
        return [self.kbi[i] for i in range(b, e)]

    def binding_keys(self, b):
        s, c = self.bind_kb[b], self.bind_kc[b]
        return self.keys[s:s + c]

    def binding_default_idx(self, b):
        """binding 的 keys 中离参数默认值最近的下标"""
        pid = self.binding_param.get(b)
        pdef = self.g('parameter.default_values')[self.pids.index(pid)] if pid else 0.0
        ks = self.binding_keys(b)
        return min(range(len(ks)), key=lambda i: abs(ks[i] - pdef))

    def combo_offsets(self, blist):
        """枚举 keyform 组合。band 列表**最后一个** binding 是主轴（stride 最大，
        已由 B_EYE 质心网格 + AngleX=-25 运行时双重验证）。
        返回 [((i0,i1,..), kf偏移)]。"""
        ns = [len(self.binding_keys(b)) for b in blist]
        out = []
        for combo in _iter_grid(ns):
            off = 0
            for axis in reversed(range(len(combo))):   # 末轴最显著
                off = off * ns[axis] + combo[axis]
            out.append((combo, off))
        return out

    def objects(self):
        """产出 (类型, 对象名, band, kf_beg, kf_cnt)"""
        m = self.m
        dids = m['deformer.ids']
        dtyp = m['deformer.types']
        dspec = m['deformer.specific_indices']
        for di, dn in enumerate(dids):
            si = dspec[di]
            if dtyp[di] == 0:
                yield ('warp', dn, m['warp_deformer.keyform_binding_band_indices'][si],
                       m['warp_deformer.keyform_begin_indices'][si],
                       m['warp_deformer.keyform_counts'][si], di)
            elif dtyp[di] == 1:
                yield ('rot', dn, m['rotation_deformer.keyform_binding_band_indices'][si],
                       m['rotation_deformer.keyform_begin_indices'][si],
                       m['rotation_deformer.keyform_counts'][si], di)
        amids = m['art_mesh.ids']
        for ai, an in enumerate(amids):
            yield ('mesh', an, m['art_mesh.keyform_binding_band_indices'][ai],
                   m['art_mesh.keyform_begin_indices'][ai],
                   m['art_mesh.keyform_counts'][ai], ai)

    # ---- 各类型的 keyform 载荷寻址（返回 section 名 + 起始下标 + 元素数）----
    # 注意：artmesh 的 keyform 位置段步长 ≠ vertex_counts*2（实测 D_PSD_05
    # 运行时 44 顶点、段内 48 对），统一改用「相邻 keyform begin 差」确定跨度，
    # 同对象内步长均匀，逐下标偏差缩放不受语义影响。
    def payload_span(self, kind, di, ai, kf):
        m = self.m
        if kind == 'warp':
            beg = m['warp_deformer_keyform.keyform_position_begin_indices'][kf]
            return 'keyform_position.xys', beg, self._pos_span(
                m['warp_deformer_keyform.keyform_position_begin_indices'], kf)
        beg = m['art_mesh_keyform.keyform_position_begin_indices'][kf]
        return 'keyform_position.xys', beg, self._pos_span(
            m['art_mesh_keyform.keyform_position_begin_indices'], kf)

    @staticmethod
    def _pos_span(begins, kf):
        total = len(begins)
        nxt = begins[kf + 1] if kf + 1 < total else None
        if nxt is None:
            # 最后一个 keyform：跨到 xys 总长（调用方换算）
            return -1
        return nxt - begins[kf]


def _iter_grid(ns):
    """生成全部组合索引（下标含义见 combo_offsets）"""
    import itertools
    for combo in itertools.product(*[range(n) for n in ns]):
        yield combo


def _offset_of(combos, target):
    for combo, off in combos:
        if list(combo) == list(target):
            return off
    raise KeyError(f'combo 不存在: {target}')


def scale_rot_object(rig, band, kb, axes, defs, factor):
    """rotation deformer：angle/origin/scale 相对默认 key 组合线性缩放"""
    m = rig.m
    blist = rig.band_bindings(band)
    combos = rig.combo_offsets(blist)
    base = tuple(defs)
    kf_base = kb + _offset_of(combos, list(base))
    for combo, off in combos:
        if not any(combo[a] != defs[a] for a in axes):
            continue
        kf = kb + off
        for sec in ('rotation_deformer_keyform.angles',
                    'rotation_deformer_keyform.origin_xs',
                    'rotation_deformer_keyform.origin_ys'):
            arr = m[sec]
            dev = arr[kf] - arr[kf_base]
            arr[kf] = arr[kf] + (factor - 1.0) * dev
        arr = m['rotation_deformer_keyform.scales']
        dev = arr[kf] - arr[kf_base]
        arr[kf] = arr[kf] + (factor - 1.0) * dev


def main():
    ap = argparse.ArgumentParser(description='moc3 keyform 写回（参数形变缩放）')
    ap.add_argument('moc3')
    ap.add_argument('--scale', action='append', default=[],
                    help='ParamId=factor，可多次')
    ap.add_argument('--out', default=None)
    ap.add_argument('--info', action='store_true')
    args = ap.parse_args()

    rig = Rig(args.moc3)

    if args.info:
        print(f'{args.moc3}: {len(rig.pids)} 参数, '
              f'{len(rig.g("deformer.ids"))} deformer, '
              f'{len(rig.g("art_mesh.ids"))} artmesh')
        for pid in rig.pids:
            print(f'  {pid}: bindings={rig.param_bindings(pid)}')
        return

    for spec in args.scale:
        pid, f = spec.split('=')
        f = float(f)
        # rotation deformer 走专门路径，warp/mesh 走位置数组路径
        bs = set(rig.param_bindings(pid))
        touched = []
        for kind, name, band, kb, kc, oi in rig.objects():
            if kc <= 0:
                continue
            blist = rig.band_bindings(band)
            if not (bs & set(blist)):
                continue
            if kind == 'rot':
                defs = [rig.binding_default_idx(b) for b in blist]
                axes = [a for a, b in enumerate(blist) if b in bs]
                scale_rot_object(rig, band, kb, axes, defs, f)
                touched.append((kind, name, kc))
            else:
                axes = [a for a, b in enumerate(blist) if b in bs]
                defs = [rig.binding_default_idx(b) for b in blist]
                combos = rig.combo_offsets(blist)
                di = oi if kind == 'warp' else -1
                ai = oi if kind == 'mesh' else -1
                for combo, off in combos:
                    kf = kb + off
                    sec, beg, n = rig.payload_span(kind, di, ai, kf)
                    arr = rig.m[sec]
                    if n < 0:
                        n = len(arr) - beg       # 数组最末 keyform
                    for axis in axes:
                        cdef = list(combo)
                        cdef[axis] = defs[axis]
                        kfd = kb + _offset_of(combos, cdef)
                        _, begd, nd = rig.payload_span(kind, di, ai, kfd)
                        if nd < 0:
                            nd = len(arr) - begd
                        for j in range(min(n, nd)):
                            dev = arr[beg + j] - arr[begd + j]
                            arr[beg + j] += (f - 1.0) * dev
                touched.append((kind, name, kc))
        kinds = {}
        for kind, name, kc in touched:
            kinds[kind] = kinds.get(kind, 0) + 1
        print(f'{pid} ×{f}: 驱动对象 {dict(kinds)}')
        for kind, name, kc in touched[:8]:
            print(f'    {kind} {name} ({kc} keyforms)')
        if len(touched) > 8:
            print(f'    … 共 {len(touched)}')

    out = args.out or args.moc3.replace('.moc3', '.patched.moc3')
    rig.m.to_file(out)
    print(f'写出: {out} ({os.path.getsize(out)} bytes)')


if __name__ == '__main__':
    main()
