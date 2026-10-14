#!/usr/bin/env python3
"""moc3 自动绑骨 —— 结构增长：给 artmesh 新增一条参数轴并生成全套 keyform

阶段 3 第四块（自动绑骨的落地）：把「关键姿势库 + 位移场」的产物写成
Cubism 能直接消费的绑骨数据，而不是只做已有绑骨的缩放（moc3_rig_write.py）。

原理（已在 Haru 上实测确认）：
  * moc3 各段是 struct-of-arrays；非 RUNTIME 段的字节长度 = len(数组)，
    RUNTIME 段（*_runtime_space）长度 = counts[idx]*8，CountInfo 由 counts[]
    单独写出。=> 追加元素只需「扩展数组 + 同步 counts[idx]」。
  * 对象的 band（keyform_binding_band）是 keyform 组合的坐标轴列表，
    keyform 组合总数 = ∏(各 binding 的 keys 数) = 该对象的 keyform_counts。
  * band 列表**最后一个 binding 是主轴**（stride 最大）=> 新轴追加到 band 末尾时，
    新布局 = 每个新 key 一层、层内照抄旧 grid：new_kf = j*old_kc + t。
  * keyform 位置段步长：artmesh = 2*ceil(position_index_counts/8)*8 floats
    （position_index_counts 实为**顶点数**，vertex_counts 实为三角索引数）；
    warp = 2*ceil(warp_deformer.vertex_counts/8)*8 floats。

本模块对目标 artmesh：
  1) 追加 1 个新参数（值域 [min,max]，默认 def），无 binding；
  2) 追加 1 个 binding（keys = 新轴关键值，如 [-30,0,30]）；
  3) 追加 1 个新 band = 旧 band 的 binding 列表 + [新 binding]（不搬运旧 band，
     旧 band 仍供其他对象使用）；kbi 追加对应条目，band begin 指向 kbi 末尾；
  4) 在 keyform_position.xys 末尾追加 3*old_kc 个位置块：j 层照抄旧 block，
     非默认 key 层叠加位移场 Δ(key, vertex)；
  5) art_mesh_keyform 的 opacities/draw_orders/keyform_position_begin_indices
     各追加 3*old_kc 项；目标 artmesh 指向新 kf 区间，keyform_counts = 3*old_kc；
  6) 同步 counts[PARAMETERS/KEYFORM_BINDINGS/KEYFORM_BINDING_BANDS/
     KEYFORM_BINDING_INDICES/KEYS/ART_MESH_KEYFORMS/KEYFORM_POSITIONS]。

默认 key 层为旧 block 的**逐位拷贝**，故 rest 姿态与位移场无关，逐位不变。

用法：
  python3 tools/agents/moc3_auto_rig.py assets/models/Haru/Haru.moc3 \
      --param ParamAutoTestX --min -30 --max 30 \
      --mesh D_PSD_09 --deform wave --amp 30 --out /tmp/HaruRig.moc3
  python3 tools/agents/moc3_auto_rig.py <moc3> --info
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.moc3_rig_write import Rig  # noqa: E402


# --- 位移场 ---------------------------------------------------------------
def _deform_offsets(kind, amp, vtx, xs, ys):
    """返回每顶点的 (dx, dy)。amp 已含 key 方向与大小（默认 key 层为 0）。

    位形以顶点序 u 为参数（不依赖绝对坐标，便于对不同网格复用）：
      translate : 整体平移 (amp, 0)
      sway      : 水平剪切，随顶点序号线性增长（像被风吹起的发梢/飘带）
      wave      : 正弦波，横竖交替（最直观的顶点级形变）
    """
    out = []
    for u in range(vtx):
        p = u / max(1, vtx - 1)
        if kind == 'translate':
            dx, dy = amp, 0.0
        elif kind == 'sway':
            dx, dy = amp * p, amp * 0.15 * p
        elif kind == 'wave':
            dx, dy = amp * math.sin(2 * math.pi * p), amp * 0.5 * math.sin(4 * math.pi * p)
        else:
            raise ValueError(f'未知位移场: {kind}')
        out.append((dx, dy))
    return out


class AutoRig:
    """结构增长型绑骨写回"""

    def __init__(self, path):
        self.rig = Rig(path)
        self.m = self.rig.m

    # ---- 基础量 ----
    @staticmethod
    def stride_floats(vtx):
        return 2 * math.ceil(vtx / 8) * 8

    def mesh_index(self, name):
        ids = self.m['art_mesh.ids']
        assert name in ids, f'artmesh 不存在: {name}（可选: {ids[:8]}…）'
        return ids.index(name)

    def mesh_vtx(self, ai):
        return self.m['art_mesh.position_index_counts'][ai]

    # ---- 四个增长步骤 ----
    def add_param(self, pid, mn, mx, dflt, dec=2, repeats=False):
        m = self.m
        n = m.counts[5]
        nb = m.counts[13]
        m['parameter.ids'].append(pid)
        m['parameter.max_values'].append(float(mx))
        m['parameter.min_values'].append(float(mn))
        m['parameter.default_values'].append(float(dflt))
        m['parameter.repeats'].append(bool(repeats))
        m['parameter.decimal_places'].append(int(dec))
        m['parameter.keyform_binding_begin_indices'].append(nb)
        m['parameter.keyform_binding_counts'].append(0)   # 占位，绑定时回填 1
        m.counts[5] = n + 1
        return n  # 新参数下标

    def add_binding(self, keys):
        m = self.m
        b = m.counts[13]
        ks = len(m['keys.values'])
        m['keys.values'].extend(float(k) for k in keys)
        m['keyform_binding.keys_begin_indices'].append(ks)
        m['keyform_binding.keys_counts'].append(len(keys))
        m.counts[13] = b + 1
        m.counts[14] = ks + len(keys)
        return b

    def add_band(self, bindings):
        """追加新 band（列表顺序保持原样，末位为主轴）"""
        m = self.m
        band = m.counts[12]
        beg = len(m['keyform_binding_index.indices'])
        m['keyform_binding_index.indices'].extend(bindings)
        m['keyform_binding_band.begin_indices'].append(beg)
        m['keyform_binding_band.counts'].append(len(bindings))
        m.counts[11] = beg + len(bindings)
        m.counts[12] = band + 1
        return band

    def bind_param(self, pidx, binding):
        self.m['parameter.keyform_binding_counts'][pidx] = 1
        return binding

    def regen_mesh_keyforms(self, ai, keys, dflt, kind, amp):
        """按 band 末端新轴重新生成该 mesh 的全套 keyform 位置"""
        m = self.m
        vtx = self.mesh_vtx(ai)
        st = self.stride_floats(vtx)
        assert st == 2 * math.ceil(vtx / 8) * 8

        kfbeg = m['art_mesh_keyform.keyform_position_begin_indices']
        obeg = m['art_mesh.keyform_begin_indices'][ai]
        okc = m['art_mesh.keyform_counts'][ai]
        opa = m['art_mesh_keyform.opacities']
        dro = m['art_mesh_keyform.draw_orders']
        xys = m['keyform_position.xys']

        # 旧 block（按旧枚举顺序，逐块拷贝；步长均匀，用已知 st 而非相邻差）
        old_blocks = []
        for t in range(okc):
            s = kfbeg[obeg + t]
            blk = xys[s:s + st]
            assert len(blk) == st, f'{ai} kf{t} block 越界: {len(blk)} != {st}'
            old_blocks.append(blk)

        # 新布局：new_kf = j*okc + t；j 对应 keys[j]
        nkf0 = len(kfbeg)
        diff = max(abs(k - dflt) for k in keys) or 1.0
        appended = 0
        for j, k in enumerate(keys):
            scale = (k - dflt) / diff
            offs = _deform_offsets(kind, amp * scale, vtx, None, None)
            for t in range(okc):
                kfbeg.append(len(xys))
                blk = old_blocks[t]
                for u in range(vtx):
                    dx, dy = offs[u]
                    xys.append(blk[2 * u] + dx)
                    xys.append(blk[2 * u + 1] + dy)
                # 补齐 padding（步长按 8 pairs 对齐，多余槽位照抄旧值）
                pad_from = 2 * vtx
                for q in range(pad_from, st):
                    xys.append(blk[q] if q < len(blk) else 0.0)
                appended += st
                opa.append(opa[obeg + t])
                dro.append(dro[obeg + t])

        m['art_mesh.keyform_begin_indices'][ai] = nkf0
        m['art_mesh.keyform_counts'][ai] = len(keys) * okc
        m.counts[9] = nkf0 + len(keys) * okc
        m.counts[10] += appended
        return dict(vtx=vtx, stride_floats=st, old_kc=okc, new_kc=len(keys) * okc)

    # ---- 组合动作 ----
    def auto_rig(self, meshes, pid, keys, kind='wave', amp=0.3,
                 mn=None, mx=None):
        """一个新增参数驱动多个 artmesh（真正的自动绑骨语义：新轴 → 多对象）

        meshes: ['D_PSD_74', ('D_PSD_04', 0.16)] —— 字符串或 (名, 幅值)；
                幅值单位是**该网格局部坐标单位**，不同网格的父级 warp 链缩放
                不同，故可逐网格覆写（否则同一 amp 在不同网格上视觉幅度差数倍）。
        """
        m = self.m
        if isinstance(meshes, str):
            meshes = [meshes]
        if mn is None:
            mn = min(keys)
        if mx is None:
            mx = max(keys)
        # 默认值：keys 中使位移为 0 者；本模块约定 keys[1] 为默认轴关键值
        dflt = float(keys[1])

        pidx = self.add_param(pid, mn, mx, dflt)
        binding = self.add_binding(keys)
        self.bind_param(pidx, binding)

        infos = []
        for spec in meshes:
            name, mamp = (spec, amp) if isinstance(spec, str) else (spec[0], spec[1])
            ai = self.mesh_index(name)
            # 一致性核验：旧 kc 应等于旧 band 各 binding keys 数之积
            old_band = m['art_mesh.keyform_binding_band_indices'][ai]
            blist = self.rig.band_bindings(old_band)
            prod = 1
            for b in blist:
                prod *= len(self.rig.binding_keys(b))
            old_kc = m['art_mesh.keyform_counts'][ai]
            assert prod == old_kc, f'旧 band 组合数 {prod} != keyform_counts {old_kc}'
            new_band = self.add_band(blist + [binding])
            m['art_mesh.keyform_binding_band_indices'][ai] = new_band
            info = self.regen_mesh_keyforms(ai, keys, dflt, kind, mamp)
            info.update(mesh=name, ai=ai, old_band=old_band, new_band=new_band,
                        old_bindings=blist, amp=mamp)
            infos.append(info)
        return {'param': pid, 'param_index': pidx, 'binding': binding,
                'keys': list(keys), 'dflt': dflt, 'deform': kind, 'amp': amp,
                'meshes': infos}


def main():
    ap = argparse.ArgumentParser(description='moc3 自动绑骨（结构增长）')
    ap.add_argument('moc3')
    ap.add_argument('--param', default='ParamAutoAxisX', help='新参数 id')
    ap.add_argument('--min', type=float, default=None)
    ap.add_argument('--max', type=float, default=None)
    ap.add_argument('--keys', default=None, help='逗号分隔，3 个（默认轴的默认值）')
    ap.add_argument('--mesh', action='append', required=False,
                    help='目标 artmesh id，可重复；可写 NAME=amp 逐网格覆写幅值')
    ap.add_argument('--deform', default='sway', choices=['translate', 'sway', 'wave'])
    ap.add_argument('--amp', type=float, default=0.3,
                    help='位移场峰值（目标网格局部单位；父级 warp 会平滑/衰减）')
    ap.add_argument('--out', default=None)
    ap.add_argument('--info', action='store_true')
    args = ap.parse_args()

    ar = AutoRig(args.moc3)

    if args.info:
        m = ar.m
        print(f'{args.moc3}: params={m.counts[5]} artmeshes={m.counts[4]} '
              f'bindings={m.counts[13]} bands={m.counts[12]} kbi={m.counts[11]} '
              f'keys={m.counts[14]} art_mesh_keyforms={m.counts[9]} '
              f'keyform_positions={m.counts[10]}')
        for i, n in enumerate(m['art_mesh.ids'][:20]):
            print(f'  [{i:3d}] {n:24s} vtx={ar.mesh_vtx(i):4d} '
                  f'kc={m["art_mesh.keyform_counts"][i]}')
        return

    assert args.mesh, '需要 --mesh'
    keys = [float(x) for x in args.keys.split(',')] if args.keys else \
        [args.min if args.min is not None else -30.0,
         0.0,
         args.max if args.max is not None else 30.0]
    # NAME 或 NAME=amp
    meshes = []
    for s in args.mesh:
        if '=' in s:
            n, a = s.split('=', 1)
            meshes.append((n, float(a)))
        else:
            meshes.append(s)

    info = ar.auto_rig(meshes, args.param, keys, args.deform, args.amp,
                       mn=args.min, mx=args.max)
    out = args.out or args.moc3.replace('.moc3', '.autorig.moc3')
    ar.m.to_file(out)
    print(f'自动绑骨: 新参数 {info["param"]} (keys={info["keys"]}, def={info["dflt"]}) '
          f'→ binding {info["binding"]}, 位移场 {info["deform"]} amp={info["amp"]}')
    for mi in info['meshes']:
        print(f'  {mi["mesh"]:22s} vtx={mi["vtx"]:4d} amp={mi["amp"]} '
              f'keyform {mi["old_kc"]}→{mi["new_kc"]} '
              f'band {mi["old_band"]}→{mi["new_band"]} (原绑定 {mi["old_bindings"]})')
    print(f'写出: {out} ({os.path.getsize(out)} bytes, 源 {os.path.getsize(args.moc3)})')


if __name__ == '__main__':
    main()
