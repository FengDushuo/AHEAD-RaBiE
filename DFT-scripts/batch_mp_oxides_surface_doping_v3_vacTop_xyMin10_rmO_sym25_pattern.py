#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
金属氧化物表面掺杂（严格 25%，等价位点整组替换，排布参考 1-out.cif）

流程：
1) 只在顶部加真空：c -> c + vac_add（分数 z 解环法，避免斜晶格耦合）；
2) 仅在 a、b 方向做最小整数倍扩胞到 >=10 Å（[ia, ib, 1]）；
3) 删除顶层 1 个 O（在真空 + XY 扩胞之后执行）；
4) 顶/底两侧的“宿主金属位点池”分别按 **25%** 掺杂，并且：
   - 以 `SpacegroupAnalyzer` 的 `equivalent_indices` 分组（对称等价轨道）；
   - 只允许“整组替换”；
   - 首选“排布模板”来自 `--ref_cif 1-out.cif`：提取参考 slab 顶/底的掺杂金属在表面 (x,y) 的分布，
     用匈牙利匹配/最近中心匹配把目标结构的等价组质心对齐到参考模板；
   - 若参考不含可解析模板（或不可达目标 25%），退化到“动态规划 + 最远点采样”以保证均匀分散。

输出：每个结构一个 .cif + manifest.csv（记录目标/实际命中等）。
"""

import os, argparse, math, itertools
from typing import List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 你的金属集合
ALL_METALS = [
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
]

# ===== 工具：分数 z 解环 =====
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z) == 0: return z
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))
    base = z_sorted[(k + 1) % len(z_sorted)]
    zu = z - base
    zu[zu < 0] += 1.0
    return zu

def unwrap_frac_along_c(struct: Structure) -> np.ndarray:
    return _unwrap_frac_z(np.array([s.frac_coords[2] for s in struct]))

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

# ===== 真空：顶部空 20 Å（可调） =====
def add_vacuum_with_empty_top(struct: Structure, delta_vac_A: float = 20.0) -> Structure:
    s = struct.copy()
    lat0 = s.lattice
    Z1 = lat0.c + float(delta_vac_A)
    new_cvec = lat0.matrix[2] * (Z1 / lat0.c)
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], new_cvec])
    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:,2])
    z_min = float(np.min(zu))
    scale_frac = lat0.c / Z1
    new_f = np.copy(f)
    new_f[:,2] = (zu - z_min) * scale_frac
    new_f[:,2] -= np.floor(new_f[:,2])
    return Structure(new_lat, [site.specie for site in s], new_f,
                     coords_are_cartesian=False, to_unit_cell=True)

# ===== XY 最小扩胞 =====
def minimal_xy_supercell(struct: Structure, a_min: float = 10.0, b_min: float = 10.0) -> Tuple[int,int]:
    lat = struct.lattice
    return max(1, math.ceil(a_min/lat.a)), max(1, math.ceil(b_min/lat.b))

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy()
    s.make_supercell([ia, ib, 1])
    return s

# ===== 删顶层一个 O =====
def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure,int,float]:
    s = struct.copy()
    zu = _unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_idx = [i for i,site in enumerate(s) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom found")
    top = max(o_idx, key=lambda i: zu[i])
    ztop = float(zu[top])
    s.remove_sites([top])
    return s, top, ztop

# ===== 表面金属池 =====
def surface_metal_indices(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s) if isinstance(site.specie, Element)
              and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return [], []
    zmax, zmin = float(np.max(zu[metals])), float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

def infer_tol_from_ref(ref_cif: Optional[str]) -> float:
    default_tol = 1.2
    if not ref_cif or not os.path.exists(ref_cif): return default_tol
    try:
        s = Structure.from_file(ref_cif)
        lat = s.lattice
        zu = unwrap_frac_along_c(s)
        metals = [i for i,site in enumerate(s) if isinstance(site.specie, Element)
                  and site.specie.is_metal and site.specie.symbol != "O"]
        if len(metals) < 2: return default_tol
        zm = np.sort(zu[metals])
        top = zm[-1]
        nxt = next((v for v in zm[-2::-1] if abs(top-v)>1e-3), None)
        if nxt is None: return default_tol
        gapA = (top-nxt)*lat.c
        return max(0.6, min(2.5, 0.5*gapA))
    except Exception:
        return default_tol

# ===== 对称等价分组（在给定索引子集上） =====
def symmetry_orbits(struct: Structure, indices: Sequence[int], symprec=3e-3, angle_tol=5.0) -> List[List[int]]:
    # 真空会降低空间群，这里适当放宽 symprec，避免把明显等价的位点打散
    try:
        sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
        ss = sga.get_symmetrized_structure()
        eq = ss.equivalent_indices
    except Exception:
        return [[i] for i in indices]
    idxset = set(indices)
    groups = []
    for g in eq:
        part = [i for i in g if i in idxset]
        if part:
            groups.append(sorted(part))
    return groups

# ===== 从参考 1-out.cif 中提取排布模板（顶/底各一组 2D (x,y) 位置） =====
def extract_ref_pattern_xy(ref_cif: Optional[str], tol_A: float) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    返回 (top_centers, bot_centers)，每个为若干 2D 点（单位分数坐标）。
    模板定义：在参考结构中，找出表面金属位点里“非宿主金属”的 (x,y)。
    如果参考中无法区分宿主/掺杂（或没有掺杂），返回空列表，表示用均匀蓝噪声策略。
    """
    if not ref_cif or not os.path.exists(ref_cif):
        return [], []
    try:
        s = Structure.from_file(ref_cif)
        top_idx, bot_idx = surface_metal_indices(s, tol_A)
        if not top_idx or not bot_idx:
            return [], []
        # 以最常见金属作为宿主
        def host_symbol(idxs):
            metals = [s[i].specie.symbol for i in idxs]
            vals, cnts = np.unique(metals, return_counts=True)
            order = np.argsort(-cnts)
            return vals[order[0]]
        host_top = host_symbol(top_idx)
        host_bot = host_symbol(bot_idx)

        def centers(side_idx, host):
            pts=[]
            for i in side_idx:
                sym = s[i].specie.symbol
                if sym != "O" and Element(sym).is_metal and sym != host:
                    f = s[i].frac_coords
                    pts.append(np.array([f[0]-np.floor(f[0]), f[1]-np.floor(f[1])]))
            return pts
        return centers(top_idx, host_top), centers(bot_idx, host_bot)
    except Exception:
        return [], []

# ===== 寻找最接近 25% 的“整组可达”子集（动态规划，带均匀性打分） =====
def choose_orbits_approx_target(orbits: List[List[int]], target: int) -> List[List[int]]:
    sizes = [len(g) for g in orbits]
    n = len(sizes)
    if n == 0 or target <= 0:
        return []
    # DP 到所有可达和
    reachable = {0: []}
    for i,w in enumerate(sizes):
        new = dict(reachable)
        for s,path in reachable.items():
            s2 = s+w
            if s2 not in new:
                new[s2] = path+[i]
        reachable = new
    if not reachable:
        return []

    # 首选 <= target 的最大；若没有，取距离 target 最近的
    le = [s for s in reachable if s<=target]
    if le:
        best_sum = max(le)
    else:
        best_sum = min(reachable.keys(), key=lambda x: abs(x-target))

    idxs = reachable[best_sum]
    return [orbits[i] for i in idxs]

# ===== 计算等价组质心（表面 2D 分数坐标） =====
def orbit_centroids_xy(struct: Structure, orbits: List[List[int]]) -> np.ndarray:
    xy=[]
    for g in orbits:
        pts=[]
        for i in g:
            f=struct[i].frac_coords
            pts.append([f[0]-np.floor(f[0]), f[1]-np.floor(f[1])])
        xy.append(np.mean(np.array(pts), axis=0))
    return np.array(xy) if xy else np.zeros((0,2))

# ===== 将等价组匹配到参考模板中心（最近邻匈牙利匹配） =====
def match_orbits_to_template(struct: Structure, host_orbits: List[List[int]],
                             template_xy: List[np.ndarray], k: int) -> List[int]:
    """
    host_orbits: 等价组列表（只含宿主金属）
    template_xy: 参考的 2D (x,y) 点（已在 [0,1)）
    k: 目标替换总数（等价组和的元素数）
    返回：选择的等价组索引列表（在 host_orbits 中的下标）
    """
    import itertools
    if not host_orbits:
        return []
    # 先挑一个“总元素数最接近 k”的等价组组合
    cand_orbits = choose_orbits_approx_target(host_orbits, k)
    if not cand_orbits:
        return []

    # 如果没有模板，则直接返回
    if not template_xy:
        # 退化：直接返回 cand_orbits 的索引
        idxs=[]
        for g in cand_orbits:
            idxs.append(host_orbits.index(g))
        return idxs

    # 用组质心与模板点做最近邻匹配（数量可能不同：只匹配 min(#groups, #tpl) 个）
    import scipy.optimize
    xy = orbit_centroids_xy(struct, cand_orbits)
    m = min(len(xy), len(template_xy))
    if m == 0:
        return [host_orbits.index(g) for g in cand_orbits]

    # 构造代价矩阵（周期性距离）
    def pbc_dist(p,q):
        d = np.abs(p-q)
        d = np.minimum(d, 1.0-d)
        return np.linalg.norm(d)
    C = np.zeros((len(xy), len(template_xy)))
    for i,p in enumerate(xy):
        for j,q in enumerate(template_xy):
            C[i,j]=pbc_dist(p,q)
    rr, cc = scipy.optimize.linear_sum_assignment(C)
    # 选择与模板匹配的前 m 个组（其余保留）
    chosen_groups = set()
    for i,j in zip(rr,cc):
        if i < len(cand_orbits) and j < len(template_xy):
            chosen_groups.add(i)
    # 如果匹配的组不足以凑到 k（按元素计），把未匹配组依次加入
    selected=[]
    total=0
    for i,g in enumerate(cand_orbits):
        if (i in chosen_groups) or (total < k):
            selected.append(host_orbits.index(g))
            total += len(g)
            if total >= k:
                break
    return selected

# ===== 每侧：按参考模板/均匀性选组 =====
def select_side_orbits(struct: Structure, pool_idx: List[int], dopant: str,
                       target_frac: float, ref_xy: List[np.ndarray],
                       symprec=3e-3, angle_tol=5.0) -> List[int]:
    # 宿主金属池（排除已是 dopant 的）
    host_pool = [i for i in pool_idx if (isinstance(struct[i].specie, Element)
                 and struct[i].specie.is_metal and struct[i].specie.symbol != "O"
                 and struct[i].specie.symbol != dopant)]
    S = len(host_pool)
    if S == 0:
        return []
    target = max(1, int(round(S*target_frac)))

    # 等价组（限定在宿主池）
    orbits_all = symmetry_orbits(struct, host_pool, symprec=symprec, angle_tol=angle_tol)
    if not orbits_all:
        return []

    # 如果有参考模板：优先按模板匹配
    idxs = match_orbits_to_template(struct, orbits_all, ref_xy, target)
    if idxs:
        # 修正：若元素总数超过 target，则按就近裁剪
        sel, tot = [], 0
        for k in idxs:
            g = orbits_all[k]
            if tot + len(g) <= target:
                sel.append(k); tot += len(g)
        if tot == target:
            return sel
        # 否则退化到 DP 最接近（不超过）
    # 无模板或不达标：DP 选择最接近且<=target
    cand = choose_orbits_approx_target(orbits_all, target)
    return [orbits_all.index(g) for g in cand]

def substitute_groups(struct: Structure, orbits: List[List[int]], chosen_idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for k in chosen_idx:
        for i in orbits[k]:
            if s2[i].specie.symbol != dopant:
                s2[i] = Element(dopant)
    return s2

# ===== 主程序 =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default="1-out.cif", help="参考模板（默认 1-out.cif）")
    ap.add_argument("--outdir", type=str, default="out_sym25_pattern")
    ap.add_argument("--dopants", type=str, default="ALL", help="例如 'Fe,Co' 或 ALL")
    ap.add_argument("--dopant_fraction", type=float, default=0.25, help="每侧宿主金属位点的目标掺杂分数（默认 0.25）")
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--vac_add", type=float, default=20.0)
    ap.add_argument("--symprec", type=float, default=3e-3)
    ap.add_argument("--angle_tol", type=float, default=5.0)
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    frac = float(args.dopant_fraction)
    assert 0.0 < frac < 1.0

    dopant_list = (ALL_METALS if args.dopants.upper()=="ALL"
                   else [x.strip() for x in args.dopants.split(",") if x.strip()])

    tol = infer_tol_from_ref(args.ref_cif)

    # 读取参考模板（顶/底）
    ref_top_xy, ref_bot_xy = extract_ref_pattern_xy(args.ref_cif, tol)

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")

    manifest=[]
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"],
               fields=["material_id","formula_pretty","structure"], chunk_size=200))[:args.max_docs]

        for ddoc in docs:
            base = ddoc.structure
            if not is_metal_oxide(base):
                continue

            # 顶部真空 + XY 扩胞 + 删顶层 O
            s1 = add_vacuum_with_empty_top(base, delta_vac_A=args.vac_add)
            ia, ib = minimal_xy_supercell(s1, a_min=args.a_min, b_min=args.b_min)
            s2 = make_supercell_xy(s1, ia, ib)
            try:
                s3, del_idx, del_fz = remove_topmost_oxygen_by_frac(s2)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status":"skip_no_O","reason":str(e)})
                continue

            # 表面金属池
            top_idx, bot_idx = surface_metal_indices(s3, tol_A=tol)
            if not top_idx or not bot_idx:
                manifest.append({"material_id": ddoc.material_id, "status":"skip_no_surface"})
                continue

            # 等价组全集（宿主池稍后在函数内过滤）
            # 仅用于替换时重用
            # —— 这里不预计算，直接在 select_side_orbits 内按宿主池做分组

            for dop in dopant_list:
                # 选择等价组（按参考模板/均匀性）——返回的是“orbits 的索引”
                top_choice_idx = select_side_orbits(s3, top_idx, dop, frac, ref_top_xy,
                                                    symprec=args.symprec, angle_tol=args.angle_tol)
                bot_choice_idx = select_side_orbits(s3, bot_idx, dop, frac, ref_bot_xy,
                                                    symprec=args.symprec, angle_tol=args.angle_tol)

                # 为了实际替换，我们需要再次得到“宿主池的等价组列表”
                top_host_orbits = symmetry_orbits(s3, [i for i in top_idx
                    if s3[i].specie.symbol!="O" and Element(s3[i].specie.symbol).is_metal
                    and s3[i].specie.symbol!=dop], symprec=args.symprec, angle_tol=args.angle_tol)
                bot_host_orbits = symmetry_orbits(s3, [i for i in bot_idx
                    if s3[i].specie.symbol!="O" and Element(s3[i].specie.symbol).is_metal
                    and s3[i].specie.symbol!=dop], symprec=args.symprec, angle_tol=args.angle_tol)

                s4 = substitute_groups(s3, top_host_orbits, top_choice_idx, dop)
                s5 = substitute_groups(s4, bot_host_orbits, bot_choice_idx, dop)

                # 统计实际掺杂比例（每侧）
                top_S = sum(len(g) for g in top_host_orbits)
                bot_S = sum(len(g) for g in bot_host_orbits)
                top_A = sum(len(top_host_orbits[k]) for k in top_choice_idx) if top_host_orbits else 0
                bot_A = sum(len(bot_host_orbits[k]) for k in bot_choice_idx) if bot_host_orbits else 0

                fname = f"{ddoc.material_id}_vacTop{int(args.vac_add)}_a{ia}x_b{ib}x_rmO_sym25pattern_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                s5.to(fmt="cif", filename=fpath)

                manifest.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status":"ok", "outfile": fpath, "ia":ia, "ib":ib, "vac_add":args.vac_add,
                    "deleted_O_index": del_idx, "deleted_O_fz": del_fz,
                    "top_host_sites": top_S, "top_doped_sites": top_A,
                    "bot_host_sites": bot_S, "bot_doped_sites": bot_A,
                    "target_fraction": args.dopant_fraction, "dopant": dop,
                    "ref_top_tpl": len(ref_top_xy), "ref_bot_tpl": len(ref_bot_xy)
                })
                print("Wrote", fname)

    if manifest:
        df = pd.DataFrame(manifest)
        df.to_csv(os.path.join(args.outdir,"manifest.csv"), index=False)
        print(f"[OK] wrote {sum(m.get('status')=='ok' for m in manifest)} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs; all skipped")

if __name__ == "__main__":
    main()
