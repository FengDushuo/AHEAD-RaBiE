#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 Materials Project 的金属氧化物；更新要点：
- 只在顶部留真空：c → c + vac_add（分数 z 解环方式，避免斜晶格耦合）。
- 仅在 a、b 扩胞到 >=10 Å 的最小整数倍 [ia, ib, 1]。
- 在“加真空 + XY 扩胞”之后 **删除 z 轴坐标最大的 1 个 O**。
- 上/下两侧金属位点池分别按 **25% 掺杂**，且 **替换位点必须为对称等价位置**（整组替换）：
  * 先用 SpacegroupAnalyzer 得到 SymmetrizedStructure 的等价类 equivalent_indices；
  * 在宿主金属池（排除本来就是掺杂元素）内按等价组切分；
  * 通过组合搜索选出一组等价类，使替换总数尽量 **等于 round(0.25 * host_pool_size)**；
    若无法整好，则取最接近目标的可达值（优先不超过目标）。
- 输出每个结构 CIF 与 manifest.csv。
"""

import os, argparse, math, itertools
from typing import List, Tuple, Optional, Sequence
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# 你给的金属集合
ALL_METALS = [
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
]

# ---------- 分数 z 解环 ----------
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
    fz = np.array([site.frac_coords[2] for site in struct.sites])
    return _unwrap_frac_z(fz)

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

# ---------- 顶部真空 ----------
def add_vacuum_with_empty_top(struct: Structure, delta_vac_A: float = 20.0) -> Structure:
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    Z1 = Z0 + float(delta_vac_A)
    new_cvec = lat0.matrix[2] * (Z1 / Z0)
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], new_cvec])

    f = np.array([site.frac_coords for site in s])
    z = f[:, 2]
    zu = _unwrap_frac_z(z)
    z_min = float(np.min(zu))
    scale_frac = Z0 / Z1
    new_z = (zu - z_min) * scale_frac

    new_f = np.copy(f)
    new_f[:, 2] = new_z
    new_f[:, 2] -= np.floor(new_f[:, 2])
    return Structure(new_lat, [site.specie for site in s], new_f,
                     coords_are_cartesian=False, to_unit_cell=True)

# ---------- XY 扩胞到 a,b >= 10 Å ----------
def minimal_xy_supercell(struct: Structure, a_min: float = 10.0, b_min: float = 10.0) -> Tuple[int,int]:
    lat = struct.lattice
    ia = max(1, math.ceil(a_min / lat.a))
    ib = max(1, math.ceil(b_min / lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy()
    s.make_supercell([ia, ib, 1])
    return s

# ---------- 删顶层 1 个 O ----------
def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float]:
    s = struct.copy()
    fz = np.array([site.frac_coords[2] for site in s])
    zu = _unwrap_frac_z(fz)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom found")
    top = max(o_idx, key=lambda i: zu[i])
    top_z = float(zu[top])
    s.remove_sites([top])
    return s, top, top_z

# ---------- 表面识别 ----------
def surface_metal_indices(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites)
              if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals:
        return [], []
    zmax, zmin = float(np.max(zu[metals])), float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

def infer_tol_from_ref(ref_cif: Optional[str]) -> float:
    default_tol = 1.2
    if not ref_cif or not os.path.exists(ref_cif):
        return default_tol
    try:
        s = Structure.from_file(ref_cif)
        lat = s.lattice
        zu = unwrap_frac_along_c(s)
        metals = [i for i,site in enumerate(s.sites)
                  if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
        if len(metals) < 2: return default_tol
        zm = np.sort(zu[metals])
        top, nxt = zm[-1], None
        for val in zm[-2::-1]:
            if abs(top - val) > 1e-3:
                nxt = val; break
        if nxt is None: return default_tol
        gapA = (top - nxt) * lat.c
        return max(0.6, min(2.5, 0.5*gapA))
    except Exception:
        return default_tol

# ---------- 对称等价 + 25% 目标组合 ----------
def symmetry_orbits(struct: Structure, indices: Sequence[int], symprec: float = 1e-3, angle_tol: float = 5.0) -> List[List[int]]:
    """
    返回 'indices' 中各位点按对称等价分组后的列表（每个子列表是一条等价轨道，含局部索引）。
    """
    sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tol)
    try:
        ss = sga.get_symmetrized_structure()
        eq = ss.equivalent_indices  # List[List[int]] 全结构的等价分组
    except Exception:
        # 如果失败，就把每个点当作单独等价类
        return [[i] for i in indices]
    idx_set = set(indices)
    groups = []
    for g in eq:
        part = [i for i in g if i in idx_set]
        if part:
            groups.append(sorted(part))
    # 可能不同等价类在 indices 上重叠为空，已过滤
    return groups

def choose_orbits_to_target(orbits: List[List[int]], target: int) -> List[List[int]]:
    """
    给定若干等价组的长度数组 w_i，选择若干组使总和尽量等于 target。
    策略：先做可达和的动态规划（子集和），优先找 <=target 的最大可达值；若没有正好命中，则找最接近的 >=target。
    """
    sizes = [len(g) for g in orbits]
    n = len(sizes)
    if n == 0 or target <= 0:
        return []

    # DP: reachable sums -> 前驱
    # 限制规模：若组数多，可用近似贪心；一般表面组数不大，DP可行
    reachable = {0: []}  # sum -> list of chosen indices
    for i, w in enumerate(sizes):
        new_reach = dict(reachable)
        for s, path in reachable.items():
            s2 = s + w
            if s2 not in new_reach:
                new_reach[s2] = path + [i]
        reachable = new_reach

    # 找到最优和
    best_sum = None
    # 先找 <= target 的最大
    le = [s for s in reachable.keys() if s <= target]
    if le:
        best_sum = max(le)
    # 若没有或太偏差，再找 >= target 的最小
    if best_sum is None or best_sum == 0:
        ge = [s for s in reachable.keys() if s >= target]
        if ge:
            cand = min(ge, key=lambda x: abs(x - target))
            if best_sum is None or abs(cand - target) < abs(best_sum - target):
                best_sum = cand
    # 兜底
    if best_sum is None:
        return []

    chosen_idx = reachable[best_sum]
    return [orbits[i] for i in chosen_idx]

def pick_symmetry_equiv_sites_per_side(struct: Structure, pool_idx: List[int], dopant: str,
                                       frac: float, symprec: float, angle_tol: float) -> Tuple[List[int], int, int, int]:
    """
    在“宿主金属池”中按对称等价组组合选择，尽量达到 25%（frac）目标。
    返回: (chosen_indices, host_pool_size, target, achieved)
    """
    # 宿主金属池（排除已是掺杂元素）
    host_pool = [i for i in pool_idx
                 if (isinstance(struct[i].specie, Element) and struct[i].specie.is_metal
                     and struct[i].specie.symbol != "O" and struct[i].specie.symbol != dopant)]
    S = len(host_pool)
    if S == 0:
        return [], 0, 0, 0
    target = max(1, int(round(S * frac)))

    # 对称等价分组（在 host_pool 内）
    orbits = symmetry_orbits(struct, host_pool, symprec=symprec, angle_tol=angle_tol)
    if not orbits:
        return [], S, target, 0

    # 只保留完全落在 host_pool 的组（已在构造时做到）
    chosen_orbits = choose_orbits_to_target(orbits, target)
    chosen = sorted(j for g in chosen_orbits for j in g)
    return chosen, S, target, len(chosen)

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for i in idx:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="用于推断表面 tol（Å）")
    ap.add_argument("--outdir", type=str, default="out_vacTop_xyMin10_rmO_sym25")
    ap.add_argument("--dopants", type=str, default="ALL", help="ALL 或逗号分隔，如 'Fe,Co'")
    ap.add_argument("--dopant_fraction", type=float, default=0.25, help="每侧宿主金属位点的目标掺杂分数（默认 0.25）")
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--vac_add", type=float, default=20.0)
    ap.add_argument("--symprec", type=float, default=1e-3)
    ap.add_argument("--angle_tol", type=float, default=5.0)
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    dopant_list = ALL_METALS if args.dopants.upper()=="ALL" else [x.strip() for x in args.dopants.split(",") if x.strip()]
    frac = float(args.dopant_fraction)
    assert 0.0 < frac < 1.0, "--dopant_fraction 应在 (0,1) 内，例如 0.25"

    tol = infer_tol_from_ref(args.ref_cif)

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")

    manifest = []

    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        docs = docs[:args.max_docs]

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue

            # 1) 顶部真空
            s_vac = add_vacuum_with_empty_top(struct, delta_vac_A=args.vac_add)

            # 2) XY 最小扩胞
            ia, ib = minimal_xy_supercell(s_vac, a_min=args.a_min, b_min=args.b_min)
            s_xy = make_supercell_xy(s_vac, ia, ib)

            # 3) 删除顶层 1 个 O
            try:
                s_rmO, del_idx, del_fz = remove_topmost_oxygen_by_frac(s_xy)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue

            # 4) 表面金属池
            top_idx, bot_idx = surface_metal_indices(s_rmO, tol_A=tol)
            if len(top_idx)==0 or len(bot_idx)==0:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_surface"})
                continue

            for dop in dopant_list:
                # 顶/底分别：按对称等价组选择，尽量达到 25%
                picks_top, S_top, T_top, A_top = pick_symmetry_equiv_sites_per_side(
                    s_rmO, top_idx, dop, frac, args.symprec, args.angle_tol
                )
                picks_bot, S_bot, T_bot, A_bot = pick_symmetry_equiv_sites_per_side(
                    s_rmO, bot_idx, dop, frac, args.symprec, args.angle_tol
                )

                doped = substitute(s_rmO, picks_top + picks_bot, dop)

                fname = f"{ddoc.material_id}_vacTop{int(args.vac_add)}_a{ia}x_b{ib}x_rmO_sym25_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                manifest.append({
                    "material_id": ddoc.material_id,
                    "formula": getattr(ddoc, "formula_pretty",""),
                    "status": "ok",
                    "outfile": fpath,
                    "ia": ia, "ib": ib,
                    "vac_add": args.vac_add,
                    "deleted_O_index": del_idx, "deleted_O_fz": del_fz,
                    "top_host_pool": S_top, "top_target": T_top, "top_achieved": A_top,
                    "bot_host_pool": S_bot, "bot_target": T_bot, "bot_achieved": A_bot,
                    "dopant": dop
                })
                print("Wrote", fname)

    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] wrote {sum(m.get('status')=='ok' for m in manifest)} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs; all skipped")

if __name__ == "__main__":
    main()
