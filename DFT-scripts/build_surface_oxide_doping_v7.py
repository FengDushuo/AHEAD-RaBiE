#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_surface_oxide_doping_v7.py

严谨构建“金属氧化物表面掺杂体系”，融合你偏好的 v3 行为与更稳的几何判定：
- 宿体几何参考 host_ref（如 0-out.cif）：继承沿 c 的真空厚度与表面层 tol；支持手动覆盖
- 掺杂风格参考 doped_ref（如 2-out.cif）：若未指定 dopants/sides，则自动推断
- 宿体来源：本地 CIF（--host_cifs）或 Materials Project 二元氧化物（--host_metals）
- 仅 x/y 扩胞（az=1），两种目标模式：
    1) target-atoms 模式（默认）：删顶层 1 个 O 后的总原子数 ≈ --target_atoms（默认 95）
    2) target-O 模式：如果给了 --target_O（如 64），则选择 (ax,ay) 使 O 数达到该目标；随后删除顶层 1 个 O
- 采用“沿 c 的分数坐标 + 解环（unwrap）”判定顶/底表面与居中；加真空并居中
- 表面金属替位掺杂（非 O 位点），全局 ≈ 1:3（--dopant_to_host 可调），避免空替换并必要回填
- 输出每个体系 CIF 与 manifest.csv

依赖：pymatgen>=2023.0.0, mp-api, numpy, pandas
"""

import os, re, glob, argparse
from typing import List, Tuple, Optional, Dict, Iterable
import numpy as np
import pandas as pd
from collections import Counter

from pymatgen.core import Structure, Element, Lattice
from mp_api.client import MPRester

# -------------------- 常量 --------------------
ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# -------------------- 基础工具（沿 c 的分数坐标 + 解环） --------------------
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))  # 最大缝隙右端
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

# -------------------- 参考模板：host_ref & doped_ref --------------------
def infer_vac_and_tol_from_host(ref_cif: Optional[str]) -> Tuple[Optional[float], float, Optional[Structure]]:
    """
    从 host_ref (如 0-out.cif) 推断：
      - total_vacuum_A：真空厚度（Å）= c - slab厚度（以分数坐标解环计）
      - tol_A：表面层厚度（Å）= 最外↔次外金属层间距的一半（夹在 0.6~2.5 Å）
    返回 (total_vacuum_A, tol_A, host_struct)
    """
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol, None
    try:
        s = Structure.from_file(ref_cif)
    except Exception:
        return None, default_tol, None

    c_len = s.lattice.c
    zu = unwrap_frac_along_c(s)
    extent_A = (zu.max() - zu.min()) * c_len
    vac = max(0.0, c_len - extent_A)

    metals = [i for i,site in enumerate(s.sites)
              if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    tol = default_tol
    if len(metals) >= 2:
        zm = np.sort(zu[metals])
        top = zm[-1]
        nxt = None
        for val in zm[-2::-1]:
            if abs(top - val) > 1e-3:
                nxt = val; break
        if nxt is not None:
            gapA = (top - nxt) * c_len
            tol = max(0.6, min(2.5, 0.5 * gapA))
    return vac, tol, s

def infer_dopants_and_sides_from_doped(doped_ref: Optional[str], host_struct: Optional[Structure]) -> Tuple[List[str], str]:
    """
    从 2-out.cif 推断掺杂元素与侧：
      - 掺杂元素：相对 host 的“非常见/少量”金属优先；若无法判断，用所有金属
      - 侧：统计掺杂金属沿 c 的分布，集中在顶/底则返回对应侧，否则 both
    """
    if (doped_ref is None) or (not os.path.exists(doped_ref)):
        return [], "both"
    try:
        ds = Structure.from_file(doped_ref)
    except Exception:
        return [], "both"

    metal_sites = [site.specie.symbol for site in ds.sites
                   if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    cnt = Counter(metal_sites)
    candidate = set()
    if host_struct is not None:
        host_metals = [site.specie.symbol for site in host_struct.sites
                       if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
        host_top = {m for m,_ in Counter(host_metals).most_common(3)}
    else:
        host_top = set()
    if cnt:
        med = np.median(list(cnt.values()))
        for m, c in cnt.items():
            if (m not in host_top and m in ALL_METALS) or (c <= 0.5*med):
                candidate.add(m)
    dopants = sorted(list(candidate)) if candidate else sorted([m for m in cnt if m in ALL_METALS])

    zu = unwrap_frac_along_c(ds)
    metals_idx = [i for i,site in enumerate(ds.sites)
                  if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals_idx:
        return dopants, "both"
    z_m = zu[metals_idx]
    sym_m = [ds[i].specie.symbol for i in metals_idx]
    dop_idx = [i for i,sy in zip(metals_idx, sym_m) if sy in dopants]
    if not dop_idx:
        return dopants, "both"
    zu_d = zu[dop_idx]
    zmin, zmax = float(np.min(z_m)), float(np.max(z_m))
    thr = (zmax - zmin) * 0.25
    top_n = np.sum((zmax - zu_d) <= thr)
    bot_n = np.sum((zu_d - zmin) <= thr)
    sides = "both"
    if top_n > 0 and bot_n == 0:
        sides = "top"
    elif bot_n > 0 and top_n == 0:
        sides = "bottom"
    return dopants, sides

# -------------------- 宿体来源：本地 or MP二元氧化物 --------------------
def iter_local_hosts(pattern: str) -> Iterable[Tuple[str, Structure]]:
    for path in sorted(glob.glob(pattern)):
        try:
            s = Structure.from_file(path)
            yield (os.path.splitext(os.path.basename(path))[0], s)
        except Exception:
            continue

def iter_mp_hosts(metals: List[str], mp_max_per_metal: int = 1) -> Iterable[Tuple[str, Structure]]:
    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("使用 Materials Project 需先 export MAPI_KEY=你的密钥")
    with MPRester(api_key) as mpr:
        for m in metals:
            docs = list(mpr.summary.search(
                elements=[m,"O"], nelements=2,
                fields=["material_id","energy_above_hull","structure","formula_pretty"],
                chunk_size=200
            ))
            if not docs: continue
            docs.sort(key=lambda d: (getattr(d,"energy_above_hull",1e9) or 1e9))
            for ddoc in docs[:mp_max_per_metal]:
                sid = f"{ddoc.material_id}_{getattr(ddoc,'formula_pretty','')}"
                yield (sid, ddoc.structure)

# -------------------- 扩胞策略（XY-only） --------------------
def best_2d_supercell_for_target_atoms(struct: Structure, target_after_remove_O: int, max_mul: int = 12) -> Tuple[int,int]:
    N0 = len(struct)
    best = (1,1); best_err = 10**9
    for ax in range(1, max_mul+1):
        for ay in range(1, max_mul+1):
            N_after = N0 * ax * ay - 1  # 删除 1 个 O 后
            err = abs(N_after - target_after_remove_O)
            if err < best_err:
                best, best_err = (ax, ay), err
    return best

def xy_supercell_for_target_O(struct: Structure, target_O: int, max_mul: int = 24) -> Optional[Tuple[int,int]]:
    """
    仅 x/y 扩胞使 O 原子数达到 target_O（若可整除）。返回 (ax,ay) 或 None
    """
    # 允许结构的 primitive/ conventional 作为候选起点
    bases = [struct]
    try:
        bases.append(struct.get_primitive_structure())
    except Exception:
        pass
    try:
        from pymatgen.transformations.standard_transformations import ConventionalCellTransformation
        bases.append(ConventionalCellTransformation().apply_transformation(struct))
    except Exception:
        pass

    for base in bases:
        count_O = sum(1 for sp in base.species if getattr(sp,"symbol",str(sp))=="O")
        if count_O == 0: continue
        if target_O % count_O != 0: 
            continue
        factor = target_O // count_O
        # 枚举 ax, ay（az=1），满足 ax*ay = factor
        for ax in range(1, max_mul+1):
            if factor % ax: continue
            ay = factor // ax
            if ay <= max_mul:
                return ax, ay
    return None

def make_xy_supercell(struct: Structure, ax:int, ay:int) -> Structure:
    s = struct.copy()
    s.make_supercell([ax, ay, 1])
    return s

# -------------------- 删 O、加真空、表面识别 --------------------
def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float, int]:
    s = struct.copy()
    fz = np.array([site.frac_coords[2] for site in s])
    zu = _unwrap_frac_z(fz)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom in structure")
    top = max(o_idx, key=lambda i: zu[i])
    top_z = float(zu[top])
    s.remove_sites([top])
    return s, top, top_z, len(s)

def add_symmetric_vacuum_along_c(struct: Structure, total_vac_A: float) -> Structure:
    if total_vac_A is None or total_vac_A <= 1e-6:
        return struct.copy()
    s = struct.copy()
    lat = s.lattice
    c_len_old = lat.c
    zu = unwrap_frac_along_c(s)
    span = zu.max() - zu.min()
    slab_thick_A = span * c_len_old
    new_c = slab_thick_A + total_vac_A
    scale = new_c / c_len_old
    new_cvec = lat.matrix[2] * scale
    new_lat = Lattice([lat.matrix[0], lat.matrix[1], new_cvec])
    center = 0.5*(zu.max()+zu.min())
    f = np.array([site.frac_coords for site in s])
    f[:,2] = f[:,2] - center + 0.5
    f[:,2] = f[:,2] - np.floor(f[:,2])
    return Structure(new_lat, [site.specie for site in s], f, coords_are_cartesian=False, to_unit_cell=True)

def get_surface_metal_indices_both_by_frac(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites)
              if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return [], []
    zmax = float(np.max(zu[metals])); zmin = float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

# -------------------- 掺杂（全局 1:3，避免空替换，可限制宿主子晶格） --------------------
def choose_sites_global_ratio(s: Structure, pool: List[int], dopant: str, d:int, h:int, rng: np.random.RandomState):
    dopable = [i for i in pool if s[i].specie.symbol != dopant]
    S = len(dopable)
    if S == 0:
        return [], 0, 0, 0
    target = int(round(S * d / (d + h)))  # e.g., 1:3 -> 25%
    k = max(0, min(S, target))
    picks = sorted(rng.choice(dopable, size=k, replace=False).tolist()) if k>0 else []
    return picks, S, target, k

def substitute_species(struct: Structure, indices: List[int], dopant: str):
    s2 = struct.copy()
    applied = []
    replaced_from = []
    for i in indices:
        if s2[i].specie.symbol != dopant:
            replaced_from.append(s2[i].specie.symbol)
            s2[i] = Element(dopant)
            applied.append(i)
    return s2, applied, replaced_from

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    # 模板
    ap.add_argument("--host_ref", type=str, default="0-out.cif", help="宿主几何模板（推断真空与 tol）")
    ap.add_argument("--doped_ref", type=str, default="2-out.cif", help="掺杂风格模板（推断掺杂元素/侧）")

    # 宿体来源
    ap.add_argument("--host_cifs", type=str, default="", help="本地 CIF 模式：通配，如 '/path/*.cif'")
    ap.add_argument("--host_metals", type=str, default="", help="MP 模式：逗号分隔金属，如 'Fe,Co,Ni'（抓二元氧化物）")
    ap.add_argument("--mp_max_per_metal", type=int, default=1)

    # 几何与模式
    ap.add_argument("--target_atoms", type=int, default=95, help="删除顶层 1 个 O 后的目标总原子数（默认 95）")
    ap.add_argument("--target_O", type=int, default=None, help="若提供，如 64，则仅 x/y 扩胞使 O 达到此数（随后会删顶层 1 个 O）")
    ap.add_argument("--surface_tol", type=float, default=None, help="表面层厚度阈值（Å）；留空则从 host_ref 推断")
    ap.add_argument("--vacuum", type=float, default=None, help="手动真空厚度（Å）；留空则继承 host_ref")

    # 掺杂
    ap.add_argument("--dopants", type=str, default="", help="手动掺杂列表，如 'Fe,Co'；留空则从 doped_ref 推断")
    ap.add_argument("--sides", type=str, default="auto", choices=["auto","top","bottom","both"], help="掺杂侧（默认 auto）")
    ap.add_argument("--host_species", type=str, default="", help="限制仅替这些宿主金属位点，如 'Ce,Zr'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="掺杂:宿主 比例（默认 1:3）")

    # 其它
    ap.add_argument("--outdir", type=str, default="out_v7")
    ap.add_argument("--max_hosts", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # 模板推断
    ref_vac, tol_infer, host_struct0 = infer_vac_and_tol_from_host(args.host_ref)
    tol = args.surface_tol if args.surface_tol is not None else tol_infer
    if args.vacuum is not None:
        ref_vac = args.vacuum

    user_dopants = [x.strip() for x in args.dopants.split(",") if x.strip()]
    if user_dopants:
        dopant_list = [d for d in user_dopants if d in ALL_METALS]
        sides_auto = "both"
    else:
        dopant_list, sides_auto = infer_dopants_and_sides_from_doped(args.doped_ref, host_struct0)
        if not dopant_list:
            dopant_list = ["Fe","Co","Ni"]  # 兜底
    sides = args.sides if args.sides != "auto" else (sides_auto or "both")

    host_species = set([x.strip() for x in args.host_species.split(",") if x.strip()]) if args.host_species else None

    try:
        d, h = [int(x) for x in args.dopant_to_host.split(":")]
        assert d>0 and h>0
    except Exception:
        raise ValueError("--dopant_to_host 必须是 'd:h' 的正整数，例如 1:3")

    print(f"[Info] vacuum={ref_vac if ref_vac is not None else 'None'} Å ; tol={tol:.3f} Å ; sides={sides} ; dopants={dopant_list}")

    # 宿体集合
    hosts: List[Tuple[str, Structure]] = []
    if args.host_cifs:
        hosts += list(iter_local_hosts(args.host_cifs))
    if args.host_metals:
        metals = [x.strip() for x in args.host_metals.split(",") if x.strip()]
        hosts += list(iter_mp_hosts(metals, args.mp_max_per_metal))
    if not hosts:
        raise RuntimeError("未提供宿体：请用 --host_cifs 或 --host_metals 至少一种来源。")
    if len(hosts) > args.max_hosts:
        hosts = hosts[:args.max_hosts]

    rows = []

    for host_id, struct in hosts:
        if not is_metal_oxide(struct):
            rows.append({"host": host_id, "status": "skip_non_oxide"})
            continue

        # ---------- XY-only supercell ----------
        if args.target_O:  # O 计数模式（更接近你喜欢的 v3 输出风格）
            xy = xy_supercell_for_target_O(struct, args.target_O)
            if not xy:
                rows.append({"host": host_id, "status": "skip_O_not_divisible", "target_O": args.target_O})
                continue
            ax, ay = xy
        else:             # 总原子数目标模式（默认）
            ax, ay = best_2d_supercell_for_target_atoms(struct, args.target_atoms)

        sup = make_xy_supercell(struct, ax, ay)

        # ---------- 删除顶层 1 个 O ----------
        try:
            sup2, del_idx, del_fz, N_after = remove_topmost_oxygen_by_frac(sup)
        except Exception as e:
            rows.append({"host": host_id, "status": "skip_no_O", "reason": str(e)})
            continue

        # ---------- 沿 c 加真空并居中 ----------
        sup3 = add_symmetric_vacuum_along_c(sup2, total_vac_A=ref_vac) if (ref_vac and ref_vac>1e-6) else sup2

        # ---------- 表面识别（上下两侧） ----------
        top_idx, bot_idx = get_surface_metal_indices_both_by_frac(sup3, tol_A=tol)
        if not top_idx and not bot_idx:
            rows.append({"host": host_id, "status": "skip_no_surface_metals"})
            continue

        # ---------- 侧向池 & 子晶格过滤 ----------
        pool = []
        if sides in ("both","top"):
            pool += top_idx
        if sides in ("both","bottom"):
            pool += bot_idx
        if host_species:
            pool = [i for i in pool if sup3[i].specie.symbol in host_species]

        if not pool:
            rows.append({"host": host_id, "status": "skip_no_dopable_pool"})
            continue

        # ---------- 按掺杂列表生成 ----------
        for dop in dopant_list:
            picks, S, target, k = choose_sites_global_ratio(sup3, pool, dop, d, h, rng)
            doped, applied, replaced_from = substitute_species(sup3, picks, dop)

            # 回填（避免因 no-op 导致低于目标）
            if len(applied) < target:
                remain = [i for i in pool if (i not in applied) and (doped[i].specie.symbol != dop)]
                need = target - len(applied)
                if need > 0 and len(remain) > 0:
                    add_more = sorted(rng.choice(remain, size=min(need,len(remain)), replace=False).tolist())
                    doped, applied2, replaced2 = substitute_species(doped, add_more, dop)
                    applied += applied2
                    replaced_from += replaced2

            # 输出
            mode_tag = f"O{args.target_O}" if args.target_O else f"N{N_after}"
            outname = f"{host_id}_2D_{ax}x{ay}_{mode_tag}_bisurf_{sides}_{dop}_d{d}_h{h}.cif"
            outpath = os.path.join(args.outdir, outname)
            doped.to(fmt="cif", filename=outpath)

            rows.append({
                "host": host_id, "status": "ok", "outfile": outpath,
                "dopant": dop, "sides": sides, "host_species_filter": ",".join(sorted(host_species)) if host_species else "",
                "supercell_2d": f"{ax}x{ay}x1",
                "deleted_O_index": del_idx, "deleted_O_fz_unwrap": del_fz,
                "final_total_atoms": N_after,
                "surface_tol_A": tol,
                "top_surface_sites": len(top_idx), "bot_surface_sites": len(bot_idx),
                "dopable_sites": S, "target_replacements": target, "picked_initial": k,
                "applied_total": len(applied), "replaced_from": ";".join(map(str, replaced_from)),
                "applied_indices": ";".join(map(str, applied))
            })

    # ---------- manifest ----------
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] Wrote {sum(r.get('status')=='ok' for r in rows)} doped slabs; total records {len(rows)}.")
    else:
        print("[WARN] No outputs; check your inputs/filters.]")

if __name__ == "__main__":
    main()
