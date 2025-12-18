#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量构建“金属氧化物表面掺杂”体系 —— v3_xy10to20
要求：
- **不再限制 O=64**；仅在 **x/y 方向扩胞**，选择 (ax, ay) 使：
  * ax*ay 介于 **[10, 20]**（若无法同时满足，自动放宽到 [5, 30] 兜底）
  * 删除顶层 1 个 O 后，**总原子数 ≈ 90**（可用 --target_atoms 改，默认 90）
- 删除沿 c（分数坐标解环）的**最顶层 1 个 O**
- 沿 c **对称加真空**并使 slab **居中**（默认从 --ref_cif 推断；也可 --vacuum 指定）
- **严格两层金属（bilayer）**：若金属层数 != 2 则**跳过**
- **上下两侧分别 1:3 掺杂**（严格 per-side，不合并），仅替换**金属位点**（非 O 位点）
- 输出：每个体系一个 CIF + manifest.csv

运行示例：
  export MAPI_KEY=你的MP密钥
  python batch_mp_oxides_all_metals_surface_doping_v3_xy10to20.py \
    --ref_cif 1-out.cif \
    --outdir out_xy10to20 \
    --dopants ALL \
    --dopant_to_host 1:3 \
    --max_docs 100 \
    --seed 123

依赖：pymatgen>=2023.0.0, mp-api, numpy, pandas
"""

import os, re, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice

ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# ---------- 工具（沿 c 分数坐标 + 解环） ----------
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    z_sorted = np.sort(z)
    if len(z_sorted) == 0:
        return z
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

# ---------- 参考 CIF 推断真空与 tol ----------
def infer_vacuum_and_tol_from_ref(ref_cif: Optional[str]) -> Tuple[Optional[float], float]:
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol
    try:
        s = Structure.from_file(ref_cif)
        c_len = s.lattice.c
        zu = unwrap_frac_along_c(s)
        extent_A = (zu.max() - zu.min()) * c_len
        ref_vac = max(0.0, c_len - extent_A)  # 参考结构的真空厚度
        # 从金属层间距估一个 tol
        metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element)
                  and site.specie.is_metal and site.specie.symbol != "O"]
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
        return ref_vac, tol
    except Exception:
        return None, default_tol

# ---------- XY-only 超胞选择 ----------
def best_xy_supercell_for_target(struct: Structure, target_atoms_after_remove_O:int=90,
                                 min_area_mul:int=10, max_area_mul:int=20,
                                 fallback_min:int=5, fallback_max:int=30) -> Tuple[int,int]:
    """
    仅在 x/y 方向扩胞（az=1）。优先在 ax*ay ∈ [min_area_mul, max_area_mul] 中
    寻找使 (len(struct)*ax*ay - 1) 接近 target 的 (ax,ay)；若无解，放宽到 [fallback_min, fallback_max]。
    """
    N0 = len(struct)
    best = (1,1); best_err = 10**9; best_area = None
    def search(lo, hi):
        nonlocal best, best_err, best_area
        for ax in range(1, hi+1):
            for ay in range(1, hi+1):
                area = ax*ay
                if area < lo or area > hi: continue
                N_after = N0*area - 1
                err = abs(N_after - target_atoms_after_remove_O)
                if err < best_err or (err==best_err and (best_area is None or area<best_area)):
                    best, best_err, best_area = (ax,ay), err, area
    search(min_area_mul, max_area_mul)
    if best_area is None:
        search(fallback_min, fallback_max)
    return best

def make_xy_supercell(struct: Structure, ax:int, ay:int) -> Structure:
    s = struct.copy()
    s.make_supercell([ax, ay, 1])
    return s

# ---------- 删 O、加真空、两层判定、表面位点 ----------
def remove_topmost_O_by_frac(struct: Structure):
    s = struct.copy()
    zu = unwrap_frac_along_c(s)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: zu[i])
    s.remove_sites([top])
    return s, top

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

def count_metal_layers_by_frac(struct: Structure, layer_merge_A: float = 1.0) -> int:
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [zu[i] for i,site in enumerate(s.sites) if isinstance(site.specie, Element)
              and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return 0
    metals = np.sort(np.array(metals))
    layers = 1
    ref = metals[0]
    for v in metals[1:]:
        if abs(v - ref) * lat.c > layer_merge_A:
            layers += 1
            ref = v
    return layers

def get_surface_metal_indices_both_by_frac(struct: Structure, tol_A: float):
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element)
              and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return [], []
    zmax, zmin = float(np.max(zu[metals])), float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

# ---------- 掺杂：严格 per-side 1:3 ----------
def choose_per_side(pool: List[int], d:int, h:int, rng) -> List[int]:
    if len(pool) == 0: return []
    frac = d/(d+h)
    k = int(round(len(pool)*frac))
    k = max(1, min(k, len(pool)))
    return sorted(rng.choice(pool, size=k, replace=False).tolist())

def substitute_metal_sites(struct: Structure, idx_list: List[int], dopant: str):
    s2 = struct.copy()
    for i in idx_list:
        if s2[i].specie.symbol != "O" and s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="参考 CIF（推断真空与表面 tol）")
    ap.add_argument("--outdir", type=str, default="out_xy10to20")
    ap.add_argument("--dopants", type=str, default="ALL", help="ALL 或逗号列表，如 'Fe,Co'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="每侧的掺杂:宿主（严格 per-side）")
    ap.add_argument("--target_atoms", type=int, default=90, help="删顶层 O 后的目标总原子数（默认 90）")
    ap.add_argument("--area_min", type=int, default=10, help="要求 ax*ay 的最小值（默认 10）")
    ap.add_argument("--area_max", type=int, default=20, help="要求 ax*ay 的最大值（默认 20）")
    ap.add_argument("--fallback_min", type=int, default=5, help="兜底 ax*ay 最小值（默认 5）")
    ap.add_argument("--fallback_max", type=int, default=30, help="兜底 ax*ay 最大值（默认 30）")
    ap.add_argument("--layer_merge_A", type=float, default=1.0, help="金属层聚类阈值（Å），用于“两层”判定")
    ap.add_argument("--vacuum", type=float, default=None, help="手动真空厚度（Å），覆盖参考推断")
    ap.add_argument("--max_docs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dopant_list = ALL_METALS if args.dopants.upper()=="ALL" else [x.strip() for x in args.dopants.split(",") if x.strip()]
    try:
        d,h = [int(x) for x in args.dopant_to_host.split(":")]
        assert d>0 and h>0
    except Exception:
        raise ValueError("--dopant_to_host 必须是 'd:h' 的正整数，例如 1:3")

    ref_vac, tol = infer_vacuum_and_tol_from_ref(args.ref_cif)
    if args.vacuum is not None:
        ref_vac = args.vacuum
    print(f"[Info] tol={tol:.3f} Å, ref_vac={ref_vac} Å, per-side={d}:{h}, target_atoms≈{args.target_atoms}, ax*ay∈[{args.area_min},{args.area_max}]")

    rng = np.random.RandomState(args.seed)

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        docs = docs[:args.max_docs]

        rows = []

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue

            # 仅 x/y 扩胞：优先 ax*ay∈[10,20] 且总原子数≈90
            ax, ay = best_xy_supercell_for_target(struct, args.target_atoms, args.area_min, args.area_max, args.fallback_min, args.fallback_max)
            sup = make_xy_supercell(struct, ax, ay)

            # 删顶层 O（分数坐标解环）
            try:
                sup2, del_o_idx = remove_topmost_O_by_frac(sup)
            except Exception as e:
                rows.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue

            # 加真空并居中
            sup3 = add_symmetric_vacuum_along_c(sup2, total_vac_A=ref_vac) if (ref_vac and ref_vac>1e-6) else sup2

            # 两层判定
            n_layers = count_metal_layers_by_frac(sup3, layer_merge_A=args.layer_merge_A)
            if n_layers != 2:
                rows.append({"material_id": ddoc.material_id, "status": "skip_not_bilayer", "metal_layers": n_layers})
                continue

            # 表面位点（上下各自池）
            top_idx, bot_idx = get_surface_metal_indices_both_by_frac(sup3, tol_A=tol)
            if len(top_idx)==0 or len(bot_idx)==0:
                rows.append({"material_id": ddoc.material_id, "status": "skip_no_top_or_bottom"})
                continue

            for dop in dopant_list:
                # 分别在两侧 1:3（严格 per-side；只替金属）
                def filter_metal(pool):
                    return [i for i in pool if isinstance(sup3[i].specie, Element) and sup3[i].specie.is_metal and sup3[i].specie.symbol != "O"]
                top_metal = filter_metal(top_idx)
                bot_metal = filter_metal(bot_idx)

                def pick_and_refill(pool):
                    # 先按比例挑；去掉本就=掺杂元素的位点；不足则回填
                    target_k = max(1, int(round(len(pool)*d/(d+h))))
                    chosen = choose_per_side(pool, d, h, rng)
                    chosen = [i for i in chosen if sup3[i].specie.symbol != dop]
                    if len(chosen) < target_k:
                        remain = [i for i in pool if (i not in chosen) and (sup3[i].specie.symbol != dop)]
                        need = target_k - len(chosen)
                        if need>0 and len(remain)>0:
                            add = sorted(rng.choice(remain, size=min(need,len(remain)), replace=False).tolist())
                            chosen = sorted(chosen + add)
                    return chosen, target_k

                picks_top, tgt_top = pick_and_refill(top_metal)
                picks_bot, tgt_bot = pick_and_refill(bot_metal)

                doped = substitute_metal_sites(sup3, picks_top + picks_bot, dop)

                # 输出
                N_after = len(doped)
                fname = f"{ddoc.material_id}_2D_{ax}x{ay}_N{N_after}_bilayer_perSide13_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "ok", "outfile": fpath, "dopant": dop,
                    "supercell_2d": f"{ax}x{ay}x1", "deleted_O_index": del_o_idx,
                    "target_atoms": args.target_atoms, "final_total_atoms": N_after,
                    "area_mul": ax*ay, "tol_A": tol, "vacuum_A": ref_vac if ref_vac else 0.0,
                    "top_pool": len(top_idx), "bot_pool": len(bot_idx),
                    "picked_top": len(picks_top), "picked_bot": len(picks_bot),
                    "target_top": tgt_top, "target_bot": tgt_bot
                })
                print("Wrote", fname)

        if rows:
            pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
            print(f"[OK] wrote {sum(r.get('status')=='ok' for r in rows)} structures; total records {len(rows)}.")
        else:
            print("[WARN] no outputs; all skipped by filters.]")

if __name__ == "__main__":
    main()
