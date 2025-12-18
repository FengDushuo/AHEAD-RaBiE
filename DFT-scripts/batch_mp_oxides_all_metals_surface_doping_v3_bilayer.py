#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 Materials Project 的金属氧化物，
扩胞到 O=64 → 删除沿 c 顶层 1 个 O → 留真空层 → 仅保留“金属层数 = 2”的双层体系 →
分别在上下两层金属位点做 **各自** 1:3 替位掺杂（严格 per-side，而非合并）。

注意：
- 为了稳健地识别“上下两层金属”，所有层厚与高度判断均基于 **沿 c 的分数坐标 + 解环 (unwrap)**。
- 若某个材料在构建后“金属层数 ≠ 2”，则跳过不输出（严格限定两层）。
"""

import os, re, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

# 全部金属元素
ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# -------------------- 几何工具（沿 c 分数坐标 + 解环） --------------------
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    z_sorted = np.sort(z)
    if len(z_sorted) == 0:
        return z
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))  # 最大缝隙右端
    base = z_sorted[(k + 1) % len(z_sorted)]
    zu = z - base
    zu[zu < 0] += 1.0
    return zu

def unwrap_frac_along_c(struct: Structure) -> np.ndarray:
    fz = np.array([site.frac_coords[2] for site in struct.sites])
    return _unwrap_frac_z(fz)

# -------------------- 判定 & 预处理 --------------------
def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def decompose_to_abc(factor: int) -> Tuple[int,int,int]:
    """把超胞因子分解成接近立方的 (a,b,c)。"""
    best = (1,1,factor); score = 1e9; root = round(factor ** (1/3))
    for a in range(1, factor+1):
        if factor % a: continue
        rem = factor // a
        for b in range(1, rem+1):
            if rem % b: continue
            c = rem // b
            s = abs(a-root) + abs(b-root) + abs(c-root)
            if s < score: best, score = (a,b,c), s
    return best

def find_supercell_for_O(struct: Structure, target_O: int = 64):
    """扩胞直到 O=64（保持与 v3 风格兼容）"""
    candidates = [struct]
    try:
        candidates.append(ConventionalCellTransformation().apply_transformation(struct))
    except Exception:
        pass
    try:
        candidates.append(struct.get_primitive_structure())
    except Exception:
        pass
    for base in candidates:
        nO = sum(1 for sp in base.species if getattr(sp, "symbol", str(sp)) == "O")
        if nO == 0 or (target_O % nO != 0):
            continue
        factor = target_O // nO
        ax,ay,az = decompose_to_abc(factor)
        sup = base.copy()
        sup.make_supercell([ax,ay,az])
        nO2 = sum(1 for sp in sup.species if sp.symbol == "O")
        if nO2 == target_O:
            return sup, (ax,ay,az)
    return None

def remove_topmost_oxygen_by_frac(struct: Structure):
    """沿 c（分数坐标解环）删除最顶层的 1 个 O"""
    s = struct.copy()
    fz = np.array([site.frac_coords[2] for site in s])
    zu = _unwrap_frac_z(fz)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: zu[i])
    top_z = float(zu[top])
    s.remove_sites([top])
    new_O = sum(1 for sp in s.species if sp.symbol == "O")
    return s, top, top_z, new_O

def parse_ref_vacuum_and_tol(ref_cif: Optional[str]):
    """从参考 CIF 粗略推断真空与表面 tol（保留 v3 接口，但更稳健）"""
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol
    try:
        s = Structure.from_file(ref_cif)
        c_len = s.lattice.c
        zu = unwrap_frac_along_c(s)
        extent_A = (zu.max() - zu.min()) * c_len
        ref_vac = max(0.0, c_len - extent_A)
        # 从金属间距估个 tol
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
        # 兜底成老逻辑
        txt = open(ref_cif, "r", encoding="utf-8", errors="ignore").read()
        m = re.search(r"_cell_length_c\s+([0-9\.\(\)Ee+\-]+)", txt, re.IGNORECASE)
        ref_vac = float(re.sub(r"\([^)]*\)", "", m.group(1))) * 0.25 if m else None
        return ref_vac, default_tol

def add_symmetric_vacuum_along_c(struct: Structure, total_vacuum_A: float) -> Structure:
    """沿 c 加对称真空并居中（基于分数坐标解环）"""
    if total_vacuum_A is None or total_vacuum_A <= 1e-6:
        return struct.copy()
    s = struct.copy()
    lat = s.lattice
    c_len_old = lat.c
    zu = unwrap_frac_along_c(s)
    span = zu.max() - zu.min()
    slab_thick_A = span * c_len_old
    new_c = slab_thick_A + total_vacuum_A
    scale = new_c / c_len_old
    new_cvec = lat.matrix[2] * scale
    new_lat = Lattice([lat.matrix[0], lat.matrix[1], new_cvec])
    center = 0.5*(zu.max()+zu.min())
    f = np.array([site.frac_coords for site in s])
    f[:,2] = f[:,2] - center + 0.5
    f[:,2] = f[:,2] - np.floor(f[:,2])  # wrap into [0,1)
    return Structure(new_lat, [site.specie for site in s], f, coords_are_cartesian=False, to_unit_cell=True)

# -------------------- 两层判定与表面位点 --------------------
def get_surface_metal_indices_both_by_frac(struct: Structure, tol_A: float):
    """按 tol_A（Å）识别上下两侧表面金属位点（沿 c 的距离阈值）"""
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element)
              and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return [], [], zu
    zmax, zmin = float(np.max(zu[metals])), float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot, zu

def count_metal_layers_by_frac(struct: Structure, layer_merge_A: float = 1.0) -> int:
    """统计“金属层层数”（将沿 c 的金属 z 用 layer_merge_A 聚类合并）。"""
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [zu[i] for i,site in enumerate(s.sites) if isinstance(site.specie, Element)
              and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals: return 0
    metals = np.sort(np.array(metals))
    # 以 Å 为单位聚类
    thr_frac = layer_merge_A / lat.c
    layers = 1
    ref = metals[0]
    for v in metals[1:]:
        if abs(v - ref) * lat.c > layer_merge_A:
            layers += 1
            ref = v
    return layers

# -------------------- 掺杂（严格 per-side 1:3） --------------------
def choose_sites_per_side(indices: List[int], frac: float, rng) -> List[int]:
    if len(indices) == 0:
        return []
    k = max(1, int(round(len(indices)*frac)))
    k = min(k, len(indices))
    return sorted(rng.choice(indices, size=k, replace=False).tolist())

def substitute_species(struct: Structure, idx_list: List[int], dopant: str):
    s2 = struct.copy()
    for i in idx_list:
        if s2[i].specie.symbol != dopant:  # 避免空替换
            s2[i] = Element(dopant)
    return s2

# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="参考 CIF（用于推断真空与表面 tol）")
    ap.add_argument("--outdir", type=str, default="out_bisurf_bilayer")
    ap.add_argument("--dopants", type=str, default="ALL", help="ALL 或逗号列表，如 'Fe,Co,Ni'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="每一侧金属层的掺杂:宿主 比例（严格 per-side）")
    ap.add_argument("--layer_merge_A", type=float, default=1.0, help="金属层聚类阈值（Å），用于判定是否为“两层”")
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dopant_list = ALL_METALS if args.dopants.upper()=="ALL" else [x.strip() for x in args.dopants.split(",") if x.strip()]
    d,h = [int(x) for x in args.dopant_to_host.split(":")]
    frac = d/(d+h)

    ref_vac, tol = parse_ref_vacuum_and_tol(args.ref_cif)
    print(f"[Info] tol={tol:.3f} Å, ref_vac={ref_vac} Å, per-side ratio={d}:{h}")

    rng = np.random.RandomState(args.seed)

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")
    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        docs = docs[:args.max_docs]

        manifest = []

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue
            found = find_supercell_for_O(struct, target_O=64)
            if found is None: 
                continue
            sup, (ax,ay,az) = found

            # 删除沿 c 顶层 O
            try:
                sup2, del_idx, topO_zu, new_O = remove_topmost_oxygen_by_frac(sup)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue
            if new_O != 63:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_O_not_63"})
                continue

            # 加真空并居中
            sup3 = add_symmetric_vacuum_along_c(sup2, total_vacuum_A=ref_vac)

            # 严格“两层”判定（金属层数必须==2）
            n_layers = count_metal_layers_by_frac(sup3, layer_merge_A=args.layer_merge_A)
            if n_layers != 2:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_not_bilayer", "metal_layers": n_layers})
                continue

            # 取上下两层金属位点池（分数坐标驱动）
            top_idx, bot_idx, zu = get_surface_metal_indices_both_by_frac(sup3, tol_A=tol)
            if len(top_idx)==0 or len(bot_idx)==0:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_top_or_bottom"})
                continue

            for dop in dopant_list:
                # 严格 per-side 1:3
                picks_top = choose_sites_per_side(top_idx, frac, rng)
                picks_bot = choose_sites_per_side(bot_idx, frac, rng)
                # 避免空替换（如果某些位点本来就是 dop，则自动回填）
                picks_top = [i for i in picks_top if sup3[i].specie.symbol != dop]
                picks_bot = [i for i in picks_bot if sup3[i].specie.symbol != dop]

                def refill(pool, chosen):
                    need = max(0, int(round(len(pool)*frac)) - len(chosen))
                    if need <= 0: return chosen
                    remain = [i for i in pool if (i not in chosen) and (sup3[i].specie.symbol != dop)]
                    if need > 0 and len(remain) > 0:
                        add = sorted(rng.choice(remain, size=min(need, len(remain)), replace=False).tolist())
                        return sorted(chosen + add)
                    return chosen

                picks_top = refill(top_idx, picks_top)
                picks_bot = refill(bot_idx, picks_bot)

                doped = substitute_species(sup3, picks_top + picks_bot, dop)

                fname = f"{ddoc.material_id}_O63_bilayer_perSide13_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)
                manifest.append({
                    "material_id": ddoc.material_id,
                    "formula": getattr(ddoc, "formula_pretty", ""),
                    "status": "ok",
                    "outfile": fpath,
                    "supercell": f"{ax}x{ay}x{az}",
                    "deleted_O_index": del_idx,
                    "top_pool": len(top_idx), "bot_pool": len(bot_idx),
                    "picked_top": len(picks_top), "picked_bot": len(picks_bot),
                    "ratio_per_side": f"{d}:{h}",
                })
                print("Wrote", fname)

        if manifest:
            pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
            print(f"[OK] wrote {sum(m.get('status')=='ok' for m in manifest)} structures; total records {len(manifest)}")
        else:
            print("[WARN] no outputs; all skipped by filters.]")

if __name__ == "__main__":
    main()
