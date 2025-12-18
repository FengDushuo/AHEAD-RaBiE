#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 Materials Project 的金属氧化物；根据你的最新要求更新版：
- z 轴不扩胞原子，而是只在顶部留出真空层：c → c + 20 Å；
  原子笛卡尔坐标保持不变，然后在新晶格下换算出新的分数坐标，
  因此延长的 20 Å 区域没有原子。
- 初始用下载单胞，原子数不变。
- 仅在 a、b 方向扩胞：若 _cell_length_a(b) < 10 Å，则取最小整数 i 使 i*a0(b0) ≥ 10，
  做 [ia, ib, 1] 超胞；z 不扩。
- 表面识别用分数坐标解环；上下两侧金属各自 1:3 替位掺杂（严格 per-side）。
- 输出每个结构的 CIF 与 manifest.csv。

依赖：pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math
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

# ---------- 基础：分数 z 解环 ----------
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

# ---------- 真空：c → c+20；保持原子笛卡尔坐标不变，映射到新分数坐标 ----------
def add_vacuum_with_empty_top(struct: Structure, delta_vac_A: float = 20.0) -> Structure:
    """
    在 z 方向只加顶部真空：c -> c + delta。
    关键：在“分数坐标 + 解环”坐标系里，仅压缩 slab 的实际占据厚度，避免斜晶格 a/b-c 耦合。
    """
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    Z1 = Z0 + float(delta_vac_A)

    # 新晶格（只放大 c 向量长度）
    new_cvec = lat0.matrix[2] * (Z1 / Z0)
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], new_cvec])

    # 原子分数坐标（先取原始分数，再做解环）
    f = np.array([site.frac_coords for site in s])  # Nx3
    z = f[:, 2]
    # 解环，把 slab 厚度放到同一切片
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))
    base = z_sorted[(k + 1) % len(z_sorted)]
    zu = z - base
    zu[zu < 0] += 1.0

    # slab 在旧晶格中的实际分数厚度
    z_min = float(np.min(zu))
    z_max = float(np.max(zu))
    span_frac = z_max - z_min
    # 旧→新厚度的分数缩放系数（只压缩占据区，不动 a/b）
    scale_frac = Z0 / Z1  # = Z0/(Z0+delta)

    # 将占据区平移到底部并压缩；顶部留出 delta 的净空
    new_z = (zu - z_min) * scale_frac  # 放到底部：起点 0
    # a,b 分数坐标保持不变（绕过笛卡尔坐标，避免耦合）
    new_f = np.copy(f)
    new_f[:, 2] = new_z

    # 保证落在 [0,1)
    new_f[:, 2] = new_f[:, 2] - np.floor(new_f[:, 2])

    return Structure(new_lat, [site.specie for site in s],
                     new_f, coords_are_cartesian=False, to_unit_cell=True)


# ---------- XY 只扩胞到 a,b≥10 Å ----------
def minimal_xy_supercell(struct: Structure, a_min: float = 10.0, b_min: float = 10.0) -> Tuple[int,int]:
    lat = struct.lattice
    a0, b0 = lat.a, lat.b
    ia = max(1, math.ceil(a_min / a0))
    ib = max(1, math.ceil(b_min / b0))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy()
    s.make_supercell([ia, ib, 1])  # z 不扩
    return s

# ---------- 表面识别与 per-side 1:3 掺杂 ----------
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

def choose_per_side(pool: List[int], frac: float, rng: np.random.RandomState) -> List[int]:
    if len(pool)==0: return []
    k = max(1, int(round(len(pool)*frac)))
    k = min(k, len(pool))
    return sorted(rng.choice(pool, size=k, replace=False).tolist())

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for i in idx:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ---------- 主程序 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None,
                    help="用于推断表面 tol（Å），不从中拷坐标")
    ap.add_argument("--outdir", type=str, default="out_vacTop_xyMin10")
    ap.add_argument("--dopants", type=str, default="ALL",
                    help="ALL 或逗号分隔，如 'Fe,Co'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3",
                    help="每一侧金属层 掺杂:宿主 比例（严格 per-side）")
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--vac_add", type=float, default=20.0,
                    help="在 c 顶部增加的真空厚度（Å）")
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    dopant_list = ALL_METALS if args.dopants.upper()=="ALL" \
                  else [x.strip() for x in args.dopants.split(",") if x.strip()]
    d,h = [int(x) for x in args.dopant_to_host.split(":")]
    assert d>0 and h>0, "--dopant_to_host 必须是正整数比"
    frac = d/(d+h)

    tol = infer_tol_from_ref(args.ref_cif)

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")

    manifest = []

    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"],
                   fields=["material_id","formula_pretty","structure"],
                   chunk_size=200))
        docs = docs[:args.max_docs]

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue

            # 1) 只在顶部加真空（c→c+vac_add），原子笛卡尔坐标不变，映射到新分数坐标
            s_vac = add_vacuum_with_empty_top(struct, delta_vac_A=args.vac_add)

            # 2) XY 最小扩胞
            ia, ib = minimal_xy_supercell(s_vac, a_min=args.a_min, b_min=args.b_min)
            s_xy = make_supercell_xy(s_vac, ia, ib)

            # 3) 表面位点（金属）
            top_idx, bot_idx = surface_metal_indices(s_xy, tol_A=tol)
            if len(top_idx)==0 or len(bot_idx)==0:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_surface"})
                continue

            for dop in dopant_list:
                picks_top = choose_per_side(
                    [i for i in top_idx if s_xy[i].specie.symbol != dop],
                    frac, rng)
                picks_bot = choose_per_side(
                    [i for i in bot_idx if s_xy[i].specie.symbol != dop],
                    frac, rng)

                doped = substitute(s_xy, picks_top + picks_bot, dop)

                fname = f"{ddoc.material_id}_vacTop{int(args.vac_add)}_a{ia}x_b{ib}x_perSide13_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                manifest.append({
                    "material_id": ddoc.material_id,
                    "formula": getattr(ddoc, "formula_pretty",""),
                    "status": "ok",
                    "outfile": fpath,
                    "ia": ia, "ib": ib,
                    "vac_add": args.vac_add,
                    "top_pool": len(top_idx), "bot_pool": len(bot_idx),
                    "picked_top": len(picks_top), "picked_bot": len(picks_bot),
                    "ratio_per_side": f"{d}:{h}", "dopant": dop
                })
                print("Wrote", fname)

    if manifest:
        pd.DataFrame(manifest).to_csv(
            os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] wrote {sum(m.get('status')=='ok' for m in manifest)} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs; all skipped")

if __name__ == "__main__":
    main()
