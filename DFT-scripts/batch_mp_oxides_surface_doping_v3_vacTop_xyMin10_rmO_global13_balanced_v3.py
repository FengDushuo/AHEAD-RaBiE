#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
要求：
- 氧化物中的“所有金属元素”（宿主金属，排除 O）必须来自 ALLOWED_METALS；
- 掺杂元素也只从 ALLOWED_METALS 里选择；
- 不对称真空（vac_top, vac_bot），只在 a,b 扩胞到 >= 指定阈值；
- 方向规整：保证 O 在上、金属在下（Otop）；
- 删除顶层分数 z 最大的 1 个氧（rmTopO，制造氧空位）；
- 表面金属位点做全局 1:3（25%）掺杂；顶/底两侧掺杂数严格相同；
- 每个表面的掺杂位点在 XY 平面上用最远点采样(FPS)均匀分布；
- 输出 .cif + manifest.csv

依赖：pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math
from typing import List, Tuple, Optional, Set
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice

# —— 允许的金属元素（宿主与掺杂统一白名单）——
ALLOWED_METALS: Set[str] = set([
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
])

# ---------- 小工具 ----------
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
    return _unwrap_frac_z(np.array([s.frac_coords[2] for s in struct.sites]))

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal and e.symbol != "O" for e in struct.composition.elements)

def metal_species_in_struct(struct: Structure) -> Set[str]:
    mets = set()
    for sp in struct.composition.elements:
        if sp.symbol != "O" and sp.is_metal:
            mets.add(sp.symbol)
    return mets

# ---------- 方向规整：O 在上、金属在下 ----------
def orient_oxygen_to_top(struct: Structure, top_frac=0.10) -> Structure:
    s = struct.copy()
    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:, 2])
    N = len(s)
    cut = max(1, int(np.ceil(N * top_frac)))
    idx = np.argsort(zu)
    top_ids = idx[-cut:]
    bot_ids = idx[:cut]
    nO_top = sum(1 for i in top_ids if s[i].specie.symbol == "O")
    nO_bot = sum(1 for i in bot_ids if s[i].specie.symbol == "O")
    if nO_top < nO_bot:
        f_new = np.copy(f)
        f_new[:, 2] = (1.0 - zu) % 1.0
        return Structure(s.lattice, [site.specie for site in s], f_new,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

# ---------- 不对称真空：顶部 vac_top、底部 vac_bot ----------
def add_vacuum_asymmetric(struct: Structure, vac_top_A: float, vac_bot_A: float) -> Structure:
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:, 2])
    zmin, zmax = float(np.min(zu)), float(np.max(zu))
    span_frac = max(1e-12, zmax - zmin)
    slab_A = span_frac * Z0
    new_c = slab_A + float(vac_top_A) + float(vac_bot_A)
    scale = new_c / Z0
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2] * scale])
    new_span = slab_A / new_c
    new_base = vac_bot_A / new_c
    new_f = np.copy(f)
    new_f[:, 2] = (new_base + (zu - zmin) * (new_span / span_frac)) % 1.0
    return Structure(new_lat, [site.specie for site in s], new_f,
                     coords_are_cartesian=False, to_unit_cell=True)

# ---------- XY 最小扩胞到 a,b ≥ 阈值 ----------
def minimal_xy_supercell(struct: Structure, a_min: float = 10.0, b_min: float = 10.0) -> Tuple[int,int]:
    lat = struct.lattice
    ia = max(1, math.ceil(a_min / lat.a))
    ib = max(1, math.ceil(b_min / lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy()
    s.make_supercell([ia, ib, 1])
    return s

# ---------- 删除顶层 1 个 O（rmTopO） ----------
def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float]:
    s = struct.copy()
    zu = _unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom found")
    top = max(o_idx, key=lambda i: zu[i])
    zval = float(zu[top])
    s.remove_sites([top])
    return s, top, zval

# ---------- 表面识别（找表面的金属位点） ----------
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
    default = 1.2
    if not ref_cif or not os.path.exists(ref_cif):
        return default
    try:
        s = Structure.from_file(ref_cif)
        lat = s.lattice
        zu = unwrap_frac_along_c(s)
        metals = [i for i,site in enumerate(s.sites)
                  if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
        zm = np.sort(zu[metals])
        if len(zm) < 2: return default
        gapA = (zm[-1] - zm[-2]) * lat.c
        return max(0.6, min(2.5, 0.5*gapA))
    except Exception:
        return default

# ---------- 最远点采样(FPS)：均匀选表面位点 ----------
def farthest_point_sampling_xy(struct: Structure, candidates: List[int], k: int) -> List[int]:
    if k <= 0 or not candidates:
        return []
    k = min(k, len(candidates))
    xy = np.array([struct[i].coords[:2] for i in candidates])
    center = np.mean(xy, axis=0)
    d2 = np.sum((xy - center)**2, axis=1)
    seed = int(np.argmax(d2))
    chosen = [seed]
    dist = np.linalg.norm(xy - xy[seed], axis=1)
    while len(chosen) < k:
        nxt = int(np.argmax(dist))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy - xy[nxt], axis=1))
    return sorted(candidates[i] for i in chosen)

# ---------- 1:3（25%）且顶/底严格相等；位点用 FPS 均匀化 ----------
def pick_hosts_global_balanced_uniform(top_idx: List[int], bot_idx: List[int],
                                       struct: Structure, dopant: str, frac: float):
    def host_pool(idxs):
        return [i for i in idxs
                if (isinstance(struct[i].specie, Element) and struct[i].specie.is_metal
                    and struct[i].specie.symbol != "O" and struct[i].specie.symbol != dopant)]
    host_top = host_pool(top_idx)
    host_bot = host_pool(bot_idx)
    S_top, S_bot = len(host_top), len(host_bot)
    S_total = S_top + S_bot
    if S_total == 0:
        return [], [], {"reason": "no_host_sites", "S_top": S_top, "S_bot": S_bot, "S_total": 0}
    T_total = max(1, int(round(S_total * frac)))
    if T_total % 2 == 1:
        T_total -= 1
    T_half = T_total // 2
    if T_half == 0 or S_top < T_half or S_bot < T_half:
        return [], [], {"reason": "capacity_insufficient_or_zero_target",
                        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
                        "T_total": T_total, "T_half": T_half}
    picks_top = farthest_point_sampling_xy(struct, host_top, T_half)
    picks_bot = farthest_point_sampling_xy(struct, host_bot, T_half)
    return picks_top, picks_bot, {
        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
        "T_total": T_total, "T_top": T_half, "T_bot": T_half
    }

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
    ap.add_argument("--outdir", type=str, default="out_Otop_rmTopO_uniform_allowedmetals")
    ap.add_argument("--dopants", type=str, default="ALL",
                    help="ALL 或逗号分隔，如 'Ag,Ni'; 将与白名单求交集")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="掺杂:宿主 全局比例（1:3=0.25）")
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--vac_top", type=float, default=20.0)
    ap.add_argument("--vac_bot", type=float, default=3.0)
    ap.add_argument("--max_docs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # 掺杂元素：限定在白名单
    if args.dopants.upper() == "ALL":
        dopant_list = sorted(ALLOWED_METALS)
    else:
        _in = [x.strip() for x in args.dopants.split(",") if x.strip()]
        dopant_list = sorted(list(ALLOWED_METALS.intersection(_in)))
        if not dopant_list:
            raise RuntimeError("给定的掺杂元素都不在白名单内。")

    d, h = [int(x) for x in args.dopant_to_host.split(":")]
    assert d > 0 and h > 0, "--dopant_to_host 必须是正整数比"
    frac = d / (d + h)  # 1:3 => 0.25

    tol = infer_tol_from_ref(args.ref_cif)
    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")

    manifest = []

    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"],
                   fields=["material_id","formula_pretty","structure"], chunk_size=200))
        docs = docs[:args.max_docs]

        for ddoc in docs:
            struct = ddoc.structure
            if not is_metal_oxide(struct):
                continue

            # —— 新增过滤：宿主中的所有金属元素必须在白名单 ——
            metals_in = metal_species_in_struct(struct)
            if (not metals_in) or (not metals_in.issubset(ALLOWED_METALS)):
                # 跳过含非白名单金属的氧化物
                manifest.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc, "formula_pretty",""),
                    "status": "skip_disallowed_metals",
                    "metals_in_struct": ",".join(sorted(metals_in))
                })
                continue

            # 1) 方向规整：Otop
            s_orient = orient_oxygen_to_top(struct, top_frac=0.10)

            # 2) 不对称真空
            s_vac = add_vacuum_asymmetric(s_orient, vac_top_A=args.vac_top, vac_bot_A=args.vac_bot)

            # 3) XY 最小扩胞
            ia, ib = minimal_xy_supercell(s_vac, a_min=args.a_min, b_min=args.b_min)
            s_xy = make_supercell_xy(s_vac, ia, ib)

            # 4) 删除顶层最高 O
            try:
                s_rmO, del_idx, del_fz = remove_topmost_oxygen_by_frac(s_xy)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue

            # 5) 表面金属位点
            top_idx, bot_idx = surface_metal_indices(s_rmO, tol_A=tol)
            if not top_idx or not bot_idx:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_surface"})
                continue

            # 6) 对每个掺杂元素都生成一个结构
            for dop in dopant_list:
                picks_top, picks_bot, info = pick_hosts_global_balanced_uniform(
                    top_idx, bot_idx, s_rmO, dop, frac
                )
                if not picks_top and not picks_bot:
                    manifest.append({
                        "material_id": ddoc.material_id,
                        "formula": getattr(ddoc, "formula_pretty",""),
                        "status": "skip_capacity",
                        "reason": info.get("reason",""),
                        "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
                        "S_top": info.get("S_top",0), "S_bot": info.get("S_bot",0),
                        "S_total": info.get("S_total",0), "T_total": info.get("T_total",0),
                        "dopant": dop, "metals_in_struct": ",".join(sorted(metals_in))
                    })
                    continue

                doped = substitute(s_rmO, picks_top + picks_bot, dop)
                fname = (f"{ddoc.material_id}_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}"
                         f"_a{ia}x_b{ib}x_rmO_top_global13_balanced_Otop_uniform_{dop}.cif")
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                manifest.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc, "formula_pretty",""),
                    "status": "ok", "outfile": fpath,
                    "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
                    "deleted_O_index": del_idx, "deleted_O_fz": del_fz,
                    "host_pool_top": info["S_top"], "host_pool_bot": info["S_bot"],
                    "host_pool_total": info["S_total"],
                    "target_total": info["T_total"],
                    "target_top": info["T_top"], "target_bot": info["T_bot"],
                    "picked_top": len(picks_top), "picked_bot": len(picks_bot),
                    "ratio_global": f"{d}:{h}", "dopant": dop,
                    "metals_in_struct": ",".join(sorted(metals_in))
                })
                print("Wrote", fname)

    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] wrote {sum(m.get('status')=='ok' for m in manifest)} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs; all skipped")

if __name__ == "__main__":
    main()
