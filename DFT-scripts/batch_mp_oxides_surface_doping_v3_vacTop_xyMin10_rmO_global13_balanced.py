#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版 + 全局1:3 & 两侧等量：
- z 轴顶部留真空：c → c + vac_add（分数 z 解环压至底部，避免斜晶格耦合）
- 仅 a、b 方向最小整数倍扩胞到 ≥10 Å（[ia, ib, 1]）
- 在“加真空 + XY 超胞”之后删除顶层 1 个 O（只删 1 个）
- 掺杂策略（与上一版不同）：
  * 在“顶/底表面金属位点”中先排除 O 与已经是 dopant 的位点，得到宿主池；
  * 以“顶+底宿主池总数 S_total”为基准，目标掺杂数 T_total = round(S_total * d/(d+h))（1:3 即 25%）；
  * 要求“顶侧掺杂数 = 底侧掺杂数 = T_total/2（四舍五入后一半）”，两侧严格相等；
  * 若某侧宿主池容量不足以实现这一半，则跳过该材料（保持比例严格）。
- 输出 .cif + manifest.csv

依赖：pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element, Lattice

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

# ---------- 顶部真空：c→c+vac_add，分数 z 压到底部 ----------
def add_vacuum_with_empty_top(struct: Structure, delta_vac_A: float = 20.0) -> Structure:
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    Z1 = Z0 + float(delta_vac_A)
    new_cvec = lat0.matrix[2] * (Z1 / Z0)
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], new_cvec])

    f = np.array([site.frac_coords for site in s])
    zu = _unwrap_frac_z(f[:, 2])
    z_min = float(np.min(zu))
    scale_frac = Z0 / Z1
    new_f = np.copy(f)
    new_f[:, 2] = (zu - z_min) * scale_frac
    new_f[:, 2] -= np.floor(new_f[:, 2])
    return Structure(new_lat, [site.specie for site in s],
                     new_f, coords_are_cartesian=False, to_unit_cell=True)

# ---------- XY 最小扩胞到 a,b ≥ 10 Å ----------
def minimal_xy_supercell(struct: Structure, a_min: float = 10.0, b_min: float = 10.0) -> Tuple[int,int]:
    lat = struct.lattice
    ia = max(1, math.ceil(a_min / lat.a))
    ib = max(1, math.ceil(b_min / lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy()
    s.make_supercell([ia, ib, 1])
    return s

# ---------- 删顶层 1 个 O（在真空+XY 超胞之后执行） ----------
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

# ---------- 全局 1:3 且顶/底数量相同 ----------
def pick_hosts_global_balanced(top_idx: List[int], bot_idx: List[int],
                               struct: Structure, dopant: str, frac: float,
                               rng: np.random.RandomState) -> Tuple[List[int], List[int], dict]:
    """
    在顶/底宿主池的合并总数基础上取 1:3（frac=0.25）的总掺杂数，
    然后强制“顶半数 / 底半数”完全相同。
    若任一侧宿主池容量不足以达到该半数，则返回空并给出原因（调用方跳过该材料）。
    """
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
    # 按“完全相同”要求拆分到两侧
    T_half = int(round(T_total / 2.0))
    # 由于四舍五入，总数可能变成 2*T_half，与 T_total 不一致；调整为偶数以严格相等
    if 2*T_half != T_total:
        # 以“不超过全局目标”为优先，向下取偶数
        if 2*math.floor(T_total/2) > 0:
            T_half = math.floor(T_total/2)
        else:
            T_half = 0
    T_top = T_half
    T_bot = T_half

    # 容量校验：每侧必须各自有足够的宿主位点
    if (S_top < T_top) or (S_bot < T_bot) or (T_top == 0 and T_bot == 0):
        return [], [], {"reason": "capacity_insufficient_or_zero_target",
                        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
                        "T_total": T_total, "T_top": T_top, "T_bot": T_bot}

    picks_top = sorted(rng.choice(host_top, size=T_top, replace=False).tolist())
    picks_bot = sorted(rng.choice(host_bot, size=T_bot, replace=False).tolist())
    return picks_top, picks_bot, {
        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
        "T_total": T_total, "T_top": T_top, "T_bot": T_bot
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
    ap.add_argument("--ref_cif", type=str, default=None,
                    help="用于推断表面 tol（Å），不从中拷坐标")
    ap.add_argument("--outdir", type=str, default="out_vacTop_xyMin10_rmO_global13_balanced")
    ap.add_argument("--dopants", type=str, default="ALL",
                    help="ALL 或逗号分隔，如 'Fe,Co'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3",
                    help="掺杂:宿主 全局比例（例如 1:3 = 0.25）")
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
    frac = d/(d+h)  # 1:3 => 0.25

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

            # 1) 顶部真空（不动 a/b 分量，沿分数 z 压到底部）
            s_vac = add_vacuum_with_empty_top(struct, delta_vac_A=args.vac_add)

            # 2) XY 最小扩胞
            ia, ib = minimal_xy_supercell(s_vac, a_min=args.a_min, b_min=args.b_min)
            s_xy = make_supercell_xy(s_vac, ia, ib)

            # 3) 删除顶层 1 个 O（在最终超胞上删）
            try:
                s_rmO, del_idx, del_fz = remove_topmost_oxygen_by_frac(s_xy)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue

            # 4) 表面位点（金属）
            top_idx, bot_idx = surface_metal_indices(s_rmO, tol_A=tol)
            if len(top_idx)==0 or len(bot_idx)==0:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_surface"})
                continue

            for dop in dopant_list:
                picks_top, picks_bot, info = pick_hosts_global_balanced(
                    top_idx, bot_idx, s_rmO, dop, frac, rng
                )
                if len(picks_top)==0 and len(picks_bot)==0:
                    manifest.append({
                        "material_id": ddoc.material_id,
                        "formula": getattr(ddoc, "formula_pretty",""),
                        "status": "skip_capacity",
                        "reason": info.get("reason",""),
                        "ia": ia, "ib": ib, "vac_add": args.vac_add,
                        "S_top": info.get("S_top",0), "S_bot": info.get("S_bot",0),
                        "S_total": info.get("S_total",0),
                        "T_total": info.get("T_total",0),
                    })
                    continue

                doped = substitute(s_rmO, picks_top + picks_bot, dop)

                fname = f"{ddoc.material_id}_vacTop{int(args.vac_add)}_a{ia}x_b{ib}x_rmO_global13_balanced_{dop}.cif"
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
                    # 宿主与目标/实际（全局 & 分侧）
                    "host_pool_top": info["S_top"], "host_pool_bot": info["S_bot"],
                    "host_pool_total": info["S_total"],
                    "target_total": info["T_total"],
                    "target_top": info["T_top"], "target_bot": info["T_bot"],
                    "picked_top": len(picks_top), "picked_bot": len(picks_bot),
                    "ratio_global": f"{d}:{h}",
                    "dopant": dop
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
