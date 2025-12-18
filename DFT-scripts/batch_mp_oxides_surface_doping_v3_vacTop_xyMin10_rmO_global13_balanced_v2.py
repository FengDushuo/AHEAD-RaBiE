#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳定版 + 全局1:3 & 两侧等量，并保证“上表面为氧、下表面为金属”，且底边预留小真空：
- 方向规整：若顶层不是氧主导，则 z -> 1 - z 翻转，使 O 在上、金属在下；
- 不对称真空：c 增加为 (vac_top + vac_bot)，把占据区压到底部上方 vac_bot 处；
- 仅 a、b 方向最小整数倍扩胞到 ≥10 Å（[ia, ib, 1]）；
- 在“加真空 + XY 超胞 + 方向规整”之后，删除**底层** 1 个 O（保留上表面 O 完整）；
- 掺杂：全局目标 1:3（25%），严格两侧相等；若某侧容量不足则跳过该材料；
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

# ---------- 方向规整：让 O 在上、金属在下 ----------
def orient_oxygen_to_top(struct: Structure, top_frac=0.10) -> Structure:
    """
    若顶层（最高 10% 分数 z）不是 O 为主，则执行 z -> 1 - z 翻转。
    """
    s = struct.copy()
    f = np.array([site.frac_coords for site in s])  # Nx3
    zu = _unwrap_frac_z(f[:, 2])
    # 取最高 top_frac 的层
    N = len(s)
    cut = max(1, int(np.ceil(N * top_frac)))
    idx_sorted = np.argsort(zu)
    top_ids = idx_sorted[-cut:]
    nO_top = sum(1 for i in top_ids if s[i].specie.symbol == "O")
    # 取最低 bottom_frac 同样数量，做个对比（非必需，仅用于更可信的判定）
    bot_ids = idx_sorted[:cut]
    nO_bot = sum(1 for i in bot_ids if s[i].specie.symbol == "O")

    # 判定：顶层 O 明显不足、且底层 O 明显较多 -> 翻转
    if nO_top < nO_bot:
        f_new = np.copy(f)
        f_new[:, 2] = (1.0 - zu)  # 直接用解环后的 zu 做 1-zu，再放回 [0,1)
        f_new[:, 2] -= np.floor(f_new[:, 2])
        s = Structure(s.lattice, [site.specie for site in s], f_new,
                      coords_are_cartesian=False, to_unit_cell=True)
    return s

# ---------- 不对称真空：顶部 vac_top、底部 vac_bot ----------
def add_vacuum_asymmetric(struct: Structure, vac_top_A: float, vac_bot_A: float) -> Structure:
    """
    在 z 方向增加不对称真空：上方 vac_top_A，下方 vac_bot_A。
    做法：新 c = 原 slab_thick + vac_top_A + vac_bot_A；
         将解环后的占据区 [zmin,zmax] 线性压缩/平移到新分数坐标的 [vac_bot_A/new_c, (vac_bot_A+slab_thick)/new_c]。
    """
    s = struct.copy()
    lat0 = s.lattice
    Z0 = lat0.c
    # 用解环分数坐标确定占据厚度
    f = np.array([site.frac_coords for site in s])  # Nx3
    zu = _unwrap_frac_z(f[:, 2])
    z_min, z_max = float(np.min(zu)), float(np.max(zu))
    occ_span_frac = z_max - z_min
    slab_thick_A = occ_span_frac * Z0

    new_c = slab_thick_A + float(vac_top_A) + float(vac_bot_A)
    if new_c <= 1e-6:
        return s

    # 新晶格：只拉伸 c 向量到 new_c
    scale = new_c / Z0
    new_cvec = lat0.matrix[2] * scale
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], new_cvec])

    # 把占据段挪到底部上方 vac_bot_A 的位置
    # 旧分数厚度 occ_span_frac -> 新分数厚度 slab_thick_A/new_c
    new_occ_span_frac = slab_thick_A / new_c
    new_base = vac_bot_A / new_c
    new_z = new_base + (zu - z_min) * (new_occ_span_frac / max(occ_span_frac, 1e-8))

    new_f = np.copy(f)
    new_f[:, 2] = new_z
    new_f[:, 2] -= np.floor(new_f[:, 2])

    return Structure(new_lat, [site.specie for site in s], new_f,
                     coords_are_cartesian=False, to_unit_cell=True)

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

# ---------- 删底层 1 个 O（而不是删顶层） ----------
def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float]:
    """
    在分数坐标解环后，删除分数 z 最高的一个氧原子。
    返回：新结构、被删原子索引、被删原子的分数 z（解环后）
    """
    s = struct.copy()
    fz = np.array([site.frac_coords[2] for site in s])
    zu = _unwrap_frac_z(fz)
    o_idx = [i for i, site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom found")
    top = max(o_idx, key=lambda i: zu[i])   # 选分数 z 最大的 O
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
                               rng: np.random.RandomState):
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

    T_total = max(1, int(round(S_total * frac)))  # 全局 25%
    # 拆到两侧严格相同
    T_half = T_total // 2  # 向下取偶，保证两侧相等
    T_total_even = 2 * T_half
    if T_half == 0:
        return [], [], {"reason": "zero_target", "S_top": S_top, "S_bot": S_bot, "S_total": S_total, "T_total": T_total}
    if S_top < T_half or S_bot < T_half:
        return [], [], {"reason": "capacity_insufficient", "S_top": S_top, "S_bot": S_bot,
                        "T_total": T_total_even, "T_half": T_half}

    picks_top = sorted(rng.choice(host_top, size=T_half, replace=False).tolist())
    picks_bot = sorted(rng.choice(host_bot, size=T_half, replace=False).tolist())
    return picks_top, picks_bot, {
        "S_top": S_top, "S_bot": S_bot, "S_total": S_total,
        "T_total": T_total_even, "T_top": T_half, "T_bot": T_half
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
    ap.add_argument("--outdir", type=str, default="out_vacAsym_xyMin10_rmO_bottom_global13_balanced_Otop")
    ap.add_argument("--dopants", type=str, default="ALL",
                    help="ALL 或逗号分隔，如 'Fe,Co'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3",
                    help="掺杂:宿主 全局比例（例如 1:3 = 0.25）")
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--vac_top", type=float, default=20.0, help="顶部真空厚度（Å）")
    ap.add_argument("--vac_bot", type=float, default=3.0,  help="底部真空厚度（Å），满足“底边空出一点距离”")
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

            # 1) 方向规整（确保 O 在上）
            s_orient = orient_oxygen_to_top(struct, top_frac=0.10)

            # 2) 不对称真空（顶部 vac_top，底部 vac_bot）
            s_vac = add_vacuum_asymmetric(s_orient, vac_top_A=args.vac_top, vac_bot_A=args.vac_bot)

            # 3) XY 最小扩胞
            ia, ib = minimal_xy_supercell(s_vac, a_min=args.a_min, b_min=args.b_min)
            s_xy = make_supercell_xy(s_vac, ia, ib)

            # 4) 删除顶层 1 个 O（注意此时顶面已按你的逻辑在上方）
            try:
                s_rmO, del_idx, del_fz = remove_topmost_oxygen_by_frac(s_xy)
            except Exception as e:
                manifest.append({"material_id": ddoc.material_id, "status": "skip_no_O", "reason": str(e)})
                continue


            # 5) 表面金属位点识别（此时上=O层、下=金属层；函数仍返回金属的 top/bot 池）
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
                        "ia": ia, "ib": ib,
                        "vac_top": args.vac_top, "vac_bot": args.vac_bot,
                        "S_top": info.get("S_top",0), "S_bot": info.get("S_bot",0),
                        "S_total": info.get("S_total",0),
                        "T_total": info.get("T_total",0),
                    })
                    continue

                doped = substitute(s_rmO, picks_top + picks_bot, dop)

                fname = f"{ddoc.material_id}_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}_a{ia}x_b{ib}x_rmO_bottom_global13_balanced_Otop_{dop}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                manifest.append({
                    "material_id": ddoc.material_id,
                    "formula": getattr(ddoc, "formula_pretty",""),
                    "status": "ok",
                    "outfile": fpath,
                    "ia": ia, "ib": ib,
                    "vac_top": args.vac_top, "vac_bot": args.vac_bot,
                    "deleted_O_index": del_idx, "deleted_O_fz": del_fz,
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
