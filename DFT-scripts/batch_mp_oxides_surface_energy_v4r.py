#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量获取 Materials Project 的金属氧化物，
仅在 x/y 方向扩胞 -> 删除沿 c (分数坐标) 顶层的 1 个 O -> 参照参考 CIF 留对称真空 ->
上下两侧表面金属位点做全局 ≈ 1:3 金属替位掺杂（避免空替换 + 必要回填）。
目标：删除 O 后总原子数 ~ 95（可调）。
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

# -------------------- 几何/工具函数（全部按沿 c 的分数坐标处理） --------------------

def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def best_2d_supercell_for_target_N(struct: Structure, N_target: int, max_mul: int = 10) -> Tuple[int,int]:
    """
    只在 x/y 扩胞（az=1），删除 1 个 O 之后的总原子数尽量接近 N_target。
    """
    N0 = len(struct)
    target_after = max(1, N_target)  # 注意：我们后面会先扩胞再删 O，因此这里就对齐 N_target
    best = (1,1); best_err = 10**9
    for ax in range(1, max_mul+1):
        for ay in range(1, max_mul+1):
            N_after_remove1O = N0 * ax * ay - 1
            err = abs(N_after_remove1O - target_after)
            if err < best_err:
                best, best_err = (ax, ay), err
    return best

def make_2d_supercell(struct: Structure, ax:int, ay:int) -> Structure:
    s = struct.copy()
    s.make_supercell([ax, ay, 1])
    return s

def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    """对分数坐标 z（0..1）进行解环（unwrap），得到 slab 的连续区间。"""
    z_sorted = np.sort(z)
    gaps = np.diff(np.r_[z_sorted, z_sorted[0] + 1.0])
    k = int(np.argmax(gaps))  # 最大缝隙
    base = z_sorted[(k + 1) % len(z_sorted)]
    zu = z - base
    zu[zu < 0] += 1.0
    return zu

def remove_topmost_oxygen_by_frac(struct: Structure) -> Tuple[Structure, int, float, int]:
    """
    基于“沿 c 的分数坐标（且解环后的）”删除最顶层的一个 O。
    返回：新结构、被删 O 的原索引、该 O 的解环分数 z、删除后总原子数
    """
    s = struct.copy()
    fcoords = np.array([site.frac_coords for site in s])
    z = fcoords[:, 2]
    zu = _unwrap_frac_z(z)
    o_idx = [i for i,site in enumerate(s.sites) if site.specie.symbol == "O"]
    if not o_idx:
        raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: zu[i])
    top_z_unwrap = float(zu[top])
    s.remove_sites([top])
    return s, top, top_z_unwrap, len(s)

def unwrap_frac_along_c(struct: Structure) -> np.ndarray:
    """返回结构里每个原子沿 c 的“解环分数坐标 z”"""
    fz = np.array([site.frac_coords[2] for site in struct.sites])
    return _unwrap_frac_z(fz)

def parse_ref_vac_and_tol(ref_cif: Optional[str]) -> Tuple[Optional[float], float]:
    """
    从参考 CIF 推断：
    - 总真空厚度（Å）= c_len - 占据厚度（用分数坐标 * c_len 求得）
    - 表面层厚度 tol（Å）= 最外金属↔次外金属 (沿 c) 的间距的一半（夹在 0.6..2.5 Å）
    """
    default_tol = 1.2
    if (ref_cif is None) or (not os.path.exists(ref_cif)):
        return None, default_tol

    txt = open(ref_cif, "r", encoding="utf-8", errors="ignore").read()

    def fval(pat):
        m = re.search(pat, txt, re.IGNORECASE)
        if not m: return None
        sval = re.sub(r"\([^)]*\)", "", m.group(1))
        try: return float(sval)
        except: return None

    c_len = fval(r"_cell_length_c\s+([0-9\.\(\)Ee+\-]+)")

    # 解析 atom_site loop
    lines = txt.splitlines()
    headers, rows = [], []
    i = 0
    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            j = i + 1; tmp = []
            while j < len(lines) and lines[j].strip().startswith("_"):
                tmp.append(lines[j].strip()); j += 1
            if any(h.lower().startswith("_atom_site_") for h in tmp):
                headers = tmp[:]; k = j
                while k < len(lines):
                    s = lines[k].strip()
                    if (not s) or s.lower().startswith("loop_") or s.lower().startswith("data_") or s.startswith("_"):
                        break
                    toks = [m.group(0).strip("'").strip('"') for m in re.finditer(r"(?:'[^']*'|\"[^\"]*\"|\S+)", s)]
                    if toks: rows.append(toks)
                    k += 1
                break
            else:
                i = j; continue
        i += 1

    if not headers or not rows:
        return None, default_tol

    headers = [h.split()[0] for h in headers]
    def col(*cands):
        for c in cands:
            if c in headers: return headers.index(c)
        return None

    c_sym = col("_atom_site_type_symbol") or col("_atom_site_label")
    c_fz  = col("_atom_site_fract_z")

    def fnum(s):
        try: return float(re.sub(r"\([^)]*\)", "", s))
        except: return None

    METALS = set(ALL_METALS)
    z_frac_all = []
    z_frac_metal = []
    for r in rows:
        if len(r) < len(headers):
            r = r + ["?"] * (len(headers) - len(r))
        sym = r[c_sym]
        fz = fnum(r[c_fz]) if c_fz is not None else None
        if fz is not None:
            z_frac_all.append(fz)
            if sym.strip("0123456789") in METALS:
                z_frac_metal.append(fz)

    total_vac = None
    if c_len is not None and len(z_frac_all) > 0:
        zf = np.array(z_frac_all)
        zu = _unwrap_frac_z(zf)
        extent_A = (zu.max() - zu.min()) * c_len
        total_vac = max(0.0, c_len - extent_A)

    tol = default_tol
    if c_len is not None and len(z_frac_metal) >= 2:
        zf_m = np.array(z_frac_metal)
        zu_m = _unwrap_frac_z(zf_m)
        zu_m.sort()
        top = zu_m[-1]
        nxt = None
        for val in zu_m[-2::-1]:
            if abs(top - val) > 1e-3:
                nxt = val; break
        if nxt is not None:
            gapA = (top - nxt) * c_len
            tol = max(0.6, min(2.5, 0.5 * gapA))

    return total_vac, tol

def add_symmetric_vacuum_along_c(struct: Structure, total_vac_A: float) -> Structure:
    """
    沿 c 加对称真空，并用解环后的分数 z 把 slab 居中在 0.5。
    """
    if total_vac_A is None or total_vac_A <= 1e-6:
        return struct.copy()
    s = struct.copy()
    lat = s.lattice
    c_len_old = lat.c
    zu = unwrap_frac_along_c(s)
    span = zu.max() - zu.min()                # in fractional
    slab_thick_A = span * c_len_old
    new_c = slab_thick_A + total_vac_A
    scale = new_c / c_len_old
    new_cvec = lat.matrix[2] * scale
    new_lat = Lattice([lat.matrix[0], lat.matrix[1], new_cvec])

    center = 0.5 * (zu.max() + zu.min())      # fractional
    f = np.array([site.frac_coords for site in s])
    f[:,2] = f[:,2] - center + 0.5
    f[:,2] = f[:,2] - np.floor(f[:,2])        # wrap入 [0,1)

    new_s = Structure(new_lat, [site.specie for site in s], f, coords_are_cartesian=False, to_unit_cell=True)
    return new_s

def get_surface_metal_indices_both_by_frac(struct: Structure, tol_A: float) -> Tuple[List[int], List[int]]:
    """
    用解环分数 z 的“沿 c 距离”来取上下两侧的金属表面层（距离阈值 = tol_A）。
    """
    s = struct
    lat = s.lattice
    zu = unwrap_frac_along_c(s)
    metals = [i for i,site in enumerate(s.sites) if isinstance(site.specie, Element) and site.specie.is_metal and site.specie.symbol != "O"]
    if not metals:
        return [], []
    zmax = float(np.max(zu[metals])); zmin = float(np.min(zu[metals]))
    thr = tol_A / lat.c
    top = sorted([i for i in metals if (zmax - zu[i]) <= thr])
    bot = sorted([i for i in metals if (zu[i] - zmin) <= thr])
    return top, bot

def substitute_species(struct: Structure, indices: List[int], dopant: str):
    s2 = struct.copy()
    applied = []
    for i in indices:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
            applied.append(i)
    return s2, applied

def choose_sites_global_13(s: Structure, top_idx: List[int], bot_idx: List[int],
                           dopant: str, d:int, h:int, rng: np.random.RandomState):
    """
    在 (top+bottom) 的“可掺杂池”（排除原本就是 dopant 的位点）上做全局 1:3：
    - 计算目标替换总数 target = round(S_total * d/(d+h))
    - 按两侧可掺杂数权重分配 top/bottom
    - 返回 picks 列表，以及各侧选择与各侧可掺杂计数
    """
    dopable_top = [i for i in top_idx if s[i].specie.symbol != dopant]
    dopable_bot = [i for i in bot_idx if s[i].specie.symbol != dopant]
    S_top, S_bot = len(dopable_top), len(dopable_bot)
    S_total = S_top + S_bot
    if S_total == 0:
        return [], [], [], S_top, S_bot

    target = int(round(S_total * d / (d + h)))
    k_top = int(round(target * (S_top / S_total))) if S_total > 0 else 0
    k_bot = target - k_top
    k_top = max(0, min(S_top, k_top))
    k_bot = max(0, min(S_bot, k_bot))

    pick_top = sorted(rng.choice(dopable_top, size=k_top, replace=False).tolist()) if k_top > 0 else []
    pick_bot = sorted(rng.choice(dopable_bot, size=k_bot, replace=False).tolist()) if k_bot > 0 else []
    picks = sorted(set(pick_top + pick_bot))
    return picks, pick_top, pick_bot, S_top, S_bot

# -------------------- 主流程 --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="用于推断真空厚度与表面 tol 的参考 CIF")
    ap.add_argument("--outdir", type=str, default="out_bisurf")
    ap.add_argument("--dopants", type=str, default="ALL", help="ALL 或逗号列表，如 'Fe,Co,Ni'")
    ap.add_argument("--dopant_to_host", type=str, default="1:3", help="全局掺杂:宿主（默认 1:3）")
    ap.add_argument("--target_atoms", type=int, default=95, help="删除 1 个顶层 O 后的目标总原子数（约 95）")
    ap.add_argument("--surface_tol", type=float, default=None, help="手动指定表面层厚度（Å）；默认按参考 CIF 推断")
    ap.add_argument("--max_docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 解析掺杂元素
    if args.dopants.strip().upper() == "ALL":
        dopant_list = ALL_METALS[:]
    else:
        dopant_list = [x.strip() for x in args.dopants.split(",") if x.strip()]
        dopant_list = [d for d in dopant_list if d in ALL_METALS]
        if not dopant_list:
            raise ValueError("未识别到有效的金属掺杂元素。")

    # 比例 d:h
    try:
        d, h = [int(x) for x in args.dopant_to_host.split(":")]
        assert d > 0 and h > 0
    except Exception:
        raise ValueError("--dopant_to_host 应为 'd:h' 的正整数，比如 1:3")

    # 真空 & tol
    ref_vac, tol_infer = parse_ref_vac_and_tol(args.ref_cif)
    tol = args.surface_tol if args.surface_tol is not None else tol_infer
    print(f"[Info] tol = {tol:.3f} Å ; ref total vacuum = {ref_vac if ref_vac is not None else 'None'} Å")

    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("未检测到 MAPI_KEY 环境变量。请先 export MAPI_KEY=你的密钥")
    rng = np.random.RandomState(args.seed)

    rows = []

    with MPRester(api_key) as mpr:
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        if args.max_docs and len(docs) > args.max_docs:
            docs = docs[:args.max_docs]

        for ddoc in docs:
            try:
                struct: Structure = ddoc.structure
            except Exception:
                continue
            if not is_metal_oxide(struct):
                continue

            # 只在 x/y 扩胞，目标删除 O 后总原子数 ~ target_atoms
            ax, ay = best_2d_supercell_for_target_N(struct, args.target_atoms)
            sup = make_2d_supercell(struct, ax, ay)

            # 删除沿 c 顶层 1 个 O（解环分数坐标）
            try:
                sup2, del_idx, topO_fz_unwrap, N_after = remove_topmost_oxygen_by_frac(sup)
            except Exception as e:
                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "skip_no_O", "reason": str(e)
                })
                continue

            # 沿 c 加对称真空并居中
            if ref_vac is not None and ref_vac > 1e-6:
                sup3 = add_symmetric_vacuum_along_c(sup2, total_vac_A=ref_vac)
            else:
                sup3 = sup2

            # 识别上下两侧表面（沿 c 的距离阈值 tol）
            top_idx, bot_idx = get_surface_metal_indices_both_by_frac(sup3, tol_A=tol)
            if not top_idx and not bot_idx:
                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "skip_no_surface_metals"
                })
                continue

            # 对每个掺杂元素生成一份
            for dop in dopant_list:
                picks, pick_top, pick_bot, S_top, S_bot = choose_sites_global_13(sup3, top_idx, bot_idx, dop, d, h, rng)
                doped, applied = substitute_species(sup3, picks, dop)

                # 若因随机/去重导致 applied 少于目标，尝试从剩余池补齐
                S_total = S_top + S_bot
                target = int(round(S_total * d / (d + h)))
                if len(applied) < target:
                    remaining = [i for i in (set(top_idx + bot_idx) - set(applied)) if doped[i].specie.symbol != dop]
                    need = target - len(applied)
                    if need > 0 and remaining:
                        add_more = sorted(rng.choice(remaining, size=min(need, len(remaining)), replace=False).tolist())
                        doped, applied2 = substitute_species(doped, add_more, dop)
                        applied += applied2

                fname = f"{ddoc.material_id}_2D_{ax}x{ay}_N{N_after}_bisurf_{dop}_d{d}_h{h}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)

                rows.append({
                    "material_id": ddoc.material_id, "formula": getattr(ddoc,"formula_pretty",""),
                    "status": "ok", "outfile": fpath, "dopant": dop,
                    "supercell_2d": f"{ax}x{ay}x1",
                    "deleted_O_index": del_idx, "deleted_O_fz_unwrap": topO_fz_unwrap,
                    "final_total_atoms": N_after,
                    "surface_tol_A": tol,
                    "top_surface_sites": S_top, "bot_surface_sites": S_bot,
                    "picked_top": len(pick_top), "picked_bot": len(pick_bot),
                    "applied_total": len(applied), "applied_indices": ";".join(map(str, applied))
                })

    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] Wrote {sum(r.get('status')=='ok' for r in rows)} doped samples; total records {len(rows)}.")
    else:
        print("[WARN] No outputs; check filters/params.]")

if __name__ == "__main__":
    main()
