#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch: all metal oxides from Materials Project -> supercell to O=64 -> drop topmost O ->
surface metal substitution (host:dopant=3:1) for ALL metallic dopants -> CIF + manifest

依赖:
  pip install "pymatgen>=2023.0.0" mp-api numpy pandas

环境:
  export MAPI_KEY=你的Materials Project密钥

示例:
  python batch_mp_oxides_all_metals_surface_doping.py \
    --ref_cif 1-out.cif \
    --outdir out_all_metals \
    --max_docs 200 \
    --host_to_dopant 3:1 \
    --seed 123
"""

import os, re, math, argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from mp_api.client import MPRester
from pymatgen.core import Structure, Element
from pymatgen.transformations.standard_transformations import ConventionalCellTransformation

# --- 全部金属元素表 ---
ALL_METALS = [
    "Li","Be","Na","Mg","K","Ca","Rb","Sr","Cs","Ba","Fr","Ra",
    "Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Al","Ga","In","Sn","Tl","Pb","Bi",
    "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu",
    "Ac","Th","Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm","Md","No","Lr",
]

# ---------- 工具函数 ----------
def is_metal_oxide(struct: Structure) -> bool:
    els = [sp.symbol for sp in struct.composition.elements]
    return ("O" in els) and any(e.is_metal for e in struct.composition.elements if e.symbol != "O")

def decompose_to_abc(factor: int) -> Tuple[int,int,int]:
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

def find_supercell_for_O(struct: Structure, target_O: int = 64) -> Optional[Tuple[Structure, Tuple[int,int,int]]]:
    candidates = [
        struct,
        ConventionalCellTransformation().apply_transformation(struct),
        struct.get_primitive_structure()
    ]
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

def remove_topmost_oxygen(struct: Structure) -> Tuple[Structure, int, float]:
    coords = np.array([s.coords for s in struct])
    o_idx = [i for i,s in enumerate(struct.sites) if s.specie.symbol == "O"]
    if not o_idx: raise RuntimeError("no oxygen atom")
    top = max(o_idx, key=lambda i: coords[i,2])
    zmax = float(coords[top,2])
    s2 = struct.copy()
    s2.remove_sites([top])
    return s2, top, zmax

def infer_surface_tol_from_ref_cif(ref_cif: Optional[str]) -> float:
    # 默认容忍度
    default_tol = 1.2
    if not ref_cif or (not os.path.exists(ref_cif)):
        return default_tol
    txt = open(ref_cif, "r", encoding="utf-8", errors="ignore").read()
    def fval(pat):
        m = re.search(pat, txt, re.IGNORECASE)
        if not m: return None
        sval = re.sub(r"\([^)]*\)", "", m.group(1))
        try: return float(sval)
        except: return None
    c_len = fval(r"_cell_length_c\s+([0-9\.\(\)Ee+\-]+)")

    # 读 atom_site loop（轻量）
    lines = txt.splitlines()
    start, headers, rows = None, [], []
    i=0
    while i < len(lines):
        if lines[i].strip().lower().startswith("loop_"):
            j = i+1; temp=[]
            while j < len(lines) and lines[j].strip().startswith("_"):
                temp.append(lines[j].strip()); j+=1
            if any(h.lower().startswith("_atom_site_") for h in temp):
                start = i; headers = temp[:]
                k = j
                while k < len(lines):
                    s = lines[k].strip()
                    if (not s) or s.lower().startswith("loop_") or s.lower().startswith("data_") or s.startswith("_"):
                        break
                    toks = [m.group(0).strip("'").strip('"') for m in re.finditer(r"(?:'[^']*'|\"[^\"]*\"|\S+)", s)]
                    if toks: rows.append(toks)
                    k+=1
                break
            else:
                i=j; continue
        i+=1
    if not headers or not rows: return default_tol
    headers = [h.split()[0] for h in headers]
    def col(*cands):
        for c in cands:
            if c in headers: return headers.index(c)
        return None
    c_sym = col("_atom_site_type_symbol") or col("_atom_site_label")
    c_fz  = col("_atom_site_fract_z")
    c_z   = col("_atom_site_Cartn_z")
    def fnum(s):
        try: return float(re.sub(r"\([^)]*\)", "", s))
        except: return None
    atoms=[]
    for r in rows:
        if len(r) < len(headers):
            r = r + ["?"]*(len(headers)-len(r))
        sym = r[c_sym]
        z   = fnum(r[c_z]) if c_z is not None else None
        fz  = fnum(r[c_fz]) if c_fz is not None else None
        atoms.append({"sym": sym, "z": z, "fz": fz})
    # 用金属分层的最上两层的间距估计 tol
    METALS = set(ALL_METALS)
    use_cart = (c_z is not None) and any(a["z"] is not None for a in atoms)
    zkey = "z" if use_cart else "fz"
    zvals_m = sorted([a[zkey] for a in atoms if a["sym"].strip("0123456789") in METALS and a[zkey] is not None])
    if len(zvals_m) < 2: return default_tol
    top = zvals_m[-1]
    nxt = None
    for val in reversed(zvals_m[:-1]):
        if abs(top - val) > 1e-3:
            nxt = val; break
    if nxt is None: return default_tol
    gapA = (top - nxt) if use_cart else ((top - nxt) * (c_len if c_len else 1.0))
    tol = max(0.6, min(2.5, 0.5 * gapA))
    return tol

def get_surface_metal_indices(struct: Structure, surface_tol: float) -> List[int]:
    coords = np.array([s.coords for s in struct])
    z = coords[:,2]
    metals = [i for i,s in enumerate(struct.sites) if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol != "O"]
    if not metals: return []
    zmax = float(np.max(z[metals]))
    return sorted([i for i in metals if (zmax - z[i]) <= surface_tol])

def choose_doping_sites(indices: List[int], host_to_dopant: Tuple[int,int], k_override: Optional[int], rng: np.random.RandomState) -> List[int]:
    if not indices: return []
    if k_override is not None:
        k = max(1, min(len(indices), int(k_override)))
    else:
        m,n = host_to_dopant
        frac = n / (m+n)  # e.g., 3:1 -> 0.25
        k = max(1, min(len(indices), int(round(len(indices)*frac))))
    return sorted(rng.choice(indices, size=k, replace=False).tolist())

def substitute_species_avoid_noop(struct: Structure, idx_list: List[int], dopant: str) -> Tuple[Structure, List[int]]:
    s2 = struct.copy()
    applied = []
    # 尽量避免把已经等于 dopant 的位点再“替换”
    cand = [i for i in idx_list if s2[i].specie.symbol != dopant]
    if not cand:
        return s2, applied
    for i in cand:
        s2[i] = Element(dopant); applied.append(i)
    return s2, applied

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_cif", type=str, default=None, help="参考CIF（用于自动估算表面层厚度 tol）")
    ap.add_argument("--outdir", type=str, default="mp_oxides_all_metals_out")
    ap.add_argument("--host_to_dopant", type=str, default="3:1", help="宿主:掺杂 比例（默认 3:1≈25%）")
    ap.add_argument("--k", type=int, default=None, help="直接指定位点个数（覆盖比例）")
    ap.add_argument("--surface_tol", type=float, default=None, help="手工指定表面层厚度（Å）；若不给则由 ref_cif 推断")
    ap.add_argument("--max_docs", type=int, default=500, help="最多处理多少个材料（防止一次拉取过大）")
    ap.add_argument("--seed", type=int, default=0, help="随机种子")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 比例解析
    try:
        m,n = [int(x) for x in args.host_to_dopant.split(":")]
        assert m>0 and n>0
    except Exception:
        raise ValueError("--host_to_dopant 应为 'm:n' 的正整数，比如 3:1")

    # 表面厚度
    tol = args.surface_tol if args.surface_tol is not None else infer_surface_tol_from_ref_cif(args.ref_cif)
    print(f"[Info] surface_tol = {tol:.3f} Å")

    # 连接 MP
    api_key = os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("未检测到 MAPI_KEY 环境变量。请先 export MAPI_KEY=你的密钥")
    rng = np.random.RandomState(args.seed)

    rows = []
    out_count = 0

    with MPRester(api_key) as mpr:
        # 先以 elements 包含 O 来搜，再本地过滤“至少一个金属”
        docs = list(mpr.summary.search(elements=["O"], fields=["material_id","formula_pretty","structure"], chunk_size=200))
        # 裁剪数量以防过大（可自行调大）
        if args.max_docs and len(docs) > args.max_docs:
            docs = docs[:args.max_docs]

        for d in docs:
            try:
                struct: Structure = d.structure
            except Exception:
                continue
            if not is_metal_oxide(struct):
                continue

            # (1) O=64 扩胞
            ok = find_supercell_for_O(struct, target_O=64)
            if ok is None:
                rows.append({
                    "material_id": d.material_id, "formula": getattr(d, "formula_pretty",""),
                    "status": "skip_O_not_divisible",
                })
                continue
            sup, (ax,ay,az) = ok

            # (2) 删除顶层 O
            try:
                sup2, del_idx, zO = remove_topmost_oxygen(sup)
            except Exception as e:
                rows.append({
                    "material_id": d.material_id, "formula": getattr(d,"formula_pretty",""),
                    "status": "skip_no_O", "reason": str(e),
                })
                continue

            # (3) 表面金属层
            surf_idx = get_surface_metal_indices(sup2, surface_tol=tol)
            if not surf_idx:
                rows.append({
                    "material_id": d.material_id, "formula": getattr(d,"formula_pretty",""),
                    "status": "skip_no_surface_metals",
                })
                continue

            # (4) 为所有金属元素分别生成一个掺杂版本
            for dop in ALL_METALS:
                # 选择掺杂位点
                picks = choose_doping_sites(surf_idx, (m,n), args.k, rng)
                doped, applied = substitute_species_avoid_noop(sup2, picks, dop)
                if not applied:
                    # 如果全是“与dop相同”的位点导致没替上，给个提醒但仍记录
                    rows.append({
                        "material_id": d.material_id, "formula": getattr(d,"formula_pretty",""),
                        "status": "noop_same_species", "dopant": dop,
                        "supercell": f"{ax}x{ay}x{az}",
                        "deleted_O_index": del_idx, "deleted_O_z": zO,
                        "surface_sites": len(surf_idx), "picked": len(picks),
                        "applied": 0
                    })
                    continue

                fname = f"{d.material_id}_O64delTopO_surface_{dop}_host{m}_dop{n}_ax{ax}ay{ay}az{az}.cif"
                fpath = os.path.join(args.outdir, fname)
                doped.to(fmt="cif", filename=fpath)
                rows.append({
                    "material_id": d.material_id, "formula": getattr(d,"formula_pretty",""),
                    "status": "ok", "outfile": fpath, "dopant": dop,
                    "supercell": f"{ax}x{ay}x{az}",
                    "deleted_O_index": del_idx, "deleted_O_z": zO,
                    "surface_sites": len(surf_idx), "picked": len(picks),
                    "applied": len(applied), "applied_indices": ";".join(map(str, applied))
                })
                out_count += 1

    # (5) manifest
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        print(f"[OK] 写出 {sum(r.get('status')=='ok' for r in rows)} 个掺杂样本；总记录 {len(rows)} 行。")
    else:
        print("[WARN] 没有生成任何样本；请检查条件。")

if __name__ == "__main__":
    main()
