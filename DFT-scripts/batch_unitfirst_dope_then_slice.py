#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
先在“元胞”上做 1:3 (=25%) 金属替位掺杂 -> 再对掺杂后的元胞按给定 (hkl) 切面生成 slab（保证原子完整） ->
加不对称真空（z 轴）-> 可选删除最顶层一个 O（造氧空位）-> 仅在 a,b 扩胞到给定阈值 -> 导出 CIF + manifest.csv

依赖：pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import pandas as pd
from pymatgen.core import Structure, Element, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# 允许的金属种类（宿主/掺杂统一白名单）
ALLOWED_METALS: Set[str] = set([
    "Li","Na","Mg","K","Ca","Ba",
    "V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd",
    "Hf","Pt","Au","Hg",
    "Al","Pb",
    "Ce","Gd"
])

# ------------------ 小工具 ------------------
def is_metal_oxide(struct: Structure) -> bool:
    els = [e.symbol for e in struct.composition.elements]
    return "O" in els and any(e.is_metal and e.symbol != "O" for e in struct.composition.elements)

def metal_species_in_struct(struct: Structure) -> Set[str]:
    mets=set()
    for e in struct.composition.elements:
        if e.symbol!="O" and e.is_metal: mets.add(e.symbol)
    return mets

def reduce_cell(bulk: Structure, mode: str) -> Structure:
    mode = (mode or "none").lower()
    try:
        if mode == "primitive":
            return bulk.get_primitive_structure()
        if mode == "symm_primitive":
            sga = SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_primitive_standard_structure()
        if mode == "conventional":
            sga = SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_conventional_standard_structure()
    except Exception:
        pass
    return bulk

def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z)==0: return z
    zs=np.sort(z)
    gaps=np.diff(np.r_[zs, zs[0]+1.0])
    k=int(np.argmax(gaps))
    base=zs[(k+1)%len(zs)]
    zu=z-base; zu[zu<0]+=1.0
    return zu

def orient_O_on_top(struct: Structure, probe_frac: float = 0.10) -> Structure:
    """若底部 O 比顶部多，则整体翻转 z 方向，使 O 更可能在上表面。"""
    s=struct.copy()
    f=np.array([site.frac_coords for site in s])
    zu=_unwrap_frac_z(f[:,2])
    N=len(s); cut=max(1,int(np.ceil(N*probe_frac)))
    idx=np.argsort(zu); top=idx[-cut:]; bot=idx[:cut]
    nO_top=sum(1 for i in top if s[i].specie.symbol=="O")
    nO_bot=sum(1 for i in bot if s[i].specie.symbol=="O")
    if nO_top < nO_bot:
        f[:,2]=(1.0-zu)%1.0
        return Structure(s.lattice, [site.specie for site in s], f,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

def add_vacuum_asymmetric(struct: Structure, vac_top: float, vac_bot: float) -> Structure:
    """仅拉长 c，并把占据区映射到 [vac_bot, vac_bot+slab]，真空中不出现原子。"""
    s=struct.copy()
    lat0=s.lattice; Z0=lat0.c
    f=np.array([site.frac_coords for site in s])
    zu=_unwrap_frac_z(f[:,2])
    zmin,zmax=float(np.min(zu)), float(np.max(zu))
    span=max(1e-12, zmax-zmin)
    slab_A=span*Z0
    new_c=slab_A+float(vac_top)+float(vac_bot)
    scale=new_c/Z0
    new_lat=Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2]*scale])
    new_span=slab_A/new_c; new_base=vac_bot/new_c
    f[:,2]=(new_base + (zu - zmin) * (new_span/span)) % 1.0
    return Structure(new_lat, [site.specie for site in s], f,
                     coords_are_cartesian=False, to_unit_cell=True)

def remove_topmost_O(struct: Structure) -> Tuple[Structure,int,float]:
    """删除分数 z 最大的 1 个 O。"""
    s=struct.copy()
    fz=_unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_ids=[i for i,site in enumerate(s.sites) if site.specie.symbol=="O"]
    if not o_ids: raise RuntimeError("No O found.")
    top=max(o_ids, key=lambda i: fz[i]); zval=float(fz[top])
    s.remove_sites([top]); return s, top, zval

def minimal_xy_supercell(struct: Structure, a_min: float, b_min: float) -> Tuple[int,int]:
    lat=struct.lattice
    ia=max(1, math.ceil(a_min/lat.a))
    ib=max(1, math.ceil(b_min/lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s=struct.copy(); s.make_supercell([ia,ib,1]); return s

# ------------------ 掺杂（先在元胞） ------------------
def pick_uc_metal_sites(struct: Structure, dopant: str, frac: float, rng: np.random.RandomState) -> List[int]:
    cand=[i for i,site in enumerate(struct.sites)
          if isinstance(site.specie, Element)
          and site.specie.is_metal
          and site.specie.symbol!="O"
          and site.specie.symbol!=dopant]
    k=max(1,int(round(len(cand)*frac))) if cand else 0
    if k<=0: return []
    # 简易均匀化：分数坐标 xy 的最远点采样
    xy=np.array([struct[i].frac_coords[:2] for i in cand])
    if len(cand)<=3 or k<=1:
        return sorted(rng.choice(cand, size=k, replace=False).tolist())
    center=np.mean(xy, axis=0); d2=np.sum((xy-center)**2, axis=1)
    seed=int(np.argmax(d2)); chosen=[seed]
    dist=np.linalg.norm(xy-xy[seed], axis=1)
    while len(chosen)<k:
        nxt=int(np.argmax(dist))
        if nxt in chosen:
            remaining=[i for i in range(len(cand)) if i not in chosen]
            if not remaining: break
            nxt=int(rng.choice(remaining))
        chosen.append(nxt)
        dist=np.minimum(dist, np.linalg.norm(xy-xy[nxt], axis=1))
    return sorted(cand[i] for i in chosen)

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2=struct.copy()
    for i in idx:
        if s2[i].specie.symbol!=dopant:
            s2[i]=Element(dopant)
    return s2

# ------------------ 切面（使用 SlabGenerator） ------------------
def parse_millers(millers_str: str) -> List[Tuple[int,int,int]]:
    # 形如 "0,1,1;1,1,0;-1,1,1"
    items=[]
    for blk in millers_str.replace("(","").replace(")","").split(";"):
        blk=blk.strip()
        if not blk: continue
        h,k,l=[int(x) for x in blk.split(",")]
        items.append((h,k,l))
    return items

def make_slab_from_doped(bulk_doped: Structure,
                         hkl: Tuple[int,int,int],
                         min_slab: float,
                         min_vac: float,
                         n_shifts: int = 8,
                         prefer_O_top: bool = True) -> Optional[Structure]:
    """
    用掺杂后的“元胞”作为 bulk 来切 slab。
    通过扫描少量 shift，尽量拿到不截断原子的整层 slab；若 prefer_O_top，则优先返回 O 在顶的切片。
    """
    gen=SlabGenerator(
        initial_structure=bulk_doped,
        miller_index=hkl,
        min_slab_size=min_slab,
        min_vacuum_size=min_vac,
        center_slab=False,
        in_unit_planes=True,         # 只在晶面整层切，避免原子被切断
        primitive=True,
        max_normal_search=1,
        reorient_lattice=True,
    )
    scores=[]; best=None
    shifts=[(i+0.5)/n_shifts for i in range(n_shifts)]
    for sh in shifts:
        try:
            slab=gen.get_slab(shift=sh)
        except TypeError:
            slabs=gen.get_slabs(ftol=0.1) if hasattr(gen,"get_slabs") else []
            for s in slabs:
                scores.append((s, 0.0))
            continue
        if slab is None: continue
        # 简单 Otop 评分：顶部 10% 原子里 O 的比例越高越好
        f=np.array([site.frac_coords for site in slab])
        zu=_unwrap_frac_z(f[:,2])
        idx=np.argsort(zu); cut=max(1,int(np.ceil(len(slab)*0.1)))
        top=idx[-cut:]; ofrac=sum(1 for i in top if slab[i].specie.symbol=="O")/cut
        score = -ofrac if prefer_O_top else 0.0
        scores.append((slab, score))
    if not scores:
        return None
    # 选择 score 最小（如果 prefer_O_top=True 即 O 顶层优先）
    scores.sort(key=lambda t: t[1])
    return scores[0][0]

# ------------------ 主程序 ------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_ucdope_then_slice")
    # 数据来源：MP 或 本地 CIF
    ap.add_argument("--ids", type=str, default="", help="逗号分隔的 mp-ids；留空则使用 --cif_dir")
    ap.add_argument("--cif_dir", type=str, default="", help="从该目录读取 CIF")
    ap.add_argument("--reduce", type=str, default="symm_primitive",
                    choices=["none","primitive","symm_primitive","conventional"])
    # 掺杂
    ap.add_argument("--dopants", type=str, default="ALL",
                    help="ALL 或逗号分隔，如 'Ag,Ni'（自动与白名单求交集）")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--seed", type=int, default=0)
    # 切面
    ap.add_argument("--millers", type=str, default="0,1,1;1,1,0",
                    help='多个 Miller 用 ";" 分隔，如 "0,1,1;1,1,0"')
    ap.add_argument("--min_slab", type=float, default=10.0)
    ap.add_argument("--min_vac", type=float, default=10.0)
    ap.add_argument("--prefer_O_top", action="store_true", help="优先选择顶层为 O 的切片")
    # 真空与后处理
    ap.add_argument("--vac_top", type=float, default=20.0)
    ap.add_argument("--vac_bot", type=float, default=3.0)
    ap.add_argument("--rm_top_O", action="store_true", help="在加真空后删除最顶层 1 个 O（造空位）")
    # XY 扩胞
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    # 其余
    ap.add_argument("--max_docs", type=int, default=50)
    ap.add_argument("--mapi_key", type=str, default=None)
    args=ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng=np.random.RandomState(args.seed)

    # 掺杂名单
    if args.dopants.upper()=="ALL":
        dopant_list=sorted(ALLOWED_METALS)
    else:
        req=[x.strip() for x in args.dopants.split(",") if x.strip()]
        dopant_list=sorted(list(ALLOWED_METALS.intersection(req)))
        if not dopant_list:
            raise RuntimeError("给定掺杂元素均不在白名单。")

    # 比例
    d,h=[int(x) for x in args.dopant_to_host.split(":")]
    assert d>0 and h>0
    frac = d/(d+h)  # 1:3 -> 0.25

    # 载入结构
    entries: List[Tuple[str,Structure]]=[]
    if args.ids.strip():
        api_key=args.mapi_key or os.environ.get("MAPI_KEY")
        if not api_key: raise RuntimeError("Missing MAPI_KEY")
        with MPRester(api_key) as mpr:
            ids=[x.strip() for x in args.ids.split(",") if x.strip()]
            docs=list(mpr.summary.search(material_ids=ids, fields=["material_id","structure"], chunk_size=50))
            mpmap={d.material_id:d.structure for d in docs}
        for mid in ids:
            if mid in mpmap: entries.append((mid, mpmap[mid]))
            else: print(f"[WARN] 未找到 {mid}")
    else:
        cifdir=args.cif_dir.strip()
        if not cifdir: raise RuntimeError("请提供 --ids 或 --cif_dir")
        for fn in os.listdir(cifdir):
            if fn.lower().endswith(".cif"):
                try:
                    s=Structure.from_file(os.path.join(cifdir, fn))
                    entries.append((os.path.splitext(fn)[0], s))
                except Exception as e:
                    print(f"[WARN] 读入失败 {fn}: {e}")

    millers=parse_millers(args.millers)
    manifest: List[Dict[str,Any]]=[]

    for bid, bulk0 in entries:
        if not is_metal_oxide(bulk0):
            manifest.append({"id": bid, "status": "skip_not_oxide"}); continue
        metals_in=metal_species_in_struct(bulk0)
        if (not metals_in) or (not metals_in.issubset(ALLOWED_METALS)):
            manifest.append({"id": bid, "status": "skip_disallowed_metals",
                             "metals_in_struct": ",".join(sorted(metals_in))}); continue

        bulk = reduce_cell(bulk0, args.reduce)

        for dop in dopant_list:
            # STEP-1: 元胞掺杂（25%）
            picks = pick_uc_metal_sites(bulk, dopant=dop, frac=frac, rng=rng)
            if not picks:
                manifest.append({"id": bid, "dopant": dop, "status": "skip_no_host_in_uc"}); continue
            bulk_doped = substitute(bulk, picks, dop)

            # STEP-2: 对掺杂元胞按 (hkl) 切 slab（保证整层，尽可能 O 顶）
            for hkl in millers:
                slab = make_slab_from_doped(
                    bulk_doped, hkl, min_slab=args.min_slab, min_vac=args.min_vac,
                    n_shifts=8, prefer_O_top=args.prefer_O_top
                )
                if slab is None:
                    manifest.append({"id": bid, "dopant": dop, "hkl": hkl, "status": "no_slab"}); continue

                # STEP-3: 方向规整（尽量 O 在上）
                slab = orient_O_on_top(slab, probe_frac=0.10)

                # STEP-4: 加不对称真空
                slab_v = add_vacuum_asymmetric(slab, vac_top=args.vac_top, vac_bot=args.vac_bot)

                # 可选：删除最顶层一个 O（造氧空位）
                deleted = {}
                if args.rm_top_O:
                    try:
                        slab_v, del_idx, del_fz = remove_topmost_O(slab_v)
                        deleted = {"deleted_O_index": del_idx, "deleted_O_fz": del_fz}
                    except Exception as e:
                        manifest.append({"id": bid, "dopant": dop, "hkl": hkl,
                                         "status": "skip_no_O_for_rm", "reason": str(e)})
                        continue

                # STEP-5: 仅在 a,b 扩胞（z 不扩）
                ia, ib = minimal_xy_supercell(slab_v, a_min=args.a_min, b_min=args.b_min)
                slab_xy = make_supercell_xy(slab_v, ia, ib)

                # 输出
                tag=(f"{bid}_hkl{hkl[0]}{hkl[1]}{hkl[2]}_ucdope_{dop}"
                     f"_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}"
                     f"_a{ia}x_b{ib}x" + ("_rmTopO" if args.rm_top_O else ""))
                outpath=os.path.join(args.outdir, tag+".cif")
                slab_xy.to(fmt="cif", filename=outpath)

                rec={"id": bid, "dopant": dop, "hkl": hkl, "status": "ok", "outfile": outpath,
                     "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
                     "picked_uc": len(picks), "ratio_global": f"{d}:{h}",
                     "metals_in_struct": ",".join(sorted(metals_in))}
                rec.update(deleted)
                manifest.append(rec)
                print("[WROTE]", outpath)

    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir, "manifest.csv"), index=False)
        okcnt=sum(1 for m in manifest if m.get("status")=="ok")
        print(f"[OK] wrote {okcnt} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs")

if __name__ == "__main__":
    main()
