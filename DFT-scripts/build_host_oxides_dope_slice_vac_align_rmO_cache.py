#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Host 金属氧化物 (host–O, 二元) → 元胞全局掺杂(25%) → 切 slab(整层, 多 Miller 可选)
→ 尽量 O 在上 → 加真空(上/下) → 底面拟合对齐到全局 xy 平面 → 仅 a,b 扩胞 → 最后删除顶层 1 个 O
并带本地缓存 (summary + CIF)，避免每次都检索 MP。

宿主元素（host）：Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au
掺杂元素（dopant）：Mg, Ca, Cr, Mn, Fe, Co, Ni, Cu, Zn, Zr, Mo, Ru, Rh, Pd, Ag, Cd, Pt, Au
比例严格按金属子晶格 1:3（掺杂:宿主 = 25%），只改金属位点，不动 O。

依赖：pymatgen, mp-api, numpy, pandas
"""

import os, argparse, math, json, gzip
from typing import List, Tuple, Optional, Dict, Any, Set
import numpy as np
import pandas as pd

from pymatgen.core import Structure, Element, Lattice
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

# ------------------ 宿主/掺杂白名单 ------------------ #
HOST_METALS: Set[str] = set([
    "Ca","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd","Pt","Au"
])
DOPANT_METALS: Set[str] = set([
    "Mg","Ca","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Zr","Mo","Ru","Rh","Pd","Ag","Cd","Pt","Au"
])

# ------------------ 几何/坐标工具 ------------------ #
def _rotation_from_u_to_v(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rodrigues：把向量 u 旋到 v 的 3x3 旋转矩阵。"""
    u = u / np.linalg.norm(u); v = v / np.linalg.norm(v)
    c = np.clip(np.dot(u, v), -1.0, 1.0)
    if np.isclose(c, 1.0):
        return np.eye(3)
    if np.isclose(c, -1.0):
        axis = np.cross(u, np.array([1.0,0.0,0.0]))
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(u, np.array([0.0,1.0,0.0]))
        axis /= np.linalg.norm(axis)
        K = np.array([[0,-axis[2],axis[1]],
                      [axis[2],0,-axis[0]],
                      [-axis[1],axis[0],0]])
        return -np.eye(3) + 2*np.outer(axis, axis)
    axis = np.cross(u, v); s = np.linalg.norm(axis); axis /= s
    K = np.array([[0,-axis[2],axis[1]],
                  [axis[2],0,-axis[0]],
                  [-axis[1],axis[0],0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s**2))

def _fit_plane_normal(points: np.ndarray) -> np.ndarray:
    """SVD 拟合平面法向（指向 +z）。"""
    P = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(P, full_matrices=False)
    n = vh[-1]
    if n[2] < 0: n = -n
    return n / np.linalg.norm(n)

def align_bottom_plane_to_xy(struct: Structure, z_shell: float = 1.0) -> Structure:
    """
    用底层原子拟合平面，将法向旋到全局 +z，并平移使底面 z=0。
    z_shell: 参与拟合的底层厚度(Å)，典型 0.5–1.5。
    """
    s = struct.copy()
    M = s.lattice.matrix
    cart = np.array([site.coords for site in s])
    z = cart[:, 2]; zmin = float(np.min(z))
    sel = (z - zmin) <= float(z_shell)
    pts = cart[sel] if np.count_nonzero(sel) >= 3 else cart
    n = _fit_plane_normal(pts)
    ez = np.array([0.0,0.0,1.0])
    R = _rotation_from_u_to_v(n, ez)
    M_new = (R @ M.T).T
    cart_new = (R @ cart.T).T
    cart_new[:, 2] -= float(np.min(cart_new[:, 2]))  # 让底面 z=0
    return Structure(Lattice(M_new), [site.specie for site in s], cart_new,
                     coords_are_cartesian=True, to_unit_cell=False)

# ------------------ 结构处理工具 ------------------ #
def _unwrap_frac_z(z: np.ndarray) -> np.ndarray:
    if len(z)==0: return z
    zs = np.sort(z); gaps = np.diff(np.r_[zs, zs[0]+1.0])
    k = int(np.argmax(gaps)); base = zs[(k+1)%len(zs)]
    zu = z - base; zu[zu<0] += 1.0
    return zu

def orient_O_on_top(struct: Structure, probe_frac: float = 0.10) -> Structure:
    """统计顶/底若干原子中的 O 含量，必要时整体翻转 z。"""
    s = struct.copy()
    f = np.array([site.frac_coords for site in s]); zu = _unwrap_frac_z(f[:,2])
    N = len(s); cut = max(1, int(np.ceil(N * probe_frac)))
    idx = np.argsort(zu); top = idx[-cut:]; bot = idx[:cut]
    nO_top = sum(1 for i in top if s[i].specie.symbol=="O")
    nO_bot = sum(1 for i in bot if s[i].specie.symbol=="O")
    if nO_top < nO_bot:
        f[:,2] = (1.0 - zu) % 1.0
        return Structure(s.lattice, [site.specie for site in s], f,
                         coords_are_cartesian=False, to_unit_cell=True)
    return s

def add_vacuum_asymmetric(struct: Structure, vac_top: float, vac_bot: float) -> Structure:
    """仅拉长 c，把占据区映射到 [vac_bot, vac_bot+slab]，真空层无原子。"""
    s = struct.copy(); lat0 = s.lattice; Z0 = lat0.c
    f = np.array([site.frac_coords for site in s]); zu = _unwrap_frac_z(f[:,2])
    zmin, zmax = float(np.min(zu)), float(np.max(zu))
    span = max(1e-12, zmax - zmin); slab_A = span * Z0
    new_c = slab_A + float(vac_top) + float(vac_bot); scale = new_c / Z0
    new_lat = Lattice([lat0.matrix[0], lat0.matrix[1], lat0.matrix[2] * scale])
    new_span = slab_A / new_c; new_base = vac_bot / new_c
    f[:,2] = (new_base + (zu - zmin) * (new_span / span)) % 1.0
    return Structure(new_lat, [site.specie for site in s], f,
                     coords_are_cartesian=False, to_unit_cell=True)

def remove_topmost_O(struct: Structure) -> Tuple[Structure,int,float]:
    s = struct.copy()
    fz = _unwrap_frac_z(np.array([site.frac_coords[2] for site in s]))
    o_ids = [i for i,site in enumerate(s.sites) if site.specie.symbol=="O"]
    if not o_ids: raise RuntimeError("No O found.")
    top = max(o_ids, key=lambda i: fz[i]); zval = float(fz[top])
    s.remove_sites([top]); return s, top, zval

def minimal_xy_supercell(struct: Structure, a_min: float, b_min: float) -> Tuple[int,int]:
    lat = struct.lattice
    ia = max(1, math.ceil(a_min / lat.a))
    ib = max(1, math.ceil(b_min / lat.b))
    return ia, ib

def make_supercell_xy(struct: Structure, ia: int, ib: int) -> Structure:
    s = struct.copy(); s.make_supercell([ia, ib, 1]); return s

def is_binary_host_oxide(struct: Structure) -> bool:
    """必须是二元 host–O（只含 O 和一个 host），且 host 在 HOST_METALS。"""
    elems = sorted(set(sp.symbol for sp in struct.composition.elements))
    if len(elems) != 2: return False
    if "O" not in elems: return False
    host = [e for e in elems if e != "O"][0]
    return host in HOST_METALS

def reduce_cell(bulk: Structure, mode: str) -> Structure:
    mode = (mode or "none").lower()
    try:
        if mode=="primitive": return bulk.get_primitive_structure()
        if mode=="symm_primitive":
            sga=SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_primitive_standard_structure()
        if mode=="conventional":
            sga=SpacegroupAnalyzer(bulk, symprec=1e-3, angle_tolerance=5)
            return sga.get_conventional_standard_structure()
    except Exception:
        pass
    return bulk

# ------------------ 掺杂（严格 1:3） ------------------ #
def pick_uc_metal_sites_strict(struct: Structure, dopant: str, frac: float,
                               rng: np.random.RandomState) -> List[int]:
    """
    金属子晶格严格 25%：目标 K_target=round(frac*M_tot)，考虑已有 dopant 数 M_d0。
    """
    metal_ids = [i for i,s in enumerate(struct.sites)
                 if isinstance(s.specie, Element) and s.specie.is_metal and s.specie.symbol!="O"]
    M_tot = len(metal_ids)
    if M_tot == 0: return []

    dop_exist = [i for i in metal_ids if struct[i].specie.symbol==dopant]
    K_target = int(round(frac * M_tot))
    k_needed = K_target - len(dop_exist)
    if k_needed <= 0: return []

    cand = [i for i in metal_ids if struct[i].specie.symbol != dopant]
    if len(cand) < k_needed: return []

    xy = np.array([struct[i].frac_coords[:2] for i in cand])
    if k_needed <= 1 or len(cand) <= 3:
        return sorted(rng.choice(cand, size=k_needed, replace=False).tolist())

    # farthest-point sampling on xy
    center = np.mean(xy, axis=0); d2 = np.sum((xy-center)**2, axis=1)
    seed = int(np.argmax(d2)); chosen=[seed]
    dist = np.linalg.norm(xy-xy[seed], axis=1)
    while len(chosen) < k_needed:
        nxt = int(np.argmax(dist))
        if nxt in chosen:
            remain=[i for i in range(len(cand)) if i not in chosen]
            if not remain: break
            nxt = int(rng.choice(remain))
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(xy-xy[nxt], axis=1))
    return sorted(cand[i] for i in chosen)

def substitute(struct: Structure, idx: List[int], dopant: str) -> Structure:
    s2 = struct.copy()
    for i in idx:
        if s2[i].specie.symbol != dopant:
            s2[i] = Element(dopant)
    return s2

# ------------------ 切 slab ------------------ #
def parse_millers(millers_str: str) -> List[Tuple[int,int,int]]:
    items=[]
    for blk in millers_str.replace("(","").replace(")","").split(";"):
        blk=blk.strip()
        if not blk: continue
        h,k,l=[int(x) for x in blk.split(",")]
        items.append((h,k,l))
    return items

def make_slab_from_bulk(bulk: Structure,
                        hkl: Tuple[int,int,int],
                        min_slab: float,
                        prefer_O_top: bool=True) -> Optional[Structure]:
    gen = SlabGenerator(
        initial_structure=bulk,
        miller_index=hkl,
        min_slab_size=min_slab,
        min_vacuum_size=0.1,       # 真空后加
        center_slab=False,
        in_unit_planes=True,
        primitive=True,
        reorient_lattice=True,
    )
    slabs = gen.get_slabs(ftol=0.1)
    if not slabs: return None
    if not prefer_O_top: return slabs[0]

    def score(slab):
        f=np.array([site.frac_coords for site in slab]); zu=_unwrap_frac_z(f[:,2])
        idx=np.argsort(zu); cut=max(1,int(np.ceil(len(slab)*0.1)))
        top=idx[-cut:]; ofrac=sum(1 for i in top if slab[i].specie.symbol=="O")/cut
        return -ofrac
    slabs.sort(key=score)
    return slabs[0]

# ------------------ MP 缓存拉取 ------------------ #
def load_or_fetch_host_binaries(api_key: str,
                                hosts: List[str],
                                per_host_max: int,
                                cache_dir: str,
                                total_max: int) -> List[Dict[str, Any]]:
    """
    对每个 host 检索 chemsys='host-O' 的二元氧化物 summary，
    缓存为 summary_{host}-O.json.gz，并把结构 CIF 缓存到 cache_cifs/ 下。
    返回 [{'material_id','cif','formula','host'}]。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cif_dir = os.path.join(cache_dir, "cache_cifs")
    os.makedirs(cif_dir, exist_ok=True)

    out: List[Dict[str, Any]] = []
    with MPRester(api_key) as mpr:
        for host in hosts:
            cache_file = os.path.join(cache_dir, f"summary_{host}-O.json.gz")
            entries = None
            if os.path.exists(cache_file):
                print(f"[CACHE] 使用 {cache_file}")
                with gzip.open(cache_file, "rt", encoding="utf-8") as f:
                    entries = json.load(f)
            if entries is None:
                print(f"[FETCH] 从 MP 拉取 {host}-O …")
                docs = list(mpr.summary.search(
                    chemsys=[f"{host}-O"],
                    fields=["material_id","structure","formula_pretty"],
                    chunk_size=200
                ))
                entries=[]
                for ddoc in docs[:per_host_max]:
                    mid = ddoc.material_id
                    cif_path = os.path.join(cif_dir, f"{mid}.cif")
                    if not os.path.exists(cif_path):
                        try:
                            ddoc.structure.to(fmt="cif", filename=cif_path)
                        except Exception:
                            continue
                    entries.append({"material_id": mid,
                                    "cif": cif_path,
                                    "formula": getattr(ddoc,"formula_pretty",""),
                                    "host": host})
                with gzip.open(cache_file, "wt", encoding="utf-8") as f:
                    json.dump(entries, f)
            out.extend(entries)
            if len(out) >= total_max:
                break
    return out[:total_max]

# ------------------ 主程序 ------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out_host_doped_slabs")

    # MP & 缓存
    ap.add_argument("--mapi_key", type=str, default=None)
    ap.add_argument("--cache_dir", type=str, default="mp_cache")
    ap.add_argument("--per_host_max", type=int, default=10)
    ap.add_argument("--total_max", type=int, default=60)

    # 掺杂
    ap.add_argument("--dopants", type=str,
        default="Mg,Ca,Cr,Mn,Fe,Co,Ni,Cu,Zn,Zr,Mo,Ru,Rh,Pd,Ag,Cd,Pt,Au")
    ap.add_argument("--dopant_to_host", type=str, default="1:3")
    ap.add_argument("--seed", type=int, default=0)

    # 减胞/切片
    ap.add_argument("--reduce", type=str, default="symm_primitive",
        choices=["none","primitive","symm_primitive","conventional"])
    ap.add_argument("--millers", type=str, default="1,1,0;1,0,1;0,1,1")
    ap.add_argument("--min_slab", type=float, default=10.0)
    ap.add_argument("--prefer_O_top", action="store_true")

    # 真空/扩胞/删O
    ap.add_argument("--vac_top", type=float, default=20.0)
    ap.add_argument("--vac_bot", type=float, default=3.0)
    ap.add_argument("--a_min", type=float, default=10.0)
    ap.add_argument("--b_min", type=float, default=10.0)
    ap.add_argument("--rm_top_O", action="store_true")

    # 对齐底面
    ap.add_argument("--z_shell", type=float, default=1.0, help="拟合底面的厚度(Å)")

    # 可选限量控制
    ap.add_argument("--sample_combos", type=int, default=0,
        help="对(结构,掺杂,Miller)组合随机抽样，仅取前 K 个；0 表示不启用")
    ap.add_argument("--limit_outputs", type=int, default=0,
        help="最多写出多少个 CIF；0 表示不限")
    ap.add_argument("--final_atoms_min", type=int, default=0)
    ap.add_argument("--final_atoms_max", type=int, default=0)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    # 掺杂元素表
    req_dops=[x.strip() for x in args.dopants.split(",") if x.strip()]
    dopant_list = sorted(list(DOPANT_METALS.intersection(req_dops)))
    if not dopant_list:
        raise RuntimeError("掺杂元素列表为空或不在白名单。")

    # 比例（严格 25%）
    d, h = [int(x) for x in args.dopant_to_host.split(":")]
    assert d>0 and h>0
    frac = d/(d+h)

    # 读取/拉取 host–O 候选（带缓存）
    api_key = args.mapi_key or os.environ.get("MAPI_KEY")
    if not api_key:
        raise RuntimeError("Missing MAPI_KEY")
    host_list = sorted(list(HOST_METALS))
    summary_entries = load_or_fetch_host_binaries(
        api_key, host_list, args.per_host_max, args.cache_dir, args.total_max
    )
    if not summary_entries:
        print("[WARN] 无候选，退出。"); return

    millers = parse_millers(args.millers)

    # 组合级抽样（限量）
    combos = []
    for ent in summary_entries:
        for dop in dopant_list:
            for hkl in millers:
                combos.append((ent, dop, hkl))
    rng.shuffle(combos)
    if args.sample_combos and args.sample_combos > 0:
        combos = combos[:args.sample_combos]

    manifest: List[Dict[str,Any]] = []
    ok_written = 0

    for (ent, dop, hkl) in combos:
        if args.limit_outputs and ok_written >= args.limit_outputs:
            print(f"[STOP] 达到上限 limit_outputs={args.limit_outputs}")
            break

        mid = ent["material_id"]; cif_path = ent["cif"]; host = ent["host"]
        try:
            bulk0 = Structure.from_file(cif_path)
        except Exception as e:
            print(f"[SKIP] 读取失败 {mid}: {e}"); continue

        if not is_binary_host_oxide(bulk0):
            manifest.append({"id": mid, "status": "skip_not_binary_host_oxide"})
            continue

        bulk = reduce_cell(bulk0, args.reduce)

        # STEP-1：元胞 1:3 掺杂（只在金属子晶格）
        picks = pick_uc_metal_sites_strict(bulk, dopant=dop, frac=frac, rng=rng)
        if not picks:
            manifest.append({"id": mid, "host": host, "dopant": dop,
                             "status": "skip_no_host_or_ratio_unreachable"})
            continue
        bulk_doped = substitute(bulk, picks, dop)

        # STEP-2：切 slab（整层，不截原子）
        slab = make_slab_from_bulk(bulk_doped, hkl, min_slab=args.min_slab,
                                   prefer_O_top=args.prefer_O_top)
        if slab is None:
            manifest.append({"id": mid, "host": host, "dopant": dop,
                             "hkl": hkl, "status": "no_slab"})
            continue

        # STEP-3：尽量 O 在上
        slab = orient_O_on_top(slab, probe_frac=0.10)

        # STEP-4：加真空（上/下）
        slab_v = add_vacuum_asymmetric(slab, vac_top=args.vac_top, vac_bot=args.vac_bot)

        # STEP-5：底面拟合对齐到 xy（关键，保证底面||xy）
        slab_v = align_bottom_plane_to_xy(slab_v, z_shell=args.z_shell)

        # STEP-6：仅在 a,b 扩胞（z 不扩）
        ia, ib = minimal_xy_supercell(slab_v, a_min=args.a_min, b_min=args.b_min)
        slab_xy = make_supercell_xy(slab_v, ia, ib)

        # STEP-7：最后删除最顶层 1 个 O（制造表面氧空位）
        deleted={}
        if args.rm_top_O:
            try:
                slab_xy, del_idx, del_fz = remove_topmost_O(slab_xy)
                # （可选）删 O 后再把底面严格贴回 z=0：
                slab_xy = align_bottom_plane_to_xy(slab_xy, z_shell=args.z_shell)
                deleted={"deleted_O_index": del_idx, "deleted_O_fz": del_fz}
            except Exception as e:
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_no_O_for_rm", "reason": str(e)})
                continue

        # 原子数范围限量（可选）
        if args.final_atoms_min or args.final_atoms_max:
            nat = len(slab_xy)
            if (args.final_atoms_min and nat < args.final_atoms_min) or \
               (args.final_atoms_max and nat > args.final_atoms_max):
                manifest.append({"id": mid, "host": host, "dopant": dop, "hkl": hkl,
                                 "status": "skip_atoms_out_of_range", "nat": nat})
                continue

        # 输出
        tag = (f"{mid}_host{host}_hkl{hkl[0]}{hkl[1]}{hkl[2]}_ucGLOBALdope_{dop}"
               f"_vacTop{int(args.vac_top)}_vacBot{int(args.vac_bot)}"
               f"_a{ia}x_b{ib}x" + ("_rmTopO" if args.rm_top_O else ""))
        outpath = os.path.join(args.outdir, tag + ".cif")
        slab_xy.to(fmt="cif", filename=outpath)

        rec = {"id": mid, "status": "ok", "outfile": outpath,
               "host": host, "dopant": dop, "hkl": hkl,
               "ia": ia, "ib": ib, "vac_top": args.vac_top, "vac_bot": args.vac_bot,
               "picked_uc": len(picks), "ratio_global": f"{d}:{h}"}
        rec.update(deleted)
        manifest.append(rec)
        ok_written += 1
        print("[WROTE]", outpath)

    if manifest:
        pd.DataFrame(manifest).to_csv(os.path.join(args.outdir,"manifest.csv"), index=False)
        print(f"[OK] wrote {ok_written} structures; total {len(manifest)}")
    else:
        print("[WARN] no outputs")

if __name__ == "__main__":
    main()
