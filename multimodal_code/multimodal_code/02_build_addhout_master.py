#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:
    raise SystemExit(f"Please install pymatgen first: {e}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Extracted addH-out directory")
    ap.add_argument("--excel-path", default=None, help="Optional xlsx with Etotal/Eslab/H吸附能")
    ap.add_argument("--output-csv", default="addH_out_master.csv")
    ap.add_argument("--adsorbate", default="H")
    ap.add_argument("--miller", default="(? ? ?)")
    return ap.parse_args()

def parse_excel(excel_path: Optional[Path]) -> Dict[Tuple[str, int], Dict[str, object]]:
    if excel_path is None or not excel_path.exists():
        return {}
    raw = pd.read_excel(excel_path)
    # expected shape from uploaded sheet: row0 headers and two 4-col blocks for CeO2 / ZnO
    if raw.empty:
        return {}
    header = raw.iloc[0].tolist()
    data = raw.iloc[1:].reset_index(drop=True)
    out = {}
    # columns 0,1 + 2,3,4 + 5,6,7 ; col8 may be EH note
    for mat, start in [("CeO2", 2), ("ZnO", 5)]:
        for _, row in data.iterrows():
            try:
                idx = int(row.iloc[0])
            except Exception:
                continue
            elem = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
            Et = row.iloc[start] if start < len(row) else None
            Es = row.iloc[start + 1] if start + 1 < len(row) else None
            Hads = row.iloc[start + 2] if start + 2 < len(row) else None
            out[(mat, idx)] = {
                "element": elem,
                "energy_total": float(Et) if pd.notna(Et) else None,
                "energy_slab": float(Es) if pd.notna(Es) else None,
                "h_ads_excel": float(Hads) if pd.notna(Hads) else None,
            }
    return out

def parse_oszicar(path: Optional[Path]):
    if path is None or not path.exists():
        return None, None
    import re
    pat = re.compile(r"F=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)\s+E0=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)")
    e0=f=None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                f = float(m.group(1).replace('D','E'))
                e0 = float(m.group(2).replace('D','E'))
    return e0, f

def parse_vasprun_energy_and_convergence(path: Optional[Path]):
    if path is None or not path.exists():
        return None, None, None
    try:
        vr = Vasprun(str(path), parse_dos=False, parse_eigen=False, parse_projected_eigen=False)
        energy = float(vr.final_energy) if getattr(vr, 'final_energy', None) is not None else None
        return energy, getattr(vr, 'converged', None), getattr(vr, 'final_structure', None)
    except Exception:
        return None, None, None

def load_structure(contcar_path: Optional[Path], fallback_structure: Optional[Structure]) -> Optional[Structure]:
    if contcar_path is not None and contcar_path.exists():
        try:
            return Poscar.from_file(str(contcar_path)).structure
        except Exception:
            pass
    return fallback_structure

def choose_adsorbate_indices(structure: Structure, ads_symbol: str) -> List[int]:
    idx = [i for i, s in enumerate(structure) if str(s.specie) == ads_symbol]
    if not idx:
        return []
    z_cart = [(i, structure[i].coords[2]) for i in idx]
    z_cart.sort(key=lambda x: x[1], reverse=True)
    return [z_cart[0][0]]

def build_formula_excluding_adsorbate(structure: Structure, ads_indices: Sequence[int]) -> str:
    ads_set = set(ads_indices)
    order_seen = []
    count_map = {}
    for i, site in enumerate(structure):
        if i in ads_set:
            continue
        sp = str(site.specie)
        if sp not in count_map:
            order_seen.append(sp)
            count_map[sp] = 0
        count_map[sp] += 1
    return "".join(f"{sp}{count_map[sp]}" for sp in order_seen if count_map[sp] > 0)

def distance_matrix_to_adsorbate(structure: Structure, ads_indices: Sequence[int], slab_indices: Sequence[int]):
    rows=[]
    for ai in ads_indices:
        for si in slab_indices:
            rows.append((ai, si, float(structure.get_distance(ai, si))))
    rows.sort(key=lambda x: x[2])
    return rows

def classify_site(anchor_count: int) -> str:
    if anchor_count <= 1:
        return "atop"
    if anchor_count == 2:
        return "bridge"
    return "hollow"

def local_env_tokens(structure: Structure, center_idx: int, env_size: int):
    dists=[]
    for j,site in enumerate(structure):
        if j == center_idx:
            continue
        dists.append((float(structure.get_distance(center_idx, j)), str(site.specie)))
    dists.sort(key=lambda x:x[0])
    toks=[str(structure[center_idx].specie)]
    toks.extend([sp for _,sp in dists[:max(env_size-1,0)]])
    return toks[:env_size]

def serialize_sample_text(structure: Structure, sample_id: str, ads_symbol="H", miller="(? ? ?)", env_size=8,
                          anchor_shell_tol=0.45, anchor_max_dist=3.0):
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No {ads_symbol} found")
    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    formula = build_formula_excluding_adsorbate(structure, ads_indices)
    rows = distance_matrix_to_adsorbate(structure, ads_indices, slab_indices)
    nearest = rows[0][2]
    anchors = [(ai,si,d) for ai,si,d in rows if d <= min(anchor_max_dist, nearest + anchor_shell_tol)]
    if not anchors:
        anchors = [rows[0]]
    anchor_sites=[]
    for _,si,_ in anchors:
        if si not in anchor_sites:
            anchor_sites.append(si)
    site_type = classify_site(len(anchor_sites))
    env_chunks=[]
    for si in anchor_sites:
        toks = local_env_tokens(structure, si, env_size)
        if ads_symbol not in toks:
            toks.append(ads_symbol)
        env_chunks.append("[" + " ".join(toks) + "]")
    lead=[ads_symbol] + [str(structure[i].specie) for i in anchor_sites] + [site_type]
    text = f"{ads_symbol}</s>{formula} {miller}</s>[{' '.join(lead + env_chunks)}]"
    return text, formula, site_type, len(anchor_sites)

def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    excel = Path(args.excel_path).resolve() if args.excel_path else (input_dir / "氢吸附能.xlsx")
    excel_map = parse_excel(excel)
    rows=[]
    for mat_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        mat = mat_dir.name
        # support both mat_dir/ and mat_dir/out/
        scan_dir = mat_dir / "out" if (mat_dir / "out").exists() else mat_dir
        contcars = sorted(scan_dir.glob("*-CONTCAR"), key=lambda p: int(p.name.split("-")[0]))
        for contcar in contcars:
            idx = int(contcar.name.split("-")[0])
            vasprun = scan_dir / f"{idx}-vasprun.xml"
            oszicar = scan_dir / f"{idx}-OSZICAR"
            cif = scan_dir / f"{idx}-out.cif"
            x = excel_map.get((mat, idx), {})
            elem = x.get("element")
            sid = f"{mat}-{idx}-{elem}" if elem else f"{mat}-{idx}"
            text=formula=site_type=None; anchor_count=None; parse_ok=False; notes=""
            osz_e0, osz_f = parse_oszicar(oszicar)
            vas_e, conv, vr_struct = parse_vasprun_energy_and_convergence(vasprun)
            try:
                struct = load_structure(contcar, vr_struct)
                if struct is None:
                    raise ValueError("No parsable structure")
                text, formula, site_type, anchor_count = serialize_sample_text(struct, sid, ads_symbol=args.adsorbate, miller=args.miller)
                parse_ok=True
            except Exception as e:
                notes=str(e)
            rows.append({
                "id": sid,
                "material": mat,
                "idx": idx,
                "element": elem,
                "energy_total": x.get("energy_total"),
                "energy_slab": x.get("energy_slab"),
                "h_ads_excel": x.get("h_ads_excel"),
                "contcar_path": str(contcar) if contcar.exists() else None,
                "vasprun_path": str(vasprun) if vasprun.exists() else None,
                "oszicar_path": str(oszicar) if oszicar.exists() else None,
                "cif_path": str(cif) if cif.exists() else None,
                "oszicar_e0": osz_e0,
                "oszicar_f": osz_f,
                "vasprun_energy": vas_e,
                "converged": conv,
                "slab_formula": formula,
                "miller": args.miller,
                "site_type": site_type,
                "anchor_count": anchor_count,
                "text": text,
                "parse_ok": parse_ok,
                "notes": notes,
            })
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(f"[OK] saved -> {args.output_csv}")
    print(f"[INFO] rows={len(df)} parse_ok={int(df['parse_ok'].fillna(False).sum())}")

if __name__ == '__main__':
    main()
