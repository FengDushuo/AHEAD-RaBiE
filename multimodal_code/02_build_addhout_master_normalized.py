#!/usr/bin/env python3
from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:
    raise SystemExit(f"Please install pymatgen first: {e}")


EH_DEFAULT = -0.0565


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Extracted addH-out directory")
    ap.add_argument("--excel-path", default=None, help="Optional xlsx with Etotal/Eslab/H吸附能")
    ap.add_argument("--output-csv", default="addH_out_master_normalized.csv")
    ap.add_argument("--adsorbate", default="H")
    ap.add_argument("--miller", default="(? ? ?)", help="Default/fallback Miller text")
    ap.add_argument(
        "--miller-map",
        default="",
        help="Per-material Miller mapping, e.g. 'CeO2=111,ZnO=100' or 'CeO2=(1 1 1),ZnO=(1 0 0)'.",
    )
    ap.add_argument("--eh-ref", type=float, default=EH_DEFAULT, help="Reference H energy used in target = delta_e_raw - eh_ref")
    ap.add_argument(
        "--target-source",
        default="excel_preferred",
        choices=["excel_preferred", "computed_preferred", "excel_only", "computed_only"],
        help="How to determine target. computed uses energy_total - energy_slab - eh_ref.",
    )
    ap.add_argument(
        "--consistency-tol",
        type=float,
        default=1e-3,
        help="Warn if excel target and computed target differ more than this tolerance.",
    )
    ap.add_argument(
        "--write-bare-from-addh",
        action="store_true",
        help="Write a temporary bare slab CONTCAR by removing the adsorbate atom from each addH-out structure.",
    )
    ap.add_argument(
        "--bare-output-dir",
        default=None,
        help="Directory for generated bare_from_addH CONTCAR files. Used only with --write-bare-from-addh.",
    )
    return ap.parse_args()


def _normalize_miller_text(x: Optional[str]) -> str:
    if x is None:
        return "(? ? ?)"
    s = str(x).strip()
    if not s:
        return "(? ? ?)"
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"\s+", " ", s)
    if re.fullmatch(r"\d{3}", s):
        return f"({s[0]} {s[1]} {s[2]})"
    m = re.fullmatch(r"\(?\s*(\d)\s+(\d)\s+(\d)\s*\)?", s)
    if m:
        return f"({m.group(1)} {m.group(2)} {m.group(3)})"
    return s


def _miller_key(miller_text: str) -> str:
    s = _normalize_miller_text(miller_text)
    m = re.fullmatch(r"\((\d)\s+(\d)\s+(\d)\)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"
    return "unknown"


def parse_miller_map(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --miller-map item: {item!r}. Expected material=miller")
        material, miller = item.split("=", 1)
        material = material.strip()
        miller = miller.strip()
        if not material:
            raise ValueError(f"Invalid --miller-map item with empty material: {item!r}")
        out[material] = _normalize_miller_text(miller)
    return out


def resolve_miller(material: str, default_miller: str, miller_map: Dict[str, str]) -> str:
    if material in miller_map:
        return miller_map[material]
    return _normalize_miller_text(default_miller)


def parse_excel(excel_path: Optional[Path]) -> Dict[Tuple[str, int], Dict[str, object]]:
    if excel_path is None or not excel_path.exists():
        return {}
    raw = pd.read_excel(excel_path)
    if raw.empty:
        return {}
    data = raw.iloc[1:].reset_index(drop=True)
    out = {}
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
    pat = re.compile(r"F=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)\s+E0=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)")
    e0 = f = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                f = float(m.group(1).replace("D", "E"))
                e0 = float(m.group(2).replace("D", "E"))
    return e0, f


def parse_vasprun_energy_and_convergence(path: Optional[Path]):
    if path is None or not path.exists():
        return None, None, None
    try:
        vr = Vasprun(str(path), parse_dos=False, parse_eigen=False, parse_projected_eigen=False)
        energy = float(vr.final_energy) if getattr(vr, "final_energy", None) is not None else None
        return energy, getattr(vr, "converged", None), getattr(vr, "final_structure", None)
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




def write_bare_structure_from_addh(
    structure: Structure,
    out_path: Path,
    ads_symbol: str = "H",
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """Remove the topmost adsorbate atom and write a VASP POSCAR/CONTCAR file."""
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        return None, None, f"No adsorbate species {ads_symbol} found for bare writing"
    bare = structure.copy()
    remove_idx = int(ads_indices[0])
    bare.remove_sites([remove_idx])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(bare).write_file(str(out_path))
    return str(out_path), remove_idx, None

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
    rows = []
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
    dists = []
    for j, site in enumerate(structure):
        if j == center_idx:
            continue
        dists.append((float(structure.get_distance(center_idx, j)), str(site.specie)))
    dists.sort(key=lambda x: x[0])
    toks = [str(structure[center_idx].specie)]
    toks.extend([sp for _, sp in dists[: max(env_size - 1, 0)]])
    return toks[:env_size]


def serialize_sample_text(
    structure: Structure,
    ads_symbol: str = "H",
    miller: str = "(? ? ?)",
    env_size: int = 8,
    anchor_shell_tol: float = 0.45,
    anchor_max_dist: float = 3.0,
):
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No {ads_symbol} found")
    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    formula = build_formula_excluding_adsorbate(structure, ads_indices)
    rows = distance_matrix_to_adsorbate(structure, ads_indices, slab_indices)
    nearest = rows[0][2]
    anchors = [(ai, si, d) for ai, si, d in rows if d <= min(anchor_max_dist, nearest + anchor_shell_tol)]
    if not anchors:
        anchors = [rows[0]]
    anchor_sites = []
    for _, si, _ in anchors:
        if si not in anchor_sites:
            anchor_sites.append(si)
    site_type = classify_site(len(anchor_sites))
    env_chunks = []
    for si in anchor_sites:
        toks = local_env_tokens(structure, si, env_size)
        if ads_symbol not in toks:
            toks.append(ads_symbol)
        env_chunks.append("[" + " ".join(toks) + "]")
    lead = [ads_symbol] + [str(structure[i].specie) for i in anchor_sites] + [site_type]
    text = f"{ads_symbol}</s>{formula} {miller}</s>[{' '.join(lead + env_chunks)}]"
    return text, formula, site_type, len(anchor_sites)


def choose_target(h_ads_excel, target_computed, mode: str):
    if mode == "excel_preferred":
        return h_ads_excel if h_ads_excel is not None and not math.isnan(h_ads_excel) else target_computed
    if mode == "computed_preferred":
        return target_computed if target_computed is not None and not math.isnan(target_computed) else h_ads_excel
    if mode == "excel_only":
        return h_ads_excel
    if mode == "computed_only":
        return target_computed
    raise ValueError(f"Unsupported target-source: {mode}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    excel = Path(args.excel_path).resolve() if args.excel_path else (input_dir / "氢吸附能.xlsx")
    bare_output_dir = Path(args.bare_output_dir).resolve() if args.bare_output_dir else Path(str(args.output_csv) + ".bare_from_addH").resolve()
    excel_map = parse_excel(excel)
    miller_map = parse_miller_map(args.miller_map)

    rows = []
    mismatch_rows = 0

    for mat_dir in sorted([p for p in input_dir.iterdir() if p.is_dir()]):
        mat = mat_dir.name
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

            miller_text = resolve_miller(mat, args.miller, miller_map)
            miller_key = _miller_key(miller_text)

            text = formula = site_type = None
            anchor_count = None
            parse_ok = False
            notes = ""
            bare_contcar_path = None
            bare_from_addh_removed_index = None

            osz_e0, osz_f = parse_oszicar(oszicar)
            vas_e, conv, vr_struct = parse_vasprun_energy_and_convergence(vasprun)

            try:
                struct = load_structure(contcar, vr_struct)
                if struct is None:
                    raise ValueError("No parsable structure")
                text, formula, site_type, anchor_count = serialize_sample_text(
                    struct,
                    ads_symbol=args.adsorbate,
                    miller=miller_text,
                )
                if args.write_bare_from_addh:
                    bare_out = bare_output_dir / mat / f"{idx}-bare-CONTCAR"
                    bare_contcar_path, bare_from_addh_removed_index, bare_note = write_bare_structure_from_addh(
                        struct, bare_out, ads_symbol=args.adsorbate
                    )
                    if bare_note:
                        notes = f"{notes}; {bare_note}" if notes else bare_note
                parse_ok = True
            except Exception as e:
                notes = str(e)

            energy_addh = x.get("energy_total")
            energy_bare = x.get("energy_slab")
            h_ads_excel = x.get("h_ads_excel")

            delta_e_raw = None
            if energy_addh is not None and energy_bare is not None:
                delta_e_raw = float(energy_addh) - float(energy_bare)

            target_computed = None if delta_e_raw is None else delta_e_raw - float(args.eh_ref)
            target = choose_target(h_ads_excel, target_computed, args.target_source)

            target_mismatch = None
            if h_ads_excel is not None and target_computed is not None:
                target_mismatch = float(h_ads_excel) - float(target_computed)
                if abs(target_mismatch) > float(args.consistency_tol):
                    mismatch_rows += 1
                    notes = f"{notes}; target_mismatch={target_mismatch:.6g}" if notes else f"target_mismatch={target_mismatch:.6g}"

            # normalize toward training schema
            family_base = mat
            family_base_miller = f"{mat}-{miller_key}" if miller_key != "unknown" else mat
            dopant = elem
            data_source = "addH_out"
            status_addH = "YES" if energy_addh is not None else None
            status_bare = "YES" if energy_bare is not None else None

            rows.append(
                {
                    # training-like schema first
                    "anchor_count": anchor_count,
                    "contcar_path": str(contcar) if contcar.exists() else None,
                    "bare_contcar_path": bare_contcar_path,
                    "bare_from_addh_removed_index": bare_from_addh_removed_index,
                    "converged": conv,
                    "data_source": data_source,
                    "delta_e_raw": delta_e_raw,
                    "dopant": dopant,
                    "eh_ref": float(args.eh_ref),
                    "energy_addH": energy_addh,
                    "energy_bare": energy_bare,
                    "family_base": family_base,
                    "family_base_miller": family_base_miller,
                    "id": sid,
                    "miller": miller_text,
                    "notes": notes if notes else None,
                    "oszicar_e0": osz_e0,
                    "oszicar_f": osz_f,
                    "oszicar_path": str(oszicar) if oszicar.exists() else None,
                    "outlier_flag_target": False,
                    "outlier_reason_target": None,
                    "parse_ok": parse_ok,
                    "site_type": site_type,
                    "slab_formula": formula,
                    "status_addH": status_addH,
                    "status_bare": status_bare,
                    "target": target,
                    "text": text,
                    "vasprun_energy": vas_e,
                    "vasprun_path": str(vasprun) if vasprun.exists() else None,

                    # keep addH-out legacy columns too
                    "material": mat,
                    "idx": idx,
                    "element": elem,
                    "energy_total": energy_addh,
                    "energy_slab_legacy": energy_bare,
                    "h_ads_excel": h_ads_excel,
                    "cif_path": str(cif) if cif.exists() else None,
                    "target_computed": target_computed,
                    "target_mismatch_excel_minus_computed": target_mismatch,
                }
            )

    df = pd.DataFrame(rows)

    # Match training table column order first, then append legacy columns.
    preferred_order = [
        "anchor_count", "contcar_path", "bare_contcar_path", "bare_from_addh_removed_index", "converged", "data_source", "delta_e_raw", "dopant",
        "eh_ref", "energy_addH", "energy_bare", "family_base", "family_base_miller", "id",
        "miller", "notes", "oszicar_e0", "oszicar_f", "oszicar_path", "outlier_flag_target",
        "outlier_reason_target", "parse_ok", "site_type", "slab_formula", "status_addH",
        "status_bare", "target", "text", "vasprun_energy", "vasprun_path",
        "material", "idx", "element", "energy_total", "energy_slab_legacy", "h_ads_excel",
        "cif_path", "target_computed", "target_mismatch_excel_minus_computed",
    ]
    ordered_cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[ordered_cols]

    df.to_csv(args.output_csv, index=False)
    print(f"[OK] saved -> {args.output_csv}")
    print(f"[INFO] rows={len(df)} parse_ok={int(df['parse_ok'].fillna(False).sum())}")
    print(f"[INFO] target non-null={int(df['target'].notna().sum())}")
    if "bare_contcar_path" in df.columns:
        print(f"[INFO] bare_contcar_nonnull={int(df['bare_contcar_path'].notna().sum())}")
    print(f"[INFO] family_base_miller values={sorted(df['family_base_miller'].dropna().astype(str).unique().tolist())}")
    print(f"[INFO] target mismatch rows over tol={mismatch_rows}")


if __name__ == "__main__":
    main()
