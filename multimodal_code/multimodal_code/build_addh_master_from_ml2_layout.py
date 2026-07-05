#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build addH master table from the new layout:

addH-2/
  ML2.xlsx
  out/
    2542-Ag-CONTCAR
    2542-Ag-OSZICAR
    2542-Ag-vasprun.xml
    ...

This script converts the new layout to the same training-ready CSV style used
previously for multi-view fine-tuning.

Key assumptions
---------------
1) The structure used for model input is the addH structure in out/*-CONTCAR
2) The label is:
       target = energy_addH - energy_bare - eh_ref
3) energy_bare / energy_addH come from the Excel table (ML2.xlsx)
4) sample id looks like:
       <family_base>-<dopant>
   e.g. 2542-Ag, 2858-Co, 643-Zr
5) Miller index is optional. If you know it, pass:
       --base-miller-map "2542=100,2858=111,643=111"
   Otherwise the text uses "(? ? ?)".

Example
-------
python build_addh_master_from_ml2_layout.py \
  --input-root addH-2 \
  --output-csv addH_master_ml2.csv \
  --base-miller-map "2542=100,2858=111,643=111"

Then optionally merge with old training data:
python merge_addh_master_tables.py \
  --old-csv addH_master_drop8.csv \
  --new-csv addH_master_ml2.csv \
  --output-csv addH_master_merged_raw.csv
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:
    raise SystemExit(f"Please install pymatgen first: {e}")

EH_DEFAULT = -0.0565
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


@dataclass
class Rec:
    sample_id: str
    energy_bare: Optional[float]
    energy_addH: Optional[float]
    contcar_path: Optional[str] = None
    vasprun_path: Optional[str] = None
    oszicar_path: Optional[str] = None
    oszicar_e0: Optional[float] = None
    oszicar_f: Optional[float] = None
    vasprun_energy: Optional[float] = None
    converged: Optional[bool] = None
    text: Optional[str] = None
    slab_formula: Optional[str] = None
    miller: Optional[str] = None
    site_type: Optional[str] = None
    anchor_count: Optional[int] = None
    parse_ok: bool = False
    notes: str = ""


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Root like addH-2/")
    ap.add_argument("--xlsx", default=None, help="Excel file path; default <input-root>/ML2.xlsx")
    ap.add_argument("--out-dir", default=None, help="Directory holding addH *-CONTCAR/*-OSZICAR/*-vasprun.xml; default <input-root>/out")
    ap.add_argument("--bare-dir", default=None, help="Directory holding bare-slab *-CONTCAR/*-OSZICAR/*-vasprun.xml; default <input-root>/withoutH if it exists")
    ap.add_argument("--output-csv", default="addH_master_ml2.csv")
    ap.add_argument("--eh-ref", type=float, default=EH_DEFAULT)
    ap.add_argument("--adsorbate", default="H")
    ap.add_argument("--base-miller-map", default="", help='Optional mapping like "2542=100,2858=111,643=111"')
    ap.add_argument("--strict", action="store_true")

    # optional outlier handling, same spirit as your previous script
    ap.add_argument("--outlier-method", default="none", choices=["none", "iqr", "id_list"])
    ap.add_argument("--outlier-action", default="flag", choices=["flag", "drop"])
    ap.add_argument("--iqr-multiplier", type=float, default=3.0)
    ap.add_argument("--drop-ids", default="")
    ap.add_argument("--outlier-report-csv", default=None)
    ap.add_argument("--outlier-summary-json", default=None)
    return ap.parse_args()


def _safe_float_fortran(s: str) -> Optional[float]:
    try:
        return float(str(s).replace("D", "E"))
    except Exception:
        return None


def _split_drop_ids(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_key_value_map(raw: str) -> Dict[str, str]:
    out = {}
    if not raw:
        return out
    for item in raw.split(","):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out




def _find_ml2_file(base_dir: Optional[Path], sample_id: str, kind: str) -> Optional[Path]:
    """Find <sample_id>-<kind> in a flat or mildly nested addH-2 style directory."""
    if base_dir is None or not base_dir.exists():
        return None
    direct = base_dir / f"{sample_id}-{kind}"
    if direct.exists():
        return direct
    cands = sorted(base_dir.rglob(f"{sample_id}-{kind}"))
    if cands:
        return cands[0]
    # permissive fallback, useful when files are nested by sample id
    cands = sorted([q for q in base_dir.rglob(kind) if sample_id in str(q)])
    return cands[0] if cands else None


def _make_family_base_miller(family_base: Optional[str], miller_raw: Optional[str]) -> Optional[str]:
    if family_base and miller_raw:
        return f"{family_base}-{miller_raw}"
    return family_base if family_base else None

def parse_sample_id(sample_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    New layout is usually:
      2542-Ag
      2858-Co
      643-Zr

    Return:
      family_base, miller_raw(optional), dopant
    """
    parts = [p for p in str(sample_id).split("-") if p]
    if len(parts) >= 2:
        family_base = parts[0]
        dopant = "-".join(parts[1:])
        return family_base, None, dopant
    if len(parts) == 1:
        return parts[0], None, None
    return None, None, None


def miller_to_string(miller: Optional[str]) -> str:
    if not miller:
        return "(? ? ?)"
    miller = str(miller).strip()
    if len(miller) == 3 and miller.isdigit():
        return f"({miller[0]} {miller[1]} {miller[2]})"
    return "(? ? ?)"


def parse_excel_energy_table(xlsx_path: Path) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    """
    Expected layout in ML2.xlsx:
      column B: sample id
      column C: surface (bare)
      column D: addH
    """
    df = pd.read_excel(xlsx_path, engine="openpyxl", header=None)

    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for _, row in df.iterrows():
        sid = row.iloc[1] if len(row) > 1 else None
        e_bare = row.iloc[2] if len(row) > 2 else None
        e_addh = row.iloc[3] if len(row) > 3 else None

        if pd.isna(sid):
            continue
        sid = str(sid).strip()
        if not sid or sid.lower() in {"surface", "addh"}:
            continue

        bare_val = None if pd.isna(e_bare) else _safe_float_fortran(e_bare)
        addh_val = None if pd.isna(e_addh) else _safe_float_fortran(e_addh)
        out[sid] = (bare_val, addh_val)

    if not out:
        raise ValueError(f"No sample energies parsed from {xlsx_path}")
    return out


def parse_oszicar(path: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if path is None or not path.exists():
        return None, None
    pat = re.compile(r"F=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)\s+E0=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)")
    e0 = None
    f = None
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                f = _safe_float_fortran(m.group(1))
                e0 = _safe_float_fortran(m.group(2))
    return e0, f


def parse_vasprun_energy_and_convergence(path: Optional[Path]) -> Tuple[Optional[float], Optional[bool], Optional[Structure]]:
    if path is None or not path.exists():
        return None, None, None
    try:
        vr = Vasprun(str(path), parse_dos=False, parse_eigen=False, parse_projected_eigen=False)
        energy = None
        if hasattr(vr, "final_energy") and vr.final_energy is not None:
            energy = float(vr.final_energy)
        elif getattr(vr, "ionic_steps", None):
            last = vr.ionic_steps[-1]
            if "e_0_energy" in last:
                energy = float(last["e_0_energy"])
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


def distance_matrix_to_adsorbate(structure: Structure, ads_indices: Sequence[int], slab_indices: Sequence[int]) -> List[Tuple[int, int, float]]:
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


def local_env_tokens(structure: Structure, center_idx: int, env_size: int) -> List[str]:
    dists = []
    for j, site in enumerate(structure):
        if j == center_idx:
            continue
        dists.append((float(structure.get_distance(center_idx, j)), str(site.specie)))
    dists.sort(key=lambda x: x[0])
    tokens = [str(structure[center_idx].specie)]
    tokens.extend([sp for _, sp in dists[: max(env_size - 1, 0)]])
    return tokens[:env_size]


def serialize_sample_text(
    structure: Structure,
    sample_id: str,
    ads_symbol: str = "H",
    env_size: int = 8,
    anchor_shell_tol: float = 0.45,
    anchor_max_dist: float = 3.0,
    miller_raw: Optional[str] = None,
):
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No adsorbate species {ads_symbol} found")
    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    if not slab_indices:
        raise ValueError("No slab atoms remained after excluding adsorbate")
    formula = build_formula_excluding_adsorbate(structure, ads_indices)
    miller_text = miller_to_string(miller_raw)
    rows = distance_matrix_to_adsorbate(structure, ads_indices, slab_indices)
    nearest = rows[0][2]
    anchors = [(ai, si, d) for ai, si, d in rows if d <= min(anchor_max_dist, nearest + anchor_shell_tol)]
    if not anchors:
        anchors = [rows[0]]
    anchor_site_indices = []
    for _, si, _ in anchors:
        if si not in anchor_site_indices:
            anchor_site_indices.append(si)
    site_type = classify_site(len(anchor_site_indices))
    env_chunks = []
    for si in anchor_site_indices:
        env_tokens = local_env_tokens(structure, si, env_size)
        if ads_symbol not in env_tokens:
            env_tokens.append(ads_symbol)
        env_chunks.append("[" + " ".join(env_tokens) + "]")
    lead_tokens = [ads_symbol] + [str(structure[i].specie) for i in anchor_site_indices] + [site_type]
    text = f"{ads_symbol}</s>{formula} {miller_text}</s>[{' '.join(lead_tokens + env_chunks)}]"
    return text, formula, site_type, len(anchor_site_indices), miller_text


def _usable_target_mask(df: pd.DataFrame) -> pd.Series:
    mask = df["target"].notna()
    if "parse_ok" in df.columns:
        mask = mask & df["parse_ok"].fillna(False)
    return mask


def _iqr_outlier_mask(y: pd.Series, iqr_multiplier: float) -> Tuple[pd.Series, Dict[str, float]]:
    q1 = float(y.quantile(0.25))
    q3 = float(y.quantile(0.75))
    iqr = q3 - q1
    low = q1 - iqr_multiplier * iqr
    high = q3 + iqr_multiplier * iqr
    mask = (y < low) | (y > high)
    meta = {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "low_bound": low,
        "high_bound": high,
        "iqr_multiplier": float(iqr_multiplier),
    }
    return mask, meta


def apply_outlier_handling(
    df: pd.DataFrame,
    outlier_method: str,
    outlier_action: str,
    iqr_multiplier: float,
    drop_ids: List[str],
    report_csv: Optional[Path],
    summary_json: Optional[Path],
) -> pd.DataFrame:
    df = df.copy()
    df["outlier_flag_target"] = False
    df["outlier_reason_target"] = ""

    usable_mask = _usable_target_mask(df)
    usable = df.loc[usable_mask].copy()

    detection_meta: Dict[str, object] = {}
    outlier_index = pd.Index([], dtype=int)

    if outlier_method == "none":
        pass
    elif outlier_method == "id_list":
        if not drop_ids:
            raise ValueError("outlier_method=id_list requires non-empty --drop-ids")
        id_mask = usable["id"].astype(str).isin(drop_ids)
        outlier_index = usable.loc[id_mask].index
        detection_meta = {"drop_ids": drop_ids}
    elif outlier_method == "iqr":
        if usable.empty:
            raise ValueError("No usable rows available for IQR outlier detection")
        out_mask, detection_meta = _iqr_outlier_mask(usable["target"].astype(float), iqr_multiplier)
        outlier_index = usable.loc[out_mask].index
    else:
        raise ValueError(f"Unsupported outlier_method: {outlier_method}")

    if len(outlier_index) > 0:
        df.loc[outlier_index, "outlier_flag_target"] = True
        reason = outlier_method if outlier_method != "id_list" else "id_list"
        df.loc[outlier_index, "outlier_reason_target"] = reason

    if report_csv is not None:
        report_cols = [c for c in [
            "id", "family_base", "family_base_miller", "dopant",
            "target", "outlier_flag_target", "outlier_reason_target",
            "notes"
        ] if c in df.columns]
        report_df = df.loc[usable_mask, report_cols].copy()
        report_df["target_abs"] = df.loc[usable_mask, "target"].abs().to_numpy()
        report_df.sort_values(["outlier_flag_target", "target"], ascending=[False, True]).to_csv(report_csv, index=False)

    summary = {
        "outlier_method": outlier_method,
        "outlier_action": outlier_action,
        "n_total_rows_before": int(len(df)),
        "n_usable_rows": int(usable_mask.sum()),
        "n_outliers": int(df["outlier_flag_target"].sum()),
        "drop_ids": drop_ids,
        "detection_meta": detection_meta,
    }

    if outlier_action == "drop":
        df = df.loc[~df["outlier_flag_target"]].copy()
    elif outlier_action == "flag":
        pass
    else:
        raise ValueError(f"Unsupported outlier_action: {outlier_action}")

    summary["n_total_rows_after"] = int(len(df))

    if summary_json is not None:
        with summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[INFO] outlier_method={outlier_method} outlier_action={outlier_action}")
    print(f"[INFO] outliers flagged={summary['n_outliers']} rows_before={summary['n_total_rows_before']} rows_after={summary['n_total_rows_after']}")
    if report_csv is not None:
        print(f"[OK] outlier report saved -> {report_csv}")
    if summary_json is not None:
        print(f"[OK] outlier summary saved -> {summary_json}")

    return df


def main():
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    xlsx_path = Path(args.xlsx).resolve() if args.xlsx else (input_root / "ML2.xlsx")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (input_root / "out")
    bare_dir = Path(args.bare_dir).resolve() if args.bare_dir else (input_root / "withoutH")
    if not bare_dir.exists():
        bare_dir = None

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")
    if not out_dir.exists():
        raise FileNotFoundError(f"Out dir not found: {out_dir}")

    base_miller_map = _parse_key_value_map(args.base_miller_map)
    energy_map = parse_excel_energy_table(xlsx_path)

    rows = []
    for sample_id in sorted(energy_map.keys()):
        e_bare, e_addh = energy_map[sample_id]
        rec = Rec(sample_id=sample_id, energy_bare=e_bare, energy_addH=e_addh)

        contcar = out_dir / f"{sample_id}-CONTCAR"
        vasprun = out_dir / f"{sample_id}-vasprun.xml"
        oszicar = out_dir / f"{sample_id}-OSZICAR"

        rec.contcar_path = str(contcar) if contcar.exists() else None
        rec.vasprun_path = str(vasprun) if vasprun.exists() else None
        rec.oszicar_path = str(oszicar) if oszicar.exists() else None

        bare_contcar = _find_ml2_file(bare_dir, sample_id, "CONTCAR")
        bare_vasprun = _find_ml2_file(bare_dir, sample_id, "vasprun.xml")
        bare_oszicar = _find_ml2_file(bare_dir, sample_id, "OSZICAR")

        try:
            rec.oszicar_e0, rec.oszicar_f = parse_oszicar(oszicar if oszicar.exists() else None)
            rec.vasprun_energy, rec.converged, vr_structure = parse_vasprun_energy_and_convergence(vasprun if vasprun.exists() else None)
            structure = load_structure(contcar if contcar.exists() else None, vr_structure)
            if structure is None:
                raise ValueError("No parsable structure")

            family_base, _, dopant = parse_sample_id(sample_id)
            miller_raw = base_miller_map.get(str(family_base), None)

            rec.text, rec.slab_formula, rec.site_type, rec.anchor_count, rec.miller = serialize_sample_text(
                structure=structure,
                sample_id=sample_id,
                ads_symbol=args.adsorbate,
                miller_raw=miller_raw,
            )
            rec.parse_ok = True
        except Exception as e:
            rec.parse_ok = False
            rec.notes = str(e)
            if args.strict:
                raise

        family_base, _, dopant = parse_sample_id(sample_id)
        miller_raw_for_group = base_miller_map.get(str(family_base), None)
        family_base_miller = _make_family_base_miller(family_base, miller_raw_for_group)
        delta = None if rec.energy_bare is None or rec.energy_addH is None else rec.energy_addH - rec.energy_bare
        target = None if delta is None else delta - args.eh_ref

        rows.append({
            "id": sample_id,
            "family_base": family_base,
            "family_base_miller": family_base_miller,
            "dopant": dopant,
            "energy_bare": rec.energy_bare,
            "energy_addH": rec.energy_addH,
            "delta_e_raw": delta,
            "target": target,
            "eh_ref": args.eh_ref,
            "contcar_path": rec.contcar_path,
            "bare_contcar_path": str(bare_contcar) if bare_contcar is not None and bare_contcar.exists() else None,
            "vasprun_path": rec.vasprun_path,
            "bare_vasprun_path": str(bare_vasprun) if bare_vasprun is not None and bare_vasprun.exists() else None,
            "oszicar_path": rec.oszicar_path,
            "bare_oszicar_path": str(bare_oszicar) if bare_oszicar is not None and bare_oszicar.exists() else None,
            "oszicar_e0": rec.oszicar_e0,
            "oszicar_f": rec.oszicar_f,
            "vasprun_energy": rec.vasprun_energy,
            "converged": rec.converged,
            "slab_formula": rec.slab_formula,
            "miller": rec.miller,
            "site_type": rec.site_type,
            "anchor_count": rec.anchor_count,
            "text": rec.text,
            "parse_ok": rec.parse_ok,
            "notes": rec.notes,
        })

    df = pd.DataFrame(rows)

    report_csv = Path(args.outlier_report_csv).resolve() if args.outlier_report_csv else None
    summary_json = Path(args.outlier_summary_json).resolve() if args.outlier_summary_json else None
    drop_ids = _split_drop_ids(args.drop_ids)

    df = apply_outlier_handling(
        df=df,
        outlier_method=args.outlier_method,
        outlier_action=args.outlier_action,
        iqr_multiplier=args.iqr_multiplier,
        drop_ids=drop_ids,
        report_csv=report_csv,
        summary_json=summary_json,
    )

    df.to_csv(args.output_csv, index=False)

    print(f"[OK] saved -> {args.output_csv}")
    print(f"[INFO] rows={len(df)} parse_ok={int(df['parse_ok'].fillna(False).sum())} target_nonnull={int(df['target'].notna().sum())}")
    print(f"[INFO] unique family_base={df['family_base'].nunique(dropna=True)}")
    if "bare_contcar_path" in df.columns:
        print(f"[INFO] bare_contcar_nonnull={int(df['bare_contcar_path'].notna().sum())}")
    print(f"[INFO] unique family_base_miller={df['family_base_miller'].nunique(dropna=True)}")
    print(f"[INFO] unique dopant={df['dopant'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()
