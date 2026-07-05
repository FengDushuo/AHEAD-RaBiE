#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, re, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:
    raise SystemExit(f"Please install pymatgen first: {e}")

EH_DEFAULT = -0.0565

@dataclass
class Rec:
    sample_id: str
    energy_bare: Optional[float]
    energy_addH: Optional[float]
    status_bare: Optional[str]
    status_addH: Optional[str]
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

_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def _is_float(s: str) -> bool:
    return bool(_FLOAT_RE.match(s))

def parse_energy_table(path: Path) -> Dict[str, Tuple[Optional[float], str, str]]:
    out = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            sid = parts[0]
            val = float(parts[1]) if len(parts) >= 2 and _is_float(parts[1]) else None
            status = parts[2] if len(parts) >= 3 else "UNKNOWN"
            out[sid] = (val, status, line)
    return out

def build_file_index(root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            idx.setdefault(p.name, []).append(p)
    return idx

def find_matching_file(root: Path, file_index: Dict[str, List[Path]], sample_id: str, kind: str) -> Optional[Path]:
    explicit_name = f"{sample_id}-{kind}"
    if explicit_name in file_index:
        return sorted(file_index[explicit_name])[0]
    candidates = file_index.get(kind, [])
    scored = []
    for p in candidates:
        s = str(p)
        score = 0
        if sample_id in s:
            score += 100
        if p.parent.name == sample_id:
            score += 50
        scored.append((-score, len(s), s, p))
    if scored:
        scored.sort()
        best = scored[0][3]
        if sample_id in str(best) or best.parent.name == sample_id:
            return best
    for p in root.rglob(kind):
        if sample_id in str(p):
            return p
    return None

def _safe_float_fortran(s: str) -> Optional[float]:
    try:
        return float(s.replace("D", "E"))
    except Exception:
        return None

def parse_oszicar(path: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if path is None or not path.exists():
        return None, None
    import re
    pat = re.compile(r"F=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)\s+E0=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)")
    e0 = None; f = None
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

def parse_sample_id(sample_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.match(r"^(?P<base>[^-]+)-(?P<miller>\d{3})-(?P<tag>.+)$", sample_id)
    if not m:
        return None, None, None
    return m.group("base"), m.group("miller"), m.group("tag")

def miller_to_string(miller: Optional[str]) -> str:
    if not miller or len(miller) != 3 or not miller.isdigit():
        return "(? ? ?)"
    return f"({miller[0]} {miller[1]} {miller[2]})"

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

def distance_matrix_to_adsorbate(structure: Structure, ads_indices: Sequence[int], slab_indices: Sequence[int]) -> List[Tuple[int,int,float]]:
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

def serialize_sample_text(structure: Structure, sample_id: str, ads_symbol: str = "H", env_size: int = 8,
                          anchor_shell_tol: float = 0.45, anchor_max_dist: float = 3.0):
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No adsorbate species {ads_symbol} found")
    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    if not slab_indices:
        raise ValueError("No slab atoms remained after excluding adsorbate")
    formula = build_formula_excluding_adsorbate(structure, ads_indices)
    _, miller_raw, _ = parse_sample_id(sample_id)
    miller_text = miller_to_string(miller_raw)
    rows = distance_matrix_to_adsorbate(structure, ads_indices, slab_indices)
    nearest = rows[0][2]
    anchors = [(ai,si,d) for ai,si,d in rows if d <= min(anchor_max_dist, nearest + anchor_shell_tol)]
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

def _split_drop_ids(raw: str) -> List[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Extracted addH directory")
    ap.add_argument("--energy-bare", default=None, help="Path to energy.dat")
    ap.add_argument("--energy-addh", default=None, help="Path to energy-addH.dat")
    ap.add_argument("--bare-structure-dir", default=None, help="Directory holding bare slab files; default <input-dir>/withoutH if it exists")
    ap.add_argument("--eh-ref", type=float, default=EH_DEFAULT, help="EH reference in target = Etotal - Eslab - EH")
    ap.add_argument("--output-csv", default="addH_master.csv")
    ap.add_argument("--adsorbate", default="H")
    ap.add_argument("--strict", action="store_true")

    ap.add_argument("--outlier-method", default="none", choices=["none", "iqr", "id_list"])
    ap.add_argument("--outlier-action", default="flag", choices=["flag", "drop"])
    ap.add_argument("--iqr-multiplier", type=float, default=3.0)
    ap.add_argument("--drop-ids", default="", help="Comma-separated IDs to drop when outlier-method=id_list")
    ap.add_argument("--outlier-report-csv", default=None, help="Optional report CSV")
    ap.add_argument("--outlier-summary-json", default=None, help="Optional summary JSON")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    bare_structure_dir = Path(args.bare_structure_dir).resolve() if args.bare_structure_dir else (input_dir / "withoutH")
    if not bare_structure_dir.exists():
        bare_structure_dir = None

    e_bare = Path(args.energy_bare).resolve() if args.energy_bare else (input_dir / "energy.dat")
    e_add = Path(args.energy_addh).resolve() if args.energy_addh else (input_dir / "energy-addH.dat")
    if not e_bare.exists():
        sib = input_dir.parent / "energy.dat"
        if sib.exists():
            e_bare = sib
    if not e_add.exists():
        sib = input_dir.parent / "energy-addH.dat"
        if sib.exists():
            e_add = sib
    if not e_bare.exists() or not e_add.exists():
        raise FileNotFoundError(f"Need both energy.dat and energy-addH.dat; got {e_bare} and {e_add}")

    bare = parse_energy_table(e_bare)
    addh = parse_energy_table(e_add)
    all_ids = sorted(set(bare) | set(addh))
    file_index = build_file_index(input_dir)
    bare_file_index = build_file_index(bare_structure_dir) if bare_structure_dir is not None else {}
    rows = []
    for sid in all_ids:
        rec = Rec(
            sid,
            bare.get(sid, (None, None, None))[0],
            addh.get(sid, (None, None, None))[0],
            bare.get(sid, (None, None, None))[1],
            addh.get(sid, (None, None, None))[1]
        )
        bare_contcar = bare_vasprun = bare_oszicar = None
        try:
            contcar = find_matching_file(input_dir, file_index, sid, "CONTCAR")
            vasprun = find_matching_file(input_dir, file_index, sid, "vasprun.xml")
            oszicar = find_matching_file(input_dir, file_index, sid, "OSZICAR")
            rec.contcar_path = str(contcar) if contcar else None
            rec.vasprun_path = str(vasprun) if vasprun else None
            rec.oszicar_path = str(oszicar) if oszicar else None
            bare_contcar = find_matching_file(bare_structure_dir, bare_file_index, sid, "CONTCAR") if bare_structure_dir is not None else None
            bare_vasprun = find_matching_file(bare_structure_dir, bare_file_index, sid, "vasprun.xml") if bare_structure_dir is not None else None
            bare_oszicar = find_matching_file(bare_structure_dir, bare_file_index, sid, "OSZICAR") if bare_structure_dir is not None else None
            rec.oszicar_e0, rec.oszicar_f = parse_oszicar(oszicar)
            rec.vasprun_energy, rec.converged, vr_structure = parse_vasprun_energy_and_convergence(vasprun)
            structure = load_structure(contcar, vr_structure)
            if structure is None:
                raise ValueError("No parsable structure")
            rec.text, rec.slab_formula, rec.site_type, rec.anchor_count, rec.miller = serialize_sample_text(
                structure, sid, ads_symbol=args.adsorbate
            )
            rec.parse_ok = True
        except Exception as e:
            rec.notes = str(e)
            rec.parse_ok = False
            if args.strict:
                raise

        base, miller_raw, dopant = parse_sample_id(sid)
        delta = None if rec.energy_bare is None or rec.energy_addH is None else rec.energy_addH - rec.energy_bare
        target = None if delta is None else delta - args.eh_ref
        rows.append({
            "id": sid,
            "family_base": base,
            "family_base_miller": f"{base}-{miller_raw}" if base and miller_raw else None,
            "dopant": dopant,
            "status_bare": rec.status_bare,
            "status_addH": rec.status_addH,
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
    ok = int(df["parse_ok"].fillna(False).sum()) if "parse_ok" in df.columns else 0
    print(f"[OK] saved -> {args.output_csv}")
    print(f"[INFO] rows={len(df)} parse_ok={ok} target_nonnull={int(df['target'].notna().sum())}")
    if "bare_contcar_path" in df.columns:
        print(f"[INFO] bare_contcar_nonnull={int(df['bare_contcar_path'].notna().sum())}")

if __name__ == "__main__":
    main()
