#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-view prediction input from an extracted addH directory containing
VASP files (CONTCAR / vasprun.xml / OSZICAR) and energy tables, then optionally
run the official `combined-ft_after_gap` checkpoint for inference.

Main outputs
------------
1) <prefix>_input.pkl            : columns = [id, text, target]
2) <prefix>_manifest.csv         : rich manifest with paths, energies, text, etc.
3) <prefix>_output/              : raw prediction output directory from regress_predict.py
4) <prefix>_merged.csv           : manifest merged with predictions
5) <prefix>_merged_compact.csv   : compact merged table sorted by prediction
6) <prefix>_failed.csv           : failed / missing-prediction rows
7) <prefix>_top15_low.csv        : 15 lowest predictions
8) <prefix>_top15_high.csv       : 15 highest predictions

New in this version
-------------------
- Reads BOTH energy.dat (bare slab) and energy-addH.dat (slab+H)
- Adds the following manifest columns automatically:
    * energy_bare
    * energy_addH
    * delta_e_raw = energy_addH - energy_bare
    * e_ads_h2ref = delta_e_raw - 0.5 * E_H2   (if --h2-energy is provided)

Designed for the `multi-view` repository by hoon-ock, where prediction inputs use
serialized text strings in the style:
    adsorbate</s>slab_formula (h k l)</s>[site/local environment ...]

For the addH dataset, the default adsorbate is H.
"""

from __future__ import annotations

import argparse
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires pymatgen. Install it first, e.g.\n"
        "  pip install pymatgen\n"
        f"Original import error: {e}"
    )


@dataclass
class SampleRecord:
    sample_id: str
    status: str
    raw_energy_line: str
    energy_dat_value: Optional[float] = None  # kept for backward compatibility; mirrors energy_addH
    energy_bare: Optional[float] = None
    energy_addH: Optional[float] = None
    delta_e_raw: Optional[float] = None
    e_ads_h2ref: Optional[float] = None
    bare_status: Optional[str] = None
    addH_status: Optional[str] = None
    raw_energy_line_bare: Optional[str] = None
    raw_energy_line_addH: Optional[str] = None
    contcar_path: Optional[str] = None
    vasprun_path: Optional[str] = None
    oszicar_path: Optional[str] = None
    oszicar_e0: Optional[float] = None
    oszicar_f: Optional[float] = None
    vasprun_energy: Optional[float] = None
    converged: Optional[bool] = None
    adsorbate: Optional[str] = None
    slab_formula: Optional[str] = None
    miller: Optional[str] = None
    site_type: Optional[str] = None
    anchor_count: Optional[int] = None
    text: Optional[str] = None
    parse_ok: bool = False
    notes: str = ""


_FLOAT_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _is_float(s: str) -> bool:
    return bool(_FLOAT_RE.match(s))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert extracted addH VASP results into multi-view prediction input and optionally run prediction."
    )
    p.add_argument("--input-dir", required=True, help="Path to the extracted addH directory.")
    p.add_argument(
        "--energy-file",
        default=None,
        help="Path to energy-addH.dat. Defaults to <input-dir>/energy-addH.dat or sibling if present.",
    )
    p.add_argument(
        "--energy-bare-file",
        default=None,
        help="Path to energy.dat. Defaults to <input-dir>/energy.dat or sibling if present.",
    )
    p.add_argument(
        "--h2-energy",
        type=float,
        default=None,
        help="Optional total energy of H2 used to compute e_ads_h2ref = delta_e_raw - 0.5*E_H2.",
    )
    p.add_argument(
        "--output-prefix",
        default="my_pred",
        help="Prefix for outputs. Example: my_pred -> my_pred_input.pkl, my_pred_manifest.csv ...",
    )
    p.add_argument(
        "--repo-root",
        default=None,
        help="Path to the local multi-view repository root. Needed for running regress_predict.py.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Path to the fine-tuned checkpoint directory, e.g. "
            "<repo-root>/multiview_data/pred_checkpoint/checkpoint/combined-ft_after_gap"
        ),
    )
    p.add_argument(
        "--adsorbate",
        default="H",
        help="Adsorbate symbol to serialize as the leading token. Default: H (for addH dataset).",
    )
    p.add_argument(
        "--target-mode",
        default="zeros",
        choices=["zeros", "energy_dat", "energy_addH", "energy_bare", "delta_e_raw", "oszicar_e0", "vasprun_energy", "e_ads_h2ref"],
        help=(
            "How to populate the required `target` column in the prediction input. "
            "Use 'zeros' for pure inference. 'energy_dat' is retained for backward compatibility and maps to energy_addH."
        ),
    )
    p.add_argument("--env-size", type=int, default=8, help="How many nearby species to keep in each local environment bracket.")
    p.add_argument("--anchor-shell-tol", type=float, default=0.45, help="Distance tolerance (Å) above nearest adsorbate-slab distance used to identify anchor atoms.")
    p.add_argument("--anchor-max-dist", type=float, default=3.0, help="Maximum adsorbate-slab distance (Å) to consider anchor atoms.")
    p.add_argument("--run-predict", action="store_true", help="If set, automatically call regress_predict.py after generating the input pickle.")
    p.add_argument("--device-python", default=sys.executable, help="Python executable used to run regress_predict.py. Default: current Python.")
    p.add_argument("--strict", action="store_true", help="Stop on the first sample that fails to parse instead of skipping it.")
    return p.parse_args()


def resolve_default_energy_file(input_dir: Path, basename: str) -> Path:
    p1 = input_dir / basename
    if p1.exists():
        return p1
    p2 = input_dir.parent / basename
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Could not find {basename}. Checked:\n  {p1}\n  {p2}")


def parse_energy_file_map(path: Path) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            sample_id = parts[0]
            energy_val: Optional[float] = None
            status = "UNKNOWN"
            if len(parts) >= 3 and _is_float(parts[1]):
                energy_val = float(parts[1])
                status = parts[2]
            else:
                status = "PARSE_FAIL"
            out[sample_id] = {"energy": energy_val, "status": status, "raw_line": line}
    return out


def make_records(addh_map: Dict[str, Dict[str, object]], bare_map: Dict[str, Dict[str, object]], h2_energy: Optional[float]) -> List[SampleRecord]:
    records: List[SampleRecord] = []
    for sample_id, addh in addh_map.items():
        bare = bare_map.get(sample_id, {})
        e_addh = addh.get("energy")
        e_bare = bare.get("energy")
        delta = None
        if e_addh is not None and e_bare is not None:
            delta = float(e_addh) - float(e_bare)
        e_ads = None
        if delta is not None and h2_energy is not None:
            e_ads = float(delta) - 0.5 * float(h2_energy)
        rec = SampleRecord(
            sample_id=sample_id,
            status=str(addh.get("status", "UNKNOWN")),
            raw_energy_line=str(addh.get("raw_line", "")),
            energy_dat_value=float(e_addh) if e_addh is not None else None,
            energy_bare=float(e_bare) if e_bare is not None else None,
            energy_addH=float(e_addh) if e_addh is not None else None,
            delta_e_raw=delta,
            e_ads_h2ref=e_ads,
            bare_status=str(bare.get("status")) if bare else None,
            addH_status=str(addh.get("status", "UNKNOWN")),
            raw_energy_line_bare=str(bare.get("raw_line")) if bare else None,
            raw_energy_line_addH=str(addh.get("raw_line", "")),
        )
        records.append(rec)
    return records


def build_file_index(root: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file():
            idx.setdefault(p.name, []).append(p)
    return idx


def find_matching_file(root: Path, file_index: Dict[str, List[Path]], sample_id: str, kind: str) -> Optional[Path]:
    target_basename = kind
    explicit_name = f"{sample_id}-{kind}"
    if explicit_name in file_index:
        return sorted(file_index[explicit_name])[0]
    candidates = file_index.get(target_basename, [])
    scored: List[Tuple[int, int, str, Path]] = []
    for p in candidates:
        s = str(p)
        score = 0
        if sample_id in s:
            score += 100
        if p.parent.name == sample_id:
            score += 50
        if p.stem == sample_id:
            score += 20
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


def parse_sample_id(sample_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    m = re.match(r"^(?P<base>[^-]+)-(?P<miller>\d{3})-(?P<tag>.+)$", sample_id)
    if not m:
        return None, None, None
    return m.group("base"), m.group("miller"), m.group("tag")


def miller_to_string(miller: Optional[str]) -> str:
    if not miller or len(miller) != 3 or not miller.isdigit():
        return "(? ? ?)"
    return f"({miller[0]} {miller[1]} {miller[2]})"


def parse_oszicar(path: Optional[Path]) -> Tuple[Optional[float], Optional[float]]:
    if path is None or not path.exists():
        return None, None
    e0_val = None
    f_val = None
    pat_e = re.compile(r"F=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)\s+E0=\s*([+-]?[0-9]*\.?[0-9]+E?[+-]?[0-9]*)")
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat_e.search(line)
            if m:
                f_val = _safe_float_fortran(m.group(1))
                e0_val = _safe_float_fortran(m.group(2))
    return e0_val, f_val


def _safe_float_fortran(s: str) -> Optional[float]:
    try:
        return float(s.replace("D", "E"))
    except Exception:
        return None


def parse_vasprun_energy_and_convergence(path: Optional[Path]) -> Tuple[Optional[float], Optional[bool], Optional[Structure]]:
    if path is None or not path.exists():
        return None, None, None
    try:
        vr = Vasprun(str(path), parse_dos=False, parse_eigen=False, parse_projected_eigen=False)
        energy = None
        if hasattr(vr, "final_energy"):
            energy = float(vr.final_energy)
        elif getattr(vr, "ionic_steps", None):
            last = vr.ionic_steps[-1]
            if "e_0_energy" in last:
                energy = float(last["e_0_energy"])
        converged = getattr(vr, "converged", None)
        structure = getattr(vr, "final_structure", None)
        return energy, converged, structure
    except Exception:
        return None, None, None


def load_structure(contcar_path: Optional[Path], fallback_structure: Optional[Structure]) -> Optional[Structure]:
    if contcar_path is not None and contcar_path.exists():
        try:
            return Poscar.from_file(str(contcar_path)).structure
        except Exception:
            pass
    return fallback_structure


def build_formula_excluding_adsorbate(structure: Structure, ads_symbol: str, ads_indices: Sequence[int]) -> str:
    ads_set = set(ads_indices)
    order_seen: List[str] = []
    count_map: Dict[str, int] = {}
    for i, site in enumerate(structure):
        if i in ads_set:
            continue
        sp = str(site.specie)
        if sp not in count_map:
            order_seen.append(sp)
            count_map[sp] = 0
        count_map[sp] += 1
    return "".join([f"{sp}{count_map[sp]}" for sp in order_seen if count_map[sp] > 0])


def choose_adsorbate_indices(structure: Structure, ads_symbol: str) -> List[int]:
    idx = [i for i, s in enumerate(structure) if str(s.specie) == ads_symbol]
    if not idx:
        return []
    z_cart = [(i, structure[i].coords[2]) for i in idx]
    z_cart.sort(key=lambda x: x[1], reverse=True)
    return [z_cart[0][0]]


def distance_matrix_to_adsorbate(structure: Structure, ads_indices: Sequence[int], slab_indices: Sequence[int]) -> List[Tuple[int, int, float]]:
    rows: List[Tuple[int, int, float]] = []
    for ai in ads_indices:
        for si in slab_indices:
            d = float(structure.get_distance(ai, si))
            rows.append((ai, si, d))
    rows.sort(key=lambda x: x[2])
    return rows


def classify_site(anchor_count: int) -> str:
    if anchor_count <= 1:
        return "atop"
    if anchor_count == 2:
        return "bridge"
    return "hollow"


def local_env_tokens(structure: Structure, center_idx: int, env_size: int) -> List[str]:
    dists: List[Tuple[float, str]] = []
    for j, site in enumerate(structure):
        if j == center_idx:
            continue
        d = float(structure.get_distance(center_idx, j))
        dists.append((d, str(site.specie)))
    dists.sort(key=lambda x: x[0])
    tokens = [str(structure[center_idx].specie)]
    tokens.extend([sp for _, sp in dists[: max(env_size - 1, 0)]])
    return tokens[:env_size]


def serialize_sample_text(structure: Structure, sample_id: str, ads_symbol: str, env_size: int, anchor_shell_tol: float, anchor_max_dist: float) -> Tuple[str, str, str, int]:
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No adsorbate species '{ads_symbol}' found in structure.")
    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    if not slab_indices:
        raise ValueError("No slab atoms remained after excluding adsorbate atoms.")
    formula = build_formula_excluding_adsorbate(structure, ads_symbol, ads_indices)
    _, miller_raw, _ = parse_sample_id(sample_id)
    miller_text = miller_to_string(miller_raw)
    rows = distance_matrix_to_adsorbate(structure, ads_indices, slab_indices)
    nearest = rows[0][2]
    anchors = [(ai, si, d) for ai, si, d in rows if d <= min(anchor_max_dist, nearest + anchor_shell_tol)]
    if not anchors:
        ai, si, d = rows[0]
        anchors = [(ai, si, d)]
    anchor_site_indices: List[int] = []
    for _, si, _ in anchors:
        if si not in anchor_site_indices:
            anchor_site_indices.append(si)
    site_type = classify_site(len(anchor_site_indices))
    env_chunks: List[str] = []
    for si in anchor_site_indices:
        env_tokens = local_env_tokens(structure, si, env_size=env_size)
        if ads_symbol not in env_tokens:
            env_tokens.append(ads_symbol)
        env_chunks.append("[" + " ".join(env_tokens) + "]")
    lead_tokens = [ads_symbol] + [str(structure[i].specie) for i in anchor_site_indices] + [site_type]
    bracket_content = " ".join(lead_tokens + env_chunks)
    text = f"{ads_symbol}</s>{formula} {miller_text}</s>[{bracket_content}]"
    return text, formula, site_type, len(anchor_site_indices)


def choose_target(rec: SampleRecord, mode: str) -> float:
    if mode == "zeros":
        return 0.0
    if mode in {"energy_dat", "energy_addH"} and rec.energy_addH is not None:
        return float(rec.energy_addH)
    if mode == "energy_bare" and rec.energy_bare is not None:
        return float(rec.energy_bare)
    if mode == "delta_e_raw" and rec.delta_e_raw is not None:
        return float(rec.delta_e_raw)
    if mode == "oszicar_e0" and rec.oszicar_e0 is not None:
        return float(rec.oszicar_e0)
    if mode == "vasprun_energy" and rec.vasprun_energy is not None:
        return float(rec.vasprun_energy)
    if mode == "e_ads_h2ref" and rec.e_ads_h2ref is not None:
        return float(rec.e_ads_h2ref)
    return 0.0


def run_prediction(repo_root: Path, checkpoint_dir: Path, input_pkl: Path, output_dir: Path, python_exe: str) -> None:
    predict_script = repo_root / "regress_predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"Cannot find regress_predict.py under repo root: {repo_root}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(predict_script),
        "--data_path", str(input_pkl),
        "--pt_ckpt_dir_path", str(checkpoint_dir),
        "--save_path", str(output_dir),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def resolve_prediction_file(output_dir: Path) -> Optional[Path]:
    if output_dir.is_file():
        return output_dir
    if output_dir.is_dir():
        cands = sorted(output_dir.glob("*-strc.pkl"))
        if cands:
            return cands[-1]
        cands = sorted(output_dir.glob("*.pkl"))
        if cands:
            return cands[-1]
    return None


def load_prediction_object(pred_file: Path):
    try:
        return pd.read_pickle(pred_file)
    except Exception:
        with pred_file.open("rb") as f:
            return pickle.load(f)


def export_post_tables(merged: pd.DataFrame, output_prefix: Path) -> None:
    compact_csv = output_prefix.with_name(output_prefix.name + "_merged_compact.csv")
    failed_csv = output_prefix.with_name(output_prefix.name + "_failed.csv")
    low_csv = output_prefix.with_name(output_prefix.name + "_top15_low.csv")
    high_csv = output_prefix.with_name(output_prefix.name + "_top15_high.csv")
    keep_cols = [c for c in [
        "id", "pred", "text", "parse_ok", "energy_bare", "energy_addH", "delta_e_raw", "e_ads_h2ref",
        "adsorbate", "slab_formula", "miller", "site_type", "contcar_path", "vasprun_path", "oszicar_path"
    ] if c in merged.columns]
    compact = merged[keep_cols].copy() if keep_cols else merged.copy()
    if "pred" in compact.columns:
        compact = compact.sort_values("pred")
    compact.to_csv(compact_csv, index=False)
    print(f"[OK] compact table saved      -> {compact_csv}")
    failed = merged[(merged.get("parse_ok") != True) | (merged.get("pred").isna() if "pred" in merged.columns else False)].copy()
    failed.to_csv(failed_csv, index=False)
    print(f"[OK] failed table saved       -> {failed_csv}")
    if "pred" in merged.columns:
        merged.sort_values("pred").head(15).to_csv(low_csv, index=False)
        merged.sort_values("pred", ascending=False).head(15).to_csv(high_csv, index=False)
        print(f"[OK] lowest-15 table saved   -> {low_csv}")
        print(f"[OK] highest-15 table saved  -> {high_csv}")


def try_merge_predictions(manifest_csv: Path, output_dir: Path, merged_csv: Path, output_prefix: Path) -> None:
    if not manifest_csv.exists():
        return
    pred_file = resolve_prediction_file(output_dir)
    if pred_file is None or not pred_file.exists():
        print(f"[WARN] Could not locate prediction output for merging under: {output_dir}")
        return
    manifest = pd.read_csv(manifest_csv)
    obj = load_prediction_object(pred_file)
    pred_df: Optional[pd.DataFrame] = None
    if isinstance(obj, dict):
        pred_df = pd.DataFrame({"id": list(obj.keys()), "pred": list(obj.values())})
    elif isinstance(obj, pd.Series):
        pred_df = obj.rename("pred").reset_index().rename(columns={"index": "id"})
    elif isinstance(obj, pd.DataFrame):
        pred_df = obj.copy()
        if "pred" not in pred_df.columns:
            num_cols = [c for c in pred_df.columns if pd.api.types.is_numeric_dtype(pred_df[c])]
            if num_cols:
                pred_df = pred_df.rename(columns={num_cols[0]: "pred"})
    elif isinstance(obj, (list, tuple)) and len(obj) == len(manifest):
        pred_df = manifest[["id"]].copy()
        pred_df["pred"] = list(obj)
    if pred_df is None:
        print(f"[WARN] Prediction object type not auto-mergeable: {type(obj)}")
        return
    if "id" not in pred_df.columns:
        if len(pred_df) == len(manifest):
            pred_df = manifest[["id"]].join(pred_df.reset_index(drop=True))
        else:
            print("[WARN] Prediction table has no 'id' column and length mismatch; skipping merge.")
            return
    merged = manifest.merge(pred_df, on="id", how="left")
    merged.to_csv(merged_csv, index=False)
    print(f"[OK] merged predictions saved -> {merged_csv}")
    if "pred" in merged.columns:
        print(f"[INFO] pred non-null count    = {int(merged['pred'].notna().sum())} / {len(merged)}")
    export_post_tables(merged, output_prefix)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    energy_addh_file = Path(args.energy_file).resolve() if args.energy_file else resolve_default_energy_file(input_dir, "energy-addH.dat")
    energy_bare_file = Path(args.energy_bare_file).resolve() if args.energy_bare_file else resolve_default_energy_file(input_dir, "energy.dat")

    output_prefix = Path(args.output_prefix).resolve() if Path(args.output_prefix).is_absolute() else (Path.cwd() / args.output_prefix)
    out_input_pkl = output_prefix.with_name(output_prefix.name + "_input.pkl")
    out_manifest_csv = output_prefix.with_name(output_prefix.name + "_manifest.csv")
    out_pred_dir = output_prefix.with_name(output_prefix.name + "_output")
    out_merged_csv = output_prefix.with_name(output_prefix.name + "_merged.csv")

    print(f"[INFO] input_dir         = {input_dir}")
    print(f"[INFO] energy_addH_file  = {energy_addh_file}")
    print(f"[INFO] energy_bare_file  = {energy_bare_file}")
    print(f"[INFO] output pkl        = {out_input_pkl}")
    print(f"[INFO] manifest csv      = {out_manifest_csv}")

    addh_map = parse_energy_file_map(energy_addh_file)
    bare_map = parse_energy_file_map(energy_bare_file)
    records = make_records(addh_map, bare_map, args.h2_energy)
    if not records:
        raise RuntimeError(f"No rows parsed from {energy_addh_file}")

    file_index = build_file_index(input_dir)
    rows_for_pkl: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for rec in records:
        try:
            contcar = find_matching_file(input_dir, file_index, rec.sample_id, "CONTCAR")
            vasprun = find_matching_file(input_dir, file_index, rec.sample_id, "vasprun.xml")
            oszicar = find_matching_file(input_dir, file_index, rec.sample_id, "OSZICAR")
            rec.contcar_path = str(contcar) if contcar else None
            rec.vasprun_path = str(vasprun) if vasprun else None
            rec.oszicar_path = str(oszicar) if oszicar else None
            rec.oszicar_e0, rec.oszicar_f = parse_oszicar(oszicar)
            rec.vasprun_energy, rec.converged, vr_structure = parse_vasprun_energy_and_convergence(vasprun)
            structure = load_structure(contcar, vr_structure)
            if structure is None:
                raise ValueError("Neither CONTCAR nor parsable vasprun.xml structure was found.")
            text, formula, site_type, anchor_count = serialize_sample_text(
                structure=structure,
                sample_id=rec.sample_id,
                ads_symbol=args.adsorbate,
                env_size=args.env_size,
                anchor_shell_tol=args.anchor_shell_tol,
                anchor_max_dist=args.anchor_max_dist,
            )
            _, miller_raw, _ = parse_sample_id(rec.sample_id)
            rec.adsorbate = args.adsorbate
            rec.slab_formula = formula
            rec.miller = miller_to_string(miller_raw)
            rec.site_type = site_type
            rec.anchor_count = anchor_count
            rec.text = text
            rec.parse_ok = True
            rows_for_pkl.append({"id": rec.sample_id, "text": rec.text, "target": choose_target(rec, args.target_mode)})
        except Exception as e:
            rec.parse_ok = False
            rec.notes = str(e)
            if args.strict:
                raise

        manifest_rows.append(
            {
                "id": rec.sample_id,
                "status": rec.status,
                "bare_status": rec.bare_status,
                "addH_status": rec.addH_status,
                "energy_dat_value": rec.energy_dat_value,
                "energy_bare": rec.energy_bare,
                "energy_addH": rec.energy_addH,
                "delta_e_raw": rec.delta_e_raw,
                "e_ads_h2ref": rec.e_ads_h2ref,
                "contcar_path": rec.contcar_path,
                "vasprun_path": rec.vasprun_path,
                "oszicar_path": rec.oszicar_path,
                "oszicar_e0": rec.oszicar_e0,
                "oszicar_f": rec.oszicar_f,
                "vasprun_energy": rec.vasprun_energy,
                "converged": rec.converged,
                "adsorbate": rec.adsorbate,
                "slab_formula": rec.slab_formula,
                "miller": rec.miller,
                "site_type": rec.site_type,
                "anchor_count": rec.anchor_count,
                "text": rec.text,
                "parse_ok": rec.parse_ok,
                "notes": rec.notes,
                "raw_energy_line": rec.raw_energy_line,
                "raw_energy_line_bare": rec.raw_energy_line_bare,
                "raw_energy_line_addH": rec.raw_energy_line_addH,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    pred_df = pd.DataFrame(rows_for_pkl)
    if pred_df.empty:
        raise RuntimeError("No valid samples were serialized into prediction inputs. Check the manifest CSV for failure reasons.")
    pred_df.to_pickle(out_input_pkl)
    manifest_df.to_csv(out_manifest_csv, index=False)
    print(f"[OK] prediction input saved -> {out_input_pkl}  (n={len(pred_df)})")
    print(f"[OK] manifest saved         -> {out_manifest_csv}")
    print(f"[INFO] parse_ok count       = {int(manifest_df['parse_ok'].sum())} / {len(manifest_df)}")

    if args.run_predict:
        repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
        checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else repo_root / "multiview_data" / "pred_checkpoint" / "checkpoint" / "combined-ft_after_gap"
        run_prediction(repo_root=repo_root, checkpoint_dir=checkpoint_dir, input_pkl=out_input_pkl, output_dir=out_pred_dir, python_exe=args.device_python)
        print(f"[OK] raw prediction saved  -> {out_pred_dir}")
        try_merge_predictions(out_manifest_csv, out_pred_dir, out_merged_csv, output_prefix)


if __name__ == "__main__":
    main()
