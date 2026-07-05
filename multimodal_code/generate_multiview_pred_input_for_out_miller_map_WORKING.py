#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate multi-view prediction input for addH-out with per-material Miller mapping.

Example
-------
python generate_multiview_pred_input_for_out_miller_map_WORKING.py \
  --input-dir addH-out \
  --repo-root . \
  --run-predict \
  --output-prefix addH_out_pred \
  --miller-map "CeO2=111,ZnO=100"
"""
from __future__ import annotations

import argparse
import os
import pickle
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Vasprun
except Exception as e:
    raise SystemExit(
        "This script requires pymatgen. Install it first, e.g.\n"
        "  pip install pymatgen\n"
        f"Original import error: {e}"
    )


@dataclass
class SampleRecord:
    sample_id: str
    material: str
    idx: int
    element: Optional[str] = None
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
    energy_total: Optional[float] = None
    energy_slab: Optional[float] = None
    h_ads_excel: Optional[float] = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate multi-view prediction input for addH-out with per-material Miller mapping.")
    p.add_argument("--input-dir", required=True, help="Path to the extracted addH-out directory.")
    p.add_argument("--repo-root", default=None, help="Path to the local multi-view repo root.")
    p.add_argument("--checkpoint-dir", default=None, help="Checkpoint dir for regress_predict.py.")
    p.add_argument("--output-prefix", default="addH_out_pred", help="Prefix for outputs.")
    p.add_argument("--adsorbate", default="H", help="Adsorbate symbol, default H.")
    p.add_argument("--miller", default="(? ? ?)", help="Default Miller text when material is not in --miller-map.")
    p.add_argument("--miller-map", default="", help='Per-material Miller mapping, e.g. "CeO2=111,ZnO=100"')
    p.add_argument("--target-value", type=float, default=0.0, help="Placeholder target value for prediction input.")
    p.add_argument("--env-size", type=int, default=8)
    p.add_argument("--anchor-shell-tol", type=float, default=0.45)
    p.add_argument("--anchor-max-dist", type=float, default=3.0)
    p.add_argument("--excel-path", default=None, help="Optional path to 氢吸附能.xlsx. Defaults to <input-dir>/氢吸附能.xlsx if present.")
    p.add_argument("--run-predict", action="store_true")
    p.add_argument("--device-python", default=sys.executable)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def safe_float_fortran(s: str) -> Optional[float]:
    try:
        return float(s.replace("D", "E"))
    except Exception:
        return None


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
                f_val = safe_float_fortran(m.group(1))
                e0_val = safe_float_fortran(m.group(2))
    return e0_val, f_val


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
    tokens.extend([sp for _, sp in dists[:max(env_size - 1, 0)]])
    return tokens[:env_size]


def serialize_sample_text(
    structure: Structure,
    ads_symbol: str,
    miller_text: str,
    env_size: int,
    anchor_shell_tol: float,
    anchor_max_dist: float,
) -> Tuple[str, str, str, int]:
    ads_indices = choose_adsorbate_indices(structure, ads_symbol)
    if not ads_indices:
        raise ValueError(f"No adsorbate species '{ads_symbol}' found in structure.")

    slab_indices = [i for i in range(len(structure)) if i not in set(ads_indices)]
    if not slab_indices:
        raise ValueError("No slab atoms remained after excluding adsorbate atoms.")

    formula = build_formula_excluding_adsorbate(structure, ads_symbol, ads_indices)

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


def scan_material_dirs(root: Path) -> List[Tuple[str, int, Path, Optional[Path], Optional[Path]]]:
    rows = []
    for mat_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        material = mat_dir.name
        contcars = sorted(mat_dir.glob("*-CONTCAR"))
        for c in contcars:
            m = re.match(r"^(\d+)-CONTCAR$", c.name)
            if not m:
                continue
            idx = int(m.group(1))
            vasprun = mat_dir / f"{idx}-vasprun.xml"
            oszicar = mat_dir / f"{idx}-OSZICAR"
            rows.append((material, idx, c, vasprun if vasprun.exists() else None, oszicar if oszicar.exists() else None))
    return rows


def parse_excel_summary(xlsx_path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    out: Dict[Tuple[str, int], Dict[str, object]] = {}
    if xlsx_path is None or not xlsx_path.exists():
        return out
    df = pd.read_excel(xlsx_path)
    cols = list(df.columns)
    material_cols = []
    for i, c in enumerate(cols):
        if isinstance(c, str) and c not in ("Unnamed: 0", "Unnamed: 1") and not c.startswith("Unnamed:") and not c.startswith("EH="):
            material_cols.append((i, c))
    for _, row in df.iloc[1:].iterrows():
        idx_val = row.iloc[0]
        elem = row.iloc[1]
        if pd.isna(idx_val):
            continue
        try:
            idx = int(idx_val)
        except Exception:
            continue
        for pos, material in material_cols:
            etotal = row.iloc[pos]
            eslab = row.iloc[pos + 1] if pos + 1 < len(row) else None
            hads = row.iloc[pos + 2] if pos + 2 < len(row) else None
            etotal = None if pd.isna(etotal) else float(etotal)
            eslab = None if pd.isna(eslab) else float(eslab)
            hads = None if pd.isna(hads) else float(hads)
            if etotal is None and eslab is None and hads is None:
                continue
            out[(material, idx)] = {
                "element": None if pd.isna(elem) else str(elem),
                "energy_total": etotal,
                "energy_slab": eslab,
                "h_ads_excel": hads,
            }
    return out


def resolve_prediction_file(pred_path: Path) -> Optional[Path]:
    if pred_path.is_file():
        return pred_path
    if pred_path.is_dir():
        cands = sorted(pred_path.glob("*-strc.pkl"))
        if cands:
            return cands[-1]
        cands = sorted(pred_path.glob("*.pkl"))
        if cands:
            return cands[-1]
    return None


def export_post_tables(merged: pd.DataFrame, output_prefix: Path) -> None:
    compact_csv = output_prefix.with_name(output_prefix.name + "_merged_compact.csv")
    failed_csv = output_prefix.with_name(output_prefix.name + "_failed.csv")
    low_csv = output_prefix.with_name(output_prefix.name + "_top15_low.csv")
    high_csv = output_prefix.with_name(output_prefix.name + "_top15_high.csv")

    keep_cols = [c for c in [
        "id", "pred", "text", "parse_ok",
        "material", "idx", "element",
        "energy_total", "energy_slab", "h_ads_excel",
        "adsorbate", "slab_formula", "miller", "site_type",
        "contcar_path", "vasprun_path", "oszicar_path"
    ] if c in merged.columns]

    compact = merged[keep_cols].copy()
    if "pred" in compact.columns:
        compact = compact.sort_values("pred")
    compact.to_csv(compact_csv, index=False)
    print(f"[OK] compact table saved      -> {compact_csv}")

    failed = merged[merged["parse_ok"] != True].copy() if "parse_ok" in merged.columns else pd.DataFrame()
    if "pred" in merged.columns:
        failed = pd.concat([failed, merged[merged["pred"].isna()]], axis=0).drop_duplicates()
    failed.to_csv(failed_csv, index=False)
    print(f"[OK] failed table saved       -> {failed_csv}")

    if "pred" in merged.columns:
        merged.sort_values("pred").head(15).to_csv(low_csv, index=False)
        merged.sort_values("pred", ascending=False).head(15).to_csv(high_csv, index=False)
        print(f"[OK] lowest-15 table saved   -> {low_csv}")
        print(f"[OK] highest-15 table saved  -> {high_csv}")


def run_prediction(repo_root: Path, checkpoint_dir: Path, input_pkl: Path, output_dir: Path, python_exe: str) -> None:
    predict_script = repo_root / "regress_predict.py"
    if not predict_script.exists():
        raise FileNotFoundError(f"Cannot find regress_predict.py under repo root: {repo_root}")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")
    cmd = [
        python_exe,
        str(predict_script),
        "--data_path", str(input_pkl),
        "--pt_ckpt_dir_path", str(checkpoint_dir),
        "--save_path", str(output_dir),
    ]
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def try_merge_predictions(manifest_csv: Path, pred_dir: Path, merged_csv: Path, output_prefix: Path) -> None:
    if not manifest_csv.exists():
        return
    real_pred = resolve_prediction_file(pred_dir)
    if real_pred is None or not real_pred.exists():
        print(f"[WARN] Could not locate prediction output for merging under: {pred_dir}")
        return
    manifest = pd.read_csv(manifest_csv)
    obj = None
    try:
        obj = pd.read_pickle(real_pred)
    except Exception:
        try:
            with real_pred.open("rb") as f:
                obj = pickle.load(f)
        except Exception:
            obj = None
    if obj is None:
        print(f"[WARN] Could not read prediction output for merging: {real_pred}")
        return
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
    else:
        raise RuntimeError(f"Unsupported prediction object type: {type(obj)}")
    merged = manifest.merge(pred_df, on="id", how="left")
    merged.to_csv(merged_csv, index=False)
    print(f"[OK] merged predictions saved -> {merged_csv}")
    print(f"[INFO] pred non-null count    = {int(merged['pred'].notna().sum()) if 'pred' in merged.columns else 'NA'} / {len(merged)}")
    export_post_tables(merged, output_prefix)


def normalize_miller_value(raw: str) -> str:
    raw = str(raw).strip()
    if raw.startswith("(") and raw.endswith(")"):
        return raw
    digits = re.sub(r"[^0-9-]", "", raw)
    if re.fullmatch(r"-?\d-?\d-?\d", digits):
        return f"({digits[0]} {digits[1]} {digits[2]})"
    if re.fullmatch(r"\d{3}", digits):
        return f"({digits[0]} {digits[1]} {digits[2]})"
    parts = [p for p in re.split(r"[,_\s]+", raw) if p]
    if len(parts) == 3 and all(re.fullmatch(r"-?\d+", p) for p in parts):
        return f"({parts[0]} {parts[1]} {parts[2]})"
    return raw


def parse_miller_map(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad --miller-map entry: {item!r}. Expected Material=111")
        material, value = item.split("=", 1)
        out[material.strip()] = normalize_miller_value(value)
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir does not exist: {input_dir}")

    output_prefix = Path(args.output_prefix).resolve() if os.path.isabs(args.output_prefix) else (Path.cwd() / args.output_prefix)
    out_input_pkl = output_prefix.with_name(output_prefix.name + "_input.pkl")
    out_manifest_csv = output_prefix.with_name(output_prefix.name + "_manifest.csv")
    out_pred_dir = output_prefix.with_name(output_prefix.name + "_output")
    out_merged_csv = output_prefix.with_name(output_prefix.name + "_merged.csv")

    excel_path = Path(args.excel_path).resolve() if args.excel_path else (input_dir / "氢吸附能.xlsx")
    excel_map = parse_excel_summary(excel_path) if excel_path.exists() else {}
    miller_map = parse_miller_map(args.miller_map)
    default_miller = normalize_miller_value(args.miller)

    print(f"[INFO] input_dir    = {input_dir}")
    print(f"[INFO] excel_path   = {excel_path if excel_path.exists() else 'None'}")
    print(f"[INFO] output pkl   = {out_input_pkl}")
    print(f"[INFO] manifest csv = {out_manifest_csv}")
    print(f"[INFO] miller_map   = {miller_map}")

    scanned = scan_material_dirs(input_dir)
    if not scanned:
        raise RuntimeError(f"No '*-CONTCAR' files found under {input_dir}")

    rows_for_pkl: List[Dict[str, object]] = []
    manifest_rows: List[Dict[str, object]] = []

    for material, idx, contcar, vasprun, oszicar in scanned:
        xinfo = excel_map.get((material, idx), {})
        element = xinfo.get("element")
        sample_id = f"{material}-{idx}" + (f"-{element}" if element else "")
        miller_text = miller_map.get(material, default_miller)

        rec = SampleRecord(sample_id=sample_id, material=material, idx=idx, element=element)
        rec.contcar_path = str(contcar)
        rec.vasprun_path = str(vasprun) if vasprun else None
        rec.oszicar_path = str(oszicar) if oszicar else None
        rec.energy_total = xinfo.get("energy_total")
        rec.energy_slab = xinfo.get("energy_slab")
        rec.h_ads_excel = xinfo.get("h_ads_excel")

        try:
            rec.oszicar_e0, rec.oszicar_f = parse_oszicar(oszicar)
            rec.vasprun_energy, rec.converged, vr_structure = parse_vasprun_energy_and_convergence(vasprun)
            structure = load_structure(contcar, vr_structure)
            if structure is None:
                raise ValueError("Neither CONTCAR nor parsable vasprun.xml structure was found.")

            text, formula, site_type, anchor_count = serialize_sample_text(
                structure=structure,
                ads_symbol=args.adsorbate,
                miller_text=miller_text,
                env_size=args.env_size,
                anchor_shell_tol=args.anchor_shell_tol,
                anchor_max_dist=args.anchor_max_dist,
            )
            rec.adsorbate = args.adsorbate
            rec.slab_formula = formula
            rec.miller = miller_text
            rec.site_type = site_type
            rec.anchor_count = anchor_count
            rec.text = text
            rec.parse_ok = True

            rows_for_pkl.append({
                "id": rec.sample_id,
                "text": rec.text,
                "target": float(args.target_value),
            })
        except Exception as e:
            rec.parse_ok = False
            rec.notes = str(e)
            if args.strict:
                raise

        manifest_rows.append({
            "id": rec.sample_id,
            "material": rec.material,
            "idx": rec.idx,
            "element": rec.element,
            "energy_total": rec.energy_total,
            "energy_slab": rec.energy_slab,
            "h_ads_excel": rec.h_ads_excel,
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
        })

    pred_df = pd.DataFrame(rows_for_pkl)
    manifest_df = pd.DataFrame(manifest_rows)
    if pred_df.empty:
        raise RuntimeError("No valid samples were serialized into prediction inputs.")

    pred_df.to_pickle(out_input_pkl)
    manifest_df.to_csv(out_manifest_csv, index=False)
    print(f"[OK] prediction input saved -> {out_input_pkl}  (n={len(pred_df)})")
    print(f"[OK] manifest saved         -> {out_manifest_csv}")
    print(f"[INFO] parse_ok count       = {int(manifest_df['parse_ok'].sum())} / {len(manifest_df)}")

    if args.run_predict:
        repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd().resolve()
        checkpoint_dir = Path(args.checkpoint_dir).resolve() if args.checkpoint_dir else (
            repo_root / "multiview_data" / "pred_checkpoint" / "checkpoint" / "combined-ft_after_gap"
        )
        run_prediction(repo_root, checkpoint_dir, out_input_pkl, out_pred_dir, args.device_python)
        print(f"[OK] raw prediction output -> {out_pred_dir}")
        try_merge_predictions(out_manifest_csv, out_pred_dir, out_merged_csv, output_prefix)


if __name__ == "__main__":
    main()
