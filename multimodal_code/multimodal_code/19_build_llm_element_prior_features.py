#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build auditable element + literature/LLM prior features for AddH -> AddH-out.

This script does not train a model. It creates:
  1) source training feature table from addH and addH-2;
  2) addH-out feature table without labels by default;
  3) optional post-hoc addH-out label file for audit only;
  4) JSONL prompts that ask an LLM to return structured materials priors.

The LLM is used as a feature generator, not as a direct adsorption-energy oracle.
If --llm-prior-jsonl is absent, deterministic chemistry priors are used so the
downstream route remains runnable and reproducible.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


EH_DEFAULT = -0.0565

# Fallback descriptors for the elements used in this project.
# Values are intentionally compact: enough for robust ML features, while pymatgen
# is used automatically when available for richer metadata.
ELEMENT_FALLBACK: Dict[str, Dict[str, float]] = {
    "H":  {"Z": 1,  "group": 1,  "period": 1, "X": 2.20, "cov_r": 0.31, "atom_r": 0.53, "mass": 1.008,   "d_count": 0,  "ox_min": -1, "ox_max": 1,  "block_s": 1, "block_p": 0, "block_d": 0, "block_f": 0},
    "O":  {"Z": 8,  "group": 16, "period": 2, "X": 3.44, "cov_r": 0.66, "atom_r": 0.48, "mass": 15.999,  "d_count": 0,  "ox_min": -2, "ox_max": -1, "block_s": 0, "block_p": 1, "block_d": 0, "block_f": 0},
    "Be": {"Z": 4,  "group": 2,  "period": 2, "X": 1.57, "cov_r": 0.96, "atom_r": 1.12, "mass": 9.0122,  "d_count": 0,  "ox_min": 2,  "ox_max": 2,  "block_s": 1, "block_p": 0, "block_d": 0, "block_f": 0},
    "Mg": {"Z": 12, "group": 2,  "period": 3, "X": 1.31, "cov_r": 1.41, "atom_r": 1.60, "mass": 24.305,  "d_count": 0,  "ox_min": 2,  "ox_max": 2,  "block_s": 1, "block_p": 0, "block_d": 0, "block_f": 0},
    "Ca": {"Z": 20, "group": 2,  "period": 4, "X": 1.00, "cov_r": 1.76, "atom_r": 1.97, "mass": 40.078,  "d_count": 0,  "ox_min": 2,  "ox_max": 2,  "block_s": 1, "block_p": 0, "block_d": 0, "block_f": 0},
    "Cr": {"Z": 24, "group": 6,  "period": 4, "X": 1.66, "cov_r": 1.39, "atom_r": 1.28, "mass": 51.9961, "d_count": 5,  "ox_min": -2, "ox_max": 6,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Mn": {"Z": 25, "group": 7,  "period": 4, "X": 1.55, "cov_r": 1.39, "atom_r": 1.27, "mass": 54.938,  "d_count": 5,  "ox_min": -3, "ox_max": 7,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Fe": {"Z": 26, "group": 8,  "period": 4, "X": 1.83, "cov_r": 1.32, "atom_r": 1.26, "mass": 55.845,  "d_count": 6,  "ox_min": -2, "ox_max": 6,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Co": {"Z": 27, "group": 9,  "period": 4, "X": 1.88, "cov_r": 1.26, "atom_r": 1.25, "mass": 58.933,  "d_count": 7,  "ox_min": -1, "ox_max": 5,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Ni": {"Z": 28, "group": 10, "period": 4, "X": 1.91, "cov_r": 1.24, "atom_r": 1.24, "mass": 58.693,  "d_count": 8,  "ox_min": -1, "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Cu": {"Z": 29, "group": 11, "period": 4, "X": 1.90, "cov_r": 1.32, "atom_r": 1.28, "mass": 63.546,  "d_count": 10, "ox_min": 1,  "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Zn": {"Z": 30, "group": 12, "period": 4, "X": 1.65, "cov_r": 1.22, "atom_r": 1.34, "mass": 65.38,   "d_count": 10, "ox_min": 2,  "ox_max": 2,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Zr": {"Z": 40, "group": 4,  "period": 5, "X": 1.33, "cov_r": 1.75, "atom_r": 1.60, "mass": 91.224,  "d_count": 2,  "ox_min": 0,  "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Mo": {"Z": 42, "group": 6,  "period": 5, "X": 2.16, "cov_r": 1.54, "atom_r": 1.39, "mass": 95.95,   "d_count": 5,  "ox_min": -2, "ox_max": 6,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Ru": {"Z": 44, "group": 8,  "period": 5, "X": 2.20, "cov_r": 1.46, "atom_r": 1.34, "mass": 101.07,  "d_count": 7,  "ox_min": -2, "ox_max": 8,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Rh": {"Z": 45, "group": 9,  "period": 5, "X": 2.28, "cov_r": 1.42, "atom_r": 1.34, "mass": 102.91,  "d_count": 8,  "ox_min": -1, "ox_max": 6,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Pd": {"Z": 46, "group": 10, "period": 5, "X": 2.20, "cov_r": 1.39, "atom_r": 1.37, "mass": 106.42,  "d_count": 10, "ox_min": 0,  "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Ag": {"Z": 47, "group": 11, "period": 5, "X": 1.93, "cov_r": 1.45, "atom_r": 1.44, "mass": 107.87,  "d_count": 10, "ox_min": 1,  "ox_max": 3,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Cd": {"Z": 48, "group": 12, "period": 5, "X": 1.69, "cov_r": 1.44, "atom_r": 1.51, "mass": 112.41,  "d_count": 10, "ox_min": 2,  "ox_max": 2,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Ce": {"Z": 58, "group": 3,  "period": 6, "X": 1.12, "cov_r": 2.04, "atom_r": 1.82, "mass": 140.12,  "d_count": 1,  "ox_min": 3,  "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 0, "block_f": 1},
    "Pt": {"Z": 78, "group": 10, "period": 6, "X": 2.28, "cov_r": 1.36, "atom_r": 1.39, "mass": 195.08,  "d_count": 9,  "ox_min": 0,  "ox_max": 6,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Au": {"Z": 79, "group": 11, "period": 6, "X": 2.54, "cov_r": 1.36, "atom_r": 1.44, "mass": 196.97,  "d_count": 10, "ox_min": -1, "ox_max": 5,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Hg": {"Z": 80, "group": 12, "period": 6, "X": 2.00, "cov_r": 1.32, "atom_r": 1.51, "mass": 200.59,  "d_count": 10, "ox_min": 1,  "ox_max": 2,  "block_s": 0, "block_p": 0, "block_d": 1, "block_f": 0},
    "Th": {"Z": 90, "group": 3,  "period": 7, "X": 1.30, "cov_r": 2.06, "atom_r": 1.79, "mass": 232.04,  "d_count": 2,  "ox_min": 4,  "ox_max": 4,  "block_s": 0, "block_p": 0, "block_d": 0, "block_f": 1},
}

LLM_NUMERIC_PRIOR_COLS = [
    "llm_prior_h_ads_eV_guess",
    "llm_expected_rank_score",
    "llm_h_binding_strength_score",
    "llm_oxygen_affinity_score",
    "llm_oxide_reducibility_score",
    "llm_charge_compensation_complexity",
    "llm_host_similarity_to_training",
    "llm_confidence",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--addh-dir", default="addH")
    ap.add_argument("--addh2-root", default="addH-2")
    ap.add_argument("--addh2-xlsx", default=None)
    ap.add_argument("--addhout-dir", default="addH-out")
    ap.add_argument("--addhout-excel", default="addH-out/氢吸附能.xlsx")
    ap.add_argument("--out-dir", default="outputs_addh_llm_element_priors")
    ap.add_argument("--eh-ref", type=float, default=EH_DEFAULT)
    ap.add_argument("--addh2-base-miller-map", default="2542=100,2858=111,643=111")
    ap.add_argument("--addhout-miller-map", default="CeO2=111,ZnO=100")
    ap.add_argument("--target-abs-max", type=float, default=10.0)
    ap.add_argument("--llm-prior-jsonl", default=None, help="Optional JSONL with structured LLM priors.")
    ap.add_argument("--strict-addhout-no-labels", action="store_true", default=True)
    ap.add_argument("--write-audit-labels", action="store_true", help="Write addH-out labels to a separate audit file.")
    return ap.parse_args()


def parse_key_value_map(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def safe_float(x) -> float:
    try:
        if pd.isna(x):
            return np.nan
        return float(str(x).replace("D", "E"))
    except Exception:
        return np.nan


def read_energy_dat(path: Path) -> Dict[str, Tuple[float, str]]:
    out: Dict[str, Tuple[float, str]] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            parts = raw.split()
            if not parts:
                continue
            sid = parts[0]
            val = safe_float(parts[1]) if len(parts) > 1 else np.nan
            status = parts[2] if len(parts) > 2 else ""
            out[sid] = (val, status)
    return out


def normalize_miller(x: Optional[str]) -> str:
    if x is None or str(x).strip() == "":
        return "unknown"
    s = str(x).strip().replace("(", "").replace(")", "").replace(" ", "")
    if re.fullmatch(r"\d{3}", s):
        return s
    return s or "unknown"


def miller_text(x: Optional[str]) -> str:
    s = normalize_miller(x)
    if re.fullmatch(r"\d{3}", s):
        return f"({s[0]} {s[1]} {s[2]})"
    return "(? ? ?)"


def parse_poscar_counts(path: Path) -> Dict[str, int]:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        species = lines[5].split()
        counts = [int(float(x)) for x in lines[6].split()]
        if len(species) == len(counts):
            return dict(zip(species, counts))
    except Exception:
        pass
    return {}


def formula_from_counts(counts: Dict[str, int], skip_h: bool = False) -> str:
    parts = []
    for el in sorted(counts):
        if skip_h and el == "H":
            continue
        n = counts[el]
        parts.append(f"{el}{n}")
    return "".join(parts)


def non_h_elements(counts: Dict[str, int]) -> List[str]:
    return sorted([el for el, n in counts.items() if el != "H" and n > 0])


def build_addh_table(addh_dir: Path, eh_ref: float) -> pd.DataFrame:
    bare = read_energy_dat(addh_dir / "energy.dat")
    addh = read_energy_dat(addh_dir / "energy-addH.dat")
    rows = []
    for sid in sorted(set(bare).intersection(addh)):
        eb, sb = bare[sid]
        ea, sa = addh[sid]
        parts = sid.split("-")
        family = parts[0] if parts else ""
        miller = parts[1] if len(parts) >= 3 else "unknown"
        dopant = "-".join(parts[2:]) if len(parts) >= 3 else parts[-1]
        contcar = addh_dir / "out" / f"{sid}-CONTCAR"
        bare_contcar = addh_dir / "withoutH" / f"{sid}-CONTCAR"
        counts = parse_poscar_counts(contcar)
        target = ea - eb - eh_ref if np.isfinite(ea) and np.isfinite(eb) else np.nan
        rows.append({
            "split_role": "train",
            "data_source": "addH",
            "id": sid,
            "family_base": family,
            "material": formula_from_counts({k: v for k, v in counts.items() if k != dopant and k != "H"}) or family,
            "miller": normalize_miller(miller),
            "miller_text": miller_text(miller),
            "dopant": dopant,
            "energy_bare": eb,
            "energy_addH": ea,
            "target": target,
            "status_bare": sb,
            "status_addH": sa,
            "contcar_path": str(contcar) if contcar.exists() else "",
            "bare_contcar_path": str(bare_contcar) if bare_contcar.exists() else "",
            "contcar_exists": bool(contcar.exists()),
            "bare_contcar_exists": bool(bare_contcar.exists()),
            "poscar_formula": formula_from_counts(counts),
            "non_h_elements": "+".join(non_h_elements(counts)),
            "n_non_h_species": len(non_h_elements(counts)),
            "n_atoms_no_h": sum(v for k, v in counts.items() if k != "H"),
        })
    return pd.DataFrame(rows)


def build_addh2_table(addh2_root: Path, xlsx: Optional[Path], eh_ref: float, miller_map: Dict[str, str]) -> pd.DataFrame:
    xlsx = xlsx or (addh2_root / "ML2.xlsx")
    raw = pd.read_excel(xlsx, engine="openpyxl", header=None)
    rows = []
    for _, r in raw.iterrows():
        sid = r.iloc[1] if len(r) > 1 else None
        if pd.isna(sid):
            continue
        sid = str(sid).strip()
        if not sid or sid.lower() in {"surface", "addh"}:
            continue
        eb = safe_float(r.iloc[2] if len(r) > 2 else np.nan)
        ea = safe_float(r.iloc[3] if len(r) > 3 else np.nan)
        parts = sid.split("-")
        family = parts[0]
        dopant = "-".join(parts[1:]) if len(parts) > 1 else ""
        miller = normalize_miller(miller_map.get(family, "unknown"))
        contcar = addh2_root / "out" / f"{sid}-CONTCAR"
        bare_contcar = addh2_root / "withoutH" / f"{sid}-CONTCAR"
        counts = parse_poscar_counts(contcar)
        rows.append({
            "split_role": "train",
            "data_source": "addH-2",
            "id": sid,
            "family_base": family,
            "material": formula_from_counts({k: v for k, v in counts.items() if k != dopant and k != "H"}) or family,
            "miller": miller,
            "miller_text": miller_text(miller),
            "dopant": dopant,
            "energy_bare": eb,
            "energy_addH": ea,
            "target": ea - eb - eh_ref if np.isfinite(ea) and np.isfinite(eb) else np.nan,
            "status_bare": "",
            "status_addH": "",
            "contcar_path": str(contcar) if contcar.exists() else "",
            "bare_contcar_path": str(bare_contcar) if bare_contcar.exists() else "",
            "contcar_exists": bool(contcar.exists()),
            "bare_contcar_exists": bool(bare_contcar.exists()),
            "poscar_formula": formula_from_counts(counts),
            "non_h_elements": "+".join(non_h_elements(counts)),
            "n_non_h_species": len(non_h_elements(counts)),
            "n_atoms_no_h": sum(v for k, v in counts.items() if k != "H"),
        })
    return pd.DataFrame(rows)


def build_addhout_table(addhout_dir: Path, excel_path: Path, miller_map: Dict[str, str], eh_ref_for_computed: float) -> pd.DataFrame:
    raw = pd.read_excel(excel_path, engine="openpyxl")
    data = raw.iloc[1:].reset_index(drop=True)
    rows = []
    for material, start in [("CeO2", 2), ("ZnO", 5)]:
        for _, r in data.iterrows():
            try:
                idx = int(r.iloc[0])
            except Exception:
                continue
            dopant = str(r.iloc[1]).strip() if pd.notna(r.iloc[1]) else ""
            etot = safe_float(r.iloc[start] if start < len(r) else np.nan)
            eslab = safe_float(r.iloc[start + 1] if start + 1 < len(r) else np.nan)
            h_ads = safe_float(r.iloc[start + 2] if start + 2 < len(r) else np.nan)
            sid = f"{material}-{idx}-{dopant}"
            contcar = addhout_dir / material / "out" / f"{idx}-CONTCAR"
            bare_contcar = addhout_dir / material / "withoutH" / f"{idx}-CONTCAR"
            counts = parse_poscar_counts(contcar)
            miller = normalize_miller(miller_map.get(material, "unknown"))
            target_computed = etot - eslab - eh_ref_for_computed if np.isfinite(etot) and np.isfinite(eslab) else np.nan
            rows.append({
                "split_role": "addH-out",
                "data_source": "addH-out",
                "id": sid,
                "family_base": material,
                "material": material,
                "idx": idx,
                "miller": miller,
                "miller_text": miller_text(miller),
                "dopant": dopant,
                "energy_total_excel": etot,
                "energy_slab_excel": eslab,
                "h_ads_excel": h_ads,
                "target_computed": target_computed,
                "contcar_path": str(contcar) if contcar.exists() else "",
                "bare_contcar_path": str(bare_contcar) if bare_contcar.exists() else "",
                "contcar_exists": bool(contcar.exists()),
                "bare_contcar_exists": bool(bare_contcar.exists()),
                "poscar_formula": formula_from_counts(counts),
                "non_h_elements": "+".join(non_h_elements(counts)),
                "n_non_h_species": len(non_h_elements(counts)),
                "n_atoms_no_h": sum(v for k, v in counts.items() if k != "H"),
            })
    return pd.DataFrame(rows)


_PYMATGEN_ELEMENT = None


def try_pymatgen_element():
    global _PYMATGEN_ELEMENT
    if _PYMATGEN_ELEMENT is not None:
        return _PYMATGEN_ELEMENT
    try:
        from pymatgen.core import Element  # type: ignore
        _PYMATGEN_ELEMENT = Element
    except Exception:
        _PYMATGEN_ELEMENT = False
    return _PYMATGEN_ELEMENT


def element_props(symbol: str) -> Dict[str, float]:
    symbol = str(symbol).strip()
    props = ELEMENT_FALLBACK.get(symbol, {}).copy()
    Element = try_pymatgen_element()
    if Element:
        try:
            e = Element(symbol)
            ox = list(getattr(e, "common_oxidation_states", []) or getattr(e, "oxidation_states", []) or [])
            block = str(getattr(e, "block", ""))
            props.update({
                "Z": float(e.Z),
                "group": float(getattr(e, "group", np.nan) or np.nan),
                "period": float(getattr(e, "row", np.nan) or np.nan),
                "X": float(getattr(e, "X", np.nan) or np.nan),
                "cov_r": float(getattr(e, "atomic_radius", np.nan) or np.nan),
                "atom_r": float(getattr(e, "atomic_radius_calculated", np.nan) or np.nan),
                "mass": float(getattr(e, "atomic_mass", np.nan) or np.nan),
                "ox_min": float(min(ox)) if ox else props.get("ox_min", np.nan),
                "ox_max": float(max(ox)) if ox else props.get("ox_max", np.nan),
                "block_s": 1.0 if block == "s" else 0.0,
                "block_p": 1.0 if block == "p" else 0.0,
                "block_d": 1.0 if block == "d" else 0.0,
                "block_f": 1.0 if block == "f" else 0.0,
            })
        except Exception:
            pass
    for k in ["Z", "group", "period", "X", "cov_r", "atom_r", "mass", "d_count", "ox_min", "ox_max", "block_s", "block_p", "block_d", "block_f"]:
        props.setdefault(k, np.nan)
    return props


def host_aggregate_props(elements: Iterable[str]) -> Dict[str, float]:
    els = [e for e in elements if e and e in ELEMENT_FALLBACK]
    cols = ["Z", "group", "period", "X", "cov_r", "atom_r", "mass", "d_count", "ox_min", "ox_max", "block_d", "block_f"]
    out: Dict[str, float] = {}
    if not els:
        for c in cols:
            out[f"host_mean_{c}"] = np.nan
            out[f"host_max_{c}"] = np.nan
        return out
    mat = pd.DataFrame([element_props(e) for e in els])
    for c in cols:
        out[f"host_mean_{c}"] = float(pd.to_numeric(mat[c], errors="coerce").mean())
        out[f"host_max_{c}"] = float(pd.to_numeric(mat[c], errors="coerce").max())
    return out


def deterministic_chemistry_prior(row: pd.Series) -> Dict[str, float]:
    d = element_props(str(row.get("dopant", "")))
    host_elements = str(row.get("non_h_elements", "")).split("+")
    h = host_aggregate_props(host_elements)
    x = d.get("X", np.nan)
    dcnt = d.get("d_count", np.nan)
    ox_span = d.get("ox_max", np.nan) - d.get("ox_min", np.nan)
    radius_mismatch = abs(d.get("cov_r", np.nan) - h.get("host_mean_cov_r", np.nan))
    noble_like = 1.0 if str(row.get("dopant")) in {"Ag", "Au", "Pt", "Pd", "Rh", "Ru"} else 0.0
    reducible_host = 1.0 if str(row.get("material")) in {"CeO2"} or "Ce" in host_elements else 0.0
    zno_host = 1.0 if str(row.get("material")) == "ZnO" or "Zn" in host_elements else 0.0

    # Higher score means weaker/more positive H adsorption tendency. It is a
    # qualitative prior only; model training learns its scale.
    rank_score = 0.22 * np.nan_to_num(x, nan=1.8)
    rank_score += 0.08 * np.nan_to_num(ox_span, nan=2.0)
    rank_score -= 0.06 * np.nan_to_num(dcnt, nan=5.0)
    rank_score += 0.18 * noble_like
    rank_score += 0.10 * reducible_host
    rank_score -= 0.12 * zno_host
    rank_score -= 0.10 * np.nan_to_num(radius_mismatch, nan=0.5)

    return {
        "llm_prior_h_ads_eV_guess": np.nan,
        "llm_expected_rank_score": float(rank_score),
        "llm_h_binding_strength_score": float(-rank_score),
        "llm_oxygen_affinity_score": float(3.0 - np.nan_to_num(x, nan=1.8) + 0.2 * np.nan_to_num(ox_span, nan=2.0)),
        "llm_oxide_reducibility_score": float(1.0 + 1.5 * reducible_host + 0.2 * np.nan_to_num(ox_span, nan=2.0)),
        "llm_charge_compensation_complexity": float(np.nan_to_num(ox_span, nan=2.0) + 0.5 * radius_mismatch),
        "llm_host_similarity_to_training": float(0.15 if str(row.get("material")) in {"CeO2", "ZnO"} else 1.0),
        "llm_confidence": 0.0,
    }


def strip_json_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?", "", s.strip(), flags=re.I).strip()
        s = re.sub(r"```$", "", s.strip()).strip()
    return s


def load_llm_priors(path: Optional[Path]) -> Dict[str, Dict[str, object]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except Exception:
                continue
            payload = rec
            # OpenAI batch/chat style.
            try:
                content = rec["response"]["body"]["choices"][0]["message"]["content"]
                payload = json.loads(strip_json_fence(content))
            except Exception:
                pass
            key = str(payload.get("id") or payload.get("custom_id") or rec.get("custom_id") or "").strip()
            if not key:
                material = payload.get("material")
                dopant = payload.get("dopant")
                if material and dopant:
                    key = f"{material}::{dopant}"
                elif dopant:
                    key = str(dopant)
            if key:
                out[key] = payload
    return out


def match_llm_prior(row: pd.Series, priors: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    keys = [
        str(row.get("id", "")),
        f"{row.get('material')}::{row.get('dopant')}",
        str(row.get("dopant", "")),
    ]
    for k in keys:
        if k in priors:
            return priors[k]
    return {}


def add_feature_columns(df: pd.DataFrame, priors: Dict[str, Dict[str, object]]) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        rec = r.to_dict()
        dop = str(r.get("dopant", ""))
        dop_props = element_props(dop)
        for k, v in dop_props.items():
            rec[f"dopant_{k}"] = v
        host_elements = [x for x in str(r.get("non_h_elements", "")).split("+") if x]
        host_props = host_aggregate_props(host_elements)
        rec.update(host_props)
        for base in ["X", "cov_r", "atom_r", "Z", "group", "period", "d_count", "ox_max"]:
            rec[f"dopant_minus_host_mean_{base}"] = rec.get(f"dopant_{base}", np.nan) - rec.get(f"host_mean_{base}", np.nan)
        for m in ["011", "101", "110", "100", "111", "unknown"]:
            rec[f"miller_is_{m}"] = 1.0 if str(r.get("miller", "unknown")) == m else 0.0
        rec["material_is_CeO2"] = 1.0 if str(r.get("material")) == "CeO2" else 0.0
        rec["material_is_ZnO"] = 1.0 if str(r.get("material")) == "ZnO" else 0.0
        rec["has_Ce"] = 1.0 if "Ce" in host_elements else 0.0
        rec["has_Zn"] = 1.0 if "Zn" in host_elements else 0.0
        rec["has_O"] = 1.0 if "O" in host_elements else 0.0
        prior = deterministic_chemistry_prior(r)
        llm = match_llm_prior(r, priors)
        for c in LLM_NUMERIC_PRIOR_COLS:
            val = llm.get(c, prior.get(c, np.nan)) if llm else prior.get(c, np.nan)
            rec[c] = safe_float(val)
        rec["llm_prior_present"] = 1.0 if llm else 0.0
        rec["llm_rationale_short"] = str(llm.get("rationale_short", "")) if llm else ""
        rec["llm_sources"] = json.dumps(llm.get("sources", []), ensure_ascii=False) if llm else "[]"
        rows.append(rec)
    return pd.DataFrame(rows)


def add_source_target_statistics(train: pd.DataFrame, addhout: pd.DataFrame, target_abs_max: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    clean = train[pd.to_numeric(train["target"], errors="coerce").abs() <= target_abs_max].copy()
    stats = clean.groupby("dopant")["target"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    stats = stats.rename(columns={
        "count": "source_dopant_target_count_full",
        "mean": "source_dopant_target_mean_full",
        "median": "source_dopant_target_median_full",
        "std": "source_dopant_target_std_full",
        "min": "source_dopant_target_min_full",
        "max": "source_dopant_target_max_full",
    })
    out_train = train.merge(stats, on="dopant", how="left")
    out_addhout = addhout.merge(stats, on="dopant", how="left")
    source_summary = {
        "source_clean_target_mean": float(clean["target"].mean()),
        "source_clean_target_median": float(clean["target"].median()),
        "source_clean_target_std": float(clean["target"].std()),
        "source_clean_target_q05": float(clean["target"].quantile(0.05)),
        "source_clean_target_q95": float(clean["target"].quantile(0.95)),
        "source_clean_n": int(len(clean)),
    }
    for k, v in source_summary.items():
        out_train[k] = v
        out_addhout[k] = v
    out_train["source_target_abs_le_max"] = pd.to_numeric(out_train["target"], errors="coerce").abs() <= target_abs_max
    return out_train, out_addhout


def make_prompt_record(row: pd.Series) -> Dict[str, object]:
    schema = {
        "id": str(row["id"]),
        "material": str(row.get("material", "")),
        "dopant": str(row.get("dopant", "")),
        "llm_prior_h_ads_eV_guess": "number or null",
        "llm_expected_rank_score": "number from -3 strong-H-adsorption to +3 weak-H-adsorption",
        "llm_h_binding_strength_score": "number from -3 weak to +3 strong",
        "llm_oxygen_affinity_score": "number from 0 to 5",
        "llm_oxide_reducibility_score": "number from 0 to 5",
        "llm_charge_compensation_complexity": "number from 0 to 5",
        "llm_host_similarity_to_training": "number from 0 to 1",
        "llm_confidence": "number from 0 to 1",
        "rationale_short": "one concise sentence",
        "sources": ["short source names or DOI/URL if known"],
    }
    user = (
        "Return JSON only. Estimate qualitative priors for H adsorption on a doped oxide surface. "
        "Do not use any hidden labels or dataset values. Use periodic trends and published materials "
        "chemistry knowledge only.\n\n"
        f"sample_id: {row.get('id')}\n"
        f"material/host: {row.get('material')}\n"
        f"dopant: {row.get('dopant')}\n"
        f"miller: {row.get('miller_text')}\n"
        f"non_H_elements_in_CONTCAR: {row.get('non_h_elements')}\n"
        f"dopant_properties: Z={row.get('dopant_Z')}, group={row.get('dopant_group')}, "
        f"period={row.get('dopant_period')}, electronegativity={row.get('dopant_X')}, "
        f"d_count={row.get('dopant_d_count')}, ox_range=[{row.get('dopant_ox_min')},{row.get('dopant_ox_max')}]\n\n"
        f"Required JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return {
        "custom_id": str(row["id"]),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4.1",
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a cautious computational materials scientist. Output strict JSON only."},
                {"role": "user", "content": user},
            ],
        },
    }


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    addh = build_addh_table(Path(args.addh_dir), args.eh_ref)
    addh2 = build_addh2_table(
        Path(args.addh2_root),
        Path(args.addh2_xlsx) if args.addh2_xlsx else None,
        args.eh_ref,
        parse_key_value_map(args.addh2_base_miller_map),
    )
    train = pd.concat([addh, addh2], ignore_index=True)
    addhout = build_addhout_table(
        Path(args.addhout_dir),
        Path(args.addhout_excel),
        parse_key_value_map(args.addhout_miller_map),
        args.eh_ref,
    )

    priors = load_llm_priors(Path(args.llm_prior_jsonl) if args.llm_prior_jsonl else None)
    train = add_feature_columns(train, priors)
    addhout = add_feature_columns(addhout, priors)
    train, addhout = add_source_target_statistics(train, addhout, args.target_abs_max)

    train_path = out_dir / "knowledge_features_train.csv"
    addhout_path = out_dir / "knowledge_features_addhout.csv"
    train.to_csv(train_path, index=False)

    label_cols = ["h_ads_excel", "target_computed", "energy_total_excel", "energy_slab_excel"]
    audit_labels = addhout[["id", "material", "dopant"] + [c for c in label_cols if c in addhout.columns]].copy()
    strict_addhout = addhout.drop(columns=[c for c in label_cols if c in addhout.columns], errors="ignore")
    strict_addhout.to_csv(addhout_path, index=False)
    if args.write_audit_labels:
        audit_labels.to_csv(out_dir / "addhout_audit_labels.csv", index=False)

    prompt_rows = pd.concat([train.drop(columns=["target"], errors="ignore"), strict_addhout], ignore_index=True)
    prompt_rows = prompt_rows.drop_duplicates(["material", "dopant", "miller"], keep="first")
    write_jsonl((make_prompt_record(r) for _, r in prompt_rows.iterrows()), out_dir / "llm_prior_prompts.jsonl")

    manifest = {
        "train_rows": int(len(train)),
        "addhout_rows": int(len(addhout)),
        "train_target_rows": int(pd.to_numeric(train["target"], errors="coerce").notna().sum()),
        "source_clean_rows_abs_max": int(train["source_target_abs_le_max"].sum()),
        "llm_prior_records_loaded": int(len(priors)),
        "strict_addhout_no_labels": True,
        "outputs": {
            "train": str(train_path),
            "addhout": str(addhout_path),
            "audit_labels": str(out_dir / "addhout_audit_labels.csv") if args.write_audit_labels else None,
            "llm_prompts": str(out_dir / "llm_prior_prompts.jsonl"),
        },
    }
    with (out_dir / "feature_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("[OK] wrote", train_path)
    print("[OK] wrote", addhout_path)
    if args.write_audit_labels:
        print("[OK] wrote", out_dir / "addhout_audit_labels.csv")
    print("[OK] wrote", out_dir / "llm_prior_prompts.jsonl")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
