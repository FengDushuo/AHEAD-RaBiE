#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_hads_publication_figures_v2_refined.py

Publication-grade plotting for the H adsorption / deprotonation literature-mining database.

Refined version: H/H* focused, OH/H2O targets are excluded by default; physical-range
filtering is used to suppress extraction artifacts; correlation labels are cleaner.

Compared with the previous version, this script focuses on mechanism-relevant descriptors only,
filters common extraction artifacts, improves the aesthetics of all figures, and redesigns the
correlation bubble matrix:
  - oxidation-state aliases are merged into oxidation_state_value and labeled Metal oxidation state;
  - lower-left triangle: bubbles, with restrained size scaling;
  - upper-right triangle: numeric correlation labels;
  - diagonal: blank summary cells (no duplicated label text).

Expected inputs:
  outputs/hads_db/paper_table.csv
  outputs/hads_db/system_table.csv
  outputs/hads_db/site_table.csv
  outputs/hads_db/adsorption_table.csv
  outputs/hads_db/model_feature_table.csv
  outputs/hads_db/dft_priority_candidates.csv
  optional: outputs/hads_step5_stats_ml/

Example:

conda activate pdfparse

python plot_hads_publication_figures_v2_refined_layoutfix10.py \
  --dbdir outputs/hads_db \
  --step5dir outputs/hads_step5_stats_ml \
  --outdir outputs/publication_figures_hads_v2_refined_layoutfix10 \
  --dpi 900 \
  --font Arial \
  --topk 10 \
  --bubble-max 260 \
  --formats png,pdf
"""

from __future__ import annotations

import argparse
import math
import os
import re
import textwrap
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# Style constants
# =============================================================================

PALETTE = {
    # User-specified endpoints: positive/red = #FF0000, negative/blue = #0000FF.
    # Other accent colors are kept on the same red-blue visual axis for consistency.
    "blue": "#0000FF",
    "light_blue": "#C9D1FF",
    "orange": "#FF6A6A",
    "red": "#FF0000",
    "green": "#335CFF",
    "purple": "#7A33FF",
    "gray": "#6E6E6E",
    "light_gray": "#E7E7E7",
    "dark": "#2B2B2B",
}

# Harmonized red-blue colormaps.  These use the exact requested endpoints,
# with a soft neutral midpoint to keep correlation matrices readable.
RED_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "red_blue_exact_diverging", ["#0000FF", "#F7F7F7", "#FF0000"], N=256
)
BLUE_TO_RED_CMAP = LinearSegmentedColormap.from_list(
    "blue_to_red_exact", ["#0000FF", "#7A33FF", "#F7F7F7", "#FF6A6A", "#FF0000"], N=256
)

TARGET_NAME_MAP = {
    "H_adsorption_free_energy_value_eV": r"$\Delta G_{\mathrm{H}^*}$ (eV)",
    "H_adsorption_energy_value_eV": r"$\Delta E_{\mathrm{H}^*}$ (eV)",
    "OH_adsorption_free_energy_value_eV": r"$\Delta G_{OH^*}$ (eV)",
    "OH_adsorption_energy_value_eV": r"$\Delta E_{OH^*}$ (eV)",
    "H2O_adsorption_energy_value_eV": r"$\Delta E_{H_2O^*}$ (eV)",
    "deprotonation_energy_value_eV": "Deprotonation energy (eV)",
    "Volmer_barrier_eV": "Volmer barrier (eV)",
    "proton_transfer_barrier_eV": "Proton-transfer barrier (eV)",
    "water_dissociation_barrier_eV": "Water-dissociation barrier (eV)",
    "H_adsorption_free_energy_value": r"$\Delta G_{\mathrm{H}^*}$",
    "H_adsorption_energy_value": r"$\Delta E_{\mathrm{H}^*}$",
    "OH_adsorption_free_energy_value": r"$\Delta G_{OH^*}$",
    "OH_adsorption_energy_value": r"$\Delta E_{OH^*}$",
    "H2O_adsorption_energy_value": r"$\Delta E_{H_2O^*}$",
    "deprotonation_energy_value": "Deprotonation energy",
    "Volmer_barrier": "Volmer barrier",
    "proton_transfer_barrier": "Proton-transfer barrier",
    "water_dissociation_barrier": "Water-dissociation barrier",
}

FEATURE_NAME_MAP = {
    "d_band_center": r"$d$-band center",
    "d_band_center_value": r"$d$-band center",
    "Bader_charge": "Bader charge",
    "bader_charge": "Bader charge",
    "coordination_number": "Coordination number",
    "oxidation_state_value": "Metal oxidation state",
    "oxidation_state": "Oxidation state",
    "work_function": "Work function",
    "PZC": "PZC",
    "U_value": "Applied potential",
    "hydrogen_bond_number": "H-bond number",
    "strong_HB_water_ratio": "Strong-HB water ratio",
    "weak_HB_water_ratio": "Weak-HB water ratio",
    "bridge_oxygen_flag_yes": "Bridge-O motif",
    "vacancy_flag_yes": "Vacancy",
    "hydroxyl_flag_yes": "Hydroxylated surface",
    "has_site": "Active-site evidence",
    "has_deprot_barrier_metric": "Deprotonation/barrier evidence",
    "has_OH_H2O_metric": "OH/H$_2$O evidence",
    "interfacial_water_flag_yes": "Interfacial water",
    "has_electronic_descriptor": "Electronic descriptor",
    "has_mechanism_descriptor": "Mechanism descriptor",
    "has_proton_transfer_metric": "Proton-transfer metric",
}

# Mechanism-related descriptors. Columns outside these patterns will not enter
# Fig5/Fig6 unless passed with --force-features.
MECHANISM_ALLOW_PATTERNS = [
    # Electronic / local structure descriptors
    r"d[_\- ]?band", r"bader", r"charge", r"coordination", r"oxidation",
    r"work[_\- ]?function", r"PZC", r"U[_\- ]?value", r"potential",
    r"ELF", r"charge[_\- ]?density", r"DOS", r"PDOS", r"electron",

    # Structural motifs that are directly relevant to H adsorption / deprotonation
    r"vacancy", r"hydroxyl", r"bridge", r"M[_\- ]?O[_\- ]?X",
    r"defect", r"dopant",

    # Elementary-step descriptors, excluding OH/H2O-specific target variables by default
    r"Volmer", r"Heyrovsky", r"Tafel", r"proton", r"deproton",
]

# Non-mechanistic or extraction-quality artifacts to exclude by default.
DEFAULT_EXCLUDE_CONTAINS = [
    # Extraction/QC/meta fields
    "coverage", "link_score", "confidence", "joint_hads_score", "DFT_priority_score",
    "target_relevance", "system_signal", "metric_signal", "mechanism_signal",
    "qc_", "record", "paper", "chunk", "evidence", "source", "filename", "title",
    "abstract", "doi", "year", "journal", "count", "n_records", "raw", "text",
    "sentence", "span", "page", "section", "relevance", "score", "rank",

    # Explicitly excluded because the requested figure set is H adsorption / deprotonation focused
    "OH_adsorption", "H2O", "H_2O", "water", "hydrogen_bond", "hbond", "HB_water",
    "has_OH_H2O_metric", "interfacial_water", "strong_HB", "weak_HB",

    # Too broad and prone to extraction artifacts in correlation figures
    "support", "surface_facet", "surface_or_facet", "adsorption_site", "active_site_nearby",
]

GENERIC_BAD_CATEGORIES = {
    "", "nan", "none", "null", "unknown", "unk", "n/a", "na",
    "for", "from", "to", "of", "in", "on", "at", "with", "by", "and", "or",
    "role", "shown", "long", "mol", "atoms", "atom", "other", "another", "the", "a", "an",
    "figure", "fig", "table", "surface", "center", "energy", "barrier", "model", "method",
}

BAD_MATERIAL_TOKENS = {
    "perdew-burke-ernzerhof", "monkhorst-pack", "gibbs", "dft", "pbe", "pbe+d2",
    "h-h", "h2", "o-o", "c-o", "c-c", "pt111", "pt(111)", "the", "water", "hydrogen",
}


# =============================================================================
# Small utilities
# =============================================================================

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def cm2inch(w: float, h: Optional[float] = None):
    if h is None:
        return w / 2.54
    return w / 2.54, h / 2.54


def sanitize_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\-.]+", "_", s)
    return s[:170]


def set_pub_style(font_family: str = "Arial", base_size: float = 7.8):
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [font_family, "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 0.6,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.3,
        "ytick.labelsize": base_size - 0.3,
        "legend.fontsize": base_size - 0.5,
        "figure.titlesize": base_size + 1.0,
        "axes.linewidth": 0.75,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.035,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })


def beautify_axes(ax, grid: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.75)
    ax.spines["bottom"].set_linewidth(0.75)
    if grid:
        ax.grid(axis="x", color="#D5D5D5", linewidth=0.45, alpha=0.75)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    return ax


def savefig_multi(fig, outbase: Path, dpi: int = 600, formats: Sequence[str] = ("png", "pdf")):
    for fmt in formats:
        fig.savefig(f"{outbase}.{fmt}", dpi=dpi)
    plt.close(fig)


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}")
        return None


def existing_columns(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def pick_first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def pretty_name(name: str) -> str:
    if name in TARGET_NAME_MAP:
        return TARGET_NAME_MAP[name]
    if name in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[name]
    out = str(name)
    out = out.replace("_value_eV", "")
    out = out.replace("_value", "")
    out = out.replace("_flag_yes", "")
    out = out.replace("qc_has_", "")
    out = out.replace("_", " ")
    out = re.sub(r"\s+", " ", out).strip()
    return out


def pretty_name_plain(name: str) -> str:
    plain_map = {
        "H_adsorption_free_energy_value_eV": "ΔG_H* (eV)",
        "H_adsorption_energy_value_eV": "ΔE_H* (eV)",
        "deprotonation_energy_value_eV": "Deprotonation energy (eV)",
        "Volmer_barrier_eV": "Volmer barrier (eV)",
        "proton_transfer_barrier_eV": "Proton-transfer barrier (eV)",
    }
    if name in plain_map:
        return plain_map[name]
    label = pretty_name(name)
    label = label.replace("$", "")
    label = label.replace("\\Delta", "Δ")
    label = label.replace("_{H^*}", "_H*")
    label = label.replace("_{OH^*}", "_OH*")
    label = label.replace("_{H_2O^*}", "_H2O*")
    label = label.replace("\\", "")
    label = re.sub(r"\{([^}]*)\}", r"\1", label)
    return label


TARGET_MATRIX_LABEL_MAP = {
    "H_adsorption_free_energy_value_eV": r"$\Delta G_{\mathrm{H}^*}$" + "\n(eV)",
    "H_adsorption_energy_value_eV": r"$\Delta E_{\mathrm{H}^*}$" + "\n(eV)",
    "H_adsorption_free_energy_value": r"$\Delta G_{\mathrm{H}^*}$",
    "H_adsorption_energy_value": r"$\Delta E_{\mathrm{H}^*}$",
    "deprotonation_energy_value_eV": "Deprotonation\nenergy (eV)",
    "Volmer_barrier_eV": "Volmer barrier\n(eV)",
    "proton_transfer_barrier_eV": "Proton-transfer\nbarrier (eV)",
}


def target_tick_label(name: str) -> str:
    tick_map = {
        "H_adsorption_free_energy_value_eV": r"$\Delta G_{\mathrm{H}^*}$" + "\n(eV)",
        "H_adsorption_energy_value_eV": r"$\Delta E_{\mathrm{H}^*}$" + "\n(eV)",
        "deprotonation_energy_value_eV": "Deprotonation\nenergy (eV)",
        "Volmer_barrier_eV": "Volmer barrier\n(eV)",
        "proton_transfer_barrier_eV": "Proton-transfer\nbarrier (eV)",
    }
    return tick_map.get(name, wrap_label_two_lines(pretty_name_plain(name), 16))


def wrap_label(s: str, width: int = 24) -> str:
    s = str(s)
    return "\n".join(textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False))


def wrap_label_two_lines(s: str, width: int = 20) -> str:
    """Wrap a label into at most two visually balanced lines."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    if not s:
        return s
    lines = textwrap.wrap(s, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) <= 2:
        return "\n".join(lines)
    words = s.split()
    if len(words) <= 2:
        return "\n".join(lines[:2])
    best = None
    for i in range(1, len(words)):
        l1 = " ".join(words[:i])
        l2 = " ".join(words[i:])
        score = abs(len(l1) - len(l2)) + max(0, len(l1) - width) * 2 + max(0, len(l2) - width) * 2
        cand = (score, l1, l2)
        if best is None or cand[0] < best[0]:
            best = cand
    return f"{best[1]}\n{best[2]}"


def compact_feature_label(name: str, mode: str = "plain") -> str:
    """Shorter labels for crowded mechanism figures."""
    low = str(name).lower()
    label = pretty_name_plain(name)
    label = label.replace("bridge structure nearby", "Bridge structure nearby")
    label = label.replace("proton transfer pathway", "Proton transfer pathway")
    if mode == "coverage":
        base = label.replace(" positive", "").replace(" coverage", "")
        if "yes" in low or "flag_yes" in low or low.endswith(" positive"):
            return f"{base}\n(+ rate)"
        return f"{base}\n(cov.)"
    return label


def axis_label_for_matrix(name: str) -> str:
    # Keep H* target labels in mathtext so H* is always displayed as a subscript/superscript,
    # e.g. $\Delta E_{\mathrm{H}^*}$ instead of plain ΔE_H*.
    if name in TARGET_MATRIX_LABEL_MAP:
        return TARGET_MATRIX_LABEL_MAP[name]
    label = compact_feature_label(name, mode="plain")
    # More stable wrapping for matrix axes, keeping the label centered.
    return wrap_label_two_lines(label, 14)


def feature_should_exclude_from_corr(name: str) -> bool:
    low = str(name).lower()
    # Remove explicit yes/flag helper columns from Fig5/Fig6.
    if "_yes" in low or low.endswith(" yes") or "flag_yes" in low or re.search(r"(^|[_\s])yes($|[_\s])", low):
        return True
    return False


def robust_limits(values: pd.Series, q: float = 0.02, pad: float = 0.05) -> Tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").dropna()
    if vals.empty:
        return -1.0, 1.0
    if len(vals) >= 10 and q > 0:
        lo = float(vals.quantile(q))
        hi = float(vals.quantile(1 - q))
    else:
        lo = float(vals.min())
        hi = float(vals.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    span = hi - lo
    return lo - pad * span, hi + pad * span


def clipped_for_plot(values: pd.Series, q: float = 0.02) -> Tuple[pd.Series, Tuple[float, float], int]:
    vals = pd.to_numeric(values, errors="coerce")
    lo, hi = robust_limits(vals, q=q, pad=0.0)
    n_clip = int(((vals < lo) | (vals > hi)).sum())
    return vals.clip(lo, hi), (lo, hi), n_clip


def physical_range_for_target(target: str) -> Optional[Tuple[float, float]]:
    """Reasonable display/analysis range to suppress obvious extraction artifacts.

    This is not used to alter the original database; it is only applied to plots and
    plot-level statistics. The ranges are intentionally broad for DFT adsorption
    descriptors and barriers.
    """
    t = str(target)
    if "H_adsorption" in t:
        return (-5.0, 5.0)
    if "deprotonation" in t:
        return (-5.0, 5.0)
    if "Volmer_barrier" in t or "proton_transfer_barrier" in t:
        return (0.0, 5.0)
    return None


def target_mask_physical(df: pd.DataFrame, target: str, apply_filter: bool = True) -> pd.Series:
    vals = to_numeric_series(df[target]) if target in df.columns else pd.Series(dtype=float)
    mask = vals.notna()
    if apply_filter:
        rng = physical_range_for_target(target)
        if rng is not None:
            lo, hi = rng
            mask = mask & vals.between(lo, hi)
    return mask


def filtered_df_for_target(df: pd.DataFrame, target: str, apply_filter: bool = True) -> pd.DataFrame:
    if target not in df.columns:
        return df.iloc[0:0].copy()
    return df.loc[target_mask_physical(df, target, apply_filter=apply_filter)].copy()


def is_binary_like(s: pd.Series) -> bool:
    x = to_numeric_series(s).dropna()
    if x.empty:
        return False
    vals = set(np.round(x.unique(), 8).tolist())
    return vals.issubset({0.0, 1.0})


def canonical_category_label(s: str) -> str:
    """Unify common capitalization and bridge-motif variants."""
    s = clean_category_value(s)
    low = s.lower().strip()
    if low in {"bridging oxygen", "bridge oxygen", "bridge-o", "bridge o"}:
        return "Bridge-O"
    if re.fullmatch(r"[A-Z]?[a-z]?[a-z]?", s.strip()):
        # Common dopant/element symbols
        elem = s.strip()
        if elem.lower() in {"pt","ni","co","cu","fe","rh","pd","ir","ru","mo","w","wc","cr","mn","zn","ag","au","ti","v"}:
            return elem[:1].upper() + elem[1:].lower()
    # Normalize common M-O-X strings.
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    # Convert yes/no and true/false categorical flags if possible.
    lower = s.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1.0, "y": 1.0, "true": 1.0, "1": 1.0, "present": 1.0, "detected": 1.0,
        "no": 0.0, "n": 0.0, "false": 0.0, "0": 0.0, "absent": 0.0, "none": 0.0,
    }
    mapped = lower.map(mapping)
    if mapped.notna().sum() >= max(5, int(0.1 * len(s))):
        return mapped
    return pd.to_numeric(s, errors="coerce")



OXIDATION_STATE_CANONICAL = "oxidation_state_value"
OXIDATION_STATE_ALIASES = (
    "oxidation_state",
    "oxidation_state_raw",
    "metal_oxidation_state",
    "metal_oxidation_state_value",
)


def _parse_single_oxidation_state(x: object) -> float:
    """Parse common oxidation-state notations into a numeric value.

    Examples: 2, +2, Fe(III), Ni2+, Co(II), IV -> numeric oxidation state.
    Values that do not look like oxidation states are returned as NaN.
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        return v if np.isfinite(v) and -8 <= v <= 8 else np.nan
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "unknown", "na", "n/a"}:
        return np.nan
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    # Direct signed/integer forms: +2, -1, 3.0
    m = re.search(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)(?![A-Za-z0-9])", s)
    if m:
        try:
            v = float(m.group(1))
            if -8 <= v <= 8:
                return v
        except Exception:
            pass
    # Chemical shorthand such as Ni2+, Fe3+, O2-
    m = re.search(r"[A-Z][a-z]?\s*(\d+(?:\.\d+)?)([+-])", s)
    if m:
        v = float(m.group(1))
        if m.group(2) == "-":
            v = -v
        return v if -8 <= v <= 8 else np.nan
    # Roman numerals, often in Fe(III), Co(II), metal(IV)
    roman_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
    m = re.search(r"\b(VIII|VII|VI|IV|V|III|II|I)\b", s.upper())
    if m:
        return float(roman_map[m.group(1)])
    return np.nan


def parse_oxidation_state_series(s: pd.Series) -> pd.Series:
    """Return numeric oxidation states, including simple Roman/signed forms."""
    num = to_numeric_series(s)
    if num.notna().sum() >= max(3, int(0.2 * len(s))):
        return num.where(num.between(-8, 8))
    return s.map(_parse_single_oxidation_state).astype(float)


def standardize_oxidation_state_columns(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Merge all oxidation-state aliases into one canonical column.

    The plotting figures should only contain one oxidation-state descriptor, shown as
    ``Metal oxidation state``. If both ``oxidation_state_value`` and aliases are
    present, missing canonical values are filled from aliases, then aliases are
    removed to avoid duplicate/correlated labels in Fig5/Fig6/Fig10.
    """
    out = df.copy()
    present_aliases = [c for c in OXIDATION_STATE_ALIASES if c in out.columns]
    if OXIDATION_STATE_CANONICAL not in out.columns and not present_aliases:
        return out
    if OXIDATION_STATE_CANONICAL not in out.columns:
        out[OXIDATION_STATE_CANONICAL] = np.nan
    out[OXIDATION_STATE_CANONICAL] = parse_oxidation_state_series(out[OXIDATION_STATE_CANONICAL])
    before = int(out[OXIDATION_STATE_CANONICAL].notna().sum())
    for alias in present_aliases:
        if alias == OXIDATION_STATE_CANONICAL:
            continue
        parsed = parse_oxidation_state_series(out[alias])
        out[OXIDATION_STATE_CANONICAL] = out[OXIDATION_STATE_CANONICAL].where(
            out[OXIDATION_STATE_CANONICAL].notna(), parsed
        )
    # Final physical sanity filter.
    out[OXIDATION_STATE_CANONICAL] = out[OXIDATION_STATE_CANONICAL].where(
        out[OXIDATION_STATE_CANONICAL].between(-8, 8)
    )
    after = int(out[OXIDATION_STATE_CANONICAL].notna().sum())
    drop_cols = [c for c in present_aliases if c != OXIDATION_STATE_CANONICAL]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    if verbose and (present_aliases or before != after):
        print(
            f"[INFO] Oxidation-state fields standardized: kept '{OXIDATION_STATE_CANONICAL}' "
            f"as 'Metal oxidation state'; filled {after - before} values; dropped {drop_cols}"
        )
    return out


def canonicalize_force_features(features: Sequence[str]) -> List[str]:
    """Map oxidation-state aliases in --force-features to the canonical column."""
    out = []
    seen = set()
    for f in features:
        ff = OXIDATION_STATE_CANONICAL if f in OXIDATION_STATE_ALIASES else f
        if ff not in seen:
            out.append(ff)
            seen.add(ff)
    return out


def coerce_numeric_columns(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = to_numeric_series(df[c])
    return df


def normalize_bool_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create *_yes numeric features from common yes/no columns when needed."""
    out = df.copy()
    for c in list(out.columns):
        lc = c.lower()
        if any(k in lc for k in ["flag", "vacancy", "hydroxyl", "bridge_oxygen", "interfacial_water"]):
            s = out[c]
            if not pd.api.types.is_numeric_dtype(s):
                num = to_numeric_series(s)
                if num.notna().sum() > 0 and num.nunique(dropna=True) <= 2:
                    newc = c if c.endswith("_yes") else f"{c}_yes"
                    if newc not in out.columns:
                        out[newc] = num
    return out


def target_columns(df: pd.DataFrame, min_n: int = 3) -> List[str]:
    """Detect plot targets.

    The refined figure set is intentionally focused on H adsorption and
    deprotonation/proton-transfer descriptors. OH/H2O/water-dissociation targets
    are not reported by default to avoid mixing different mechanistic axes.
    """
    priority = [
        "H_adsorption_free_energy_value_eV",
        "H_adsorption_energy_value_eV",
        "deprotonation_energy_value_eV",
        "Volmer_barrier_eV",
        "proton_transfer_barrier_eV",
        "H_adsorption_free_energy_value",
        "H_adsorption_energy_value",
        "deprotonation_energy_value",
        "Volmer_barrier",
        "proton_transfer_barrier",
    ]
    out = []
    for c in priority:
        if c in df.columns:
            vals = to_numeric_series(df[c])
            if vals.notna().sum() >= min_n and vals.nunique(dropna=True) > 1:
                out.append(c)
    return out


def parse_comma_list(text: str) -> List[str]:
    if text is None:
        return []
    return [x.strip() for x in str(text).split(",") if x.strip()]


def has_any_pattern(name: str, patterns: Sequence[str]) -> bool:
    name = str(name)
    return any(re.search(p, name, flags=re.I) for p in patterns)


def contains_any(name: str, terms: Sequence[str]) -> bool:
    low = str(name).lower()
    return any(str(t).lower() in low for t in terms if str(t).strip())


def is_mechanism_feature_name(name: str, exclude_terms: Sequence[str]) -> bool:
    if contains_any(name, exclude_terms):
        return False
    if has_any_pattern(name, MECHANISM_ALLOW_PATTERNS):
        return True
    return False


def numeric_mechanism_features(
    df: pd.DataFrame,
    targets: Sequence[str],
    force_features: Sequence[str] = (),
    exclude_terms: Sequence[str] = (),
    min_non_null: int = 10,
) -> List[str]:
    out = []
    target_set = set(targets)
    force_set = set(force_features)
    for c in df.columns:
        if c in target_set:
            continue
        # For correlation figures, exclude explicit *_yes helper features unless forced.
        if feature_should_exclude_from_corr(c) and c not in force_set:
            continue
        if c in force_set:
            pass
        elif not is_mechanism_feature_name(c, exclude_terms):
            continue
        s = to_numeric_series(df[c])
        if s.notna().sum() >= min_non_null and s.nunique(dropna=True) > 1:
            out.append(c)
    # Add force features even if non-null threshold is low, when possible.
    for c in force_features:
        if c in df.columns and c not in out and c not in target_set:
            s = to_numeric_series(df[c])
            if s.notna().sum() >= 3 and s.nunique(dropna=True) > 1:
                out.append(c)
    return out


def clean_category_value(x: object, max_len: int = 60) -> str:
    if pd.isna(x):
        return "Unknown"
    s = str(x).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[,;:.\-\s]+|[,;:.\-\s]+$", "", s)
    if not s:
        return "Unknown"
    if len(s) > max_len:
        s = s[:max_len].rstrip() + "..."
    return s


def category_is_valid(s: str, material: bool = False) -> bool:
    low = str(s).strip().lower()
    if low in GENERIC_BAD_CATEGORIES:
        return False
    if material and low in BAD_MATERIAL_TOKENS:
        return False
    if len(low) <= 1:
        return False
    # Remove fragments that are likely extraction artifacts.
    if re.fullmatch(r"[a-z]{1,4}", low) and low not in {"pt", "ni", "co", "cu", "fe", "rh", "pd", "ir", "ru", "mo", "wc"}:
        return False
    if len(low.split()) > 9:
        return False
    return True


def clean_category_series(s: pd.Series, material: bool = False, drop_unknown: bool = True) -> pd.Series:
    out = s.map(lambda x: canonical_category_label(clean_category_value(x)))
    valid = out.map(lambda x: category_is_valid(x, material=material))
    out = out.where(valid, "Unknown")
    if drop_unknown:
        out = out.where(out.str.lower() != "unknown")
    return out


# =============================================================================
# Figure 1: overview dashboard
# =============================================================================

def plot_overview_dashboard(
    paper_df: Optional[pd.DataFrame],
    system_df: Optional[pd.DataFrame],
    site_df: Optional[pd.DataFrame],
    ads_df: Optional[pd.DataFrame],
    model_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
):
    counts = []
    if paper_df is not None:
        counts.append(("Papers", len(paper_df)))
    if system_df is not None:
        counts.append(("Systems", len(system_df)))
    if site_df is not None:
        counts.append(("Sites", len(site_df)))
    if ads_df is not None:
        counts.append(("Adsorption records", len(ads_df)))
    counts.append(("Model-ready rows", len(model_df)))

    targets = target_columns(model_df, min_n=3)
    target_ns = [(t, int(to_numeric_series(model_df[t]).notna().sum())) for t in targets]
    target_ns = target_ns[:8]

    mech_cols = [
        "d_band_center", "Bader_charge", "coordination_number", "oxidation_state",
        "bridge_oxygen_flag_yes", "vacancy_flag_yes", "hydroxyl_flag_yes",
        "defect_type", "dopant", "work_function", "PZC", "U_value",
    ]
    mech_cols = [c for c in mech_cols if c in model_df.columns]
    cov = []
    for c in mech_cols:
        if is_binary_like(model_df[c]):
            value = 100 * (to_numeric_series(model_df[c]).fillna(0) > 0).mean()
            label = pretty_name(c) + " positive"
        else:
            value = 100 * model_df[c].notna().mean()
            label = pretty_name(c) + " coverage"
        cov.append((label, value))
    cov = sorted(cov, key=lambda x: x[1], reverse=True)[:7]

    fig = plt.figure(figsize=cm2inch(19.6, 13.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.08, 1.20], height_ratios=[0.84, 1.16], wspace=0.58, hspace=0.42)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    labels, vals = zip(*counts)
    x = np.arange(len(vals))
    ax1.bar(x, vals, color=PALETTE["blue"], edgecolor=PALETTE["dark"], linewidth=0.65, width=0.78)
    ax1.set_xticks(x)
    ax1.set_xticklabels([wrap_label_two_lines({"Adsorption records":"Adsorption records", "Model-ready rows":"Model-ready rows"}.get(lb, lb), 14) for lb in labels], rotation=18, ha="center", rotation_mode="anchor")
    for lbl in ax1.get_xticklabels():
        lbl.set_fontsize(6.9)
    ax1.tick_params(axis="x", pad=16)
    ax1.set_xlim(-0.70, len(vals) - 0.30)
    ax1.set_ylabel("Count")
    ax1.set_title("A  Database scale", loc="left", fontweight="bold", pad=8)
    beautify_axes(ax1, grid=True)
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ymax = max(vals) if vals else 1
    ax1.margins(y=0.08)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + ymax * 0.025, f"{int(v)}", ha="center", va="bottom", fontsize=7.2)

    if target_ns:
        lab, val = zip(*target_ns)
        yy = np.arange(len(val))
        ax2.barh(yy, val, color=PALETTE["green"], edgecolor=PALETTE["dark"], linewidth=0.55, height=0.72)
        ax2.set_yticks(yy)
        ax2.set_yticklabels([target_tick_label(x) for x in lab])
        ax2.tick_params(axis="y", pad=10)
        ax2.invert_yaxis()
        ax2.set_xlabel("Non-missing records")
        ax2.set_title("B  Target availability", loc="left", fontweight="bold", pad=8)
        beautify_axes(ax2, grid=True)
        xmax = max(val) if val else 1
        ax2.set_xlim(0, xmax * 1.18)
        for yi, v in zip(yy, val):
            ax2.text(v + xmax * 0.018, yi, f"{v}", va="center", fontsize=7.0)
    else:
        ax2.axis("off")

    if cov:
        lab, val = zip(*cov)
        yy = np.arange(len(val))
        # Panel C: slightly thicker bars, but with a taller subplot so rows remain well separated
        ax3.barh(yy, val, color=PALETTE["orange"], edgecolor=PALETTE["dark"], linewidth=0.55, height=0.62)
        ax3.set_yticks(yy)
        ax3.set_yticklabels([compact_feature_label(x, mode="coverage") for x in lab])
        for lbl in ax3.get_yticklabels():
            lbl.set_fontsize(6.2)
        ax3.tick_params(axis="y", pad=15)
        ax3.invert_yaxis()
        ax3.set_xlabel("Positive rate / coverage (%)")
        ax3.set_xlim(0, 112)
        ax3.set_title("C  Mechanistic descriptor summary", loc="left", fontweight="bold", pad=10)
        beautify_axes(ax3, grid=True)
        ax3.margins(y=0.06)
        for yi, v in zip(yy, val):
            ax3.text(v + 1.2, yi, f"{v:.1f}%", va="center", fontsize=7.0)
    else:
        ax3.axis("off")

    fig.subplots_adjust(left=0.06, right=0.985, top=0.965, bottom=0.13)
    savefig_multi(fig, outdir / "Fig1_overview_dashboard", dpi=dpi, formats=formats)


# =============================================================================
# Figure 2: target distribution panel
# =============================================================================

def plot_target_distribution_panel(
    model_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    robust_q: float,
    max_targets: int = 5,
    apply_physical_filter: bool = True,
):
    targets = target_columns(model_df, min_n=3)[:max_targets]
    if not targets:
        return

    rows = []
    target_meta = {}
    for t in targets:
        fdf = filtered_df_for_target(model_df, t, apply_filter=apply_physical_filter)
        vals = to_numeric_series(fdf[t]).dropna()
        if vals.empty:
            continue
        plot_vals, limits, n_clip = clipped_for_plot(vals, q=robust_q)
        target_meta[t] = {
            "n": len(vals),
            "n_removed_physical": int(to_numeric_series(model_df[t]).notna().sum() - len(vals)),
            "n_clip": n_clip,
        }
        for v in plot_vals.dropna():
            rows.append({"target": t, "pretty": pretty_name(t), "value": v})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    targets = [t for t in targets if t in target_meta]

    n = len(targets)
    fig_h = max(6.2, 0.98 * n + 1.65)
    fig, ax = plt.subplots(figsize=cm2inch(15.8, fig_h))
    positions = np.arange(n)

    data = [plot_df.loc[plot_df["target"] == t, "value"].values for t in targets]
    bp = ax.boxplot(
        data,
        vert=False,
        positions=positions,
        widths=0.42,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color=PALETTE["dark"], linewidth=1.05),
        boxprops=dict(linewidth=0.75, color=PALETTE["dark"]),
        whiskerprops=dict(linewidth=0.75, color=PALETTE["dark"]),
        capprops=dict(linewidth=0.75, color=PALETTE["dark"]),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(PALETTE["light_blue"])
        patch.set_alpha(0.90)

    rng = np.random.default_rng(7)
    for i, vals in enumerate(data):
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            continue
        if len(vals) > 120:
            vals = rng.choice(vals, size=120, replace=False)
        jitter = rng.normal(0, 0.045, len(vals))
        ax.scatter(vals, np.full_like(vals, i, dtype=float) + jitter,
                   s=8, color=PALETTE["blue"], alpha=0.45, linewidth=0, zorder=2)

    pretty_labels = []
    for t in targets:
        meta = target_meta[t]
        base_label = wrap_label_two_lines(pretty_name(t), 24)
        pretty_labels.append(base_label + f"\n(n={meta['n']})")

    ax.set_yticks(positions)
    ax.set_yticklabels(pretty_labels)
    ax.tick_params(axis="y", pad=11)
    ax.invert_yaxis()
    ax.axvline(0, color=PALETTE["gray"], linewidth=0.75, linestyle="--", alpha=0.75)
    ax.set_xlabel("Extracted value (eV)")
    ax.set_title("H adsorption / deprotonation target distributions", loc="left", fontweight="bold", pad=8)
    ax.margins(y=0.10)
    beautify_axes(ax, grid=True)

    fig.subplots_adjust(left=0.20, right=0.985, top=0.95, bottom=0.09)
    savefig_multi(fig, outdir / "Fig2_target_distribution_panel", dpi=dpi, formats=formats)

# =============================================================================
# Figure 3: mechanism descriptor coverage
# =============================================================================

def plot_mechanism_coverage(
    model_df: pd.DataFrame,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
):
    candidates = [
        "d_band_center", "Bader_charge", "coordination_number", "oxidation_state",
        "bridge_oxygen_flag_yes", "vacancy_flag_yes", "hydroxyl_flag_yes",
        "defect_type", "dopant", "work_function", "PZC", "U_value",
        "ELF_descriptor", "charge_density_difference_claim",
        "proton_transfer_pathway", "rate_determining_step",
    ]
    cols = [c for c in candidates if c in model_df.columns]
    if not cols:
        return
    rows = []
    for c in cols:
        if is_binary_like(model_df[c]):
            num = to_numeric_series(model_df[c]).fillna(0)
            value = 100 * (num > 0).mean()
            label_type = "positive rate"
        else:
            value = 100 * model_df[c].notna().mean()
            label_type = "coverage"
        rows.append({
            "feature": c,
            "pretty": compact_feature_label(c, mode="coverage"),
            "coverage": value,
        })
    df = pd.DataFrame(rows).sort_values("coverage", ascending=True)

    fig_h = max(9.6, 0.88 * len(df) + 2.3)
    fig, ax = plt.subplots(figsize=cm2inch(15.6, fig_h))
    y = np.arange(len(df))
    norm = Normalize(vmin=0, vmax=max(100, df["coverage"].max()))
    colors = BLUE_TO_RED_CMAP(norm(df["coverage"].values))
    ax.barh(y, df["coverage"].values, color=colors, edgecolor=PALETTE["dark"], linewidth=0.45, height=0.54)
    ax.set_yticks(y)
    ax.set_yticklabels(df["pretty"].tolist())
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(6.4)
    ax.tick_params(axis="y", pad=11)
    ax.set_xlabel("Positive rate / coverage (%)")
    ax.set_xlim(0, 112)
    ax.set_title("Mechanism-relevant descriptor summary", loc="left", fontweight="bold", pad=8)
    beautify_axes(ax, grid=True)
    ax.margins(y=0.06)
    for yi, cov in zip(y, df["coverage"].values):
        ax.text(cov + 1.2, yi, f"{cov:.1f}%", va="center", fontsize=6.9)

    fig.subplots_adjust(left=0.37, right=0.985, top=0.95, bottom=0.08)
    savefig_multi(fig, outdir / "Fig3_mechanistic_descriptor_summary", dpi=dpi, formats=formats)

# =============================================================================
# Figure 4: category distribution, cleaned
# =============================================================================

def plot_top_categories_cleaned(
    df: pd.DataFrame,
    col: str,
    outbase: Path,
    title: str,
    dpi: int,
    formats: Sequence[str],
    topn: int = 15,
    material: bool = False,
):
    if col not in df.columns:
        return
    s = clean_category_series(df[col], material=material, drop_unknown=True).dropna()
    if s.empty:
        return
    counts = s.value_counts().head(topn)
    if counts.empty:
        return
    counts = counts.iloc[::-1]
    fig_h = max(6.2, 0.43 * len(counts) + 1.4)
    fig, ax = plt.subplots(figsize=cm2inch(13.8, fig_h))
    y = np.arange(len(counts))
    ax.barh(y, counts.values, color=PALETTE["blue"], edgecolor=PALETTE["dark"], linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, 28) for x in counts.index])
    ax.set_xlabel("Count")
    ax.set_title(title, loc="left", fontweight="bold")
    beautify_axes(ax, grid=True)
    xmax = max(counts.values) if len(counts) else 1
    for yi, v in zip(y, counts.values):
        ax.text(v + xmax * 0.015, yi, f"{int(v)}", va="center", fontsize=7.2)
    savefig_multi(fig, outbase, dpi=dpi, formats=formats)


def plot_mechanism_category_summary(
    model_df: pd.DataFrame,
    system_df: Optional[pd.DataFrame],
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
):
    # Keep this panel mechanism-focused. Material-name panels are omitted by default
    # because they are more prone to extraction fragments and are less central to the mechanism story.
    fig = plt.figure(figsize=cm2inch(18.0, 10.5))
    gs = fig.add_gridspec(1, 3, wspace=0.48)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    panels = [
        ("bridge_structure", "A  Bridge/interface motifs", False, PALETTE["green"]),
        ("dopant", "B  Dopants", False, PALETTE["purple"]),
        ("defect_type", "C  Defect types", False, PALETTE["orange"]),
    ]
    for ax, (col, title, is_mat, color) in zip(axes, panels):
        src = model_df
        if col not in src.columns:
            ax.axis("off")
            continue
        s = clean_category_series(src[col], material=is_mat, drop_unknown=True).dropna()
        counts = s.value_counts().head(10).iloc[::-1]
        if counts.empty:
            ax.axis("off")
            continue
        y = np.arange(len(counts))
        ax.barh(y, counts.values, color=color, edgecolor=PALETTE["dark"], linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(x, 20) for x in counts.index])
        ax.set_xlabel("Count")
        ax.set_title(title, loc="left", fontweight="bold")
        beautify_axes(ax, grid=True)
        xmax = max(counts.values) if len(counts) else 1
        for yi, v in zip(y, counts.values):
            ax.text(v + xmax * 0.02, yi, str(int(v)), va="center", fontsize=7.0)
    savefig_multi(fig, outdir / "Fig4_mechanism_category_summary", dpi=dpi, formats=formats)

# =============================================================================
# Figure 5: mechanism-only target correlation strip
# =============================================================================

def compute_feature_target_correlations(
    df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    min_pair_n: int = 8,
    apply_physical_filter: bool = True,
) -> pd.DataFrame:
    base = filtered_df_for_target(df, target, apply_filter=apply_physical_filter)
    if base.empty:
        return pd.DataFrame()
    y = to_numeric_series(base[target])
    rows = []
    for f in features:
        if f not in base.columns:
            continue
        x = to_numeric_series(base[f])
        mask = x.notna() & y.notna()
        n = int(mask.sum())
        if n < min_pair_n:
            continue
        xv = x.loc[mask].astype(float)
        yv = y.loc[mask].astype(float)
        if xv.nunique() <= 1 or yv.nunique() <= 1:
            continue
        r = float(np.corrcoef(xv, yv)[0, 1])
        if np.isfinite(r):
            rows.append({"feature": f, "r": r, "abs_r": abs(r), "n": n})
    return pd.DataFrame(rows)

def plot_corr_strip(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topk: int = 12,
):
    use_features = [f for f in features if not feature_should_exclude_from_corr(f)]
    corr = compute_feature_target_correlations(model_df, target, use_features)
    if corr.empty:
        return
    corr["display"] = corr["feature"].map(lambda x: compact_feature_label(x, mode="plain"))
    corr = corr.sort_values("abs_r", ascending=False).drop_duplicates(subset=["display"], keep="first")
    corr = corr.head(topk).sort_values("r")
    n = len(corr)
    fig_h = max(5.8, 0.46 * n + 1.65)
    fig, ax = plt.subplots(figsize=cm2inch(12.0, fig_h))
    y = np.arange(n)
    rvals = corr["r"].values
    colors = [PALETTE["red"] if r >= 0 else PALETTE["blue"] for r in rvals]
    ax.hlines(y, xmin=0, xmax=rvals, color=colors, linewidth=1.0, alpha=0.88)
    sizes = 44 + 300 * np.abs(rvals)
    ax.scatter(rvals, y, s=sizes, color=colors, edgecolor=PALETTE["dark"], linewidth=0.42, zorder=3, clip_on=False)

    text_pad = 0.075
    xmin = min(-1.0, float(np.min(rvals)) - text_pad - 0.04)
    xmax = max(1.0, float(np.max(rvals)) + text_pad + 0.04)
    xmin = max(-1.20, xmin)
    xmax = min(1.20, xmax)
    for yi, r in zip(y, corr["r"].values):
        if r >= 0:
            xt = min(r + text_pad, xmax - 0.02)
            ha = "left"
        else:
            xt = max(r - text_pad, xmin + 0.02)
            ha = "right"
        ax.text(xt, yi, f"{r:.2f}", va="center", ha=ha, fontsize=6.8, clip_on=False)

    ax.axvline(0, color=PALETTE["dark"], linewidth=0.75)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label_two_lines(x, 20) for x in corr["display"]])
    ax.tick_params(axis="y", pad=5)
    ax.set_xlim(xmin, xmax)
    ax.margins(y=0.10)
    ax.set_xlabel(f"Pearson correlation with {pretty_name(target)}")
    ax.set_title(f"Mechanism feature correlations with {pretty_name(target)}", loc="left", fontweight="bold", pad=8)
    beautify_axes(ax, grid=True)

    fig.subplots_adjust(left=0.24, right=0.97, top=0.93, bottom=0.12)
    savefig_multi(fig, outdir / f"Fig5_mechanism_corr_strip__{sanitize_filename(target)}", dpi=dpi, formats=formats)

# =============================================================================
# Figure 6: lower-left bubbles + upper-right numeric labels
# =============================================================================

def plot_corr_bubble_matrix_v2(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topk: int = 10,
    bubble_max: float = 260.0,
):
    use_features = [f for f in features if not feature_should_exclude_from_corr(f)]
    corr_rank = compute_feature_target_correlations(model_df, target, use_features)
    if corr_rank.empty or len(corr_rank) < 3:
        return
    corr_rank["display"] = corr_rank["feature"].map(lambda x: compact_feature_label(x, mode="plain"))
    corr_rank = corr_rank.sort_values("abs_r", ascending=False).drop_duplicates(subset=["display"], keep="first")
    top_feats = corr_rank.head(topk)["feature"].tolist()
    use_cols = top_feats + [target]

    base = filtered_df_for_target(model_df, target, apply_filter=True)
    tmp = pd.DataFrame({c: to_numeric_series(base[c]) for c in use_cols if c in base.columns})
    corr = tmp.corr(numeric_only=True)
    valid_cols = [c for c in use_cols if c in corr.columns and corr[c].notna().sum() > 1]
    corr = corr.loc[valid_cols, valid_cols]
    if corr.shape[0] < 4:
        return

    if target in corr.columns:
        feats = [c for c in corr.columns if c != target]
        feats = sorted(feats, key=lambda c: corr.loc[c, target] if pd.notna(corr.loc[c, target]) else 0)
        corr = corr.loc[feats + [target], feats + [target]]

    display_cols = [axis_label_for_matrix(c) for c in corr.columns]
    n = len(corr)
    fig_size = max(13.0, 0.92 * n + 4.2)
    fig, ax = plt.subplots(figsize=cm2inch(fig_size, fig_size))

    for i in range(n + 1):
        ax.axhline(i - 0.5, color="#E4E4E4", linewidth=0.42, zorder=0)
        ax.axvline(i - 0.5, color="#E4E4E4", linewidth=0.42, zorder=0)

    xs, ys, sizes, colors = [], [], [], []
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if i > j:
                xs.append(j)
                ys.append(i)
                colors.append(val)
                sizes.append(10 + bubble_max * (abs(val) ** 1.34))
            elif i < j:
                color = PALETTE["red"] if val >= 0 else PALETTE["blue"]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6.1, color=color, fontweight="bold" if abs(val) >= 0.55 else "normal")
            else:
                face = "#F4F4F4" if corr.index[i] != target else "#FFF1E6"
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=face,
                                           edgecolor="#D0D0D0", linewidth=0.42, zorder=1))

    if xs:
        sc = ax.scatter(xs, ys, s=sizes, c=colors, cmap=RED_BLUE_CMAP, vmin=-1, vmax=1,
                        edgecolor=PALETTE["dark"], linewidth=0.28, alpha=0.92, zorder=3)
    else:
        sc = ScalarMappable(norm=Normalize(-1, 1), cmap=RED_BLUE_CMAP)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(display_cols, rotation=32, ha="center", rotation_mode="anchor")
    ax.set_yticklabels(display_cols)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(5.7)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(6.6)
    ax.tick_params(axis="x", pad=11, length=0)
    ax.tick_params(axis="y", pad=5, length=0)
    ax.set_title(f"Mechanism correlation matrix for {pretty_name(target)}", loc="left", fontweight="bold", pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(sc, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("Pearson r")

    fig.subplots_adjust(left=0.28, right=0.93, top=0.94, bottom=0.29)
    savefig_multi(fig, outdir / f"Fig6_corr_bubble_matrix_v2__{sanitize_filename(target)}", dpi=dpi, formats=formats)


# =============================================================================
# Figure 10/11/12: detailed ΔE_H* feature-direction analysis
# =============================================================================

def feature_type_for_effect(x: pd.Series) -> str:
    """Classify numeric feature as binary-like or continuous for effect-direction plots."""
    xv = to_numeric_series(x).dropna()
    if xv.empty:
        return "empty"
    vals = set(np.round(xv.unique(), 8).tolist())
    if vals.issubset({0.0, 1.0}) or len(vals) <= 2:
        return "binary"
    return "continuous"


def safe_iqr(x: pd.Series) -> float:
    xv = to_numeric_series(x).dropna()
    if xv.empty:
        return np.nan
    q1 = float(xv.quantile(0.25))
    q3 = float(xv.quantile(0.75))
    return float(q3 - q1)


def cliffs_delta(x_high: Sequence[float], x_low: Sequence[float]) -> float:
    """Cliff's delta for two groups: positive means high-group target values are larger."""
    a = np.asarray(pd.Series(x_high).dropna(), dtype=float)
    b = np.asarray(pd.Series(x_low).dropna(), dtype=float)
    if a.size == 0 or b.size == 0:
        return np.nan
    # Pairwise compare. For current dataset sizes this is still lightweight and dependency-free.
    gt = (a[:, None] > b[None, :]).sum()
    lt = (a[:, None] < b[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


def compute_deltae_feature_effects(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    min_pair_n: int = 8,
    apply_physical_filter: bool = True,
) -> pd.DataFrame:
    """Estimate how increasing each feature shifts ΔE_H*.

    For continuous features, the reported effect is median(target | feature in top quartile)
    minus median(target | feature in bottom quartile). For binary features, it is
    median(target | feature=1) minus median(target | feature=0). A positive value means
    the feature increase / presence is associated with higher ΔE_H*; a negative value means
    it is associated with lower ΔE_H*.

    Additional unitless effect sizes are also reported:
      - effect_sd_norm: raw median-shift divided by the global SD of the target.
      - effect_iqr_norm: raw median-shift divided by the global IQR of the target.
      - cliffs_delta: binary-feature effect size in [-1, 1].
    """
    if target not in model_df.columns:
        return pd.DataFrame()
    base = filtered_df_for_target(model_df, target, apply_filter=apply_physical_filter)
    if base.empty:
        return pd.DataFrame()
    y_all = to_numeric_series(base[target])
    target_std = float(y_all.std(ddof=0)) if y_all.notna().sum() >= 2 else np.nan
    target_iqr = safe_iqr(y_all)
    rows = []
    for f in features:
        if f not in base.columns or feature_should_exclude_from_corr(f):
            continue
        x_all = to_numeric_series(base[f])
        use = pd.DataFrame({"x": x_all, "y": y_all}).dropna()
        if len(use) < min_pair_n or use["x"].nunique() <= 1 or use["y"].nunique() <= 1:
            continue
        ftype = feature_type_for_effect(use["x"])
        min_group_n = max(3, min_pair_n // 3)
        group_method = ""
        if ftype == "binary":
            low = use.loc[use["x"] <= use["x"].min()]
            high = use.loc[use["x"] >= use["x"].max()]
            low_label = "0 / low"
            high_label = "1 / high"
            group_method = "binary_min_max"
        else:
            q25 = use["x"].quantile(0.25)
            q75 = use["x"].quantile(0.75)
            low = use.loc[use["x"] <= q25]
            high = use.loc[use["x"] >= q75]
            low_label = "bottom quartile"
            high_label = "top quartile"
            group_method = "quartile"

            # For heavily discrete but non-binary descriptors, fall back to min/max or ranked tails
            # so that valid directionality is not dropped just because quartile groups are tiny.
            if len(low) < min_group_n or len(high) < min_group_n:
                low = use.loc[use["x"] <= use["x"].min()]
                high = use.loc[use["x"] >= use["x"].max()]
                low_label = "minimum group"
                high_label = "maximum group"
                group_method = "min_max_discrete"

            if len(low) < min_group_n or len(high) < min_group_n:
                k = min(max(min_group_n, int(math.ceil(0.25 * len(use)))), len(use) // 2)
                if k >= 2:
                    low = use.nsmallest(k, "x")
                    high = use.nlargest(k, "x")
                    low_label = "ranked low group"
                    high_label = "ranked high group"
                    group_method = "ranked_low_high"

        if len(low) < 2 or len(high) < 2:
            continue
        med_low = float(low["y"].median())
        med_high = float(high["y"].median())
        effect = med_high - med_low
        try:
            pearson_r = float(use["x"].corr(use["y"], method="pearson"))
        except Exception:
            pearson_r = np.nan
        try:
            spearman_r = float(use["x"].corr(use["y"], method="spearman"))
        except Exception:
            spearman_r = np.nan
        # Standardized linear slope: ΔE_H* shift per one standard-deviation increase in feature.
        x_std = float(use["x"].std(ddof=0))
        y_std = float(use["y"].std(ddof=0))
        if np.isfinite(x_std) and x_std > 0 and np.isfinite(y_std) and y_std > 0:
            slope = float(np.polyfit((use["x"] - use["x"].mean()) / x_std, use["y"], 1)[0])
            beta_unitless = float(slope / y_std)
        else:
            slope = np.nan
            beta_unitless = np.nan
        effect_sd_norm = float(effect / target_std) if np.isfinite(target_std) and target_std > 0 else np.nan
        effect_iqr_norm = float(effect / target_iqr) if np.isfinite(target_iqr) and target_iqr > 0 else np.nan
        binary_cliffs = cliffs_delta(high["y"].values, low["y"].values) if ftype == "binary" else np.nan
        rows.append({
            "feature": f,
            "label": compact_feature_label(f, mode="plain"),
            "feature_type": ftype,
            "n_pair": int(len(use)),
            "n_low": int(len(low)),
            "n_high": int(len(high)),
            "median_low_DeltaE_H_eV": med_low,
            "median_high_DeltaE_H_eV": med_high,
            "effect_high_minus_low_eV": float(effect),
            "abs_effect_eV": float(abs(effect)),
            "effect_sd_norm": effect_sd_norm,
            "abs_effect_sd_norm": float(abs(effect_sd_norm)) if np.isfinite(effect_sd_norm) else np.nan,
            "effect_iqr_norm": effect_iqr_norm,
            "abs_effect_iqr_norm": float(abs(effect_iqr_norm)) if np.isfinite(effect_iqr_norm) else np.nan,
            "pearson_r": pearson_r,
            "spearman_r": spearman_r,
            "standardized_slope_eV_per_SD": slope,
            "standardized_beta_unitless": beta_unitless,
            "cliffs_delta": binary_cliffs,
            "abs_cliffs_delta": float(abs(binary_cliffs)) if np.isfinite(binary_cliffs) else np.nan,
            "target_global_std_eV": target_std,
            "target_global_iqr_eV": target_iqr,
            "low_group": low_label,
            "high_group": high_label,
            "group_method": group_method,
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("abs_effect_eV", ascending=False).reset_index(drop=True)
    return out


def _plot_deltae_effect_bar_generic(
    effects: pd.DataFrame,
    value_col: str,
    label_col: str,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int,
    filename_stub: str,
    xlabel: str,
    title: str,
    left_guide: str,
    right_guide: str,
    xlim: tuple | None = None,
):
    if effects.empty or value_col not in effects.columns:
        return
    tmp = effects.loc[np.isfinite(effects[value_col].values)].copy()
    if tmp.empty:
        return
    tmp["_abs_sort"] = np.abs(tmp[value_col].values)
    tmp = tmp.sort_values("_abs_sort", ascending=False).head(topn).sort_values(value_col)
    n = len(tmp)
    fig_h = max(6.2, 0.52 * n + 1.65)
    fig, ax = plt.subplots(figsize=cm2inch(14.2, fig_h))
    y = np.arange(n)
    vals = tmp[value_col].values
    colors = [PALETTE["red"] if v >= 0 else PALETTE["blue"] for v in vals]
    ax.barh(y, vals, color=colors, edgecolor=PALETTE["dark"], linewidth=0.48, height=0.68)
    ax.axvline(0, color=PALETTE["dark"], linewidth=0.80)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label_two_lines(x, 23) for x in tmp[label_col]])
    ax.tick_params(axis="y", pad=7)
    ax.set_xlabel(xlabel)
    fig.suptitle(title, x=0.125, y=0.975, ha="left", va="top", fontweight="bold", fontsize=10.8)
    beautify_axes(ax, grid=True)
    xmax = max(0.01, float(np.nanmax(np.abs(vals))))
    if xlim is None:
        ax.set_xlim(-1.18 * xmax, 1.18 * xmax)
    else:
        ax.set_xlim(*xlim)
    for yi, v in zip(y, vals):
        if v >= 0:
            ha, xt = "left", v + max(xmax, 1e-6) * 0.035
        else:
            ha, xt = "right", v - max(xmax, 1e-6) * 0.035
        ax.text(xt, yi, f"{v:+.2f}", ha=ha, va="center", fontsize=6.8, clip_on=False)
    def _guide_two_lines(s: str) -> str:
        s = s.replace("  -> ", "\n→ ").replace(" -> ", "\n→ ")
        return s

    left_guide_disp = _guide_two_lines(left_guide)
    right_guide_disp = _guide_two_lines(right_guide)
    ax.text(0.24, 1.018, left_guide_disp, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.8, linespacing=1.05,
            color=PALETTE["blue"])
    ax.text(0.76, 1.018, right_guide_disp, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5.8, linespacing=1.05,
            color=PALETTE["red"])
    fig.subplots_adjust(left=0.28, right=0.965, top=0.84, bottom=0.12)
    savefig_multi(fig, outdir / filename_stub, dpi=dpi, formats=formats)



def plot_deltae_feature_effect_direction(
    effects: pd.DataFrame,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 12,
):
    if effects.empty:
        return
    _plot_deltae_effect_bar_generic(
        effects=effects,
        value_col="effect_high_minus_low_eV",
        label_col="label",
        target=target,
        outdir=outdir,
        dpi=dpi,
        formats=formats,
        topn=topn,
        filename_stub=f"Fig10_DeltaEH_feature_direction_effect__{sanitize_filename(target)}",
        xlabel=r"Median shift in $\Delta E_{H^*}$ from low to high feature group (eV)",
        title=r"Feature increase / presence associated with $\Delta E_{H^*}$ shift",
        left_guide=r"Lower $\Delta E_{H^*}$  -> stronger H* binding",
        right_guide=r"Higher $\Delta E_{H^*}$  -> weaker H* binding",
    )



def plot_deltae_feature_effect_standardized(
    effects: pd.DataFrame,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 12,
    norm: str = "sd",
):
    if effects.empty:
        return
    if norm == "sd":
        value_col = "effect_sd_norm"
        filename_stub = f"Fig10b_DeltaEH_feature_direction_effect_SDnorm__{sanitize_filename(target)}"
        xlabel = r"Standardized median shift in $\Delta E_{H^*}$ (high minus low; SD units)"
        title = r"Unitless directionality of mechanism features on $\Delta E_{H^*}$ (SD normalized)"
    else:
        value_col = "effect_iqr_norm"
        filename_stub = f"Fig10c_DeltaEH_feature_direction_effect_IQRnorm__{sanitize_filename(target)}"
        xlabel = r"Standardized median shift in $\Delta E_{H^*}$ (high minus low; IQR units)"
        title = r"Unitless directionality of mechanism features on $\Delta E_{H^*}$ (IQR normalized)"
    _plot_deltae_effect_bar_generic(
        effects=effects,
        value_col=value_col,
        label_col="label",
        target=target,
        outdir=outdir,
        dpi=dpi,
        formats=formats,
        topn=topn,
        filename_stub=filename_stub,
        xlabel=xlabel,
        title=title,
        left_guide=r"Negative  -> lower $\Delta E_{H^*}$ after feature increase / presence",
        right_guide=r"Positive  -> higher $\Delta E_{H^*}$ after feature increase / presence",
    )



def plot_deltae_binary_cliffs(
    effects: pd.DataFrame,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 12,
):
    if effects.empty or "cliffs_delta" not in effects.columns:
        return
    tmp = effects[(effects["feature_type"] == "binary") & np.isfinite(effects["cliffs_delta"])].copy()
    if tmp.empty:
        return
    tmp = tmp.sort_values("abs_cliffs_delta", ascending=False).head(topn)
    tmp.to_csv(outdir / f"DeltaEH_binary_presence_effect__{sanitize_filename(target)}.csv", index=False)
    _plot_deltae_effect_bar_generic(
        effects=tmp,
        value_col="cliffs_delta",
        label_col="label",
        target=target,
        outdir=outdir,
        dpi=dpi,
        formats=formats,
        topn=min(topn, len(tmp)),
        filename_stub=f"Fig10d_DeltaEH_binary_presence_effect_CliffsDelta__{sanitize_filename(target)}",
        xlabel=r"Cliff's $\delta$ for $\Delta E_{H^*}$ (presence/high vs absence/low)",
        title=r"Binary-feature effect size on $\Delta E_{H^*}$ (unitless Cliff's $\delta$)",
        left_guide=r"Negative  -> presence associated with lower $\Delta E_{H^*}$",
        right_guide=r"Positive  -> presence associated with higher $\Delta E_{H^*}$",
        xlim=(-1.05, 1.05),
    )


def make_binned_response_table(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    min_pair_n: int = 8,
    n_bins: int = 5,
) -> pd.DataFrame:
    if target not in model_df.columns:
        return pd.DataFrame()
    base = filtered_df_for_target(model_df, target, apply_filter=True)
    y_all = to_numeric_series(base[target])
    rows = []
    for f in features:
        if f not in base.columns or feature_should_exclude_from_corr(f):
            continue
        x_all = to_numeric_series(base[f])
        use = pd.DataFrame({"x": x_all, "y": y_all}).dropna()
        if len(use) < min_pair_n or use["x"].nunique() <= 1:
            continue
        ftype = feature_type_for_effect(use["x"])
        if ftype == "binary" or use["x"].nunique() <= n_bins:
            bins = sorted(use["x"].dropna().unique())
            for i, b in enumerate(bins):
                sub = use.loc[use["x"] == b]
                if len(sub) < 2:
                    continue
                rows.append({
                    "feature": f, "label": compact_feature_label(f, mode="plain"),
                    "bin_order": i, "bin_label": str(round(float(b), 3)),
                    "x_median": float(sub["x"].median()), "n": int(len(sub)),
                    "y_median": float(sub["y"].median()),
                    "y_q25": float(np.nanpercentile(sub["y"], 25)),
                    "y_q75": float(np.nanpercentile(sub["y"], 75)),
                })
        else:
            try:
                use["bin"] = pd.qcut(use["x"], q=min(n_bins, use["x"].nunique()), duplicates="drop")
            except Exception:
                continue
            grouped = use.groupby("bin", observed=True)
            for i, (b, sub) in enumerate(grouped):
                if len(sub) < 2:
                    continue
                rows.append({
                    "feature": f, "label": compact_feature_label(f, mode="plain"),
                    "bin_order": i, "bin_label": f"Q{i+1}",
                    "x_median": float(sub["x"].median()), "n": int(len(sub)),
                    "y_median": float(sub["y"].median()),
                    "y_q25": float(np.nanpercentile(sub["y"], 25)),
                    "y_q75": float(np.nanpercentile(sub["y"], 75)),
                })
    return pd.DataFrame(rows)


def plot_deltae_binned_responses(
    model_df: pd.DataFrame,
    effects: pd.DataFrame,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 6,
    min_pair_n: int = 8,
):
    if effects.empty:
        return
    feats = effects.head(topn)["feature"].tolist()
    binned = make_binned_response_table(model_df, target, feats, min_pair_n=min_pair_n, n_bins=5)
    if binned.empty:
        return
    binned.to_csv(outdir / f"DeltaEH_binned_response_table__{sanitize_filename(target)}.csv", index=False)
    feats = [f for f in feats if f in set(binned["feature"])]
    if not feats:
        return
    n = len(feats)
    ncols = 2
    nrows = int(math.ceil(n / ncols))
    fig_w = 16.4
    fig_h = max(7.8, 4.0 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=cm2inch(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()
    for ax, f in zip(axes_flat, feats):
        sub = binned[binned["feature"] == f].sort_values("bin_order")
        x = np.arange(len(sub))
        y = sub["y_median"].values
        lo = np.maximum(0, y - sub["y_q25"].values)
        hi = np.maximum(0, sub["y_q75"].values - y)
        ax.errorbar(x, y, yerr=[lo, hi], fmt="o-", color=PALETTE["blue"],
                    ecolor=PALETTE["gray"], elinewidth=0.8, capsize=2.2,
                    markersize=4.3, markeredgecolor=PALETTE["dark"], markeredgewidth=0.35)
        ax.axhline(0, color=PALETTE["gray"], linestyle="--", linewidth=0.65, alpha=0.7)
        ax.set_xticks(x)
        # Use compact low-to-high ordinal labels; exact x medians are saved in CSV.
        ax.set_xticklabels(sub["bin_label"].tolist())
        ax.set_title(wrap_label_two_lines(sub["label"].iloc[0], 24), loc="left", fontweight="bold", pad=5)
        ax.set_ylabel(r"Median $\Delta E_{H^*}$ (eV)")
        ax.set_xlabel("Feature group from low to high")
        beautify_axes(ax, grid=True)
    for ax in axes_flat[n:]:
        ax.axis("off")
    fig.suptitle(r"Binned response of $\Delta E_{H^*}$ to key mechanism features", fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    savefig_multi(fig, outdir / f"Fig11_DeltaEH_binned_feature_response__{sanitize_filename(target)}", dpi=dpi, formats=formats)


def plot_deltae_ml_importance(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 12,
    min_pair_n: int = 8,
):
    """Optional model-based importance for ΔE_H*. Direction-free but useful for ranking influence."""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
    except Exception as exc:
        print(f"[INFO] Skip ΔE_H* ML importance because scikit-learn is unavailable: {exc}")
        return
    if target not in model_df.columns:
        return
    base = filtered_df_for_target(model_df, target, apply_filter=True)
    y = to_numeric_series(base[target])
    use_features = [f for f in features if f in base.columns and not feature_should_exclude_from_corr(f)]
    X = pd.DataFrame({f: to_numeric_series(base[f]) for f in use_features})
    valid_cols = [c for c in X.columns if X[c].notna().sum() >= min_pair_n and X[c].nunique(dropna=True) > 1]
    X = X[valid_cols]
    mask = y.notna() & X.notna().any(axis=1)
    X = X.loc[mask]
    y = y.loc[mask]
    if len(y) < max(30, min_pair_n * 3) or X.shape[1] < 2:
        print("[INFO] Skip ΔE_H* ML importance: insufficient data after filtering.")
        return
    test_size = 0.25 if len(y) >= 80 else 0.30
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=13)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=500, random_state=13, min_samples_leaf=3, n_jobs=-1)
    )
    model.fit(X_train, y_train)
    try:
        score = float(model.score(X_test, y_test))
    except Exception:
        score = np.nan
    perm = permutation_importance(model, X_test, y_test, n_repeats=25, random_state=13, n_jobs=-1)
    imp = pd.DataFrame({
        "feature": X.columns,
        "label": [compact_feature_label(c, mode="plain") for c in X.columns],
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
        "test_R2": score,
        "n_train": len(y_train),
        "n_test": len(y_test),
    }).sort_values("permutation_importance_mean", ascending=False)
    imp.to_csv(outdir / f"DeltaEH_ML_permutation_importance__{sanitize_filename(target)}.csv", index=False)
    tmp = imp.head(topn).iloc[::-1]
    if tmp.empty:
        return
    fig_h = max(5.8, 0.46 * len(tmp) + 1.55)
    fig, ax = plt.subplots(figsize=cm2inch(13.6, fig_h))
    y_pos = np.arange(len(tmp))
    vals = tmp["permutation_importance_mean"].values
    err = tmp["permutation_importance_std"].values
    ax.barh(y_pos, vals, xerr=err, color=PALETTE["purple"], edgecolor=PALETTE["dark"], linewidth=0.45, height=0.68)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([wrap_label_two_lines(x, 23) for x in tmp["label"]])
    ax.set_xlabel(r"Permutation importance for $\Delta E_{\mathrm{H}^*}$ prediction")
    title = r"Model-based feature importance for $\Delta E_{\mathrm{H}^*}$"
    if np.isfinite(score):
        title += f" (test R²={score:.2f})"
    ax.set_title(title, loc="left", fontweight="bold", pad=8)
    beautify_axes(ax, grid=True)
    fig.subplots_adjust(left=0.28, right=0.965, top=0.92, bottom=0.12)
    savefig_multi(fig, outdir / f"Fig12_DeltaEH_ML_permutation_importance__{sanitize_filename(target)}", dpi=dpi, formats=formats)


def run_deltae_detailed_analysis(
    model_df: pd.DataFrame,
    target: str,
    features: Sequence[str],
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    topn: int = 12,
    min_pair_n: int = 8,
    run_ml: bool = True,
):
    if target not in model_df.columns:
        print(f"[INFO] Skip detailed ΔE_H* analysis: target not found: {target}")
        return
    effects = compute_deltae_feature_effects(model_df, target, features, min_pair_n=min_pair_n)
    if effects.empty:
        print(f"[INFO] Skip detailed ΔE_H* analysis for {target}: no valid effects.")
        return
    # Main summary with raw, standardized, and binary effect-size columns.
    effects.to_csv(outdir / f"DeltaEH_feature_effect_summary__{sanitize_filename(target)}.csv", index=False)
    # Convenience ranking tables.
    effects.sort_values("abs_effect_sd_norm", ascending=False).to_csv(
        outdir / f"DeltaEH_feature_effect_summary_SDnorm_ranked__{sanitize_filename(target)}.csv", index=False
    )
    effects.sort_values("abs_effect_iqr_norm", ascending=False).to_csv(
        outdir / f"DeltaEH_feature_effect_summary_IQRnorm_ranked__{sanitize_filename(target)}.csv", index=False
    )
    plot_deltae_feature_effect_direction(effects, target, outdir, dpi, formats, topn=topn)
    plot_deltae_feature_effect_standardized(effects, target, outdir, dpi, formats, topn=topn, norm="sd")
    plot_deltae_feature_effect_standardized(effects, target, outdir, dpi, formats, topn=topn, norm="iqr")
    plot_deltae_binary_cliffs(effects, target, outdir, dpi, formats, topn=topn)
    plot_deltae_binned_responses(model_df, effects, target, outdir, dpi, formats, topn=min(6, topn), min_pair_n=min_pair_n)
    if run_ml:
        plot_deltae_ml_importance(model_df, target, features, outdir, dpi, formats, topn=topn, min_pair_n=min_pair_n)

# =============================================================================
# Figure 7: categorical effect, cleaned and robust
# =============================================================================

def category_effect_table(
    model_df: pd.DataFrame,
    cat_col: str,
    target: str,
    min_count: int,
    topn: int,
    robust_q: float,
) -> pd.DataFrame:
    if cat_col not in model_df.columns or target not in model_df.columns:
        return pd.DataFrame()
    base = filtered_df_for_target(model_df, target, apply_filter=True)
    if base.empty:
        return pd.DataFrame()
    material = cat_col == "material_name"
    cats = clean_category_series(base[cat_col], material=material, drop_unknown=True)
    y = to_numeric_series(base[target])
    plot_y, _, _ = clipped_for_plot(y, q=robust_q)
    use = pd.DataFrame({"cat": cats, "target": y, "plot_target": plot_y}).dropna(subset=["cat", "target"])
    if use.empty:
        return pd.DataFrame()
    grp = use.groupby("cat", dropna=False).agg(
        n=("target", "size"),
        mean=("target", "mean"),
        median=("target", "median"),
        q25=("target", lambda x: float(np.nanpercentile(x, 25))),
        q75=("target", lambda x: float(np.nanpercentile(x, 75))),
        plot_median=("plot_target", "median"),
    ).reset_index()
    grp = grp[grp["n"] >= min_count]
    if grp.empty:
        return pd.DataFrame()
    grp = grp.sort_values("n", ascending=False).head(topn)
    grp = grp.sort_values("plot_median")
    return grp

def plot_categorical_effect_v2(
    model_df: pd.DataFrame,
    cat_col: str,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    min_count: int,
    topn: int,
    robust_q: float,
):
    grp = category_effect_table(model_df, cat_col, target, min_count, topn, robust_q)
    if grp.empty or len(grp) < 2:
        return
    n = len(grp)
    fig_h = max(5.4, 0.44 * n + 1.4)
    fig, ax = plt.subplots(figsize=cm2inch(12.3, fig_h))
    y = np.arange(n)

    x = grp["median"].values
    xerr_low = np.maximum(0, x - grp["q25"].values)
    xerr_high = np.maximum(0, grp["q75"].values - x)
    ax.errorbar(x, y, xerr=[xerr_low, xerr_high], fmt="o",
                color=PALETTE["blue"], ecolor=PALETTE["gray"],
                elinewidth=0.8, capsize=2.0, markersize=4.6,
                markeredgecolor=PALETTE["dark"], markeredgewidth=0.45)
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=0.75, alpha=0.75)
    ax.set_yticks(y)
    labels = [wrap_label(c, 28) for c in grp["cat"]]
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Median {pretty_name(target)} with IQR")
    ax.set_title(f"{pretty_name(cat_col)} effect on {pretty_name(target)}", loc="left", fontweight="bold")
    beautify_axes(ax, grid=True)

    # Put sample size on the far right to avoid label overlap.
    xlim = ax.get_xlim()
    xr = xlim[1] - xlim[0]
    for yi, nv in zip(y, grp["n"].values):
        ax.text(xlim[1] + 0.02 * xr, yi, f"n={int(nv)}", va="center", fontsize=6.5, color=PALETTE["gray"])
    ax.set_xlim(xlim[0], xlim[1] + 0.12 * xr)
    savefig_multi(fig, outdir / f"Fig7_cat_effect_v2__{sanitize_filename(cat_col)}__{sanitize_filename(target)}", dpi=dpi, formats=formats)

# =============================================================================
# Figure 8: DFT candidate priority, improved
# =============================================================================

def pick_label_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["candidate_id", "model_to_build", "material_name", "catalyst", "system_name", "catalyst_system", "record_id"]:
        if c in df.columns:
            return c
    return None


def plot_dft_priority_v2(dft_df: pd.DataFrame, outdir: Path, dpi: int, formats: Sequence[str], topn: int = 20):
    if "DFT_priority_score" not in dft_df.columns:
        return
    tmp = dft_df.copy()
    tmp["DFT_priority_score"] = to_numeric_series(tmp["DFT_priority_score"])
    tmp = tmp.dropna(subset=["DFT_priority_score"])
    if tmp.empty:
        return
    lab_col = pick_label_column(tmp)
    if lab_col is None:
        tmp["_label"] = [f"Candidate {i+1}" for i in range(len(tmp))]
        lab_col = "_label"

    tmp[lab_col] = tmp[lab_col].map(lambda x: canonical_category_label(clean_category_value(x, max_len=80)))
    tmp = tmp[tmp[lab_col].map(lambda x: category_is_valid(x, material=False))]
    if tmp.empty:
        return

    # Deduplicate candidate names by retaining the highest score.
    tmp = tmp.sort_values("DFT_priority_score", ascending=False).drop_duplicates(subset=[lab_col], keep="first").head(topn)
    tmp = tmp.iloc[::-1]
    n = len(tmp)
    fig_h = max(6.2, 0.38 * n + 1.35)
    fig, ax = plt.subplots(figsize=cm2inch(14.0, fig_h))
    y = np.arange(n)
    score = tmp["DFT_priority_score"].values
    vmin = np.nanmin(score)
    vmax = np.nanmax(score) if np.nanmax(score) > vmin else vmin + 1
    colors = BLUE_TO_RED_CMAP(Normalize(vmin=vmin, vmax=vmax)(score))
    ax.hlines(y, xmin=0, xmax=score, color="#BDBDBD", linewidth=0.85)
    ax.scatter(score, y, s=42, color=colors, edgecolor=PALETTE["dark"], linewidth=0.40, zorder=3)
    labels = [wrap_label(clean_category_value(x, max_len=70), 36) for x in tmp[lab_col]]
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("DFT priority score")
    ax.set_title("Top unique DFT candidates from H adsorption / deprotonation evidence", loc="left", fontweight="bold")
    beautify_axes(ax, grid=True)
    xmax = max(score) if len(score) else 1
    for yi, v in zip(y, score):
        ax.text(v + xmax * 0.012, yi, f"{v:.2f}", va="center", fontsize=6.8)
    savefig_multi(fig, outdir / "Fig8_top_dft_priority_candidates_v2", dpi=dpi, formats=formats)

# =============================================================================
# Figure 9: priority-target scatter, robust and cleaner labels
# =============================================================================

def plot_priority_scatter_v2(
    model_df: pd.DataFrame,
    target: str,
    outdir: Path,
    dpi: int,
    formats: Sequence[str],
    robust_q: float,
):
    if target not in model_df.columns or "DFT_priority_score" not in model_df.columns:
        return
    base = filtered_df_for_target(model_df, target, apply_filter=True)
    if base.empty:
        return

    y = to_numeric_series(base["DFT_priority_score"])
    x = to_numeric_series(base[target])
    use = pd.DataFrame({"x": x, "y": y})
    if "confidence_score" in base.columns:
        use["color"] = to_numeric_series(base["confidence_score"])
        color_label = "Confidence score"
    elif "joint_hads_score" in base.columns:
        use["color"] = to_numeric_series(base["joint_hads_score"])
        color_label = "Evidence score"
    else:
        use["color"] = y
        color_label = "DFT priority score"

    use = use.dropna(subset=["x", "y"])
    if len(use) < 8:
        return
    x_plot, _, _ = clipped_for_plot(use["x"], q=robust_q)
    use["x_plot"] = x_plot
    cvals = use["color"].fillna(use["color"].median() if use["color"].notna().any() else 0)

    fig, ax = plt.subplots(figsize=cm2inch(11.2, 8.2))
    sc = ax.scatter(use["x_plot"], use["y"], c=cvals, cmap=BLUE_TO_RED_CMAP, s=30,
                    edgecolor=PALETTE["dark"], linewidth=0.32, alpha=0.86)
    ax.axvline(0, color=PALETTE["gray"], linestyle="--", linewidth=0.75, alpha=0.75)
    ax.set_xlabel(pretty_name(target))
    ax.set_ylabel("DFT priority score")
    ax.set_title(f"DFT priority score versus {pretty_name(target)}", loc="left", fontweight="bold")
    beautify_axes(ax, grid=True)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(color_label)

    # No point labels by default to avoid overlap and visual clutter.
    savefig_multi(fig, outdir / f"Fig9_priority_scatter_v2__{sanitize_filename(target)}", dpi=dpi, formats=formats)

# =============================================================================
# Optional Step5 importance / interaction plots, with better fallback
# =============================================================================

def find_csvs(root: Path, patterns: Sequence[str]) -> List[Path]:
    out = []
    if not root.exists():
        return out
    for pat in patterns:
        out.extend(root.rglob(pat))
    seen, uniq = set(), []
    for p in sorted(out):
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def plot_importance_files(step5dir: Path, outdir: Path, dpi: int, formats: Sequence[str], topn: int = 18):
    files = find_csvs(step5dir, ["*importance*.csv", "*permutation*.csv", "*coef*.csv"])
    plotted = 0
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception:
            continue
        feat_col = pick_first_existing(df, ["feature", "Feature", "variable", "descriptor", "name"])
        val_col = pick_first_existing(df, ["importance", "Importance", "importance_mean", "mean_importance", "abs_coef", "coef_abs", "coefficient", "score", "value"])
        if not feat_col or not val_col:
            continue
        tmp = df[[feat_col, val_col]].copy()
        tmp[val_col] = to_numeric_series(tmp[val_col])
        tmp = tmp.dropna(subset=[val_col]).sort_values(val_col, ascending=False).head(topn).iloc[::-1]
        if tmp.empty:
            continue
        fig_h = max(6.0, 0.40 * len(tmp) + 1.4)
        fig, ax = plt.subplots(figsize=cm2inch(13.5, fig_h))
        y = np.arange(len(tmp))
        ax.barh(y, tmp[val_col].values, color=PALETTE["purple"], edgecolor=PALETTE["dark"], linewidth=0.45)
        ax.set_yticks(y)
        ax.set_yticklabels([wrap_label(pretty_name(x), 26) for x in tmp[feat_col]])
        ax.set_xlabel(val_col)
        ax.set_title(f"Predictive feature importance: {fp.stem}", loc="left", fontweight="bold")
        beautify_axes(ax, grid=True)
        savefig_multi(fig, outdir / f"Fig10_predictive_importance_v2__{sanitize_filename(fp.stem)}", dpi=dpi, formats=formats)
        plotted += 1
    if plotted == 0:
        print("[INFO] No compatible Step5 importance CSV found.")


# =============================================================================
# Summary table
# =============================================================================

def export_summary_stats(model_df: pd.DataFrame, outdir: Path, robust_q: float):
    rows = []
    for t in target_columns(model_df):
        raw_vals = to_numeric_series(model_df[t]).dropna()
        fdf = filtered_df_for_target(model_df, t, apply_filter=True)
        vals = to_numeric_series(fdf[t]).dropna()
        if vals.empty:
            continue
        _, _, nclip = clipped_for_plot(vals, q=robust_q)
        rows.append({
            "target": t,
            "label": pretty_name(t),
            "n_raw_non_missing": int(len(raw_vals)),
            "n_after_physical_filter": int(len(vals)),
            "n_removed_by_physical_filter": int(len(raw_vals) - len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "q02": float(vals.quantile(0.02)) if len(vals) >= 10 else float(vals.min()),
            "q98": float(vals.quantile(0.98)) if len(vals) >= 10 else float(vals.max()),
            "n_clipped_for_display": int(nclip),
        })
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "publication_summary_stats_v2_refined.csv", index=False)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Publication-grade mechanism-focused figures for HADS/deprotonation database.")
    parser.add_argument("--dbdir", type=str, default="outputs/hads_db")
    parser.add_argument("--step5dir", type=str, default="outputs/hads_step5_stats_ml")
    parser.add_argument("--outdir", type=str, default="outputs/publication_figures_hads_v2_refined")
    parser.add_argument("--dpi", type=int, default=900)
    parser.add_argument("--font", type=str, default="Arial")
    parser.add_argument("--formats", type=str, default="png,pdf")
    parser.add_argument("--topk", type=int, default=10, help="Top mechanism features for Fig5/Fig6.")
    parser.add_argument("--min-numeric-n", type=int, default=10, help="Minimum non-null records for numeric mechanism features.")
    parser.add_argument("--min-pair-n", type=int, default=8, help="Minimum pairwise records for correlation.")
    parser.add_argument("--cat-min-count", type=int, default=8, help="Minimum category size for categorical effect plots.")
    parser.add_argument("--cat-topn", type=int, default=10)
    parser.add_argument("--robust-q", type=float, default=0.02, help="Quantile clipping for display only; set 0 to disable.")
    parser.add_argument("--force-features", type=str, default="", help="Comma-separated extra feature columns to include in correlations.")
    parser.add_argument("--exclude-contains", type=str, default=",".join(DEFAULT_EXCLUDE_CONTAINS), help="Comma-separated substrings excluded from mechanism correlations.")
    parser.add_argument("--primary-targets", type=str, default="", help="Comma-separated targets to plot first. Default auto-detects.")
    parser.add_argument("--bubble-max", type=float, default=260.0, help="Maximum-ish bubble size scaling for Fig6.")
    parser.add_argument("--deltae-detail", action="store_true", default=True, help="Add detailed ΔE_H* feature-direction figures and CSV summaries.")
    parser.add_argument("--no-deltae-detail", dest="deltae_detail", action="store_false", help="Disable detailed ΔE_H* feature-direction analysis.")
    parser.add_argument("--deltae-target", type=str, default="H_adsorption_energy_value_eV", help="Target column for detailed ΔE_H* analysis.")
    parser.add_argument("--deltae-topn", type=int, default=12, help="Top features for detailed ΔE_H* raw / normalized direction plots and ML-importance figures.")
    parser.add_argument("--no-deltae-ml", dest="deltae_ml", action="store_false", default=True, help="Disable optional RandomForest permutation-importance analysis for ΔE_H*.")
    args = parser.parse_args()

    dbdir = Path(args.dbdir)
    step5dir = Path(args.step5dir)
    outdir = ensure_dir(Path(args.outdir))
    formats = tuple(x.strip() for x in args.formats.split(",") if x.strip())
    set_pub_style(args.font, base_size=7.8)

    paper_df = read_csv_if_exists(dbdir / "paper_table.csv")
    system_df = read_csv_if_exists(dbdir / "system_table.csv")
    site_df = read_csv_if_exists(dbdir / "site_table.csv")
    ads_df = read_csv_if_exists(dbdir / "adsorption_table.csv")
    model_df = read_csv_if_exists(dbdir / "model_feature_table.csv")
    dft_df = read_csv_if_exists(dbdir / "dft_priority_candidates.csv")
    if model_df is None:
        raise FileNotFoundError(f"Cannot find model table: {dbdir / 'model_feature_table.csv'}")

    model_df = normalize_bool_flags(model_df)
    model_df = standardize_oxidation_state_columns(model_df)
    targets_all = target_columns(model_df, min_n=3)
    forced_targets = [t for t in parse_comma_list(args.primary_targets) if t in model_df.columns]
    primary_targets = forced_targets + [t for t in targets_all if t not in forced_targets]
    primary_targets = primary_targets[:4]

    exclude_terms = parse_comma_list(args.exclude_contains)
    force_features = canonicalize_force_features(parse_comma_list(args.force_features))

    # Numeric mechanism features only.
    mech_feats = numeric_mechanism_features(
        model_df,
        targets=targets_all,
        force_features=force_features,
        exclude_terms=exclude_terms,
        min_non_null=args.min_numeric_n,
    )
    print(f"[INFO] Targets detected: {targets_all}")
    print(f"[INFO] Mechanism numeric features used for correlation: {len(mech_feats)}")
    if mech_feats:
        print("[INFO] First features:", ", ".join(mech_feats[:30]))

    # Fig1-4: overview and data quality.
    plot_overview_dashboard(paper_df, system_df, site_df, ads_df, model_df, outdir, args.dpi, formats)
    plot_target_distribution_panel(model_df, outdir, args.dpi, formats, robust_q=args.robust_q)
    plot_mechanism_coverage(model_df, outdir, args.dpi, formats)
    plot_mechanism_category_summary(model_df, system_df, outdir, args.dpi, formats)

    # Optional separate cleaned top-category plots for mechanism-focused categories.
    for c, title, material in [
        ("bridge_structure", "Top bridge/interface motifs", False),
        ("M_O_X_configuration", "Top M-O-X configurations", False),
        ("dopant", "Top dopants", False),
        ("defect_type", "Top defect types", False),
    ]:
        src = model_df if c in model_df.columns else (system_df if system_df is not None else model_df)
        if src is not None and c in src.columns:
            plot_top_categories_cleaned(src, c, outdir / f"Fig4_top_cleaned_{sanitize_filename(c)}", title, args.dpi, formats, topn=15, material=material)

    # Fig5/Fig6: mechanism-only correlations.
    for t in primary_targets:
        if not mech_feats:
            continue
        # Recompute selected features for each target and keep pairwise n.
        corr_df = compute_feature_target_correlations(model_df, t, mech_feats, min_pair_n=args.min_pair_n)
        if corr_df.empty:
            print(f"[INFO] Skip correlations for {t}: no valid mechanism feature pairs.")
            continue
        selected = corr_df.sort_values("abs_r", ascending=False).head(args.topk)["feature"].tolist()
        plot_corr_strip(model_df, t, selected, outdir, args.dpi, formats, topk=args.topk)
        plot_corr_bubble_matrix_v2(model_df, t, selected, outdir, args.dpi, formats, topk=args.topk, bubble_max=args.bubble_max)

    # Detailed ΔE_H* analysis: effect direction, binned response, and optional ML importance.
    if args.deltae_detail:
        run_deltae_detailed_analysis(
            model_df=model_df,
            target=args.deltae_target,
            features=mech_feats,
            outdir=outdir,
            dpi=args.dpi,
            formats=formats,
            topn=args.deltae_topn,
            min_pair_n=args.min_pair_n,
            run_ml=args.deltae_ml,
        )

    # Fig7: categorical mechanism effects. Avoid generic material/support fields by default.
    cat_cols = [
        "bridge_structure", "M_O_X_configuration", "dopant", "defect_type",
        "vacancy_flag", "hydroxyl_flag",
    ]
    cat_cols = existing_columns(model_df, cat_cols)
    for t in primary_targets[:2]:
        for c in cat_cols[:6]:
            plot_categorical_effect_v2(
                model_df, c, t, outdir, args.dpi, formats,
                min_count=args.cat_min_count, topn=args.cat_topn, robust_q=args.robust_q
            )

    # Fig8/Fig9: DFT priority.
    if dft_df is not None:
        plot_dft_priority_v2(dft_df, outdir, args.dpi, formats, topn=22)
    for t in primary_targets[:3]:
        plot_priority_scatter_v2(model_df, t, outdir, args.dpi, formats, robust_q=args.robust_q)

    # Optional Step5 importance plots.
    if step5dir.exists():
        plot_importance_files(step5dir, outdir, args.dpi, formats, topn=18)

    export_summary_stats(model_df, outdir, robust_q=args.robust_q)
    print(f"[DONE] Publication-grade v2 figures saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
