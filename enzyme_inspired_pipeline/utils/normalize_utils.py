#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# canonical vocab
# -----------------------------------------------------------------------------

_NULLS = {
    "", "unknown", "none", "null", "nan", "n/a", "na", "not reported",
    "not mentioned", "missing", "not available", "n.r.", "nr"
}

# Canonical ions commonly seen in ALP/CES and related hydrolase assays
COFACTOR_MAP: Dict[str, str] = {
    "MG2+": "Mg2+",
    "ZN2+": "Zn2+",
    "CA2+": "Ca2+",
    "MN2+": "Mn2+",
    "CO2+": "Co2+",
    "NI2+": "Ni2+",
    "FE2+": "Fe2+",
    "FE3+": "Fe3+",
    "CU2+": "Cu2+",
    "CD2+": "Cd2+",
    "SR2+": "Sr2+",
    "BA2+": "Ba2+",
    "METALION": "metal ion",
    "DIVALENTMETAL": "divalent metal",
    "COFACTOR": "cofactor",
}

# More complete enzyme assay buffer normalization
BUFFER_MAP: Dict[str, str] = {
    "TRIS": "Tris",
    "TRISHCL": "Tris-HCl",
    "HEPES": "HEPES",
    "MOPS": "MOPS",
    "MES": "MES",
    "PBS": "PBS",
    "PHOSPHATE": "phosphate buffer",
    "PHOSPHATEBUFFER": "phosphate buffer",
    "SODIUMPHOSPHATE": "sodium phosphate",
    "POTASSIUMPHOSPHATE": "potassium phosphate",
    "BICINE": "Bicine",
    "GLYCINE": "glycine",
    "GLYCINENAOH": "glycine-NaOH",
    "CARBONATE": "carbonate buffer",
    "CARBONATEBICARBONATE": "carbonate-bicarbonate",
    "BICARBONATE": "bicarbonate buffer",
    "BORATE": "borate",
    "ACETATE": "acetate",
    "CITRATE": "citrate",
    "SUCCINATE": "succinate",
    "CHES": "CHES",
    "CAPS": "CAPS",
    "CAPSO": "CAPSO",
    "TAPS": "TAPS",
    "HEPPS": "HEPPS",
    "HEPPSO": "HEPPSO",
    "IMIDAZOLE": "imidazole",
    "DIETHANOLAMINE": "diethanolamine",
    "DEA": "diethanolamine",
    "AMPD": "AMPD",
}

# full amino-acid coverage + several residue-category placeholders
RESIDUE_MAP: Dict[str, str] = {
    "A": "Ala", "ALA": "Ala", "ALANINE": "Ala",
    "R": "Arg", "ARG": "Arg", "ARGININE": "Arg",
    "N": "Asn", "ASN": "Asn", "ASPARAGINE": "Asn",
    "D": "Asp", "ASP": "Asp", "ASPARTATE": "Asp", "ASPARTICACID": "Asp",
    "C": "Cys", "CYS": "Cys", "CYSTEINE": "Cys",
    "Q": "Gln", "GLN": "Gln", "GLUTAMINE": "Gln",
    "E": "Glu", "GLU": "Glu", "GLUTAMATE": "Glu", "GLUTAMICACID": "Glu",
    "G": "Gly", "GLY": "Gly", "GLYCINE": "Gly",
    "H": "His", "HIS": "His", "HISTIDINE": "His",
    "I": "Ile", "ILE": "Ile", "ISOLEUCINE": "Ile",
    "L": "Leu", "LEU": "Leu", "LEUCINE": "Leu",
    "K": "Lys", "LYS": "Lys", "LYSINE": "Lys",
    "M": "Met", "MET": "Met", "METHIONINE": "Met",
    "F": "Phe", "PHE": "Phe", "PHENYLALANINE": "Phe",
    "P": "Pro", "PRO": "Pro", "PROLINE": "Pro",
    "S": "Ser", "SER": "Ser", "SERINE": "Ser",
    "T": "Thr", "THR": "Thr", "THREONINE": "Thr",
    "W": "Trp", "TRP": "Trp", "TRYPTOPHAN": "Trp",
    "Y": "Tyr", "TYR": "Tyr", "TYROSINE": "Tyr",
    "V": "Val", "VAL": "Val", "VALINE": "Val",
}

_BUFFER_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btris(?:[-\s]*hcl)?\b", re.I), "Tris-HCl"),
    (re.compile(r"\bhepes\b", re.I), "HEPES"),
    (re.compile(r"\bmops\b", re.I), "MOPS"),
    (re.compile(r"\bmes\b", re.I), "MES"),
    (re.compile(r"\bpbs\b", re.I), "PBS"),
    (re.compile(r"\b(?:sodium|potassium)?\s*phosphate(?:\s+buffer)?\b", re.I), "phosphate buffer"),
    (re.compile(r"\bbicine\b", re.I), "Bicine"),
    (re.compile(r"\bglycine(?:[-\s]*naoh)?\b", re.I), "glycine-NaOH"),
    (re.compile(r"\bcarbonate[-\s]*bicarbonate\b", re.I), "carbonate-bicarbonate"),
    (re.compile(r"\bborate\b", re.I), "borate"),
    (re.compile(r"\bacetate\b", re.I), "acetate"),
    (re.compile(r"\bcitrate\b", re.I), "citrate"),
    (re.compile(r"\bsuccinate\b", re.I), "succinate"),
    (re.compile(r"\bches\b", re.I), "CHES"),
    (re.compile(r"\bcaps[o]?\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\btaps\b", re.I), "TAPS"),
    (re.compile(r"\bhepps[o]?\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bimidazole\b", re.I), "imidazole"),
    (re.compile(r"\b(?:diethanolamine|dea)\b", re.I), "diethanolamine"),
    (re.compile(r"\bampd\b", re.I), "AMPD"),
]

# -----------------------------------------------------------------------------
# generic helpers
# -----------------------------------------------------------------------------

def to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    return str(x).strip()


def clean_scalar(x: Any) -> Optional[str]:
    s = to_str(x)
    if not s or s.lower() in _NULLS:
        return None
    return re.sub(r"\s+", " ", s).strip() or None


def clean_listlike(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        arr = x
    else:
        s = to_str(x)
        if not s:
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                arr = j if isinstance(j, list) else [s]
            except Exception:
                arr = re.split(r"[;,|/]+|\band\b", s, flags=re.I)
        else:
            arr = re.split(r"[;,|/]+|\band\b", s, flags=re.I)
    out: List[str] = []
    seen = set()
    for a in arr:
        t = clean_scalar(a)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def to_num(x: Any) -> Optional[float]:
    try:
        v = pd.to_numeric(x, errors="coerce")
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def maybe_round(x: Optional[float], nd: int = 6) -> Optional[float]:
    return None if x is None else round(float(x), nd)

# -----------------------------------------------------------------------------
# enzyme / cofactor / buffer normalization
# -----------------------------------------------------------------------------

def infer_enzyme_family(x: Any) -> Optional[str]:
    s = (clean_scalar(x) or "").lower()
    if not s:
        return None
    if "alkaline phosphatase" in s or re.search(r"\balp\b", s):
        return "ALP"
    if ("carboxyl" in s and "esterase" in s) or re.search(r"\bces\b", s) or "carboxylesterase" in s:
        return "CES"
    if "esterase" in s:
        return "Esterase"
    if "phosphatase" in s:
        return "Phosphatase"
    if "hydrolase" in s:
        return "Hydrolase"
    return None


def _canonicalize_charge_text(s: str) -> str:
    s = s.replace("²⁺", "2+").replace("³⁺", "3+").replace("⁺", "+").replace("⁻", "-")
    s = s.replace("(", "").replace(")", "")
    return s


def normalize_cofactor(x: Any) -> Optional[str]:
    s = clean_scalar(x)
    if not s:
        return None
    raw = _canonicalize_charge_text(s)
    compact = re.sub(r"[^A-Za-z0-9+]+", "", raw).upper()
    if compact in COFACTOR_MAP:
        return COFACTOR_MAP[compact]

    low = raw.lower()
    alias_patterns = [
        (r"\bmagnesium\b|\bmg\b|\bmg2\+\b|\bmgii\b|\bmgcl2\b|\bmgso4\b", "Mg2+"),
        (r"\bzinc\b|\bzn\b|\bzn2\+\b|\bznii\b|\bzncl2\b|\bznso4\b", "Zn2+"),
        (r"\bcalcium\b|\bca\b|\bca2\+\b|\bcaii\b|\bcacl2\b", "Ca2+"),
        (r"\bmanganese\b|\bmn\b|\bmn2\+\b|\bmnii\b|\bmncl2\b", "Mn2+"),
        (r"\bcobalt\b|\bco2\+\b|\bcoii\b|\bcocl2\b", "Co2+"),
        (r"\bnickel\b|\bni2\+\b|\bniii?\b|\bnicl2\b", "Ni2+"),
        (r"\biron\(ii\)|\bferrous\b|\bfe2\+\b", "Fe2+"),
        (r"\biron\(iii\)|\bferric\b|\bfe3\+\b", "Fe3+"),
        (r"\bcopper\b|\bcu2\+\b|\bcuii\b", "Cu2+"),
        (r"\bcadmium\b|\bcd2\+\b", "Cd2+"),
        (r"\bstrontium\b|\bsr2\+\b", "Sr2+"),
        (r"\bbarium\b|\bba2\+\b", "Ba2+"),
        (r"\bdivalent\s+metal\b", "divalent metal"),
        (r"\bmetal\s+ion\b|\bmetal\s+center\b|\bbinuclear\s+metal\s+center\b", "metal ion"),
        (r"\bcofactor\b", "cofactor"),
    ]
    for pat, canon in alias_patterns:
        if re.search(pat, low, re.I):
            return canon
    return s


def normalize_buffer_name(x: Any) -> Optional[str]:
    s = clean_scalar(x)
    if not s:
        return None
    raw = s.strip()
    compact = re.sub(r"[^A-Za-z0-9]+", "", raw).upper()
    if compact in BUFFER_MAP:
        return BUFFER_MAP[compact]
    for pat, canon in _BUFFER_PATTERNS:
        m = pat.search(raw)
        if m:
            return canon(m) if callable(canon) else canon
    return s

# -----------------------------------------------------------------------------
# residue normalization
# -----------------------------------------------------------------------------

def _canonical_residue(code: str) -> Optional[str]:
    return RESIDUE_MAP.get(code.upper().replace(" ", ""))


def normalize_residue_list(x: Any) -> List[str]:
    out: List[str] = []
    for a in clean_listlike(x):
        aa = a.strip()
        up = aa.upper().replace(" ", "").replace("-", "")

        # categorical placeholders
        if up in {"CATALYCTRIAD", "CATALYTICTRIAD"}:
            name = "catalytic triad"
        elif up == "ACTIVESITE":
            name = "active site"
        elif up == "NUCLEOPHILE":
            name = "nucleophile"
        elif up == "OXYANIONHOLE":
            name = "oxyanion hole"
        else:
            # Ser102 / SER102 / Ser-102 / Ser 102 / S102 / serine102
            m = re.match(r"^([A-Z]{1}|[A-Z]{3}|[A-Z][a-z]+)[\s\-]*?(\d+)([A-Z])?$", aa)
            if m:
                code = _canonical_residue(m.group(1))
                if code:
                    name = f"{code}{m.group(2)}"
                else:
                    name = aa
            else:
                # naked residue names/codes
                mapped = _canonical_residue(up)
                name = mapped if mapped else aa
        if name not in out:
            out.append(name)
    return out


def normalize_residue_state(x: Any) -> Optional[str]:
    s = (clean_scalar(x) or "").lower()
    if not s:
        return None
    if "deproton" in s or "nucleophilic serine" in s or "alkoxide" in s:
        return "deprotonated"
    if "proton" in s:
        return "protonated"
    if "neutral" in s:
        return "neutral"
    return s


def normalize_mechanism_type(x: Any) -> Optional[str]:
    s = (clean_scalar(x) or "").lower()
    if not s:
        return None
    if "qm/mm" in s or "dft" in s or "free energy" in s or "computed" in s or "simulation" in s:
        return "computed_mechanism"
    if "mutagen" in s or "kinetic isotope" in s or "kinetic" in s or "experiment" in s:
        return "experiment_supported"
    if "proposed" in s or "review" in s or "hypothesis" in s:
        return "proposed_mechanism"
    return s

# -----------------------------------------------------------------------------
# unit normalization
# -----------------------------------------------------------------------------

def normalize_concentration_to_mM(value: Any, unit: Any) -> Tuple[Optional[float], Optional[str], int]:
    v = to_num(value)
    u = clean_scalar(unit)
    if v is None:
        return None, u, 0
    if not u:
        return maybe_round(v), None, 0
    uu = u.lower().replace(" ", "")
    if uu in {"m", "mol/l", "moll-1", "mol·l-1", "molar"}:
        return maybe_round(v * 1000.0), "mM", 0
    if uu in {"mm", "mmol/l", "mmoll-1", "mmol·l-1", "mmolar", "mm"}:
        return maybe_round(v), "mM", 0
    if uu in {"um", "μm", "µm", "umol/l", "μmol/l", "µmol/l", "micromolar"}:
        return maybe_round(v / 1000.0), "mM", 0
    if uu in {"nm", "nmol/l", "nanomolar"}:
        return maybe_round(v / 1_000_000.0), "mM", 0
    return maybe_round(v), u, 1


def normalize_km_to_mM(value: Any, unit: Any) -> Tuple[Optional[float], Optional[str], int]:
    return normalize_concentration_to_mM(value, unit)


def normalize_kcat(value: Any, unit: Any) -> Tuple[Optional[float], Optional[str], int]:
    v = to_num(value)
    u = clean_scalar(unit)
    if v is None:
        return None, u, 0
    if not u:
        return maybe_round(v), None, 0
    uu = u.lower().replace(" ", "")
    if uu in {"s-1", "/s", "sec-1", "s^-1"}:
        return maybe_round(v), "s^-1", 0
    if uu in {"min-1", "/min", "min^-1"}:
        return maybe_round(v / 60.0), "s^-1", 0
    if uu in {"h-1", "/h", "hr-1", "hour-1"}:
        return maybe_round(v / 3600.0), "s^-1", 0
    return maybe_round(v), u, 1


def normalize_kcat_over_km(value: Any, unit: Any) -> Tuple[Optional[float], Optional[str], int]:
    v = to_num(value)
    u = clean_scalar(unit)
    if v is None:
        return None, u, 0
    if not u:
        return maybe_round(v), None, 0
    uu = u.lower().replace(" ", "")
    if "m-1s-1" in uu or "m^-1s^-1" in uu:
        return maybe_round(v), "M^-1 s^-1", 0
    if "m-1min-1" in uu or "m^-1min^-1" in uu:
        return maybe_round(v / 60.0), "M^-1 s^-1", 0
    if "mm-1s-1" in uu or "mm^-1s^-1" in uu:
        return maybe_round(v * 1000.0), "M^-1 s^-1", 0
    if "mm-1min-1" in uu or "mm^-1min^-1" in uu:
        return maybe_round(v * 1000.0 / 60.0), "M^-1 s^-1", 0
    return maybe_round(v), u, 1

# -----------------------------------------------------------------------------
# H adsorption / deprotonation energy normalization utilities
# -----------------------------------------------------------------------------

ENERGY_UNIT_TO_EV = {
    "ev": 1.0,
    "electronvolt": 1.0,
    "electronvolts": 1.0,
    "kj/mol": 1.0 / 96.4853321233,
    "kjmol-1": 1.0 / 96.4853321233,
    "kj mol-1": 1.0 / 96.4853321233,
    "kj mol^-1": 1.0 / 96.4853321233,
    "kj mol−1": 1.0 / 96.4853321233,
    "kcal/mol": 1.0 / 23.0605478306,
    "kcalmol-1": 1.0 / 23.0605478306,
    "kcal mol-1": 1.0 / 23.0605478306,
    "kcal mol^-1": 1.0 / 23.0605478306,
    "kcal mol−1": 1.0 / 23.0605478306,
    "j/mol": 1.0 / 96485.3321233,
    "jmol-1": 1.0 / 96485.3321233,
    "j mol-1": 1.0 / 96485.3321233,
    "j mol^-1": 1.0 / 96485.3321233,
}


def clean_unit(unit: Any) -> Optional[str]:
    """Normalize energy units used in DFT adsorption/deprotonation literature."""
    u = clean_scalar(unit)
    if u is None:
        return None
    s = str(u).strip()
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)
    low = s.lower().replace("·", " ").strip()
    compact = low.replace(" ", "")
    if low in {"ev", "e v"} or compact in {"ev", "electronvolt", "electronvolts"}:
        return "eV"
    if low in {"mev", "m ev"} or compact in {"mev", "millielectronvolt", "millielectronvolts"}:
        return "meV"
    if compact in {"kjmol-1", "kjmol^-1", "kj/mol", "kjmol−1"} or low in {"kj mol-1", "kj mol^-1", "kj mol−1", "kj/mol"}:
        return "kJ mol-1"
    if compact in {"kcalmol-1", "kcalmol^-1", "kcal/mol", "kcalmol−1"} or low in {"kcal mol-1", "kcal mol^-1", "kcal mol−1", "kcal/mol"}:
        return "kcal mol-1"
    if compact in {"jmol-1", "jmol^-1", "j/mol", "jmol−1"} or low in {"j mol-1", "j mol^-1", "j mol−1", "j/mol"}:
        return "J mol-1"
    return s


def to_float(x: Any) -> Optional[float]:
    """Robust float parser that accepts Unicode minus and textual numeric snippets."""
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if pd.isna(x):
            return None
        return float(x)
    s = str(x).strip()
    if not s or s.lower() in _NULLS:
        return None
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace(",", "")
    # common forms: ca. -0.12, ~0.03, = -10.5
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def to_ev(value: Any, unit: Any) -> Optional[float]:
    """Convert an energy-like value to eV when the unit is known."""
    v = to_float(value)
    if v is None:
        return None
    cu = clean_unit(unit)
    if not cu:
        return maybe_round(v)
    low = str(cu).lower().replace("−", "-").replace("·", " ")
    compact = low.replace(" ", "")
    key = None
    if compact == "ev":
        key = "ev"
    elif compact == "mev":
        return maybe_round(v / 1000.0)
    elif compact in {"kjmol-1", "kjmol^-1", "kj/mol"}:
        key = "kj/mol"
    elif compact in {"kcalmol-1", "kcalmol^-1", "kcal/mol"}:
        key = "kcal/mol"
    elif compact in {"jmol-1", "jmol^-1", "j/mol"}:
        key = "j/mol"
    if key is None:
        # preserve the original value if unit already looks dimensionless/unknown
        return maybe_round(v)
    return maybe_round(v * ENERGY_UNIT_TO_EV[key])
