#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validation and normalization helpers for H adsorption/deprotonation extraction.

These functions are intentionally conservative: they prefer missing values over
unsupported values. This reduces GLM/vLLM extraction noise such as mismatched
material names, hallucinated numeric descriptors, and cross-system leakage.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from utils.text_utils import normalize_text
except Exception:
    def normalize_text(x: Any) -> str:
        return re.sub(r"\s+", " ", str(x or "")).strip()

DASH_TRANS = str.maketrans({"−": "-", "–": "-", "—": "-", "﹣": "-", "－": "-", "‐": "-"})

METHOD_TERMS = {
    "dft", "gga", "pbe", "pbe0", "rpbe", "b3lyp", "scan", "hse06", "vasp", "dmol3", "gaussian",
    "xps", "xas", "xanes", "exafs", "xrd", "tem", "hrtem", "sem", "eds", "edx", "icp", "seiras",
    "her", "hor", "orr", "oer", "edl", "rhe", "cv", "lsv", "rde", "eis", "pzc", "elf", "dos", "pdos",
    "koh", "naoh", "hcl", "h2o", "oh", "h2", "o2", "n2", "ar", "co", "co2", "h", "o", "c", "n", "p", "s",
    "table", "figure", "fig", "scheme", "equation", "supporting", "information", "supplementary",
    "adsorption", "desorption", "volmer", "heyrovsky", "tafel", "surface", "catalyst", "sample", "model",
}

BAD_MATERIAL_FRAGMENTS = [
    "perdew", "burke", "ernzerhof", "grimme", "monkhorst", "methfessel", "paxton", "koutecky", "levich",
    "butler", "volmer", "gibbs", "hydrogen binding", "adsorption energy", "free energy", "this work",
]

COMMON_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs","Ba","La","Ce","Pr","Nd","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi"
}
METAL_ELEMENTS = {
    "Li","Na","K","Mg","Al","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Y","Zr","Nb","Mo","Ru","Rh","Pd","Ag","Cd","In","Sn","La","Ce","Hf","Ta","W","Re","Os","Ir","Pt","Au","Bi"
}

MATERIAL_TOKEN_RE = re.compile(
    r"(?:[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*){0,6})(?:[@/\-][A-Za-z0-9()+/\-]+)*(?:\([0-9]{3,4}\))?"
)

NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


ROMAN_OXIDATION = {
    "0": 0, "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8,
    "IX": 9, "X": 10,
}
SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
    "⁺": "+", "⁻": "-",
})


def normalize_yes_no(x: Any) -> Optional[str]:
    """Normalize presence/absence style fields without fabricating values."""
    s = norm_text(x).strip().lower()
    if not s or s in {"none", "null", "nan", "unknown", "not mentioned", "n/a", "na"}:
        return None
    if s in {"yes", "y", "true", "present", "presence", "1", "positive", "nearby"}:
        return "yes"
    if s in {"no", "n", "false", "absent", "absence", "0", "negative"}:
        return "no"
    # Keep descriptive evidence-supported phrases rather than forcing binary.
    return norm_text(x)


def parse_oxidation_state(x: Any, default_element: Any = None) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Parse active-metal oxidation state from text.

    Returns (raw, element, numeric_value). Handles Fe(III), Fe3+, Fe³⁺, Ni2+, Cu0,
    and phrases such as "Ni(II)/Ni(III)" by returning the first explicit valence.
    The raw field is preserved for traceability.
    """
    raw = norm_text(x).strip()
    if not raw or raw.lower() in {"none", "null", "nan", "unknown", "not mentioned", "n/a", "na"}:
        return None, None, None
    s = raw.translate(SUPERSCRIPT_TRANS).replace("−", "-").replace("–", "-")
    default_el = norm_text(default_element).strip() or None

    # Element + Roman numeral, e.g. Fe(III), Ni II, Co(II/III)
    m = re.search(r"\b([A-Z][a-z]?)\s*\(?\s*(0|I{1,3}|IV|V|VI{0,3}|IX|X)\s*\)?\b", s)
    if m and m.group(1) in COMMON_ELEMENTS:
        roman = m.group(2).upper()
        if roman in ROMAN_OXIDATION:
            return raw, m.group(1), float(ROMAN_OXIDATION[roman])

    # Element + arabic valence, e.g. Fe3+, Fe+3, Cu0, Ni2+
    m = re.search(r"\b([A-Z][a-z]?)[\s\-]*(?:\(?\s*)?([0-9]+(?:\.\d+)?)(?:\s*[+\-])?", s)
    if m and m.group(1) in COMMON_ELEMENTS:
        return raw, m.group(1), float(m.group(2))

    # Generic valence value after phrase, e.g. oxidation state: +2, valence state of +3
    m = re.search(r"(?:oxidation state|valence state|valence)\D{0,20}([+\-]?\d+(?:\.\d+)?)", s, re.I)
    if m:
        try:
            return raw, default_el, float(m.group(1).replace("+", ""))
        except Exception:
            pass

    # Roman without element; use default active atom if available.
    m = re.search(r"\b(0|I{1,3}|IV|V|VI{0,3}|IX|X)\b", s)
    if m and m.group(1).upper() in ROMAN_OXIDATION:
        return raw, default_el, float(ROMAN_OXIDATION[m.group(1).upper()])

    return raw, default_el, None


def norm_text(x: Any) -> str:
    return normalize_text(x).translate(DASH_TRANS)


def normalize_dash(x: Any) -> str:
    return str(x or "").translate(DASH_TRANS)


def evidence_text(evs: Any) -> str:
    parts: List[str] = []
    if isinstance(evs, list):
        for e in evs:
            if isinstance(e, dict):
                parts.append(str(e.get("quote") or e.get("text") or ""))
    elif isinstance(evs, str):
        parts.append(evs)
    return norm_text(" ".join(parts))


def packed_evidence_text(*ev_lists: Any) -> str:
    parts: List[str] = []
    for evs in ev_lists:
        parts.append(evidence_text(evs))
    return norm_text(" ".join(parts))


def token_overlap(a: str, b: str, min_len: int = 3) -> int:
    ta = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{%d,}" % (min_len-1), norm_text(a))}
    tb = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9+\-]{%d,}" % (min_len-1), norm_text(b))}
    return len(ta & tb)


def _has_element_like(token: str) -> bool:
    return any(el in token for el in COMMON_ELEMENTS)


def is_bad_material_name(x: Any) -> bool:
    s0 = norm_text(x).strip()
    if not s0:
        return True
    s = s0.lower().strip(" .,:;()[]{}")
    if len(s) < 2 or len(s) > 80:
        return True
    if s in METHOD_TERMS:
        return True
    if any(frag in s for frag in BAD_MATERIAL_FRAGMENTS):
        return True
    if re.fullmatch(r"[A-Za-z]{1,2}", s0) and s0 not in COMMON_ELEMENTS:
        return True
    if re.fullmatch(r"[A-Z]{2,6}", s0) and s0.lower() in METHOD_TERMS:
        return True
    # Pure bond/species tokens are usually not material systems.
    if re.fullmatch(r"[A-Z][a-z]?[-=][A-Z][a-z]?", s0) and s0.split("-")[0] in {"H", "O", "C", "N"}:
        return True
    # Long plain English phrases are not material names unless they contain formula delimiters.
    if " " in s0 and not re.search(r"[@/()\-]|\d|doped|coated|supported|alloy|oxide|nitride|carbide|phosphide", s0, re.I):
        return True
    # Must contain at least one element symbol/formula delimiter/numeric/facet marker.
    if not (MATERIAL_TOKEN_RE.search(s0) and (_has_element_like(s0) or re.search(r"[@/()\d\-]", s0))):
        return True
    return False


def clean_material_name(x: Any, evidence: str = "") -> Optional[str]:
    s = norm_text(x).strip()
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" ,.;:")
    if is_bad_material_name(s):
        return None
    # If evidence is supplied, require literal support or at least token overlap for non-formula phrases.
    ev = norm_text(evidence)
    if ev and s.lower() not in ev.lower():
        if token_overlap(s, ev) == 0 and not re.search(r"[@/()\d\-]", s):
            return None
    return s


def is_bridge_oxygen_motif(x: Any) -> bool:
    s = norm_text(x)
    if not s:
        return False
    if re.search(r"bridg(?:e|ing) oxygen|oxygen bridge", s, re.I):
        return True
    # Accept M-O-X where M is a metal element and X is element/C/P/N/S etc.
    m = re.search(r"\b([A-Z][a-z]?)[- ]?O[- ]?([A-Z][a-z]?)\b", s)
    if m and m.group(1) in METAL_ELEMENTS and m.group(2) in COMMON_ELEMENTS:
        return True
    m = re.search(r"\b([A-Z][a-z]?)[- ]?O[- ]?C/P\b", s)
    if m and m.group(1) in METAL_ELEMENTS:
        return True
    return False


def clean_bridge(x: Any, evidence: str = "") -> Optional[str]:
    s = norm_text(x).strip()
    if not s:
        return None
    low = s.lower()
    bad = {"h-o-h", "o-o", "c-o-c", "c=o", "o-h", "h-o", "water", "h2o", "oh"}
    if low in bad:
        return None
    if not is_bridge_oxygen_motif(s):
        return None
    ev = norm_text(evidence)
    if ev and s.lower() not in ev.lower() and token_overlap(s, ev) == 0:
        # Motif may be normalized from Ni-O-P while paper has Ni–O–P, but dash normalization handles this.
        return None
    return s


def numeric_variants(value: Any) -> List[str]:
    try:
        v = float(value)
    except Exception:
        return []
    variants = {f"{v:g}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"}
    if v >= 0:
        variants.add("+" + f"{v:g}")
    return sorted(variants, key=len, reverse=True)


def number_supported(value: Any, evidence: str, rel_tol: float = 0.005, abs_tol: float = 0.015) -> bool:
    if value is None or evidence is None:
        return False
    ev = norm_text(evidence)
    try:
        v = float(value)
    except Exception:
        return False
    # Fast exact-ish string support.
    for vv in numeric_variants(v):
        if re.search(r"(?<![0-9.])" + re.escape(vv) + r"(?![0-9.])", ev):
            return True
    # Tolerance support against all numbers in evidence.
    for m in NUMBER_RE.finditer(ev):
        try:
            x = float(m.group(0))
        except Exception:
            continue
        if math.isclose(x, v, rel_tol=rel_tol, abs_tol=abs_tol):
            return True
    return False


def text_supported(x: Any, evidence: str, min_overlap: int = 1) -> bool:
    s = norm_text(x)
    ev = norm_text(evidence)
    if not s:
        return False
    if not ev:
        return True
    if s.lower() in ev.lower():
        return True
    return token_overlap(s, ev) >= min_overlap


def validate_numeric_fields(rec: Dict[str, Any], fields: Sequence[str], evidence: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    support: Dict[str, str] = {}
    for f in fields:
        val = rec.get(f)
        if val is None or val == "":
            support[f] = "missing"
            continue
        if number_supported(val, evidence):
            support[f] = "supported"
        else:
            rec[f] = None
            unit_f = f + "_unit"
            # Known exceptions where field name does not end with _value
            unit_f = {
                "proton_transfer_barrier": "proton_transfer_barrier_unit",
                "Volmer_barrier": "Volmer_barrier_unit",
                "water_dissociation_barrier": "water_dissociation_barrier_unit",
            }.get(f, unit_f)
            if unit_f in rec:
                rec[unit_f] = None
            support[f] = "dropped_not_in_evidence"
    return rec, support


def system_match_score(system_obj: Dict[str, Any], evidence: str) -> float:
    ev = norm_text(evidence).lower()
    terms = []
    for k in ["material_name", "composition", "surface_facet", "dopant", "defect_type", "bridge_structure", "M_O_X_configuration", "support"]:
        val = norm_text(system_obj.get(k))
        if val and val.lower() not in {"unknown", "none", "null"}:
            terms.append(val)
    if not terms:
        return 0.0
    weights = []
    for t in terms:
        if t.lower() in ev:
            weights.append(1.0)
        elif token_overlap(t, ev) > 0:
            weights.append(0.5)
        else:
            weights.append(0.0)
    return round(sum(weights) / max(1, len(weights)), 4)


def field_support_summary(rec: Dict[str, Any], evidence: str, fields: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in fields:
        v = rec.get(f)
        if v is None or v == "":
            out[f] = "missing"
        elif isinstance(v, (int, float)):
            out[f] = "supported" if number_supported(v, evidence) else "unsupported"
        else:
            out[f] = "supported" if text_supported(v, evidence) else "weak_or_unsupported"
    return out
