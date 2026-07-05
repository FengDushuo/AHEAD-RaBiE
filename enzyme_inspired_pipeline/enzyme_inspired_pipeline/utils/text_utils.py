#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Dict, List

SECTION_PATTERNS = [
    ("abstract", re.compile(r"\babstract\b", re.I)),
    ("experimental", re.compile(r"\b(experimental|experiment|methods?|materials? and methods?|assay|kinetics?)\b", re.I)),
    ("results_discussion", re.compile(r"\b(results?|discussion|results and discussion)\b", re.I)),
    ("conclusion", re.compile(r"\b(conclusion|conclusions|summary|outlook)\b", re.I)),
    ("references", re.compile(r"\b(references|bibliography)\b", re.I)),
    ("introduction", re.compile(r"\bintroduction\b", re.I)),
]
TABLE_SEMANTICS = {
    "has_enzyme": re.compile(r"\b(alkaline phosphatase|carboxyl(?:ic)? esterase|hydrolase|phosphatase|esterase)\b", re.I),
    "has_residue": re.compile(r"\b(serine|histidine|lysine|aspartate|glutamate|catalytic triad|active site)\b", re.I),
    "has_pka": re.compile(r"\bpK[aA]\b|deprotonation|protonation", re.I),
    "has_ph": re.compile(r"\bpH\b|buffer", re.I),
    "has_kinetics": re.compile(r"\bkcat\b|\bK[Mm]\b|kcat\s*/\s*K[Mm]|turnover|activity", re.I),
    "has_conformation": re.compile(r"conformation|conformational|loop closure|open state|closed state", re.I),
    "has_cofactor": re.compile(r"\b(Mg2\+|Zn2\+|Ca2\+|Mn2\+|metal ion|cofactor)\b", re.I),
}


UNICODE_TRANSLATION = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "Δ": "Delta", "δ": "delta",
    "∗": "*", "⁎": "*",
    "·": " ", "×": "x",
    "μ": "u", "µ": "u",
})


def normalize_text(s: object) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).translate(UNICODE_TRANSLATION).strip())


# H adsorption / deprotonation specific cues. Kept together with the original
# enzyme/pKa cues so older enzyme scripts remain compatible.
TABLE_SEMANTICS.update({
    "has_H_adsorption": re.compile(r"\b(HBE|hydrogen binding energy|hydrogen adsorption|H adsorption|Delta\s*G[_ -]?H|Delta\s*E[_ -]?H|\bH\*)\b", re.I),
    "has_OH_water": re.compile(r"\b(OHBE|OH adsorption|hydroxyl adsorption|water adsorption|H2O adsorption|water dissociation)\b", re.I),
    "has_deprotonation": re.compile(r"\b(deprotonation|proton transfer|Volmer|Heyrovsky|Tafel step|PCET|proton-coupled electron transfer)\b", re.I),
    "has_interfacial_water": re.compile(r"hydrogen[- ]bond network|interfacial water|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water|double layer|EDL", re.I),
    "has_electronic_descriptor": re.compile(r"d[- ]band center|Bader charge|charge density difference|electron localization function|\bELF\b|work function|PZC|DOS|PDOS", re.I),
    "has_catalyst_system": re.compile(r"catalyst|surface|facet|slab|dopant|doped|vacancy|defect|support|interface|bridge oxygen|M[-–]O[-–][CP]|Ni[-–]O[-–]P", re.I),
})


def clip_text(s: str, max_chars: int) -> str:
    s = normalize_text(s)
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rstrip() + " ...[TRUNCATED]"


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    return [p.strip() for p in re.split(r"(?<=[\.!?;])\s+", text) if p.strip()]


def infer_section_type(section_path: str, text: str = "") -> str:
    probe = f"{section_path} {text[:200]}"
    for name, pat in SECTION_PATTERNS:
        if pat.search(probe):
            return name
    if re.match(r"^PAGE_\d+$", str(section_path or "")):
        return "page_fallback"
    return "body"


def match_table_semantics(text: str) -> Dict[str, int]:
    t = normalize_text(text)
    return {k: int(bool(p.search(t))) for k, p in TABLE_SEMANTICS.items()}
