#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detailed add-on extraction for ΔE_H* + DFT mechanism descriptors.

This script is intended for pdf-files-add after the main HADS corpus has already
been processed. Compared with the lightweight 3g extractor, this version uses a
more detailed two-stage strategy per paper:

1) Target extraction: find ΔE_H* / H adsorption energy records and their
   material/site linkage.
2) Descriptor extraction: for each extracted target record, link same-system/site
   DFT descriptors such as d-band center, Bader charge, coordination number,
   oxidation state, vacancy/defect, work function, PZC, facet, and bridge oxygen.

It still avoids re-running the full 3b/3c/3d pipeline and remains compatible with
4_build_hads_normalized_database.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from utils.evidence_utils import dedup_evidence, normalize_evidence_list
from utils.io_utils import append_jsonl, load_progress, read_jsonl, save_progress
from utils.llm_client import MultiEndpointClient, safe_json_extract
from utils.normalize_utils import clean_unit, to_ev, to_float
from utils.text_utils import clip_text, normalize_text
from utils.hads_validation import (
    clean_material_name,
    clean_bridge,
    field_support_summary,
    number_supported,
    parse_oxidation_state,
    text_supported,
    validate_numeric_fields,
)

SEP = r"(?:[^0-9+\-−–.;\n]{0,110})"
NUM_UNIT = r"([-+−–]?\d+(?:\.\d+)?)\s*(eV|meV|kJ\s*mol[-−–]?[1l]?|kcal\s*mol[-−–]?[1l]?|J\s*mol[-−–]?[1l]?)?"
RE_H_E = re.compile(
    r"(?:H adsorption energy|adsorption energy of H|adsorption energy of hydrogen|hydrogen adsorption energy|"
    r"E[_\s-]?ads\s*\(?H\)?|Eads\s*\(?H\)?|E[_\s-]?ads[,\s]*H|"
    r"(?:Delta|Δ)\s*E[_\-\s]?H\*?|DE[_\-\s]?H\*?|"
    r"binding energy of H|H binding energy|hydrogen binding energy|HBE)" + SEP + NUM_UNIT,
    re.I,
)
RE_H_G = re.compile(r"(?:free energy of H adsorption|(?:Delta|Δ)\s*G[_\-\s]?H\*?|DG[_\-\s]?H\*?)" + SEP + NUM_UNIT, re.I)
RE_DESCRIPTOR = re.compile(
    r"\b(d[- ]?band center|Bader charge|Bader analysis|charge transfer|charge density difference|"
    r"electron density difference|electron localization function|ELF|work function|PZC|point of zero charge|"
    r"oxidation state|valence state|coordination number|coordination environment|local coordination|"
    r"DOS|PDOS|density of states|projected density of states|COHP|ICOHP|bond length|M[- ]?H bond|"
    r"vacancy|defect|dopant|doping|surface facet|active site|adsorption site|bridge oxygen|bridging oxygen|M[-–—]?O)\b",
    re.I,
)
RE_DFT = re.compile(r"\b(DFT|density functional theory|first[- ]principles|first principles|ab initio|VASP|Quantum ESPRESSO|PBE|RPBE|SCAN|HSE06|PBE\+U|DFT\+U|GGA\+U)\b", re.I)
RE_FUNC = re.compile(r"\b(PBE0?|RPBE|B3LYP|HSE06|SCAN|M06[- ]?L|PW91|PBE\+U|GGA\+U|DFT\+U|revPBE)\b", re.I)
RE_U = re.compile(r"(?:U\s*=\s*|Hubbard\s*U\s*(?:value)?\s*[:=]?\s*)([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_BADER = re.compile(r"(?:Bader charge|charge on [A-Z][a-z]?|Delta\s*q|\bΔq\b)" + SEP + r"([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_DBAND = re.compile(r"(?:d[- ]?band center|epsilon[_\- ]?d|ε_d)" + SEP + r"([-+−–]?\d+(?:\.\d+)?)\s*(?:eV)?", re.I)
RE_WORK = re.compile(r"(?:work function|Φ|Phi)" + SEP + r"([-+−–]?\d+(?:\.\d+)?)\s*(?:eV)?", re.I)
RE_PZC = re.compile(r"(?:PZC|point of zero charge)" + SEP + r"([-+−–]?\d+(?:\.\d+)?)\s*(?:V)?", re.I)
RE_CN = re.compile(r"(?:coordination number|coordination|\bCN\b)" + SEP + r"([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_OX = re.compile(r"(?:oxidation state|valence state|valence)" + SEP + r"([A-Z][a-z]?\s*\(?\s*(?:0|I{1,3}|IV|V|VI{0,3}|IX|X|[0-9]+(?:\.[0-9]+)?)[+\-⁺⁻]?\s*\)?|[+\-]?\d+(?:\.\d+)?)", re.I)
RE_VAC = re.compile(r"\b(oxygen vacancy|sulfur vacancy|nitrogen vacancy|vacancy|defect site|near vacancy|defect)\b", re.I)
RE_HYDROX = re.compile(r"\b(hydroxylated|surface hydroxyl|OH[- ]terminated|hydroxyl[- ]covered)\b", re.I)
RE_BRIDGE = re.compile(r"\b(bridging oxygen|bridge oxygen|(?:Ni|Fe|Co|Cu|Zn|Pt|Pd|Ru|Rh|Ir|Ce|Ti|Mo|W|Mn|Cr|Zr|La|Ga|In|Sn)[-–—]?O[-–—]?[A-Z][a-z]?|M[-–—]?O[-–—]?[A-Z][a-z]?|M[-–—]?O[-–—]?C/P)\b", re.I)
RE_SITE = re.compile(r"\b([A-Z][a-z]?(?:[- ]?(?:top|bridge|hollow))|O[- ]top|metal[- ]top|bridge site|hollow site|top site|active site|adsorption site|interface site|vacancy site)\b", re.I)
RE_FACET = re.compile(r"\([0-9]{3,4}\)|\b(?:facet|surface|plane)\s*(?:of\s*)?\(?([0-9]{3,4})\)?", re.I)
RE_COVERAGE = re.compile(r"(?:coverage|H coverage)\s*[:=]?\s*([0-9./]+\s*(?:ML|monolayer|monolayers))", re.I)
RE_MATERIAL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9@/\-()]{1,45}(?:\([0-9]{3,4}\))?\b")

# Tokens that often come from equations, labels, vibrational modes, methods, or table markers
# rather than real catalyst/material names.  We keep this conservative but stricter than
# the previous version because bad material names directly pollute downstream linking.
BAD_MATERIAL_RE = re.compile(
    r"^(?:"
    r"X\d+|S\d+|H\d+|I\d+|Hs\d+|TS\d+|IS\d+|FS\d+|IM\d+|R\d+|P\d+|"
    r"Fig\.?\d*|Figure\.?\d*|Table\.?\d*|Scheme\.?\d*|Eq\.?\d*|"
    r"B3LYP|PBE|RPBE|SCAN|HSE06|VASP|DFT|DOS|PDOS|X[- ]?ray|XPS|XANES|EXAFS|LO[- ]TO|"
    r"NH2|NH3|OH|H2O|H2|HER|HOR|HBE|PCET|Ava|Ur"
    r")$", re.I
)

ELEMENT_SYMBOLS = {
    'H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca',
    'Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr','Rb','Sr',
    'Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','I','Xe','Cs','Ba',
    'La','Ce','Pr','Nd','Pm','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Hf','Ta','W',
    'Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','At','Rn'
}
KNOWN_MATERIAL_WORDS = re.compile(
    r"\b(graphene|carbon|carbide|nitride|oxide|sulfide|sulphide|phosphide|boride|selenide|"
    r"hydroxide|oxyhydroxide|perovskite|spinel|mxene|mof|cof|zeolite|alloy|single[- ]atom|"
    r"nanotube|nanoribbon|nanosheet|slab|surface|catalyst|electrode|substrate|support)\b", re.I
)
SIMPLE_BOND_RE = re.compile(r"^(?:[HCONSPBFI]|Cl|Br)(?:[-–—](?:[HCONSPBFI]|Cl|Br)){1,4}$", re.I)
YEAR_FACET_RE = re.compile(r"^\(?((?:19|20)\d{2})\)?$")
ALLOWED_DESCRIPTOR_LINK_METHODS = {
    "same_table_row",
    "same_evidence_window",
    "two_stage_same_paper_link",
    "heuristic_same_paper_descriptor",
    "target_only_no_descriptor",
    "same_caption_or_paragraph",
    "explicit_same_catalyst_site",
}

TARGET_PROMPT = """
You are extracting TARGET H adsorption energy records from DFT literature evidence.
Use ONLY the evidence blocks. Return STRICT JSON only.

Return JSON with keys: paper_id, targets[].
Each item in targets[] must contain:
target_id, material_name, material_class, composition, surface_facet, support, dopant, defect_type,
site_label, ads_site_type, active_atom, adsorption_configuration,
H_adsorption_energy_value, H_adsorption_energy_unit,
H_adsorption_free_energy_value, H_adsorption_free_energy_unit,
coverage, DFT_functional, U_value, dispersion_correction, calculation_model, reference_surface,
confidence_score, evidence_ids[].

Rules:
- Focus ONLY on ΔE_H* / H adsorption energy / E_ads(H) / HBE / H binding energy.
- Do NOT put ΔG_H* into H_adsorption_energy_value; ΔG_H* may go only to H_adsorption_free_energy_value.
- H_adsorption_energy_value must be copied from evidence near the H adsorption phrase or a clearly labelled table column.
- Preserve the unit near the number. If unit is absent, use null.
- Reject potentials, pH, temperatures, current densities, overpotentials, particle sizes, frequencies, page numbers, figure/table numbers, reference citation numbers, and unrelated catalysts.
- Reject molecular H2 adsorption / storage records, eV/H2 values, and H2 binding energies; this pipeline is for adsorbed atomic H* on a catalyst surface.
- Do not use method names, spectral labels, bond names, transition-state labels, or table labels as material names. Examples to reject as material names: B3LYP, X-ray, LO-TO, C-H, O-H, C-O, H-C-C, TS32, NH2.
- If the value is not explicitly linked to a material/site, return null for uncertain linkage fields rather than guessing.
- Use target_id values TGT1, TGT2, ...
""".strip()

DESCRIPTOR_PROMPT = """
You are linking DFT mechanism descriptors to already extracted H adsorption energy target records.
Use ONLY the evidence blocks and the TARGET_RECORDS. Return STRICT JSON only.

Return JSON with keys: paper_id, descriptors[].
Each item in descriptors[] must contain:
target_id, material_name, site_label, active_atom,
coordination_number, vacancy_nearby, hydroxylated_surface,
oxidation_state_raw, oxidation_state_element, oxidation_state_value,
bader_charge, d_band_center, charge_transfer_direction, bridge_structure_nearby,
ELF_descriptor, work_function, PZC, bridge_structure, M_O_X_configuration,
descriptor_claim, descriptor_link_method, descriptor_link_confidence, evidence_ids[].

Rules:
- Link descriptors ONLY when they belong to the same material/system/site as the target record.
- Prefer descriptor values in the same table row, same paragraph, same figure caption, or explicitly same catalyst/site.
- If descriptor linkage is uncertain, set the descriptor to null.
- Descriptors include d-band center, Bader charge, charge transfer, coordination number, oxidation/valence state, DOS/PDOS, work function, PZC, vacancy/defect, dopant, facet, bridge oxygen, and adsorption site.
- Do not invent descriptors from general discussion or from a different catalyst.
- descriptor_link_method must be one of: same_table_row, same_evidence_window, same_caption_or_paragraph, explicit_same_catalyst_site, two_stage_same_paper_link, target_only_no_descriptor. Do not write a free-text explanation in descriptor_link_method.
""".strip()


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    if not s or s.lower() in {"none", "null", "nan", "unknown", "not mentioned", "n/a", "na"}:
        return None
    return s


def clean_listlike(x: Any) -> List[str]:
    if x is None:
        return []
    arr = x if isinstance(x, list) else re.split(r"[;,|/]+", str(x))
    out, seen = [], set()
    for a in arr:
        t = clean_scalar(a)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def to_num(x: Any) -> Optional[float]:
    return to_float(x)


def bool_presence(x: Any) -> Optional[str]:
    s = normalize_text(x).lower()
    if not s or s in {"none", "null", "nan", "unknown", "not mentioned", "n/a", "na"}:
        return None
    if s in {"yes", "true", "present", "presence", "1", "nearby"}:
        return "yes"
    if s in {"no", "false", "absent", "absence", "0"}:
        return "no"
    return normalize_text(x)


def contains_element_or_material_word(s: str) -> bool:
    ss = normalize_text(s)
    if KNOWN_MATERIAL_WORDS.search(ss):
        return True
    # Direct formula-like element detection. This deliberately requires a real element
    # symbol, so tokens such as Ava/Ur/LO-TO are rejected.
    for sym in ELEMENT_SYMBOLS:
        if re.search(rf"(?<![a-z]){re.escape(sym)}(?![a-z])", ss):
            return True
    return False


def is_bad_material_name(s: Optional[str]) -> bool:
    if not s:
        return True
    ss = normalize_text(s).strip()
    if not ss or BAD_MATERIAL_RE.match(ss):
        return True
    # Reject pure bond / fragment labels that are not catalysts. Fe-N-C is retained
    # because it contains a transition metal; C-H/O-H/C-O/H-C-C are rejected.
    if SIMPLE_BOND_RE.match(ss) and not re.search(r"\b(Fe|Co|Ni|Cu|Mn|Mo|W|Ru|Rh|Pd|Pt|Ir|Au|Ag|Zn|Ti|V|Cr|Zr|Ce)\b", ss):
        return True
    if re.fullmatch(r"[A-Z][a-z]?[-–—][A-Z][a-z]?(?:[-–—][A-Z][a-z]?){0,2}", ss) and not re.search(r"Fe|Co|Ni|Cu|Mn|Mo|W|Ru|Rh|Pd|Pt|Ir|Au|Ag|Zn|Ti|V|Cr|Zr|Ce", ss):
        return True
    if len(ss) <= 3 and ss not in ELEMENT_SYMBOLS and not re.search(r"\d", ss):
        return True
    if len(ss) <= 4 and re.match(r"^[A-Za-z]+\d+$", ss):
        return True
    if ss.lower() in {"reported catalyst", "surface", "catalyst", "sample", "model", "system", "method", "basis", "state"}:
        return True
    if not contains_element_or_material_word(ss):
        return True
    return False


def sanitize_material(raw: Any, ev_txt: str) -> Optional[str]:
    cm = clean_material_name(raw, ev_txt)
    if cm and not is_bad_material_name(cm):
        return cm
    s = clean_scalar(raw)
    if s and not is_bad_material_name(s):
        return s
    return None


def sanitize_surface_facet(raw: Any) -> Optional[str]:
    s = clean_scalar(raw)
    if not s:
        return None
    m = re.search(r"\(?([0-9]{3,4}|[0-9]{1,2}[-−–][0-9]{1,2}[-−–]?[0-9]{0,2})\)?", s)
    if not m:
        return None
    val = m.group(1).replace("−", "-").replace("–", "-")
    if YEAR_FACET_RE.match(val):
        return None
    # Common Miller indices are generally 3 digits; allow 0001 for hexagonal surfaces.
    if len(val) == 4 and val != "0001" and re.fullmatch(r"[0-9]{4}", val):
        return None
    return f"({val})"


def sanitize_site_like(raw: Any) -> Optional[str]:
    s = clean_scalar(raw)
    if not s:
        return None
    if BAD_MATERIAL_RE.match(s) or YEAR_FACET_RE.match(s):
        return None
    if s.lower() in {"x-ray", "xps", "xanes", "b3lyp", "pbe", "dft", "dos", "pdos", "lo-to"}:
        return None
    return s


def canonicalize_descriptor_link_method(x: Any, default: str = "two_stage_same_paper_link") -> str:
    s = normalize_text(x).strip()
    return s if s in ALLOWED_DESCRIPTOR_LINK_METHODS else default


def unit_mentions_h2(raw_unit: Any) -> bool:
    return bool(re.search(r"\bH\s*2\b|/\s*H2|per\s+H2", normalize_text(raw_unit), re.I))


def is_h2_adsorption_context(txt: str) -> bool:
    t = normalize_text(txt)
    bad = re.search(r"\b(H2|H\s*2|molecular hydrogen|hydrogen molecule)\b.{0,80}\b(adsorption|binding|storage)", t, re.I)
    bad = bad or re.search(r"\b(adsorption|binding)\b.{0,80}\b(H2|H\s*2|molecular hydrogen|hydrogen molecule)\b", t, re.I)
    good = re.search(r"H\*|H ads|E[_\s-]?ads\s*\(?H\)?|Delta\s*E[_\s-]?H|Δ\s*E[_\s-]?H", t, re.I)
    return bool(bad and not good)


def has_any_dft_descriptor(rec: Dict[str, Any]) -> int:
    keys = ["coordination_number", "oxidation_state_value", "bader_charge", "d_band_center", "work_function", "PZC", "vacancy_nearby", "hydroxylated_surface", "bridge_structure_nearby", "ELF_descriptor", "charge_transfer_direction"]
    return int(any(rec.get(k) not in (None, "", [], {}) for k in keys))


def finalize_qc_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    rec = dict(rec)
    rec["surface_facet"] = sanitize_surface_facet(rec.get("surface_facet"))
    rec["site_label"] = sanitize_site_like(rec.get("site_label"))
    rec["ads_site_type"] = sanitize_site_like(rec.get("ads_site_type"))
    rec["adsorption_configuration"] = sanitize_site_like(rec.get("adsorption_configuration"))
    rec["active_atom"] = sanitize_site_like(rec.get("active_atom"))
    rec["material_name"] = sanitize_material(rec.get("material_name"), evidence_text(rec.get("evidence_all", []))) or "reported catalyst/surface"
    rec["composition"] = sanitize_material(rec.get("composition"), evidence_text(rec.get("evidence_all", []))) or (rec.get("material_name") if rec.get("material_name") != "reported catalyst/surface" else None)
    rec["descriptor_link_method"] = canonicalize_descriptor_link_method(rec.get("descriptor_link_method"), "target_only_no_descriptor" if not has_any_dft_descriptor(rec) else "two_stage_same_paper_link")
    h2_unit = unit_mentions_h2(rec.get("H_adsorption_energy_unit"))
    h2_ctx = is_h2_adsorption_context(evidence_text(rec.get("evidence_all", [])))
    valid_mat = int(rec.get("material_name") not in (None, "", "reported catalyst/surface") and not is_bad_material_name(rec.get("material_name")))
    valid_facet = int(rec.get("surface_facet") is not None)
    is_hstar = int(not h2_unit and not h2_ctx and str(rec.get("adsorbate") or "H*").strip().upper() != "H2")
    any_desc = has_any_dft_descriptor(rec)
    rec["qc_valid_material_name"] = valid_mat
    rec["qc_valid_surface_facet"] = valid_facet
    rec["qc_is_Hstar_adsorption"] = is_hstar
    rec["qc_has_any_dft_descriptor"] = any_desc
    rec["qc_keep_for_deltaEH_descriptor_analysis"] = int(is_hstar and any_desc and to_float(rec.get("H_adsorption_energy_value_eV")) is not None)
    rec["qc_deltaeh_descriptor_filter_reason"] = ";".join([
        r for r, flag in [
            ("not_Hstar_or_H2_record", not is_hstar),
            ("no_dft_descriptor", not any_desc),
            ("missing_deltaE", to_float(rec.get("H_adsorption_energy_value_eV")) is None),
            ("invalid_material_name", not valid_mat),
        ] if flag
    ]) or "pass"
    return rec


def load_chunks(parsed_root: Path, paper_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(parsed_root / "papers" / paper_id / "chunks.jsonl")


def evidence_from_chunk(ch: Dict[str, Any], eid: str, score: float) -> Dict[str, Any]:
    return {
        "eid": eid,
        "paper_id": ch.get("paper_id"),
        "chunk_id": ch.get("chunk_id"),
        "chunk_type": normalize_text(ch.get("chunk_type") or "text"),
        "section_path": normalize_text(ch.get("section_path")),
        "page_start": ch.get("page_start"),
        "page_end": ch.get("page_end"),
        "source": normalize_text(ch.get("source")),
        "text": normalize_text(ch.get("text", "")),
        "score": float(score),
    }


def target_score(ch: Dict[str, Any]) -> float:
    txt = normalize_text(ch.get("text", ""))
    if not txt:
        return 0.0
    ctype = normalize_text(ch.get("chunk_type") or "text")
    s = 0.0
    if RE_H_E.search(txt): s += 10.0
    if RE_H_G.search(txt): s += 0.6
    if RE_DFT.search(txt): s += 1.0
    if RE_DESCRIPTOR.search(txt): s += 1.0
    if ctype == "table": s += 5.0
    elif ctype == "caption": s += 2.5
    if RE_H_E.search(txt) and RE_DESCRIPTOR.search(txt): s += 4.0
    return s


def descriptor_score(ch: Dict[str, Any]) -> float:
    txt = normalize_text(ch.get("text", ""))
    if not txt:
        return 0.0
    ctype = normalize_text(ch.get("chunk_type") or "text")
    s = 0.0
    if RE_DESCRIPTOR.search(txt): s += 8.0
    if RE_DFT.search(txt): s += 2.0
    if RE_H_E.search(txt): s += 2.0
    if ctype == "table": s += 4.0
    elif ctype == "caption": s += 2.0
    if RE_H_E.search(txt) and RE_DESCRIPTOR.search(txt): s += 8.0
    return s


def select_evidence(chunks: List[Dict[str, Any]], score_fn, limits: Dict[str, int]) -> List[Dict[str, Any]]:
    buckets = {"table": [], "caption": [], "text": []}
    for ch in chunks:
        sc = score_fn(ch)
        if sc <= 0:
            continue
        ctype = normalize_text(ch.get("chunk_type") or "text")
        key = "table" if ctype == "table" else ("caption" if ctype == "caption" else "text")
        buckets[key].append((sc, ch))
    out: List[Dict[str, Any]] = []
    counters = {"T": 0, "C": 0, "M": 0}
    for key, prefix in [("table", "T"), ("caption", "C"), ("text", "M")]:
        for sc, ch in sorted(buckets[key], key=lambda x: x[0], reverse=True)[:limits.get(key, 0)]:
            counters[prefix] += 1
            out.append(evidence_from_chunk(ch, f"{prefix}{counters[prefix]}", sc))
    return dedup_evidence(out)


def trim_evidence(evs: List[Dict[str, Any]], budgets: Dict[str, int], max_quote_chars: int) -> List[Dict[str, Any]]:
    used = {"text": 0, "table": 0, "caption": 0}
    out = []
    for ev in sorted(evs, key=lambda e: float(e.get("score") or 0), reverse=True):
        ctype = "table" if ev.get("chunk_type") == "table" else ("caption" if ev.get("chunk_type") == "caption" else "text")
        quote = clip_text(ev.get("text", ""), max_quote_chars)
        if not quote:
            continue
        if used[ctype] + len(quote) > budgets.get(ctype, 0):
            continue
        e2 = dict(ev)
        e2["quote"] = quote
        e2.pop("text", None)
        used[ctype] += len(quote)
        out.append(e2)
    return out


def pack_prompt_evidence(evs: List[Dict[str, Any]]) -> str:
    parts = []
    for ev in evs:
        meta = f"[{ev.get('eid')}] type={ev.get('chunk_type')} page={ev.get('page_start')} section={ev.get('section_path')} chunk={ev.get('chunk_id')}"
        parts.append(meta + "\n" + str(ev.get("quote") or ""))
    return "\n\n".join(parts)


def evidence_text(evs: List[Dict[str, Any]]) -> str:
    return normalize_text(" ".join(str(e.get("quote") or e.get("text") or "") for e in evs))


def evidence_for_ids(evs: List[Dict[str, Any]], ids: Sequence[Any]) -> List[Dict[str, Any]]:
    idset = {str(x).strip() for x in ids if str(x).strip()}
    if not idset:
        return evs[:10]
    selected = [e for e in evs if str(e.get("eid")) in idset]
    return selected[:14] if selected else evs[:10]


def infer_material_from_text(txt: str) -> Optional[str]:
    candidates = []
    for m in RE_MATERIAL_TOKEN.finditer(txt):
        token = m.group(0)
        cm = sanitize_material(token, txt)
        if cm:
            candidates.append(cm)
    candidates = sorted(set(candidates), key=lambda x: (0 if re.search(r"[@/()\d-]", x) else 1, len(x)))
    return candidates[0] if candidates else None


def build_target_prompt(paper_id: str, evs: List[Dict[str, Any]]) -> str:
    return TARGET_PROMPT + "\n\n" + f"paper_id={paper_id}\n\nTARGET_EVIDENCE:\n{pack_prompt_evidence(evs)}\n\nTASK: Extract target ΔE_H* / H adsorption energy records. Return JSON only."


def build_descriptor_prompt(paper_id: str, targets: List[Dict[str, Any]], evs: List[Dict[str, Any]]) -> str:
    target_payload = []
    for i, t in enumerate(targets, 1):
        target_payload.append({
            "target_id": t.get("target_id") or f"TGT{i}",
            "material_name": t.get("material_name"),
            "composition": t.get("composition"),
            "surface_facet": t.get("surface_facet"),
            "site_label": t.get("site_label"),
            "active_atom": t.get("active_atom"),
            "adsorption_configuration": t.get("adsorption_configuration"),
            "H_adsorption_energy_value": t.get("H_adsorption_energy_value"),
            "H_adsorption_energy_unit": t.get("H_adsorption_energy_unit"),
            "DFT_functional": t.get("DFT_functional"),
            "coverage": t.get("coverage"),
        })
    return DESCRIPTOR_PROMPT + "\n\n" + f"paper_id={paper_id}\n\nTARGET_RECORDS:\n{json.dumps(target_payload, ensure_ascii=False, indent=2)}\n\nDESCRIPTOR_EVIDENCE:\n{pack_prompt_evidence(evs)}\n\nTASK: Link DFT descriptors to the target records. Return JSON only."


def is_plausible_deltae(value: Any, unit: Any, max_abs_ev: float) -> bool:
    ev = to_ev(value, unit)
    if ev is None:
        return False
    if not math.isfinite(ev):
        return False
    return abs(ev) <= max_abs_ev


def postprocess_target(raw: Dict[str, Any], evs: List[Dict[str, Any]], ev_text_all: str, paper_id: str, idx: int, max_abs_ev: float, keep_h2_adsorption: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ids = raw.get("evidence_ids") if isinstance(raw.get("evidence_ids"), list) else []
    rec_evs = evidence_for_ids(evs, ids)
    ev_txt = evidence_text(rec_evs) or ev_text_all
    rec = dict(raw)
    rec["target_id"] = clean_scalar(rec.get("target_id")) or f"TGT{idx}"
    mat = sanitize_material(rec.get("material_name"), ev_txt) or sanitize_material(rec.get("composition"), ev_txt) or infer_material_from_text(ev_txt)
    rec["material_name"] = mat or "reported catalyst/surface"
    rec["composition"] = sanitize_material(rec.get("composition"), ev_txt) or (mat if mat else None)
    rec["H_adsorption_energy_value"] = to_num(rec.get("H_adsorption_energy_value"))
    raw_h_unit = rec.get("H_adsorption_energy_unit")
    if (not keep_h2_adsorption) and (unit_mentions_h2(raw_h_unit) or str(rec.get("adsorbate") or "").strip().upper() == "H2" or is_h2_adsorption_context(ev_txt)):
        return None
    rec["H_adsorption_energy_unit"] = clean_unit(raw_h_unit)
    rec["surface_facet"] = sanitize_surface_facet(rec.get("surface_facet"))
    rec["site_label"] = sanitize_site_like(rec.get("site_label"))
    rec["ads_site_type"] = sanitize_site_like(rec.get("ads_site_type"))
    rec["adsorption_configuration"] = sanitize_site_like(rec.get("adsorption_configuration"))
    rec["active_atom"] = sanitize_site_like(rec.get("active_atom"))
    rec["H_adsorption_free_energy_value"] = to_num(rec.get("H_adsorption_free_energy_value"))
    rec["H_adsorption_free_energy_unit"] = clean_unit(rec.get("H_adsorption_free_energy_unit"))
    rec["U_value"] = to_num(rec.get("U_value"))
    rec["confidence_score"] = to_num(rec.get("confidence_score"))
    rec, supp = validate_numeric_fields(rec, ["H_adsorption_energy_value"], ev_txt)
    if rec.get("H_adsorption_energy_value") is None:
        return None
    if rec.get("H_adsorption_energy_unit") is None:
        rec["H_adsorption_energy_unit"] = "eV"
    if not is_plausible_deltae(rec.get("H_adsorption_energy_value"), rec.get("H_adsorption_energy_unit"), max_abs_ev):
        return None
    rec["H_adsorption_energy_value_eV"] = to_ev(rec.get("H_adsorption_energy_value"), rec.get("H_adsorption_energy_unit"))
    rec["target_evidence"] = rec_evs
    rec["target_numeric_support"] = supp
    return rec


def heuristic_targets(paper_id: str, evs: List[Dict[str, Any]], relpath: str, max_abs_ev: float) -> List[Dict[str, Any]]:
    txt = evidence_text(evs)
    out = []
    for i, m in enumerate(RE_H_E.finditer(txt), 1):
        win = txt[max(0, m.start() - 1000): min(len(txt), m.end() + 1200)]
        raw = {
            "target_id": f"TGT{i}",
            "material_name": infer_material_from_text(win) or infer_material_from_text(txt),
            "composition": infer_material_from_text(win),
            "surface_facet": sanitize_surface_facet(RE_FACET.search(win).group(0)) if RE_FACET.search(win) else None,
            "site_label": RE_SITE.search(win).group(1) if RE_SITE.search(win) else None,
            "active_atom": None,
            "adsorption_configuration": RE_SITE.search(win).group(1) if RE_SITE.search(win) else None,
            "H_adsorption_energy_value": to_num(m.group(1)),
            "H_adsorption_energy_unit": clean_unit(m.group(2)) or "eV",
            "coverage": RE_COVERAGE.search(win).group(1) if RE_COVERAGE.search(win) else None,
            "DFT_functional": RE_FUNC.search(win).group(1) if RE_FUNC.search(win) else None,
            "U_value": to_num(RE_U.search(win).group(1)) if RE_U.search(win) else None,
            "evidence_ids": [],
        }
        pp = postprocess_target(raw, evs, txt, paper_id, i, max_abs_ev, False)
        if pp:
            out.append(pp)
    return out


def descriptor_from_window(win: str) -> Dict[str, Any]:
    ox_raw = RE_OX.search(win).group(1) if RE_OX.search(win) else None
    ox_raw2, ox_el, ox_val = parse_oxidation_state(ox_raw, None)
    bridge = clean_bridge(RE_BRIDGE.search(win).group(1), win) if RE_BRIDGE.search(win) else None
    return {
        "coordination_number": to_num(RE_CN.search(win).group(1)) if RE_CN.search(win) else None,
        "vacancy_nearby": "yes" if RE_VAC.search(win) else None,
        "hydroxylated_surface": "yes" if RE_HYDROX.search(win) else None,
        "oxidation_state_raw": ox_raw2,
        "oxidation_state_element": ox_el,
        "oxidation_state_value": ox_val,
        "bader_charge": to_num(RE_BADER.search(win).group(1)) if RE_BADER.search(win) else None,
        "d_band_center": to_num(RE_DBAND.search(win).group(1)) if RE_DBAND.search(win) else None,
        "work_function": to_num(RE_WORK.search(win).group(1)) if RE_WORK.search(win) else None,
        "PZC": to_num(RE_PZC.search(win).group(1)) if RE_PZC.search(win) else None,
        "bridge_structure_nearby": bridge,
        "bridge_structure": bridge,
        "M_O_X_configuration": bridge,
    }


def normalize_descriptor(raw: Dict[str, Any], evs: List[Dict[str, Any]], ev_text_all: str) -> Dict[str, Any]:
    ids = raw.get("evidence_ids") if isinstance(raw.get("evidence_ids"), list) else []
    rec_evs = evidence_for_ids(evs, ids)
    ev_txt = evidence_text(rec_evs) or ev_text_all
    ox_raw = clean_scalar(raw.get("oxidation_state_raw") or raw.get("oxidation_state"))
    ox_el = clean_scalar(raw.get("oxidation_state_element") or raw.get("active_atom"))
    ox_raw2, ox_el2, ox_val2 = parse_oxidation_state(ox_raw, ox_el)
    ox_val = to_num(raw.get("oxidation_state_value"))
    if ox_val is None:
        ox_val = ox_val2
    bridge = clean_bridge(raw.get("bridge_structure_nearby"), ev_txt) or clean_bridge(raw.get("bridge_structure"), ev_txt) or clean_bridge(raw.get("M_O_X_configuration"), ev_txt)
    d = {
        "target_id": clean_scalar(raw.get("target_id")),
        "site_label": sanitize_site_like(raw.get("site_label")),
        "active_atom": sanitize_site_like(raw.get("active_atom")) or ox_el2 or ox_el,
        "coordination_number": to_num(raw.get("coordination_number")),
        "vacancy_nearby": bool_presence(raw.get("vacancy_nearby")),
        "hydroxylated_surface": bool_presence(raw.get("hydroxylated_surface")),
        "oxidation_state_raw": ox_raw2 or ox_raw,
        "oxidation_state_element": ox_el2 or ox_el,
        "oxidation_state_value": ox_val,
        "bader_charge": to_num(raw.get("bader_charge")),
        "d_band_center": to_num(raw.get("d_band_center")),
        "charge_transfer_direction": clean_scalar(raw.get("charge_transfer_direction")),
        "bridge_structure_nearby": bridge,
        "bridge_structure": bridge,
        "M_O_X_configuration": clean_bridge(raw.get("M_O_X_configuration"), ev_txt) or bridge,
        "ELF_descriptor": clean_scalar(raw.get("ELF_descriptor")),
        "work_function": to_num(raw.get("work_function")),
        "PZC": to_num(raw.get("PZC")),
        "descriptor_claim": clean_scalar(raw.get("descriptor_claim")),
        "descriptor_link_method": canonicalize_descriptor_link_method(raw.get("descriptor_link_method"), "two_stage_same_paper_link"),
        "descriptor_link_confidence": to_num(raw.get("descriptor_link_confidence")) or 0.70,
        "descriptor_evidence": rec_evs,
    }
    d, desc_support = validate_numeric_fields(d, ["coordination_number", "bader_charge", "d_band_center", "work_function", "PZC"], ev_txt)
    if d.get("oxidation_state_value") is not None and not (number_supported(d.get("oxidation_state_value"), ev_txt) or text_supported(d.get("oxidation_state_raw"), ev_txt)):
        d["oxidation_state_value"] = None
    desc_support["oxidation_state_value"] = "supported" if d.get("oxidation_state_value") is not None else "missing"
    d["descriptor_numeric_support"] = desc_support
    return d


def build_canonical(paper_id: str, relpath: str, target: Dict[str, Any], desc: Dict[str, Any], idx: int) -> Dict[str, Any]:
    target_evs = target.get("target_evidence", [])
    desc_evs = desc.get("descriptor_evidence", []) if desc else []
    ev_all = normalize_evidence_list(target_evs + desc_evs)[:40]
    ev_txt = evidence_text(ev_all)
    mat = sanitize_material(target.get("material_name"), ev_txt) or sanitize_material(target.get("composition"), ev_txt) or "reported catalyst/surface"
    rec: Dict[str, Any] = {
        "paper_id": paper_id,
        "canonical_id": f"{paper_id}_detailed_add_{idx:04d}",
        "reaction_family": "H_adsorption",
        "paper_type": "theoretical",
        "priority_hint": "deltaE_descriptor_detailed_add",
        "relpath": relpath,
        "bucket": "pdf-files-add",
        "system_local_id": f"{paper_id}_sys_{idx:03d}",
        "site_local_id": f"{paper_id}_sys_{idx:03d}_site_001",
        "material_name": mat,
        "material_class": clean_scalar(target.get("material_class")),
        "composition": sanitize_material(target.get("composition"), ev_txt) or mat,
        "surface_facet": sanitize_surface_facet(target.get("surface_facet")),
        "termination": None,
        "phase": None,
        "support": clean_scalar(target.get("support")),
        "dopant": clean_scalar(target.get("dopant")),
        "interface_type": None,
        "defect_type": (clean_scalar(target.get("defect_type")) or (clean_scalar(desc.get("defect_type")) if desc else None)),
        "bridge_structure": clean_bridge(desc.get("bridge_structure"), ev_txt) if desc else None,
        "M_O_X_configuration": clean_bridge(desc.get("M_O_X_configuration"), ev_txt) if desc else None,
        "model_type": clean_scalar(target.get("calculation_model")),
        "ads_site_type": sanitize_site_like(target.get("ads_site_type")),
        "site_label": sanitize_site_like(desc.get("site_label")) if desc and desc.get("site_label") else sanitize_site_like(target.get("site_label")),
        "active_atom": sanitize_site_like(desc.get("active_atom")) if desc and desc.get("active_atom") else sanitize_site_like(target.get("active_atom")),
        "neighbor_atoms": [],
        "coordination_number": desc.get("coordination_number") if desc else None,
        "local_geometry": None,
        "vacancy_nearby": desc.get("vacancy_nearby") if desc else None,
        "hydroxylated_surface": desc.get("hydroxylated_surface") if desc else None,
        "oxidation_state": desc.get("oxidation_state_raw") if desc else None,
        "oxidation_state_raw": desc.get("oxidation_state_raw") if desc else None,
        "oxidation_state_element": desc.get("oxidation_state_element") if desc else None,
        "oxidation_state_value": desc.get("oxidation_state_value") if desc else None,
        "bader_charge": desc.get("bader_charge") if desc else None,
        "d_band_center": desc.get("d_band_center") if desc else None,
        "charge_transfer_direction": desc.get("charge_transfer_direction") if desc else None,
        "bridge_structure_nearby": desc.get("bridge_structure_nearby") if desc else None,
        "ELF_descriptor": desc.get("ELF_descriptor") if desc else None,
        "work_function": desc.get("work_function") if desc else None,
        "PZC": desc.get("PZC") if desc else None,
        "hbond_acceptor_site": None,
        "H2O_binding_mode": None,
        "interfacial_water_role": None,
        "hydrogen_bond_network": None,
        "strong_HB_water_signal": None,
        "weak_HB_water_signal": None,
        "site_descriptor_claim": desc.get("descriptor_claim") if desc else None,
        "adsorbate": "H*",
        "H_adsorption_energy_value": target.get("H_adsorption_energy_value"),
        "H_adsorption_energy_unit": target.get("H_adsorption_energy_unit"),
        "H_adsorption_energy_value_eV": target.get("H_adsorption_energy_value_eV"),
        "H_adsorption_free_energy_value": target.get("H_adsorption_free_energy_value"),
        "H_adsorption_free_energy_unit": target.get("H_adsorption_free_energy_unit"),
        "H_adsorption_free_energy_value_eV": to_ev(target.get("H_adsorption_free_energy_value"), target.get("H_adsorption_free_energy_unit")),
        "coverage": clean_scalar(target.get("coverage")),
        "coadsorbates": [],
        "solvation_model": clean_scalar(target.get("solvation_model")),
        "spin_state": None,
        "DFT_functional": clean_scalar(target.get("DFT_functional")),
        "U_value": to_num(target.get("U_value")),
        "dispersion_correction": clean_scalar(target.get("dispersion_correction")),
        "calculation_model": clean_scalar(target.get("calculation_model")),
        "adsorption_configuration": sanitize_site_like(target.get("adsorption_configuration")),
        "reference_surface": clean_scalar(target.get("reference_surface")),
        "adsorption_strength_trend": None,
        "proton_transfer_pathway": None,
        "rate_determining_step": None,
        "adsorption_descriptor_claim": desc.get("descriptor_claim") if desc else None,
        "system_confidence_score": target.get("confidence_score"),
        "site_confidence_score": desc.get("descriptor_link_confidence") if desc else None,
        "adsorption_confidence_score": target.get("confidence_score"),
        "site_system_match_score": 0.55 if desc else 0.35,
        "adsorption_system_match_score": 0.60,
        "site_extraction_source": "detailed_two_stage_llm" if desc else "detailed_target_only_llm",
        "adsorption_extraction_source": "detailed_two_stage_llm",
        "descriptor_link_method": canonicalize_descriptor_link_method(desc.get("descriptor_link_method") if desc else "target_only_no_descriptor", "target_only_no_descriptor" if not desc else "two_stage_same_paper_link"),
        "descriptor_link_confidence": desc.get("descriptor_link_confidence") if desc else 0.0,
        "descriptor_link_is_soft": 0 if desc else 1,
        "descriptor_repair_method": None,
        "descriptor_repair_confidence": None,
        "descriptor_repair_source": None,
        "descriptor_repaired_fields": [],
        "link_score": 0.90 if desc else 0.72,
        "target_relevance": 1.0,
        "system_signal": 0.7,
        "metric_signal": 1.0,
        "mechanism_signal": 0.8 if desc else 0.2,
        "site_qc_numeric_support": desc.get("descriptor_numeric_support") if desc else {},
        "adsorption_qc_numeric_support": target.get("target_numeric_support"),
        "site_qc_field_support": field_support_summary(desc if desc else {}, ev_txt, ["site_label", "active_atom", "bridge_structure_nearby", "oxidation_state_raw"]),
        "adsorption_qc_field_support": field_support_summary(target, ev_txt, ["adsorbate", "coverage", "adsorption_configuration", "reference_surface"]),
        "evidence_all": ev_all,
    }
    return finalize_qc_record(rec)


def process_paper(row: Dict[str, Any], args: argparse.Namespace, llm: Optional[MultiEndpointClient]) -> Tuple[str, str, int, Optional[str]]:
    paper_id = normalize_text(row.get("paper_id"))
    relpath = normalize_text(row.get("relpath"))
    chunks = load_chunks(Path(args.parsed_root), paper_id)
    if not chunks:
        return "skip", paper_id, 0, "no_chunks"
    # Full local prefilter, no API call if no ΔE_H-like cue anywhere.
    if not any(RE_H_E.search(normalize_text(c.get("text", ""))) for c in chunks):
        return "skip", paper_id, 0, "no_deltaE_cue"

    target_evs = select_evidence(chunks, target_score, {"text": args.target_topk_text, "table": args.target_topk_table, "caption": args.target_topk_caption})
    target_evs = trim_evidence(target_evs, {"text": args.target_text_budget_chars, "table": args.target_table_budget_chars, "caption": args.target_caption_budget_chars}, args.max_quote_chars)
    if not target_evs:
        return "skip", paper_id, 0, "no_target_evidence"
    desc_evs = select_evidence(chunks, descriptor_score, {"text": args.desc_topk_text, "table": args.desc_topk_table, "caption": args.desc_topk_caption})
    desc_evs = trim_evidence(desc_evs, {"text": args.desc_text_budget_chars, "table": args.desc_table_budget_chars, "caption": args.desc_caption_budget_chars}, args.max_quote_chars)
    if not desc_evs:
        desc_evs = target_evs

    target_obj = None
    if llm is not None:
        target_obj = safe_json_extract(llm.completions(build_target_prompt(paper_id, target_evs), temperature=0.0, max_tokens=args.target_max_tokens))
    target_rows: List[Dict[str, Any]] = []
    ev_target_all = evidence_text(target_evs)
    if isinstance(target_obj, dict) and isinstance(target_obj.get("targets"), list):
        for i, raw in enumerate(target_obj.get("targets", []), 1):
            pp = postprocess_target(raw, target_evs, ev_target_all, paper_id, i, args.energy_abs_max_ev, args.keep_h2_adsorption)
            if pp:
                target_rows.append(pp)
    if not target_rows and args.heuristic_fallback:
        target_rows = heuristic_targets(paper_id, target_evs, relpath, args.energy_abs_max_ev)
    if not target_rows:
        return "skip", paper_id, 0, "no_valid_deltaE_record"

    # De-duplicate target rows before descriptor linking.
    uniq_targets, seen = [], set()
    for t in target_rows:
        key = (str(t.get("material_name")), str(t.get("site_label")), str(t.get("active_atom")), f"{t.get('H_adsorption_energy_value_eV')}", str(t.get("DFT_functional")), str(t.get("coverage")))
        if key in seen:
            continue
        seen.add(key); uniq_targets.append(t)
    target_rows = uniq_targets[:args.max_targets_per_paper]

    desc_map: Dict[str, Dict[str, Any]] = {}
    if llm is not None and target_rows:
        try:
            desc_obj = safe_json_extract(llm.completions(build_descriptor_prompt(paper_id, target_rows, desc_evs), temperature=0.0, max_tokens=args.desc_max_tokens))
            if isinstance(desc_obj, dict) and isinstance(desc_obj.get("descriptors"), list):
                ev_desc_all = evidence_text(desc_evs)
                for raw in desc_obj.get("descriptors", []):
                    if isinstance(raw, dict):
                        dd = normalize_descriptor(raw, desc_evs, ev_desc_all)
                        tid = clean_scalar(dd.get("target_id"))
                        if tid:
                            desc_map[tid] = dd
        except Exception as e:
            if args.verbose:
                print(f"[WARN] descriptor LLM failed for {paper_id}: {e}", flush=True)
    # Heuristic descriptor fallback from combined evidence, only fills if LLM did not link.
    if args.heuristic_fallback:
        all_txt = evidence_text(target_evs + desc_evs)
        fallback_desc = descriptor_from_window(all_txt[:40000])
        for t in target_rows:
            tid = t.get("target_id")
            if tid not in desc_map and any(fallback_desc.get(k) is not None for k in ["coordination_number", "bader_charge", "d_band_center", "work_function", "PZC", "oxidation_state_value", "vacancy_nearby", "hydroxylated_surface", "bridge_structure_nearby"]):
                dd = dict(fallback_desc)
                dd.update({"target_id": tid, "descriptor_link_method": "heuristic_same_paper_descriptor", "descriptor_link_confidence": 0.45, "descriptor_evidence": (target_evs + desc_evs)[:10], "descriptor_numeric_support": {}})
                desc_map[tid] = dd

    records = []
    for i, t in enumerate(target_rows, 1):
        tid = t.get("target_id") or f"TGT{i}"
        records.append(build_canonical(paper_id, relpath, t, desc_map.get(tid, {}), i))

    return "ok", paper_id, len(records), json.dumps(records, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parsed-root", required=True)
    ap.add_argument("--api-bases", default="")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=360)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--target-max-tokens", type=int, default=3000)
    ap.add_argument("--desc-max-tokens", type=int, default=4500)
    ap.add_argument("--target-topk-text", type=int, default=16)
    ap.add_argument("--target-topk-table", type=int, default=30)
    ap.add_argument("--target-topk-caption", type=int, default=8)
    ap.add_argument("--desc-topk-text", type=int, default=24)
    ap.add_argument("--desc-topk-table", type=int, default=32)
    ap.add_argument("--desc-topk-caption", type=int, default=10)
    ap.add_argument("--target-text-budget-chars", type=int, default=14000)
    ap.add_argument("--target-table-budget-chars", type=int, default=22000)
    ap.add_argument("--target-caption-budget-chars", type=int, default=6000)
    ap.add_argument("--desc-text-budget-chars", type=int, default=22000)
    ap.add_argument("--desc-table-budget-chars", type=int, default=26000)
    ap.add_argument("--desc-caption-budget-chars", type=int, default=8000)
    ap.add_argument("--max-quote-chars", type=int, default=2200)
    ap.add_argument("--max-targets-per-paper", type=int, default=24)
    ap.add_argument("--energy-abs-max-ev", type=float, default=5.0, help="Drop suspect H adsorption energies outside this absolute eV range.")
    ap.add_argument("--keep-h2-adsorption", action="store_true", help="Keep molecular H2 adsorption records. Default is to exclude them from ΔE_H* extraction.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default="outputs_add_deltaeh_detailed/progress_3g_detailed_extract.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--heuristic-fallback", action="store_true", default=True)
    ap.add_argument("--no-heuristic-fallback", dest="heuristic_fallback", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "paper_id" in manifest.columns:
        manifest["paper_id"] = manifest["paper_id"].astype(str).str.strip()
        manifest = manifest[manifest["paper_id"].ne("")].drop_duplicates(subset=["paper_id"], keep="last").copy()
    rows = manifest.to_dict(orient="records")
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress); progress_path.parent.mkdir(parents=True, exist_ok=True)
    done = set(load_progress(progress_path).get("done", [])) if progress_path.exists() and not args.force else set()
    if args.force and outp.exists():
        outp.unlink()

    llm = None
    if args.api_bases.strip() and args.model_id.strip():
        llm = MultiEndpointClient([x.strip() for x in args.api_bases.split(",") if x.strip()], args.model_id, timeout=args.timeout, max_retries=args.max_retries)
    else:
        print("[WARN] No LLM configured; heuristic extraction only.", flush=True)

    lock = threading.Lock()
    counter = {"done": 0, "ok": 0, "skip": 0, "err": 0, "records": 0}
    prog = {"done": sorted(done)}

    def handle(row):
        pid = normalize_text(row.get("paper_id"))
        if pid in done:
            return "skip", pid, 0, "already_done"
        try:
            return process_paper(row, args, llm)
        except Exception as e:
            return "err", pid, 0, str(e)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(handle, row): row for row in rows}
        for fut in as_completed(futs):
            status, paper_id, n, payload = fut.result()
            with lock:
                counter["done"] += 1
                if status == "ok":
                    counter["ok"] += 1; counter["records"] += int(n)
                    if payload:
                        for r in json.loads(payload):
                            append_jsonl(r, outp)
                    done.add(paper_id); prog["done"] = sorted(done); save_progress(prog, progress_path)
                    print(f"[OK] {paper_id} detailed_deltaE_records={n} progress={counter['done']}/{len(rows)} total_records={counter['records']}", flush=True)
                elif status == "skip":
                    counter["skip"] += 1
                    done.add(paper_id); prog["done"] = sorted(done); save_progress(prog, progress_path)
                    print(f"[SKIP] {paper_id} reason={payload} progress={counter['done']}/{len(rows)}", flush=True)
                else:
                    counter["err"] += 1
                    print(f"[ERR] {paper_id} error={payload} progress={counter['done']}/{len(rows)}", flush=True)
    print(f"[DONE] wrote {args.out}; records={counter['records']} ok_papers={counter['ok']} skipped={counter['skip']} errors={counter['err']}", flush=True)


if __name__ == "__main__":
    main()
