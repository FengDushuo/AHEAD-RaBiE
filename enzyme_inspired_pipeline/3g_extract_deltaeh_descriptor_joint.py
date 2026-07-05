#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focused add-on extraction for ΔE_H* + DFT mechanism descriptors.

This script is designed for supplemental PDFs that were added after the main
HADS corpus was already processed. It intentionally uses ONE compact extraction
pass per paper instead of the full 3b/3c/3d pipeline to control API tokens.

Output is canonical-record JSONL compatible with 4_build_hads_normalized_database.py.
The focus is joint coverage: records must contain H_adsorption_energy_value
(ΔE_H*, E_ads(H), HBE, H binding energy) and are enriched with same-system/site
DFT descriptors when evidence supports them.
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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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

# -----------------------------------------------------------------------------
# Focused patterns
# -----------------------------------------------------------------------------
SEP = r"(?:[^0-9+\-−–.;\n]{0,90})"
NUM_UNIT = r"([-+−–]?\d+(?:\.\d+)?)\s*(eV|meV|kJ\s*mol[-−–]?[1l]?|kcal\s*mol[-−–]?[1l]?|J\s*mol[-−–]?[1l]?)?"
RE_H_E = re.compile(
    r"(?:H adsorption energy|adsorption energy of H|adsorption energy of hydrogen|hydrogen adsorption energy|"
    r"E[_\s-]?ads\s*\(?H\)?|Eads\s*\(?H\)?|(?:Delta|Δ)\s*E[_\-\s]?H\*?|DE[_\-\s]?H\*?|"
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
RE_MATERIAL_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9@/\-()]{1,35}(?:\([0-9]{3,4}\))?\b")

NUMERIC_FIELDS = [
    "H_adsorption_energy_value",
    "H_adsorption_free_energy_value",
    "coordination_number",
    "oxidation_state_value",
    "bader_charge",
    "d_band_center",
    "work_function",
    "PZC",
    "U_value",
]

SYSTEM_PROMPT = """
You are extracting ONLY records that jointly connect ΔE_H* / H adsorption energy with DFT mechanism descriptors.
Use ONLY the provided evidence blocks. Return STRICT JSON only.

Return JSON with keys: paper_id, records[].
Each record must contain these keys:
paper_id, material_name, material_class, composition, surface_facet, support, dopant, defect_type, bridge_structure, M_O_X_configuration,
site_label, ads_site_type, active_atom, coordination_number, vacancy_nearby, hydroxylated_surface,
oxidation_state_raw, oxidation_state_element, oxidation_state_value, bader_charge, d_band_center, charge_transfer_direction,
bridge_structure_nearby, ELF_descriptor, work_function, PZC,
adsorbate, H_adsorption_energy_value, H_adsorption_energy_unit, H_adsorption_free_energy_value, H_adsorption_free_energy_unit,
coverage, solvation_model, DFT_functional, U_value, dispersion_correction, calculation_model, adsorption_configuration, reference_surface,
descriptor_claim, confidence_score, evidence_ids[].

Rules:
- Focus on ΔE_H* / H adsorption energy / E_ads(H) / HBE. Do NOT put ΔG_H* into H_adsorption_energy_value.
- H_adsorption_energy_value must be a number copied from evidence. Preserve unit near the number; use null if unit is absent.
- Extract DFT descriptors only when they belong to the same material/system/site as the H adsorption energy.
- Descriptor examples: d-band center, Bader charge, charge transfer, coordination number, oxidation/valence state, DOS/PDOS, work function, PZC, vacancy/defect, dopant, facet, bridge oxygen, adsorption site.
- Do not use potentials, temperatures, pH, current density, overpotential, particle size, figure number, page number, or unrelated reference catalyst values as adsorption energies or descriptors.
- If a paper reports several catalysts/sites with H adsorption energies, return one record for each explicitly supported catalyst/site.
- If same-system linkage is uncertain, return null for the uncertain descriptor rather than guessing.
- evidence_ids must list the evidence block IDs supporting the record, such as ["T1", "M3"].
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


def chunk_score(ch: Dict[str, Any]) -> float:
    txt = normalize_text(ch.get("text", ""))
    if not txt:
        return 0.0
    ctype = normalize_text(ch.get("chunk_type") or "text")
    s = 0.0
    if RE_H_E.search(txt): s += 8.0
    if RE_DESCRIPTOR.search(txt): s += 3.0
    if RE_DFT.search(txt): s += 1.5
    if RE_H_G.search(txt): s += 0.5
    if ctype == "table": s += 4.0
    elif ctype == "caption": s += 2.0
    # Prioritize evidence that has both target and descriptors in one block.
    if RE_H_E.search(txt) and RE_DESCRIPTOR.search(txt): s += 6.0
    if RE_H_E.search(txt) and ctype == "table": s += 4.0
    return s


def select_evidence(chunks: List[Dict[str, Any]], topk_text: int, topk_table: int, topk_caption: int) -> List[Dict[str, Any]]:
    buckets = {"table": [], "caption": [], "text": []}
    for ch in chunks:
        sc = chunk_score(ch)
        if sc <= 0:
            continue
        ctype = normalize_text(ch.get("chunk_type") or "text")
        key = "table" if ctype == "table" else ("caption" if ctype == "caption" else "text")
        buckets[key].append((sc, ch))
    out: List[Dict[str, Any]] = []
    counters = {"T": 0, "C": 0, "M": 0}
    for key, limit, prefix in [("table", topk_table, "T"), ("caption", topk_caption, "C"), ("text", topk_text, "M")]:
        arr = sorted(buckets[key], key=lambda x: x[0], reverse=True)[:limit]
        for sc, ch in arr:
            counters[prefix] += 1
            out.append(evidence_from_chunk(ch, f"{prefix}{counters[prefix]}", sc))
    return dedup_evidence(out)


def trim_evidence_to_budgets(evs: List[Dict[str, Any]], text_budget: int, table_budget: int, caption_budget: int, max_quote_chars: int) -> List[Dict[str, Any]]:
    used = {"text": 0, "table": 0, "caption": 0}
    budgets = {"text": text_budget, "table": table_budget, "caption": caption_budget}
    out = []
    # Keep score order within original table/caption/text ordering, sorted globally for compactness.
    for ev in sorted(evs, key=lambda e: float(e.get("score") or 0), reverse=True):
        ctype = "table" if ev.get("chunk_type") == "table" else ("caption" if ev.get("chunk_type") == "caption" else "text")
        quote = clip_text(ev.get("text", ""), max_quote_chars)
        if not quote:
            continue
        if used[ctype] + len(quote) > budgets[ctype]:
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
        return evs[:8]
    selected = [e for e in evs if str(e.get("eid")) in idset]
    return selected[:12] if selected else evs[:8]


def infer_material_from_text(txt: str) -> Optional[str]:
    candidates = []
    for m in RE_MATERIAL_TOKEN.finditer(txt):
        token = m.group(0)
        cm = clean_material_name(token, txt)
        if cm:
            candidates.append(cm)
    # Prefer tokens with formula separators or facets; otherwise first candidate.
    candidates = sorted(set(candidates), key=lambda x: (0 if re.search(r"[@/()\d-]", x) else 1, len(x)))
    return candidates[0] if candidates else None


def heuristic_records(paper_id: str, evs: List[Dict[str, Any]], relpath: str = "") -> List[Dict[str, Any]]:
    txt = evidence_text(evs)
    rows = []
    for i, m in enumerate(RE_H_E.finditer(txt), 1):
        st, en = m.start(), m.end()
        win = txt[max(0, st - 850): min(len(txt), en + 1100)]
        mat = infer_material_from_text(win) or infer_material_from_text(txt) or "reported catalyst/surface"
        ox_raw = None
        ox_m = RE_OX.search(win)
        if ox_m:
            ox_raw = ox_m.group(1)
        ox_raw2, ox_el, ox_val = parse_oxidation_state(ox_raw, None)
        bridge = clean_bridge(RE_BRIDGE.search(win).group(1), win) if RE_BRIDGE.search(win) else None
        rec = {
            "paper_id": paper_id,
            "canonical_id": f"{paper_id}_add_{i:04d}",
            "reaction_family": "H_adsorption",
            "paper_type": "theoretical" if RE_DFT.search(win) else None,
            "priority_hint": "deltaE_descriptor_add",
            "relpath": relpath,
            "bucket": "pdf-files-add",
            "system_local_id": f"{paper_id}_sys_{i:03d}",
            "site_local_id": f"{paper_id}_sys_{i:03d}_site_001",
            "material_name": mat,
            "composition": mat if mat != "reported catalyst/surface" else None,
            "surface_facet": (RE_FACET.search(win).group(0) if RE_FACET.search(win) else None),
            "defect_type": RE_VAC.search(win).group(1) if RE_VAC.search(win) else None,
            "bridge_structure": bridge,
            "M_O_X_configuration": bridge,
            "site_label": RE_SITE.search(win).group(1) if RE_SITE.search(win) else None,
            "active_atom": None,
            "coordination_number": to_num(RE_CN.search(win).group(1)) if RE_CN.search(win) else None,
            "vacancy_nearby": "yes" if RE_VAC.search(win) else None,
            "hydroxylated_surface": "yes" if RE_HYDROX.search(win) else None,
            "oxidation_state": ox_raw2,
            "oxidation_state_raw": ox_raw2,
            "oxidation_state_element": ox_el,
            "oxidation_state_value": ox_val,
            "bader_charge": to_num(RE_BADER.search(win).group(1)) if RE_BADER.search(win) else None,
            "d_band_center": to_num(RE_DBAND.search(win).group(1)) if RE_DBAND.search(win) else None,
            "work_function": to_num(RE_WORK.search(win).group(1)) if RE_WORK.search(win) else None,
            "PZC": to_num(RE_PZC.search(win).group(1)) if RE_PZC.search(win) else None,
            "bridge_structure_nearby": bridge,
            "adsorbate": "H*",
            "H_adsorption_energy_value": to_num(m.group(1)),
            "H_adsorption_energy_unit": clean_unit(m.group(2)) or "eV",
            "H_adsorption_free_energy_value": None,
            "H_adsorption_free_energy_unit": None,
            "coverage": RE_COVERAGE.search(win).group(1) if RE_COVERAGE.search(win) else None,
            "DFT_functional": RE_FUNC.search(win).group(1) if RE_FUNC.search(win) else None,
            "U_value": to_num(RE_U.search(win).group(1)) if RE_U.search(win) else None,
            "adsorption_configuration": RE_SITE.search(win).group(1) if RE_SITE.search(win) else None,
            "reference_surface": None,
            "descriptor_claim": None,
            "system_confidence_score": None,
            "site_confidence_score": None,
            "adsorption_confidence_score": None,
            "site_system_match_score": 0.30,
            "adsorption_system_match_score": 0.30,
            "site_extraction_source": "focused_heuristic",
            "adsorption_extraction_source": "focused_heuristic",
            "descriptor_link_method": "same_evidence_window",
            "descriptor_link_confidence": 0.65,
            "descriptor_link_is_soft": 0,
            "link_score": 0.75,
            "target_relevance": 1.0,
            "system_signal": 0.6,
            "metric_signal": 1.0,
            "mechanism_signal": 0.6,
            "evidence_all": normalize_evidence_list(evs[:8]),
        }
        rows.append(postprocess_record(rec, evs, evidence_text(evs), paper_id, i))
    return [r for r in rows if r is not None]


def build_prompt(paper_id: str, evs: List[Dict[str, Any]]) -> str:
    return SYSTEM_PROMPT + "\n\n" + f"paper_id={paper_id}\n\nEVIDENCE:\n{pack_prompt_evidence(evs)}\n\nTASK: Extract only ΔE_H* / H adsorption energy records with same-system DFT descriptors. Return JSON only."


def postprocess_record(raw: Dict[str, Any], evs: List[Dict[str, Any]], ev_text_all: str, paper_id: str, idx: int) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    evidence_ids = raw.get("evidence_ids") or raw.get("evidence") or []
    rec_evs = evidence_for_ids(evs, evidence_ids if isinstance(evidence_ids, list) else [])
    ev_txt = evidence_text(rec_evs) or ev_text_all

    mat = clean_material_name(raw.get("material_name"), ev_txt) or clean_material_name(raw.get("composition"), ev_txt)
    if not mat:
        mat = infer_material_from_text(ev_txt) or "reported catalyst/surface"
    ox_raw = clean_scalar(raw.get("oxidation_state_raw") or raw.get("oxidation_state"))
    ox_el = clean_scalar(raw.get("oxidation_state_element") or raw.get("active_atom"))
    ox_raw2, ox_el2, ox_val2 = parse_oxidation_state(ox_raw, ox_el)
    ox_val = to_num(raw.get("oxidation_state_value"))
    if ox_val is None:
        ox_val = ox_val2
    ox_raw = ox_raw2 or ox_raw
    ox_el = ox_el2 or ox_el
    bridge = clean_bridge(raw.get("bridge_structure_nearby"), ev_txt) or clean_bridge(raw.get("bridge_structure"), ev_txt) or clean_bridge(raw.get("M_O_X_configuration"), ev_txt)

    rec: Dict[str, Any] = {
        "paper_id": paper_id,
        "canonical_id": clean_scalar(raw.get("canonical_id")) or f"{paper_id}_add_{idx:04d}",
        "reaction_family": "H_adsorption",
        "paper_type": "theoretical",
        "priority_hint": "deltaE_descriptor_add",
        "relpath": clean_scalar(raw.get("relpath")),
        "bucket": clean_scalar(raw.get("bucket")) or "pdf-files-add",
        "system_local_id": clean_scalar(raw.get("system_local_id")) or f"{paper_id}_sys_{idx:03d}",
        "site_local_id": clean_scalar(raw.get("site_local_id")) or f"{paper_id}_sys_{idx:03d}_site_001",
        "material_name": mat,
        "material_class": clean_scalar(raw.get("material_class")),
        "composition": clean_scalar(raw.get("composition")) or mat,
        "surface_facet": clean_scalar(raw.get("surface_facet")),
        "termination": clean_scalar(raw.get("termination")),
        "phase": clean_scalar(raw.get("phase")),
        "support": clean_scalar(raw.get("support")),
        "dopant": clean_scalar(raw.get("dopant")),
        "interface_type": clean_scalar(raw.get("interface_type")),
        "defect_type": clean_scalar(raw.get("defect_type")),
        "bridge_structure": bridge,
        "M_O_X_configuration": clean_bridge(raw.get("M_O_X_configuration"), ev_txt) or bridge,
        "model_type": clean_scalar(raw.get("model_type")) or clean_scalar(raw.get("calculation_model")),
        "ads_site_type": clean_scalar(raw.get("ads_site_type")),
        "site_label": clean_scalar(raw.get("site_label")),
        "active_atom": clean_scalar(raw.get("active_atom")) or ox_el,
        "neighbor_atoms": clean_listlike(raw.get("neighbor_atoms")),
        "coordination_number": to_num(raw.get("coordination_number")),
        "local_geometry": clean_scalar(raw.get("local_geometry")),
        "vacancy_nearby": bool_presence(raw.get("vacancy_nearby")),
        "hydroxylated_surface": bool_presence(raw.get("hydroxylated_surface")),
        "oxidation_state": ox_raw,
        "oxidation_state_raw": ox_raw,
        "oxidation_state_element": ox_el,
        "oxidation_state_value": ox_val,
        "bader_charge": to_num(raw.get("bader_charge")),
        "d_band_center": to_num(raw.get("d_band_center")),
        "charge_transfer_direction": clean_scalar(raw.get("charge_transfer_direction")),
        "bridge_structure_nearby": bridge,
        "ELF_descriptor": clean_scalar(raw.get("ELF_descriptor")),
        "work_function": to_num(raw.get("work_function")),
        "PZC": to_num(raw.get("PZC")),
        "hbond_acceptor_site": clean_scalar(raw.get("hbond_acceptor_site")),
        "H2O_binding_mode": None,
        "interfacial_water_role": None,
        "hydrogen_bond_network": None,
        "strong_HB_water_signal": None,
        "weak_HB_water_signal": None,
        "site_descriptor_claim": clean_scalar(raw.get("descriptor_claim")),
        "adsorbate": clean_scalar(raw.get("adsorbate")) or "H*",
        "H_adsorption_energy_value": to_num(raw.get("H_adsorption_energy_value")),
        "H_adsorption_energy_unit": clean_unit(raw.get("H_adsorption_energy_unit")) or None,
        "H_adsorption_free_energy_value": to_num(raw.get("H_adsorption_free_energy_value")),
        "H_adsorption_free_energy_unit": clean_unit(raw.get("H_adsorption_free_energy_unit")) or None,
        "coverage": clean_scalar(raw.get("coverage")),
        "coadsorbates": clean_listlike(raw.get("coadsorbates")),
        "solvation_model": clean_scalar(raw.get("solvation_model")),
        "spin_state": clean_scalar(raw.get("spin_state")),
        "DFT_functional": clean_scalar(raw.get("DFT_functional")),
        "U_value": to_num(raw.get("U_value")),
        "dispersion_correction": clean_scalar(raw.get("dispersion_correction")),
        "calculation_model": clean_scalar(raw.get("calculation_model")),
        "adsorption_configuration": clean_scalar(raw.get("adsorption_configuration")),
        "reference_surface": clean_scalar(raw.get("reference_surface")),
        "adsorption_strength_trend": clean_scalar(raw.get("adsorption_strength_trend")),
        "proton_transfer_pathway": None,
        "rate_determining_step": None,
        "adsorption_descriptor_claim": clean_scalar(raw.get("descriptor_claim")),
        "system_confidence_score": to_num(raw.get("confidence_score")),
        "site_confidence_score": to_num(raw.get("confidence_score")),
        "adsorption_confidence_score": to_num(raw.get("confidence_score")),
        "site_system_match_score": 0.35,
        "adsorption_system_match_score": 0.35,
        "site_extraction_source": "focused_llm" if raw.get("extraction_source") != "focused_heuristic" else "focused_heuristic",
        "adsorption_extraction_source": "focused_llm" if raw.get("extraction_source") != "focused_heuristic" else "focused_heuristic",
        "descriptor_link_method": "same_prompt_joint_extraction",
        "descriptor_link_confidence": 0.75,
        "descriptor_link_is_soft": 0,
        "descriptor_repair_method": None,
        "descriptor_repair_confidence": None,
        "descriptor_repair_source": None,
        "descriptor_repaired_fields": [],
        "link_score": 0.82,
        "target_relevance": 1.0,
        "system_signal": 0.6,
        "metric_signal": 1.0,
        "mechanism_signal": 0.6,
        "evidence_all": normalize_evidence_list(rec_evs),
    }

    # Validate the target first. Skip if no real ΔE_H* value is present in evidence.
    rec, numeric_support = validate_numeric_fields(rec, ["H_adsorption_energy_value"], ev_txt)
    if rec.get("H_adsorption_energy_value") is None:
        return None
    if rec.get("H_adsorption_energy_unit") is None:
        # Most H adsorption energy tables use eV; still mark unit as inferred only if value is supported.
        rec["H_adsorption_energy_unit"] = "eV"
    # Validate descriptors, but do not drop oxidation_state_value if the raw oxidation-state phrase is supported.
    rec, desc_support = validate_numeric_fields(rec, ["coordination_number", "bader_charge", "d_band_center", "work_function", "PZC", "U_value"], ev_txt)
    if rec.get("oxidation_state_value") is not None and not (number_supported(rec.get("oxidation_state_value"), ev_txt) or text_supported(rec.get("oxidation_state_raw"), ev_txt)):
        rec["oxidation_state_value"] = None
    desc_support["oxidation_state_value"] = "supported" if rec.get("oxidation_state_value") is not None else "missing"
    rec["site_qc_numeric_support"] = desc_support
    rec["adsorption_qc_numeric_support"] = numeric_support
    rec["site_qc_field_support"] = field_support_summary(rec, ev_txt, ["material_name", "site_label", "active_atom", "DFT_functional", "bridge_structure_nearby", "oxidation_state_raw"])
    rec["adsorption_qc_field_support"] = field_support_summary(rec, ev_txt, ["adsorbate", "coverage", "adsorption_configuration", "reference_surface"])
    for f, uf in [("H_adsorption_energy_value", "H_adsorption_energy_unit"), ("H_adsorption_free_energy_value", "H_adsorption_free_energy_unit")]:
        rec[f + "_eV"] = to_ev(rec.get(f), rec.get(uf))
    return rec


def process_paper(row: Dict[str, Any], args: argparse.Namespace, llm: Optional[MultiEndpointClient]) -> Tuple[str, str, int, Optional[str]]:
    paper_id = normalize_text(row.get("paper_id"))
    relpath = normalize_text(row.get("relpath"))
    chunks = load_chunks(Path(args.parsed_root), paper_id)
    if not chunks:
        return "skip", paper_id, 0, "no_chunks"
    # Cheap prefilter: do not call LLM if paper lacks ΔE_H-like cues.
    text_small = " ".join(normalize_text(c.get("text", ""))[:1200] for c in chunks[:120])
    if not RE_H_E.search(text_small) and not any(RE_H_E.search(normalize_text(c.get("text", ""))) for c in chunks):
        return "skip", paper_id, 0, "no_deltaE_cue"
    evs = select_evidence(chunks, args.topk_text, args.topk_table, args.topk_caption)
    evs = trim_evidence_to_budgets(evs, args.text_budget_chars, args.table_budget_chars, args.caption_budget_chars, args.max_quote_chars)
    if not evs:
        return "skip", paper_id, 0, "no_evidence"
    obj = None
    if llm is not None:
        try:
            prompt = build_prompt(paper_id, evs)
            obj = safe_json_extract(llm.completions(prompt, temperature=0.0, max_tokens=args.max_tokens))
        except Exception as e:
            if args.verbose:
                print(f"[WARN] LLM failed for {paper_id}: {e}", flush=True)
            obj = None
    rows: List[Dict[str, Any]] = []
    ev_text_all = evidence_text(evs)
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        for i, raw in enumerate(obj.get("records", []), 1):
            if isinstance(raw, dict):
                raw["relpath"] = relpath
                r = postprocess_record(raw, evs, ev_text_all, paper_id, i)
                if r is not None:
                    rows.append(r)
    if not rows and args.heuristic_fallback:
        rows = heuristic_records(paper_id, evs, relpath)
    # De-duplicate records within paper.
    out, seen = [], set()
    for r in rows:
        key = tuple(str(r.get(k) or "") for k in ["material_name", "surface_facet", "site_label", "active_atom", "H_adsorption_energy_value_eV", "DFT_functional", "coverage"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return "ok", paper_id, len(out), json.dumps(out, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parsed-root", required=True)
    ap.add_argument("--api-bases", default="")
    ap.add_argument("--model-id", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=3200)
    ap.add_argument("--topk-text", type=int, default=10)
    ap.add_argument("--topk-table", type=int, default=18)
    ap.add_argument("--topk-caption", type=int, default=6)
    ap.add_argument("--text-budget-chars", type=int, default=9000)
    ap.add_argument("--table-budget-chars", type=int, default=12000)
    ap.add_argument("--caption-budget-chars", type=int, default=4500)
    ap.add_argument("--max-quote-chars", type=int, default=1800)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", default="outputs_add_deltaeh/progress_3g_extract.json")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--heuristic-fallback", action="store_true", default=True)
    ap.add_argument("--no-heuristic-fallback", dest="heuristic_fallback", action="store_false")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
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
                    print(f"[OK] {paper_id} deltaE_records={n} progress={counter['done']}/{len(rows)} total_records={counter['records']}", flush=True)
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
