#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3d: extract H/OH/H2O adsorption and deprotonation/proton-transfer metrics."""
from __future__ import annotations

import argparse
import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.evidence_utils import dedup_evidence, normalize_evidence_list, pack_evidence
from utils.io_utils import append_jsonl, load_progress, read_jsonl, save_progress
from utils.llm_client import MultiEndpointClient, safe_json_extract
from utils.extraction_verifier import verify_extraction_json
from utils.normalize_utils import clean_unit, to_ev, to_float
from utils.text_utils import normalize_text
from utils.hads_validation import (
    packed_evidence_text, validate_numeric_fields, field_support_summary,
    text_supported, system_match_score, number_supported,
)

SEARCH_LOCK = threading.Lock()

QUERY_MAIN = "hydrogen adsorption energy free energy Delta G_H HBE Delta E_H OH adsorption OHBE water adsorption H2O deprotonation proton transfer Volmer barrier water dissociation DFT functional U coverage solvation"
QUERY_TABLE = "table adsorption energy Delta G_H Delta E_H HBE OHBE water adsorption Volmer barrier deprotonation energy functional coverage"
QUERY_ENTITY = "H adsorption OH adsorption H2O adsorption deprotonation proton transfer Volmer barrier adsorption free energy"

SYSTEM_PROMPT = """
You are extracting ADSORPTION / DEPROTONATION / PROTON-TRANSFER metric records.
Use ONLY the provided evidence. Output STRICT JSON only. If unsupported, return null or [].

Return JSON with keys: paper_id, records[].
Each record in records[] must contain:
system_local_id, site_local_id, adsorbate,
H_adsorption_energy_value, H_adsorption_energy_unit, H_adsorption_free_energy_value, H_adsorption_free_energy_unit,
OH_adsorption_energy_value, OH_adsorption_energy_unit, OH_adsorption_free_energy_value, OH_adsorption_free_energy_unit,
H2O_adsorption_energy_value, H2O_adsorption_energy_unit,
deprotonation_energy_value, deprotonation_energy_unit, proton_transfer_barrier, proton_transfer_barrier_unit, Volmer_barrier, Volmer_barrier_unit, water_dissociation_barrier, water_dissociation_barrier_unit,
coverage, coadsorbates, solvation_model, spin_state, DFT_functional, U_value, dispersion_correction, calculation_model, adsorption_configuration, reference_surface, adsorption_strength_trend, proton_transfer_pathway, rate_determining_step, descriptor_claim, confidence_score, evidence[].

Rules:
- Keep Delta E_H and Delta G_H separate.
- Extract OHBE/OH adsorption and H2O adsorption if present.
- Extract Volmer/deprotonation/proton transfer/water dissociation barriers if present.
- Numeric values must be copied exactly from the evidence blocks; do not infer values from trends, plots, or unrelated reference systems.
- Do not use potentials, temperatures, particle sizes, current densities, page numbers, figure numbers, or literature-reference values as adsorption energies/barriers.
- Preserve units exactly; eV, kJ mol-1, kcal mol-1 are all acceptable. If the unit is not stated near the number, use null.
- If site linkage is uncertain, site_local_id may be null.
- If the evidence contains multiple catalysts/surfaces, extract only metrics explicitly linked to TARGET_SYSTEM; otherwise return null or [].
""".strip()

SEP = r"(?:[^0-9+\-−–.;\n]{0,80})"
NUM_UNIT = r"([-+−–]?\d+(?:\.\d+)?)\s*(eV|meV|kJ\s*mol[-−–]?[1l]?|kcal\s*mol[-−–]?[1l]?|J\s*mol[-−–]?[1l]?)?"
RE_H_E = re.compile(r"(?:H adsorption energy|adsorption energy of H|E[_ ]?ads\(?H\)?|(?:Delta|Δ)\s*E[_\- ]?H\*?|DE[_\- ]?H\*?|binding energy of H|hydrogen binding energy|HBE)" + SEP + NUM_UNIT, re.I)
RE_H_G = re.compile(r"(?:free energy of H adsorption|adsorption free energy of hydrogen|(?:Delta|Δ)\s*G[_\- ]?H\*?|DG[_\- ]?H\*?|Gibbs free energy of H\* adsorption|ΔG[_\- ]?H\*?)" + SEP + NUM_UNIT, re.I)
RE_OH_E = re.compile(r"(?:OH adsorption energy|adsorption energy of OH|E[_ ]?ads\(?OH\)?|OHBE|hydroxyl binding energy)" + SEP + NUM_UNIT, re.I)
RE_OH_G = re.compile(r"(?:free energy of OH adsorption|OH adsorption free energy|Delta\s*G[_\- ]?OH\*?|DG[_\- ]?OH\*?)" + SEP + NUM_UNIT, re.I)
RE_H2O_E = re.compile(r"(?:H2O adsorption energy|water adsorption energy|adsorption energy of water|E[_ ]?ads\(?H2O\)?)" + SEP + NUM_UNIT, re.I)
RE_DEPROT = re.compile(r"(?:deprotonation energy|proton removal energy|protonation energy|pKa-related energy)" + SEP + NUM_UNIT, re.I)
RE_PROTON_BARRIER = re.compile(r"(?:proton transfer barrier|proton-transfer barrier|PCET barrier|deprotonation barrier|activation barrier for proton transfer)" + SEP + NUM_UNIT, re.I)
RE_VOLMER = re.compile(r"(?:Volmer(?: step)?(?: barrier| energy barrier| activation barrier)?|barrier for Volmer step)" + SEP + NUM_UNIT, re.I)
RE_WD = re.compile(r"(?:water dissociation barrier|H2O dissociation barrier|barrier for water dissociation)" + SEP + NUM_UNIT, re.I)
RE_FUNC = re.compile(r"\b(PBE0?|RPBE|B3LYP|HSE06|SCAN|M06[- ]?L|PW91|PBE\+U|GGA\+U|DFT\+U|revPBE)\b", re.I)
RE_U = re.compile(r"(?:U\s*=\s*|Hubbard\s*U\s*(?:value)?\s*[:=]?\s*)([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_DISP = re.compile(r"\b(D3|D2|vdW|Grimme|TS correction|DFT-D3|DFT-D2)\b", re.I)
RE_SOLV = re.compile(r"(implicit solvation|VASPsol|PCM|continuum solvation|solvation model|explicit water|water layer)", re.I)
RE_COV = re.compile(r"(?:coverage|H coverage)\s*[:=]?\s*([0-9./]+\s*(?:ML|monolayer|monolayers))", re.I)
RE_SPIN = re.compile(r"(ferromagnetic|antiferromagnetic|spin-polarized|triplet|singlet)", re.I)
RE_MODEL = re.compile(r"(slab model|cluster model|supercell|periodic DFT|DFT calculation|first-principles|periodic slab|computational model)", re.I)
RE_CFG = re.compile(r"(H\*|OH\*|H2O|adsorbed H|adsorbed OH|top site|bridge site|hollow site|O-top|metal-top|Volmer step|water dissociation derived H|molecular water|dissociated water)", re.I)
RE_REF = re.compile(r"(pristine surface|clean surface|defect-free surface|undoped surface|reference surface|bare surface|pure surface)", re.I)
RE_TREND = re.compile(r"(stronger H adsorption|weaker H adsorption|near[- ]thermoneutral|optimal H adsorption|too strong|too weak|weaken(?:ed)? hydrogen binding|strengthen(?:ed)? hydrogen binding|facilitated adsorption|stabilized H adsorption|promote(?:d)? H desorption|accelerate(?:d)? proton transfer)", re.I)
RE_COADS = re.compile(r"(OH\*|O\*|H2O|OH|OOH\*|CO|COOH\*|NHx|NOx|K\+|Li\+|Na\+)", re.I)
RE_SITE_LINK = re.compile(r"\b([A-Z][a-z]?(?:-top|-bridge|-hollow)?|O-top|metal-top|bridge site|hollow site|top site|interface site|vacancy site)\b", re.I)
RE_PATHWAY = re.compile(r"(H\*\s*\+\s*OH[-−–]?\s*->\s*H2O\s*\+\s*e[-−–]?|Volmer step|Heyrovsky step|Tafel step|proton transfer pathway|Grotthuss|hydrogen-bond network)", re.I)
RE_RDS = re.compile(r"(?:rate[- ]determining step|RDS)\s*(?:is|was|:)?\s*([A-Za-z0-9\-+* >_/]+)", re.I)

PATTERN_FIELDS = [
    (RE_H_E, 'H_adsorption_energy_value', 'H_adsorption_energy_unit'),
    (RE_H_G, 'H_adsorption_free_energy_value', 'H_adsorption_free_energy_unit'),
    (RE_OH_E, 'OH_adsorption_energy_value', 'OH_adsorption_energy_unit'),
    (RE_OH_G, 'OH_adsorption_free_energy_value', 'OH_adsorption_free_energy_unit'),
    (RE_H2O_E, 'H2O_adsorption_energy_value', 'H2O_adsorption_energy_unit'),
    (RE_DEPROT, 'deprotonation_energy_value', 'deprotonation_energy_unit'),
    (RE_PROTON_BARRIER, 'proton_transfer_barrier', 'proton_transfer_barrier_unit'),
    (RE_VOLMER, 'Volmer_barrier', 'Volmer_barrier_unit'),
    (RE_WD, 'water_dissociation_barrier', 'water_dissociation_barrier_unit'),
]


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    if not s or s.lower() in {'none','null','nan','unknown','not mentioned','n/a','na'}:
        return None
    return s


def clean_listlike(x: Any) -> List[str]:
    if x is None:
        return []
    arr = x if isinstance(x, list) else re.split(r'[;,|/]+', str(x))
    out, seen = [], set()
    for a in arr:
        t = clean_scalar(a)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def unique_keep(xs: List[Any]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        t = clean_scalar(x)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def strict_confidence_score(x: Any) -> Tuple[Optional[float], Optional[str], str]:
    if x is None:
        return None, None, 'missing'
    raw = str(x).strip()
    if raw == '':
        return None, None, 'missing'
    v = to_float(raw)
    if v is None:
        return None, raw, 'non_numeric'
    if 0 <= v <= 1:
        return v, raw, 'ok'
    return None, raw, 'out_of_range'


def load_triage_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        pid = str(r.get('paper_id', '')).strip()
        if pid and int(pd.to_numeric(r.get('should_extract'), errors='coerce') or 0) == 1:
            out[pid] = r
    return out


def load_grouped(path: Path, key_cols: Tuple[str, ...]) -> Dict[Tuple[str, ...], List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in read_jsonl(path):
        out[tuple(str(r.get(c, '')).strip() for c in key_cols)].append(r)
    return out


def load_chunks(parsed_root: Path, paper_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(parsed_root / 'papers' / paper_id / 'chunks.jsonl')


def evidence_from_chunk(ch: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    return {'paper_id': ch.get('paper_id'), 'chunk_id': ch.get('chunk_id'), 'chunk_type': ch.get('chunk_type'), 'section_path': ch.get('section_path'), 'page_start': ch.get('page_start'), 'page_end': ch.get('page_end'), 'source': ch.get('source'), 'text': normalize_text(ch.get('text', '')), 'score': score}


def local_keyword_search(parsed_root: Path, paper_id: str, keywords: List[str], allow_types: List[str], topk: int, sys_terms: str = '') -> List[Dict[str, Any]]:
    rows = []
    sys_tokens = [t for t in re.split(r'\W+', sys_terms.lower()) if len(t) >= 2]
    for ch in load_chunks(parsed_root, paper_id):
        ctype = normalize_text(ch.get('chunk_type') or 'text')
        if allow_types and ctype not in allow_types:
            continue
        txt = normalize_text(ch.get('text','')); low = txt.lower()
        score = sum(1 for kw in keywords if kw.lower() in low)
        score += sum(1 for t in sys_tokens if t in low[:4000])
        if any(p.search(txt) for p,_,_ in PATTERN_FIELDS) or RE_TREND.search(txt) or RE_PATHWAY.search(txt):
            score += 4
        if ctype == 'table':
            score += 3
        if score > 0:
            rows.append(evidence_from_chunk(ch, float(score)))
    rows.sort(key=lambda r: float(r.get('score') or 0), reverse=True)
    return rows[:topk]


def is_local_qdrant_prefix(prefix: str, shard_ids: List[int]) -> bool:
    return any(Path(f'{prefix}{sid}').exists() for sid in shard_ids)


def qdrant_search_or_fallback(parsed_root: Path, qdrant_prefix: str, collection_prefix: str, shard_ids: List[int], qvec: Optional[List[float]], paper_id: str, allow_types: List[str], topk: int, query_text: str, use_lock: bool, sys_terms: str) -> List[Dict[str, Any]]:
    """High-recall search: combine semantic Qdrant hits with exact keyword/system hits."""
    rows: List[Dict[str, Any]] = []
    if qvec is not None:
        try:
            from utils.qdrant_utils import search_multishard
            if use_lock:
                with SEARCH_LOCK:
                    rows.extend(search_multishard(qdrant_prefix, collection_prefix, shard_ids, qvec, paper_id, allow_types, topk=topk))
            else:
                rows.extend(search_multishard(qdrant_prefix, collection_prefix, shard_ids, qvec, paper_id, allow_types, topk=topk))
        except Exception:
            pass
    kws = [w for w in re.split(r'\s+', query_text) if len(w) >= 3][:45]
    try:
        rows.extend(local_keyword_search(parsed_root, paper_id, kws, allow_types, topk=max(topk, 14), sys_terms=sys_terms))
    except Exception:
        pass
    rows = dedup_evidence(rows)
    rows.sort(key=lambda r: float(r.get('score') or 0), reverse=True)
    return rows[:max(topk, min(len(rows), topk + 10))]


def score_hit(ev: Dict[str, Any], sys_terms: str) -> int:
    txt = normalize_text(ev.get('text') or ev.get('quote')).lower(); s = 0
    for kw in ['hydrogen adsorption', 'h adsorption', 'delta g', 'delta e', 'hbe', 'ohbe', 'oh adsorption', 'water adsorption', 'deprotonation', 'proton transfer', 'volmer', 'barrier', 'h*', 'oh*', 'dft', 'functional', 'coverage', 'solvation']:
        if kw in txt: s += 1
    for token in re.split(r'\W+', sys_terms.lower()):
        if len(token) >= 2 and token in txt:
            s += 1
    if ev.get('chunk_type') == 'table': s += 4
    return s


def trim_evidence(evs_main, evs_table, evs_entity, args, sys_terms: str):
    evs_main = sorted(dedup_evidence(evs_main), key=lambda x: score_hit(x, sys_terms), reverse=True)[:max(8, args.max_items_main + 3)]
    evs_table = sorted(dedup_evidence(evs_table), key=lambda x: score_hit(x, sys_terms), reverse=True)[:max(8, args.max_items_table + 3)]
    evs_entity = sorted(dedup_evidence(evs_entity), key=lambda x: score_hit(x, sys_terms), reverse=True)[:min(max(4, args.max_items_entity), 8)]
    return evs_main, evs_table, evs_entity


def site_id_by_hint(site_rows: List[Dict[str, Any]], hint_text: str) -> Optional[str]:
    hint = hint_text.lower()
    for s in site_rows:
        label = str(s.get('site_label') or '').lower(); stype = str(s.get('ads_site_type') or '').lower(); atom = str(s.get('active_atom') or '').lower()
        if label and label in hint:
            return clean_scalar(s.get('site_local_id'))
        if stype and stype in hint and (not atom or atom in hint):
            return clean_scalar(s.get('site_local_id'))
    return None


def all_metric_matches(text: str) -> List[Tuple[int, int, re.Pattern, str, str]]:
    matches = []
    for pat, vfield, ufield in PATTERN_FIELDS:
        for m in pat.finditer(text):
            matches.append((m.start(), m.end(), pat, vfield, ufield))
    matches.sort(key=lambda x: x[0])
    return matches


def fill_match_fields(rec: Dict[str, Any], window: str) -> None:
    for pat, vfield, ufield in PATTERN_FIELDS:
        m = pat.search(window)
        if m:
            rec[vfield] = to_float(m.group(1))
            rec[ufield] = clean_unit(m.group(2)) or 'eV'


def infer_adsorbate(rec: Dict[str, Any], window: str) -> str:
    if rec.get('OH_adsorption_energy_value') is not None or rec.get('OH_adsorption_free_energy_value') is not None:
        return 'OH*'
    if rec.get('H2O_adsorption_energy_value') is not None:
        return 'H2O'
    if rec.get('deprotonation_energy_value') is not None or rec.get('proton_transfer_barrier') is not None or rec.get('Volmer_barrier') is not None or rec.get('water_dissociation_barrier') is not None:
        return 'proton/deprotonation step'
    if re.search(r'OH\*|OH adsorption|OHBE', window, re.I):
        return 'OH*'
    if re.search(r'H2O|water adsorption', window, re.I):
        return 'H2O'
    return 'H*'


def build_base_record(paper_id: str, system_obj: Dict[str, Any], site_rows: List[Dict[str, Any]], window: str, idx: int, evs_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    site_hint = ' '.join(m.group(0) for m in RE_SITE_LINK.finditer(window))
    rec: Dict[str, Any] = {
        'paper_id': paper_id,
        'system_local_id': system_obj.get('system_local_id'),
        'site_local_id': site_id_by_hint(site_rows, site_hint),
        'adsorbate': None,
        'H_adsorption_energy_value': None, 'H_adsorption_energy_unit': None, 'H_adsorption_free_energy_value': None, 'H_adsorption_free_energy_unit': None,
        'OH_adsorption_energy_value': None, 'OH_adsorption_energy_unit': None, 'OH_adsorption_free_energy_value': None, 'OH_adsorption_free_energy_unit': None,
        'H2O_adsorption_energy_value': None, 'H2O_adsorption_energy_unit': None,
        'deprotonation_energy_value': None, 'deprotonation_energy_unit': None,
        'proton_transfer_barrier': None, 'proton_transfer_barrier_unit': None,
        'Volmer_barrier': None, 'Volmer_barrier_unit': None,
        'water_dissociation_barrier': None, 'water_dissociation_barrier_unit': None,
    }
    fill_match_fields(rec, window)
    rec.update({
        'adsorbate': infer_adsorbate(rec, window),
        'coverage': clean_scalar(RE_COV.search(window).group(1)) if RE_COV.search(window) else None,
        'coadsorbates': unique_keep(RE_COADS.findall(window)),
        'solvation_model': clean_scalar(RE_SOLV.search(window).group(1)) if RE_SOLV.search(window) else None,
        'spin_state': clean_scalar(RE_SPIN.search(window).group(1)) if RE_SPIN.search(window) else None,
        'DFT_functional': clean_scalar(RE_FUNC.search(window).group(1)) if RE_FUNC.search(window) else None,
        'U_value': to_float(RE_U.search(window).group(1)) if RE_U.search(window) else None,
        'dispersion_correction': clean_scalar(RE_DISP.search(window).group(1)) if RE_DISP.search(window) else None,
        'calculation_model': clean_scalar(RE_MODEL.search(window).group(1)) if RE_MODEL.search(window) else None,
        'adsorption_configuration': clean_scalar(RE_CFG.search(window).group(1)) if RE_CFG.search(window) else None,
        'reference_surface': clean_scalar(RE_REF.search(window).group(1)) if RE_REF.search(window) else None,
        'adsorption_strength_trend': clean_scalar(RE_TREND.search(window).group(1)) if RE_TREND.search(window) else None,
        'proton_transfer_pathway': clean_scalar(RE_PATHWAY.search(window).group(1)) if RE_PATHWAY.search(window) else None,
        'rate_determining_step': clean_scalar(RE_RDS.search(window).group(1)) if RE_RDS.search(window) else None,
        'descriptor_claim': clean_scalar(RE_TREND.search(window).group(1)) if RE_TREND.search(window) else clean_scalar(RE_PATHWAY.search(window).group(1)) if RE_PATHWAY.search(window) else None,
        'confidence_score': None,
        'system_match_score': system_match_score(system_obj, window),
        'qc_numeric_support': {},
        'qc_field_support': {},
        'extraction_source': 'heuristic',
        'evidence': normalize_evidence_list([{'chunk_id': e.get('chunk_id'), 'quote': str(e.get('text') or e.get('quote') or '')[:650], 'source': e.get('source'), 'chunk_type': e.get('chunk_type'), 'page_start': e.get('page_start'), 'page_end': e.get('page_end'), 'section_path': e.get('section_path')} for e in evs_all[:8]]),
    })
    metric_fields = [v for _, v, _ in PATTERN_FIELDS]
    rec, numeric_support = validate_numeric_fields(rec, metric_fields, window)
    rec['qc_numeric_support'] = numeric_support
    rec['qc_field_support'] = field_support_summary(rec, window, ['adsorbate','DFT_functional','coverage','solvation_model','adsorption_configuration','reference_surface','proton_transfer_pathway','rate_determining_step'])
    for _, vfield, ufield in PATTERN_FIELDS:
        rec[vfield + '_eV'] = to_ev(rec.get(vfield), rec.get(ufield))
    return rec


def heuristic_ads_records(paper_id: str, system_obj: Dict[str, Any], site_rows: List[Dict[str, Any]], evs_main, evs_table, evs_entity) -> List[Dict[str, Any]]:
    evs_all = evs_table + evs_main + evs_entity
    text = normalize_text(' '.join(str(e.get('text') or e.get('quote') or '') for e in evs_all))
    anchors = [(st, en) for st, en, _, _, _ in all_metric_matches(text)]
    if not anchors and (RE_TREND.search(text) or RE_PATHWAY.search(text)):
        anchors = [(m.start(), m.end()) for m in list(RE_TREND.finditer(text)) + list(RE_PATHWAY.finditer(text))]
    rows = []
    for idx, (st, en) in enumerate(anchors[:18], 1):
        window = text[max(0, st-520):min(len(text), en+760)]
        rec = build_base_record(paper_id, system_obj, site_rows, window, idx, evs_all)
        rows.append(rec)
    out, seen = [], set()
    metric_fields = [v for _, v, _ in PATTERN_FIELDS]
    for r in rows:
        key = tuple(r.get(k) for k in ['system_local_id','site_local_id','adsorbate'] + metric_fields + ['DFT_functional','coverage','reference_surface','adsorption_configuration'])
        if key in seen:
            continue
        seen.add(key)
        if any(r.get(k) is not None for k in metric_fields) or r.get('adsorption_strength_trend') or r.get('proton_transfer_pathway'):
            out.append(r)
    return out


def postprocess_records(obj: Dict[str, Any], paper_id: str, system_obj: Dict[str, Any], evidence_txt: str = "") -> List[Dict[str, Any]]:
    recs = obj.get('records', []) if isinstance(obj, dict) else []
    if not isinstance(recs, list): recs = []
    out, seen = [], set()
    for r in recs:
        if not isinstance(r, dict): continue
        conf_num, conf_raw, conf_status = strict_confidence_score(r.get('confidence_score'))
        rec: Dict[str, Any] = {
            'paper_id': paper_id,
            'system_local_id': clean_scalar(r.get('system_local_id')) or clean_scalar(system_obj.get('system_local_id')),
            'site_local_id': clean_scalar(r.get('site_local_id')),
            'adsorbate': clean_scalar(r.get('adsorbate')) or 'H*',
            'coverage': clean_scalar(r.get('coverage')),
            'coadsorbates': unique_keep(clean_listlike(r.get('coadsorbates'))),
            'solvation_model': clean_scalar(r.get('solvation_model')),
            'spin_state': clean_scalar(r.get('spin_state')),
            'DFT_functional': clean_scalar(r.get('DFT_functional')),
            'U_value': to_float(r.get('U_value')),
            'dispersion_correction': clean_scalar(r.get('dispersion_correction')),
            'calculation_model': clean_scalar(r.get('calculation_model')),
            'adsorption_configuration': clean_scalar(r.get('adsorption_configuration')),
            'reference_surface': clean_scalar(r.get('reference_surface')),
            'adsorption_strength_trend': clean_scalar(r.get('adsorption_strength_trend')),
            'proton_transfer_pathway': clean_scalar(r.get('proton_transfer_pathway')),
            'rate_determining_step': clean_scalar(r.get('rate_determining_step')),
            'descriptor_claim': clean_scalar(r.get('descriptor_claim')),
            'confidence_score': conf_num,
            'confidence_score_raw': conf_raw,
            'confidence_score_status': conf_status,
            'evidence': normalize_evidence_list(r.get('evidence', [])),
        }
        for _, vfield, ufield in PATTERN_FIELDS:
            rec[vfield] = to_float(r.get(vfield))
            rec[ufield] = clean_unit(r.get(ufield))
        # Drop hallucinated or cross-system numeric values not present in the evidence prompt.
        metric_fields = [v for _, v, _ in PATTERN_FIELDS]
        rec, numeric_support = validate_numeric_fields(rec, metric_fields, evidence_txt)
        rec['qc_numeric_support'] = numeric_support
        rec['system_match_score'] = system_match_score(system_obj, evidence_txt)
        rec['qc_field_support'] = field_support_summary(rec, evidence_txt, ['adsorbate','DFT_functional','coverage','solvation_model','adsorption_configuration','reference_surface','proton_transfer_pathway','rate_determining_step'])
        # Drop unsupported textual mechanism claims.
        for tf in ['adsorption_strength_trend','proton_transfer_pathway','rate_determining_step','descriptor_claim','adsorption_configuration','reference_surface']:
            if rec.get(tf) and not text_supported(rec.get(tf), evidence_txt):
                rec[tf] = None
        # Add normalized eV columns here too; Step4 also recomputes these.
        for _, vfield, ufield in PATTERN_FIELDS:
            rec[vfield + '_eV'] = to_ev(rec.get(vfield), rec.get(ufield))
        rec['extraction_source'] = 'llm'
        key = tuple(rec.get(k) for k in ['system_local_id','site_local_id','adsorbate'] + [v for _, v, _ in PATTERN_FIELDS] + ['DFT_functional','coverage','reference_surface','adsorption_configuration'])
        if key in seen: continue
        seen.add(key); out.append(rec)
    return out


def build_prompt(paper_id: str, system_obj: Dict[str, Any], site_rows: List[Dict[str, Any]], evs_main, evs_table, evs_entity, args) -> str:
    payload = {'system_local_id': system_obj.get('system_local_id'), 'material_name': system_obj.get('material_name'), 'surface_facet': system_obj.get('surface_facet'), 'defect_type': system_obj.get('defect_type'), 'dopant': system_obj.get('dopant'), 'bridge_structure': system_obj.get('bridge_structure'), 'candidate_sites': [{k: s.get(k) for k in ['site_local_id','site_label','ads_site_type','active_atom','d_band_center','bader_charge']} for s in site_rows[:8]]}
    return SYSTEM_PROMPT + '\n\n' + (
        f'paper_id={paper_id}\nTARGET_SYSTEM={json.dumps(payload, ensure_ascii=False)}\n\n'
        f'EVIDENCE_TABLE:\n{pack_evidence(evs_table, max_items=args.max_items_table, total_chars=args.table_budget_chars, max_quote_chars=args.max_table_chars)}\n\n'
        f'EVIDENCE_ENTITY:\n{pack_evidence(evs_entity, max_items=args.max_items_entity, total_chars=args.entity_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        f'EVIDENCE_MAIN:\n{pack_evidence(evs_main, max_items=args.max_items_main, total_chars=args.main_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        'TASK: Extract adsorption and deprotonation/proton-transfer records for this system only.\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument('--triage', required=True); ap.add_argument('--systems', required=True); ap.add_argument('--sites', required=True); ap.add_argument('--parsed-root', required=True)
    ap.add_argument('--api-bases', default=''); ap.add_argument('--model-id', default=''); ap.add_argument('--embed-model', required=True); ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--qdrant-prefix', required=True); ap.add_argument('--collection-prefix', required=True); ap.add_argument('--shard-ids', default='0,1,2')
    ap.add_argument('--topk-main', type=int, default=14); ap.add_argument('--topk-table', type=int, default=24); ap.add_argument('--topk-entity', type=int, default=6)
    ap.add_argument('--main-budget-chars', type=int, default=9500); ap.add_argument('--table-budget-chars', type=int, default=7500); ap.add_argument('--entity-budget-chars', type=int, default=1800)
    ap.add_argument('--max-items-main', type=int, default=10); ap.add_argument('--max-items-table', type=int, default=9); ap.add_argument('--max-items-entity', type=int, default=4)
    ap.add_argument('--max-text-chars', type=int, default=850); ap.add_argument('--max-table-chars', type=int, default=1500); ap.add_argument('--max-caption-chars', type=int, default=1200)
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--timeout', type=int, default=180); ap.add_argument('--max-retries', type=int, default=3); ap.add_argument('--verify-llm', action='store_true', help='Run a second LLM pass to verify extracted JSON against evidence before deterministic QC.')
    ap.add_argument('--verifier-api-bases', default='', help='Optional API bases for verifier; defaults to --api-bases.')
    ap.add_argument('--verifier-model-id', default='', help='Optional verifier model; defaults to --model-id.')
    ap.add_argument('--verify-max-tokens', type=int, default=2600, help='Max tokens for second-pass verification.')
    ap.add_argument('--max-tokens', type=int, default=2200)
    ap.add_argument('--out', required=True); ap.add_argument('--progress', default='outputs/progress_hads_step3d_adsorption.json'); ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    triage_map = load_triage_rows(Path(args.triage))
    systems_by_key = load_grouped(Path(args.systems), ('paper_id',))
    sites_by_key = load_grouped(Path(args.sites), ('paper_id','system_local_id'))
    work_items = []
    for pid in triage_map:
        for s in systems_by_key.get((pid,), []):
            work_items.append((pid, s, sites_by_key.get((pid, str(s.get('system_local_id','')).strip()), [])))
    total = len(work_items); print(f'[INFO] HADS Step3d work_items={total}', flush=True)
    shard_ids = [int(x) for x in args.shard_ids.split(',') if x.strip()]
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress)
    done = set(load_progress(progress_path).get('done', [])) if progress_path.exists() and not args.force else set()
    if args.force and outp.exists(): outp.unlink()

    qvecs: Dict[str, Optional[List[float]]] = {'main': None, 'table': None, 'entity': None}
    try:
        from sentence_transformers import SentenceTransformer
        embed = SentenceTransformer(args.embed_model, device=args.device)
        qvecs['main'] = embed.encode([QUERY_MAIN], normalize_embeddings=True, batch_size=1)[0].tolist()
        qvecs['table'] = embed.encode([QUERY_TABLE], normalize_embeddings=True, batch_size=1)[0].tolist()
        qvecs['entity'] = embed.encode([QUERY_ENTITY], normalize_embeddings=True, batch_size=1)[0].tolist()
    except Exception as e:
        print(f'[WARN] embedding/qdrant disabled; using local keyword fallback. reason={e}', flush=True)

    llm = MultiEndpointClient([x.strip() for x in args.api_bases.split(',') if x.strip()], args.model_id, timeout=args.timeout, max_retries=args.max_retries) if args.api_bases.strip() and args.model_id.strip() else None
    verifier_bases = args.verifier_api_bases.strip() or args.api_bases.strip()
    verifier_model = args.verifier_model_id.strip() or args.model_id.strip()
    verifier_llm = MultiEndpointClient([x.strip() for x in verifier_bases.split(',') if x.strip()], verifier_model, timeout=args.timeout, max_retries=args.max_retries) if args.verify_llm and verifier_bases and verifier_model else None
    use_qdrant_lock = is_local_qdrant_prefix(args.qdrant_prefix, shard_ids)
    progress_lock = threading.Lock(); counter = {'done':0,'ok':0,'skip':0,'err':0}

    def process_item(item):
        paper_id, system_obj, site_rows = item; work_key = f"{paper_id}::{system_obj.get('system_local_id')}"
        if work_key in done: return 'skip', work_key, 0, None
        try:
            sys_terms = ' '.join([str(system_obj.get(k,'') or '') for k in ['material_name','composition','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration']] + [str(s.get('site_label','') or '') for s in site_rows[:6]])
            evs_main = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_main_text_shard', shard_ids, qvecs['main'], paper_id, ['text'], args.topk_main, QUERY_MAIN, use_qdrant_lock, sys_terms)
            evs_table = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_table_caption_shard', shard_ids, qvecs['table'], paper_id, ['caption','table'], args.topk_table, QUERY_TABLE, use_qdrant_lock, sys_terms)
            evs_entity = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_entity_shard', shard_ids, qvecs['entity'], paper_id, ['text','caption','table'], args.topk_entity, QUERY_ENTITY, use_qdrant_lock, sys_terms)
            evs_main, evs_table, evs_entity = trim_evidence(evs_main, evs_table, evs_entity, args, sys_terms)
            evidence_txt = packed_evidence_text(evs_main, evs_table, evs_entity)
            obj = None
            if llm is not None and (evs_main or evs_table or evs_entity):
                try: obj = safe_json_extract(llm.completions(build_prompt(paper_id, system_obj, site_rows, evs_main, evs_table, evs_entity, args), temperature=0.0, max_tokens=args.max_tokens))
                except Exception as e: print(f'[WARN] LLM failed for {work_key}: {e}', flush=True)
            if obj is not None and verifier_llm is not None:
                vobj = verify_extraction_json(verifier_llm, paper_id, 'adsorption_deprotonation_metrics', obj, evidence_txt, target_context={'system': {k: system_obj.get(k) for k in ['system_local_id','material_name','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration']}, 'candidate_sites': [{k: s.get(k) for k in ['site_local_id','site_label','ads_site_type','active_atom']} for s in site_rows[:8]]}, max_tokens=args.verify_max_tokens)
                if vobj is not None:
                    obj = vobj
            rows = postprocess_records(obj, paper_id, system_obj, evidence_txt) if obj is not None else []
            if not rows:
                rows = heuristic_ads_records(paper_id, system_obj, site_rows, evs_main, evs_table, evs_entity)
            for r in rows: append_jsonl(r, outp)
            return 'ok', work_key, len(rows), None
        except Exception as e:
            return 'err', work_key, 0, str(e)

    prog = {'done': sorted(done)}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_item, item): item for item in work_items}
        for fut in as_completed(futs):
            status, work_key, n_rows, err = fut.result()
            with progress_lock:
                counter['done'] += 1
                if status == 'ok':
                    counter['ok'] += 1; done.add(work_key); prog['done'] = sorted(done); save_progress(prog, progress_path)
                    print(f'[OK] {work_key} metric_records={n_rows} progress={counter["done"]}/{total}', flush=True)
                elif status == 'skip':
                    counter['skip'] += 1; print(f'[SKIP] {work_key} progress={counter["done"]}/{total}', flush=True)
                else:
                    counter['err'] += 1; print(f'[ERR] {work_key} error={err} progress={counter["done"]}/{total}', flush=True)
    print(f'[DONE] wrote {args.out}', flush=True)

if __name__ == '__main__':
    main()
