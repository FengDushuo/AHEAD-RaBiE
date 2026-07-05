#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3c: extract adsorption-site and local-mechanism descriptors."""
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
from utils.text_utils import normalize_text
from utils.hads_validation import (
    packed_evidence_text, validate_numeric_fields, field_support_summary,
    clean_bridge, text_supported, system_match_score, parse_oxidation_state, normalize_yes_no,
)

SEARCH_LOCK = threading.Lock()

QUERY_MAIN = "adsorption site active atom top bridge hollow interface vacancy coordination Bader charge d-band center ELF work function PZC hydrogen bond interfacial water bridge oxygen"
QUERY_TABLE = "table active site coordination Bader d-band ELF work function PZC hydrogen bond water site descriptor"
QUERY_ENTITY = "adsorption site active atom vacancy bridge oxygen d-band center Bader charge ELF interfacial water"

SYSTEM_PROMPT = """
You are extracting ADSORPTION SITE and LOCAL DESCRIPTOR records for H adsorption / deprotonation literature mining.
Use ONLY the provided evidence. Output STRICT JSON only.
If unsupported, return null or [].

Return JSON with keys: paper_id, records[].
Each record in records[] must contain:
system_local_id, site_local_id, ads_site_type, site_label, active_atom, neighbor_atoms, coordination_number, local_geometry, vacancy_nearby, hydroxylated_surface, oxidation_state, oxidation_state_raw, oxidation_state_element, oxidation_state_value, bader_charge, d_band_center, charge_transfer_direction, bridge_structure_nearby, M_O_X_configuration, ELF_descriptor, work_function, PZC, hbond_acceptor_site, H2O_binding_mode, interfacial_water_role, hydrogen_bond_network, strong_HB_water_signal, weak_HB_water_signal, descriptor_claim, confidence_score, evidence[].

Rules:
- Extract one record per distinct active/adsorption site if supported.
- Keep top / bridge / hollow / O-top / metal-top / vacancy-nearby / interface labels explicit.
- Do not invent numeric descriptors; null is better than guessing.
- oxidation_state refers to the active-site metal oxidation state. Preserve the original text in oxidation_state_raw, and use oxidation_state_value only when the valence is explicitly stated (e.g., Fe(III)=3, Ni2+=2).
- Numeric fields such as coordination_number, bader_charge, d_band_center, work_function and PZC must appear in the provided evidence.
- Do not copy numeric values from unrelated systems, reference catalysts, experimental potentials, page numbers, temperatures or figure labels.
- Capture interfacial-water and hydrogen-bond-network descriptors only when the motif/site is explicitly linked to proton transfer or Volmer chemistry.
- If a field is uncertain or only implied, return null rather than a guess.
""".strip()

RE_SITE_LABEL = re.compile(r"\b([A-Z][a-z]?(?:[- ]?(?:top|bridge|hollow))|O[- ]top|metal[- ]top|bridge site|hollow site|top site|interfacial site|interface site|vacancy[- ]nearby site)\b", re.I)
RE_SITE_TOKEN = re.compile(r"\b(?:(?P<atom>[A-Z][a-z]?)\s*[- ]?(?P<stype>top|bridge|hollow)|(?P<stype2>bridge|hollow|top site|atop|metal[- ]top|oxygen[- ]top|o[- ]top|interfacial site|interface site|subsurface|vacancy[- ]nearby))\b", re.I)
RE_CN = re.compile(r"(?:coordination number|coordination|\bCN\b)\s*[:=]?\s*([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_BADER = re.compile(r"(?:Bader charge|charge on [A-Z][a-z]?|Delta\s*q|\bΔq\b)\s*[:=]?\s*([-+−–]?\d+(?:\.\d+)?)", re.I)
RE_DBAND = re.compile(r"(?:d[- ]?band center|epsilon[_\- ]?d|ε_d)\s*(?:of\s*[A-Z][a-z]?)?\s*[:=]?\s*([-+−–]?\d+(?:\.\d+)?)\s*(?:eV)?", re.I)
RE_WORK = re.compile(r"(?:work function|Φ|Phi)\s*[:=]?\s*([-+−–]?\d+(?:\.\d+)?)\s*(?:eV)?", re.I)
RE_PZC = re.compile(r"(?:PZC|point of zero charge)\s*[:=]?\s*([-+−–]?\d+(?:\.\d+)?)\s*(?:V)?", re.I)
RE_OX = re.compile(r"(?:oxidation state|valence state|valence)\s*[:=]?\s*([A-Za-z0-9+\-()]+)", re.I)
RE_CHARGE_DIR = re.compile(r"(electron[- ]rich|electron[- ]deficient|charge transfer from .*? to .*?|electron donation|electron withdrawal|charge redistribution|electron accumulation|electron depletion|downshift(?:ed)? d[- ]?band|upshift(?:ed)? d[- ]?band)", re.I)
RE_GEOM = re.compile(r"(tetrahedral|octahedral|square planar|undercoordinated|distorted|trigonal|square pyramidal|low[- ]coordination)", re.I)
RE_VAC = re.compile(r"(oxygen vacancy|sulfur vacancy|nitrogen vacancy|vacancy|near vacancy|defect site)", re.I)
RE_HYDROX = re.compile(r"(hydroxylated|surface hydroxyl|OH[- ]terminated|hydroxyl[- ]covered)", re.I)
RE_BRIDGE = re.compile(r"(bridging oxygen|bridge oxygen|(?:Ni|Fe|Co|Cu|Zn|Pt|Pd|Ru|Rh|Ir|Ce|Ti|Mo|W|Mn|Cr|Zr|La|Ga|In|Sn)[-–—]?O[-–—]?[A-Z][a-z]?|M[-–—]?O[-–—]?[A-Z][a-z]?|Ni[-–—]?O[-–—]?[CP]|M[-–—]?O[-–—]?C/P|P[-–—]?O[-–—]?Ni)", re.I)
RE_ELF = re.compile(r"(ELF|electron localization function|localized electron|electron localization|unpaired valence electron)", re.I)
RE_HBOND = re.compile(r"(hydrogen[- ]bond network|H[- ]bond network|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water|symmetric HB water|asymmetric HB water|interfacial water|water network|hydrogen bond)", re.I)
RE_H2O_MODE = re.compile(r"(molecular water|dissociated water|H2O adsorption|water adsorption|OH[- ]covered|water layer|explicit water)", re.I)
RE_NEIGH = re.compile(r"(?:neighbor(?:ing)?|adjacent to|bonded to|coordinated to)\s+([A-Z][a-z]?(?:\s*(?:and|,|/)\s*[A-Z][a-z]?)*)", re.I)


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    if not s or s.lower() in {'none', 'null', 'nan', 'unknown', 'not mentioned', 'n/a', 'na'}:
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


def to_num(x: Any) -> Optional[float]:
    try:
        s = str(x).replace('−', '-').replace('–', '-').replace(',', '').strip()
        v = pd.to_numeric(s, errors='coerce')
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def strict_confidence_score(x: Any) -> Tuple[Optional[float], Optional[str], str]:
    if x is None:
        return None, None, 'missing'
    raw = str(x).strip()
    if not raw:
        return None, None, 'missing'
    v = to_num(x)
    if v is None:
        return None, raw, 'non_numeric'
    if 0 <= v <= 1:
        return v, raw, 'ok'
    return None, raw, 'out_of_range'


def normalize_site_type(x: Any) -> Optional[str]:
    s = normalize_text(x).lower()
    if not s:
        return None
    mapping = [('bridge', ['bridge']), ('hollow', ['hollow']), ('metal-top', ['metal-top', 'metal top', 'm-top']), ('oxygen-top', ['oxygen-top', 'oxygen top', 'o-top']), ('top', ['top', 'atop']), ('vacancy-nearby', ['vacancy-nearby', 'near vacancy', 'vacancy']), ('interface', ['interfacial site', 'interface site', 'interface']), ('subsurface', ['subsurface'])]
    for canon, pats in mapping:
        if any(p in s for p in pats):
            return canon
    return clean_scalar(x)


def load_triage_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        pid = str(r.get('paper_id', '')).strip()
        if pid and int(pd.to_numeric(r.get('should_extract'), errors='coerce') or 0) == 1:
            out[pid] = r
    return out


def load_system_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in read_jsonl(path):
        pid = str(r.get('paper_id', '')).strip()
        if pid:
            out[pid].append(r)
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
        txt = normalize_text(ch.get('text', '')); low = txt.lower()
        score = sum(1 for kw in keywords if kw.lower() in low)
        score += sum(1 for t in sys_tokens if t in low[:3000])
        if any(p.search(txt) for p in [RE_SITE_LABEL, RE_CN, RE_BADER, RE_DBAND, RE_WORK, RE_PZC, RE_ELF, RE_HBOND, RE_BRIDGE]):
            score += 3
        if ctype == 'table':
            score += 2
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
    for kw in ['site', 'bridge', 'hollow', 'top', 'vacancy', 'coordination', 'bader', 'd-band', 'charge transfer', 'oxidation state', 'pdos', 'dos', 'elf', 'work function', 'pzc', 'hydrogen bond', 'interfacial water', 'bridge oxygen']:
        if kw in txt:
            s += 1
    for token in re.split(r'\W+', sys_terms.lower()):
        if len(token) >= 2 and token in txt:
            s += 1
    if ev.get('chunk_type') == 'table':
        s += 3
    return s


def trim_evidence(evs_main, evs_table, evs_entity, args, sys_terms: str):
    evs_main = sorted(dedup_evidence(evs_main), key=lambda x: score_hit(x, sys_terms), reverse=True)[:max(8, args.max_items_main + 3)]
    evs_table = sorted(dedup_evidence(evs_table), key=lambda x: score_hit(x, sys_terms), reverse=True)[:max(6, args.max_items_table + 2)]
    evs_entity = sorted(dedup_evidence(evs_entity), key=lambda x: score_hit(x, sys_terms), reverse=True)[:min(max(3, args.max_items_entity), 6)]
    return evs_main, evs_table, evs_entity


def split_neighbors(text: str) -> List[str]:
    out = []
    for m in RE_NEIGH.finditer(text):
        out.extend(re.findall(r'[A-Z][a-z]?', m.group(1)))
    seen, final = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); final.append(x)
    return final


def parse_active_atom(label: Optional[str], system_obj: Dict[str, Any]) -> Optional[str]:
    if label:
        m = re.match(r'([A-Z][a-z]?)\s*[- ]?(?:top|bridge|hollow)', label)
        if m:
            return m.group(1)
        if label.lower().startswith('o'):
            return 'O'
    comp = clean_scalar(system_obj.get('composition') or system_obj.get('material_name')) or ''
    m = re.match(r'([A-Z][a-z]?)', comp)
    return m.group(1) if m else None


def oxidation_state_fields(raw: Any, default_element: Any = None) -> Dict[str, Any]:
    ox_raw, ox_el, ox_val = parse_oxidation_state(raw, default_element=default_element)
    return {
        'oxidation_state': ox_raw,
        'oxidation_state_raw': ox_raw,
        'oxidation_state_element': ox_el,
        'oxidation_state_value': ox_val,
    }


def infer_site_mentions(text: str) -> List[Tuple[Optional[str], Optional[str], int, int]]:
    out: List[Tuple[Optional[str], Optional[str], int, int]] = []
    for m in RE_SITE_LABEL.finditer(text):
        label = clean_scalar(m.group(1)); out.append((label, normalize_site_type(label), m.start(), m.end()))
    for m in RE_SITE_TOKEN.finditer(text):
        atom = clean_scalar(m.group('atom')); stype_raw = clean_scalar(m.group('stype') or m.group('stype2')); stype = normalize_site_type(stype_raw)
        label = f'{atom}-{stype_raw}' if atom and stype_raw and stype_raw.lower() in {'top', 'bridge', 'hollow'} else stype_raw
        out.append((clean_scalar(label), stype, m.start(), m.end()))
    final, seen = [], set()
    for item in out:
        key = (item[0], item[1], item[2] // 80)
        if key not in seen:
            seen.add(key); final.append(item)
    return final[:10]


def match_num(pattern: re.Pattern, text: str) -> Optional[float]:
    m = pattern.search(text)
    return to_num(m.group(1)) if m else None


def match_text(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    if not m:
        return None
    return clean_scalar(m.group(1) if m.lastindex else m.group(0))


def build_record(paper_id: str, system_obj: Dict[str, Any], label: Optional[str], stype: Optional[str], window: str, idx: int, evs_all: List[Dict[str, Any]]) -> Dict[str, Any]:
    bridge = clean_bridge(match_text(RE_BRIDGE, window), window) or clean_bridge(system_obj.get('bridge_structure'), window)
    hbond = match_text(RE_HBOND, window)
    strong = 'yes' if re.search(r'strongly hydrogen[- ]bonded|symmetric HB|strong HB', window, re.I) else None
    weak = 'yes' if re.search(r'weakly hydrogen[- ]bonded|weak HB', window, re.I) else None
    active_atom = parse_active_atom(label, system_obj)
    ox = oxidation_state_fields(match_text(RE_OX, window), default_element=active_atom)
    return {
        'paper_id': paper_id,
        'system_local_id': system_obj.get('system_local_id'),
        'site_local_id': f"{system_obj.get('system_local_id')}_site_{idx:03d}",
        'ads_site_type': stype,
        'site_label': label,
        'active_atom': active_atom,
        'neighbor_atoms': split_neighbors(window),
        'coordination_number': match_num(RE_CN, window),
        'local_geometry': match_text(RE_GEOM, window),
        'vacancy_nearby': 'yes' if RE_VAC.search(window) else None,
        'hydroxylated_surface': 'yes' if RE_HYDROX.search(window) else None,
        'oxidation_state': ox['oxidation_state'],
        'oxidation_state_raw': ox['oxidation_state_raw'],
        'oxidation_state_element': ox['oxidation_state_element'],
        'oxidation_state_value': ox['oxidation_state_value'],
        'bader_charge': match_num(RE_BADER, window),
        'd_band_center': match_num(RE_DBAND, window),
        'charge_transfer_direction': match_text(RE_CHARGE_DIR, window),
        'bridge_structure_nearby': bridge,
        'M_O_X_configuration': bridge or clean_scalar(system_obj.get('M_O_X_configuration')),
        'ELF_descriptor': match_text(RE_ELF, window),
        'work_function': match_num(RE_WORK, window),
        'PZC': match_num(RE_PZC, window),
        'hbond_acceptor_site': 'bridge oxygen' if bridge and re.search(r'oxygen|O', bridge, re.I) else None,
        'H2O_binding_mode': match_text(RE_H2O_MODE, window),
        'interfacial_water_role': hbond,
        'hydrogen_bond_network': hbond,
        'strong_HB_water_signal': strong,
        'weak_HB_water_signal': weak,
        'descriptor_claim': match_text(RE_CHARGE_DIR, window) or match_text(RE_HBOND, window) or match_text(RE_GEOM, window) or ('bridge oxygen motif' if bridge else None),
        'confidence_score': None,
        'system_match_score': system_match_score(system_obj, window),
        'qc_numeric_support': {},
        'qc_field_support': field_support_summary({'site_label': label, 'active_atom': active_atom, 'bridge_structure_nearby': bridge, 'hydrogen_bond_network': hbond}, window, ['site_label','active_atom','bridge_structure_nearby','hydrogen_bond_network']),
        'extraction_source': 'heuristic',
        'evidence': normalize_evidence_list([{'chunk_id': e.get('chunk_id'), 'quote': str(e.get('text') or e.get('quote') or '')[:520], 'source': e.get('source'), 'chunk_type': e.get('chunk_type'), 'page_start': e.get('page_start'), 'page_end': e.get('page_end'), 'section_path': e.get('section_path')} for e in evs_all[:7]]),
    }


def heuristic_site_records(paper_id: str, system_obj: Dict[str, Any], evs_main, evs_table, evs_entity) -> List[Dict[str, Any]]:
    evs_all = evs_table + evs_entity + evs_main
    text = normalize_text(' '.join(str(e.get('text') or e.get('quote') or '') for e in evs_all))
    mentions = infer_site_mentions(text)
    anchors = []
    for pat in [RE_CN, RE_BADER, RE_DBAND, RE_WORK, RE_PZC, RE_BRIDGE, RE_ELF, RE_HBOND, RE_VAC, RE_HYDROX]:
        anchors.extend([(None, None, m.start(), m.end()) for m in pat.finditer(text)])
    mentions = mentions + anchors
    if not mentions and text:
        mentions = [(None, None, 0, min(len(text), 1))]
    rows = []
    for idx, (label, stype, st, en) in enumerate(mentions[:10], 1):
        window = text[max(0, st-450):min(len(text), en+650)]
        rows.append(build_record(paper_id, system_obj, label, stype, window, idx, evs_all))
    # de-duplicate and remove fully empty records except evidence
    out, seen = [], set()
    for r in rows:
        key = (r['site_label'], r['ads_site_type'], r['active_atom'], r['d_band_center'], r['bader_charge'], r['bridge_structure_nearby'], r['hydrogen_bond_network'])
        if key in seen:
            continue
        seen.add(key)
        if any(r.get(k) for k in ['site_label','ads_site_type','active_atom','coordination_number','d_band_center','bader_charge','bridge_structure_nearby','ELF_descriptor','work_function','PZC','hydrogen_bond_network','vacancy_nearby','hydroxylated_surface']):
            out.append(r)
    return out[:10]


def postprocess_records(obj: Dict[str, Any], paper_id: str, system_obj: Dict[str, Any], evidence_txt: str = "") -> List[Dict[str, Any]]:
    recs = obj.get('records', []) if isinstance(obj, dict) else []
    if not isinstance(recs, list):
        recs = []
    out, seen = [], set()
    for i, r in enumerate(recs, 1):
        if not isinstance(r, dict):
            continue
        conf_num, conf_raw, conf_status = strict_confidence_score(r.get('confidence_score'))
        bridge = clean_bridge(r.get('bridge_structure_nearby'), evidence_txt) or clean_bridge(r.get('M_O_X_configuration'), evidence_txt) or clean_bridge(system_obj.get('bridge_structure'), evidence_txt)
        active_atom = clean_scalar(r.get('active_atom'))
        ox_raw_in = clean_scalar(r.get('oxidation_state_raw')) or clean_scalar(r.get('oxidation_state'))
        ox_raw, ox_el, ox_val = parse_oxidation_state(ox_raw_in, default_element=active_atom)
        if to_num(r.get('oxidation_state_value')) is not None:
            ox_val = to_num(r.get('oxidation_state_value'))
        if clean_scalar(r.get('oxidation_state_element')):
            ox_el = clean_scalar(r.get('oxidation_state_element'))
        rec = {
            'paper_id': paper_id,
            'system_local_id': clean_scalar(r.get('system_local_id')) or clean_scalar(system_obj.get('system_local_id')),
            'site_local_id': clean_scalar(r.get('site_local_id')) or f"{system_obj.get('system_local_id')}_site_{i:03d}",
            'ads_site_type': normalize_site_type(r.get('ads_site_type')),
            'site_label': clean_scalar(r.get('site_label')),
            'active_atom': active_atom,
            'neighbor_atoms': clean_listlike(r.get('neighbor_atoms')),
            'coordination_number': to_num(r.get('coordination_number')),
            'local_geometry': clean_scalar(r.get('local_geometry')),
            'vacancy_nearby': normalize_yes_no(r.get('vacancy_nearby')),
            'hydroxylated_surface': normalize_yes_no(r.get('hydroxylated_surface')),
            'oxidation_state': ox_raw,
            'oxidation_state_raw': ox_raw,
            'oxidation_state_element': ox_el,
            'oxidation_state_value': ox_val,
            'bader_charge': to_num(r.get('bader_charge')),
            'd_band_center': to_num(r.get('d_band_center')),
            'charge_transfer_direction': clean_scalar(r.get('charge_transfer_direction')),
            'bridge_structure_nearby': bridge,
            'M_O_X_configuration': clean_bridge(r.get('M_O_X_configuration'), evidence_txt) or bridge,
            'ELF_descriptor': clean_scalar(r.get('ELF_descriptor')),
            'work_function': to_num(r.get('work_function')),
            'PZC': to_num(r.get('PZC')),
            'hbond_acceptor_site': clean_scalar(r.get('hbond_acceptor_site')),
            'H2O_binding_mode': clean_scalar(r.get('H2O_binding_mode')),
            'interfacial_water_role': clean_scalar(r.get('interfacial_water_role')),
            'hydrogen_bond_network': clean_scalar(r.get('hydrogen_bond_network')),
            'strong_HB_water_signal': normalize_yes_no(r.get('strong_HB_water_signal')),
            'weak_HB_water_signal': normalize_yes_no(r.get('weak_HB_water_signal')),
            'descriptor_claim': clean_scalar(r.get('descriptor_claim')),
            'confidence_score': conf_num,
            'confidence_score_raw': conf_raw,
            'confidence_score_status': conf_status,
            'evidence': normalize_evidence_list(r.get('evidence', [])),
            'system_match_score': system_match_score(system_obj, evidence_txt),
            'qc_field_support': {},
            'extraction_source': 'llm',
        }
        # Drop hallucinated numeric descriptors that are not present in the evidence prompt.
        rec, numeric_support = validate_numeric_fields(rec, ['coordination_number','bader_charge','d_band_center','work_function','PZC'], evidence_txt)
        rec['qc_numeric_support'] = numeric_support
        rec['qc_field_support'] = field_support_summary(rec, evidence_txt, ['site_label','active_atom','bridge_structure_nearby','ELF_descriptor','hydrogen_bond_network','interfacial_water_role'])
        # Drop weak textual claims that are unsupported by evidence.
        for tf in ['ELF_descriptor','interfacial_water_role','hydrogen_bond_network','H2O_binding_mode','descriptor_claim']:
            if rec.get(tf) and not text_supported(rec.get(tf), evidence_txt):
                rec[tf] = None
        key = (rec['system_local_id'], rec['site_label'], rec['ads_site_type'], rec['active_atom'], rec['d_band_center'], rec['bader_charge'], rec['bridge_structure_nearby'], rec.get('oxidation_state_value'))
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return out


def build_prompt(paper_id: str, system_obj: Dict[str, Any], evs_main, evs_table, evs_entity, args) -> str:
    payload = {k: system_obj.get(k) for k in ['system_local_id','material_name','surface_facet','defect_type','dopant','bridge_structure','M_O_X_configuration','support']}
    return SYSTEM_PROMPT + '\n\n' + (
        f'paper_id={paper_id}\nTARGET_SYSTEM={json.dumps(payload, ensure_ascii=False)}\n\n'
        f'EVIDENCE_TABLE:\n{pack_evidence(evs_table, max_items=args.max_items_table, total_chars=args.table_budget_chars, max_quote_chars=args.max_table_chars)}\n\n'
        f'EVIDENCE_ENTITY:\n{pack_evidence(evs_entity, max_items=args.max_items_entity, total_chars=args.entity_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        f'EVIDENCE_MAIN:\n{pack_evidence(evs_main, max_items=args.max_items_main, total_chars=args.main_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        'TASK: Extract site-level descriptors for this system only.\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--triage', required=True); ap.add_argument('--systems', required=True); ap.add_argument('--parsed-root', required=True)
    ap.add_argument('--api-bases', default=''); ap.add_argument('--model-id', default=''); ap.add_argument('--embed-model', required=True); ap.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    ap.add_argument('--qdrant-prefix', required=True); ap.add_argument('--collection-prefix', required=True); ap.add_argument('--shard-ids', default='0,1,2')
    ap.add_argument('--topk-main', type=int, default=14); ap.add_argument('--topk-table', type=int, default=20); ap.add_argument('--topk-entity', type=int, default=6)
    ap.add_argument('--main-budget-chars', type=int, default=9000); ap.add_argument('--table-budget-chars', type=int, default=6500); ap.add_argument('--entity-budget-chars', type=int, default=1800)
    ap.add_argument('--max-items-main', type=int, default=10); ap.add_argument('--max-items-table', type=int, default=8); ap.add_argument('--max-items-entity', type=int, default=4)
    ap.add_argument('--max-text-chars', type=int, default=850); ap.add_argument('--max-table-chars', type=int, default=1500); ap.add_argument('--max-caption-chars', type=int, default=1200)
    ap.add_argument('--workers', type=int, default=1); ap.add_argument('--timeout', type=int, default=180); ap.add_argument('--max-retries', type=int, default=3); ap.add_argument('--verify-llm', action='store_true', help='Run a second LLM pass to verify extracted JSON against evidence before deterministic QC.')
    ap.add_argument('--verifier-api-bases', default='', help='Optional API bases for verifier; defaults to --api-bases.')
    ap.add_argument('--verifier-model-id', default='', help='Optional verifier model; defaults to --model-id.')
    ap.add_argument('--verify-max-tokens', type=int, default=2600, help='Max tokens for second-pass verification.')
    ap.add_argument('--max-tokens', type=int, default=1800)
    ap.add_argument('--out', required=True); ap.add_argument('--progress', default='outputs/progress_hads_step3c_sites.json'); ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    triage_map = load_triage_rows(Path(args.triage))
    systems_by_paper = load_system_rows(Path(args.systems))
    work_items = [(pid, s) for pid in triage_map for s in systems_by_paper.get(pid, [])]
    total = len(work_items); print(f'[INFO] HADS Step3c work_items={total}', flush=True)
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
    progress_lock = threading.Lock(); counter = {'done': 0, 'ok': 0, 'skip': 0, 'err': 0}

    def process_item(item):
        paper_id, system_obj = item; work_key = f"{paper_id}::{system_obj.get('system_local_id')}"
        if work_key in done: return 'skip', work_key, 0, None
        try:
            sys_terms = ' '.join(str(system_obj.get(k, '') or '') for k in ['material_name','composition','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration','support'])
            evs_main = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_main_text_shard', shard_ids, qvecs['main'], paper_id, ['text'], args.topk_main, QUERY_MAIN, use_qdrant_lock, sys_terms)
            evs_table = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_table_caption_shard', shard_ids, qvecs['table'], paper_id, ['caption','table'], args.topk_table, QUERY_TABLE, use_qdrant_lock, sys_terms)
            evs_entity = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_entity_shard', shard_ids, qvecs['entity'], paper_id, ['text','caption','table'], args.topk_entity, QUERY_ENTITY, use_qdrant_lock, sys_terms)
            evs_main, evs_table, evs_entity = trim_evidence(evs_main, evs_table, evs_entity, args, sys_terms)
            evidence_txt = packed_evidence_text(evs_main, evs_table, evs_entity)
            obj = None
            if llm is not None and (evs_main or evs_table or evs_entity):
                try: obj = safe_json_extract(llm.completions(build_prompt(paper_id, system_obj, evs_main, evs_table, evs_entity, args), temperature=0.0, max_tokens=args.max_tokens))
                except Exception as e: print(f'[WARN] LLM failed for {work_key}: {e}', flush=True)
            if obj is not None and verifier_llm is not None:
                vobj = verify_extraction_json(verifier_llm, paper_id, 'site_descriptors', obj, evidence_txt, target_context={k: system_obj.get(k) for k in ['system_local_id','material_name','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration','support']}, max_tokens=args.verify_max_tokens)
                if vobj is not None:
                    obj = vobj
            rows = postprocess_records(obj, paper_id, system_obj, evidence_txt) if obj is not None else []
            if not rows:
                rows = heuristic_site_records(paper_id, system_obj, evs_main, evs_table, evs_entity)
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
                    print(f'[OK] {work_key} site_records={n_rows} progress={counter["done"]}/{total}', flush=True)
                elif status == 'skip':
                    counter['skip'] += 1; print(f'[SKIP] {work_key} progress={counter["done"]}/{total}', flush=True)
                else:
                    counter['err'] += 1; print(f'[ERR] {work_key} error={err} progress={counter["done"]}/{total}', flush=True)
    print(f'[DONE] wrote {args.out}', flush=True)

if __name__ == '__main__':
    main()
