#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3f: repair missing site/mechanism descriptors for metric-bearing records.

Purpose
-------
After Step3e, adsorption metrics (e.g. ΔE_H*) and site descriptors may still live
in different canonical rows because papers often report them in different tables or
paragraphs. This repair step increases the joint coverage of ΔE_H* and mechanism
descriptors while keeping provenance explicit.

Repair layers
-------------
1. exact_site_id: fill missing descriptors from raw/merged site rows with the same site_local_id.
2. unique_system_site: if a paper+system has only one informative site descriptor, use it.
3. fuzzy_site_match: match site rows to adsorption rows using active atom, site label/type,
   adsorption configuration, and evidence overlap.
4. optional llm_repair: for records still missing key descriptors, search the parsed paper and
   ask an OpenAI-compatible model to extract only evidence-supported descriptors.

No existing non-empty descriptor is overwritten unless --overwrite is given.
Every repaired row receives descriptor_repair_* metadata and repaired field names.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from utils.evidence_utils import normalize_evidence_list
from utils.io_utils import append_jsonl, read_jsonl
from utils.llm_client import MultiEndpointClient, safe_json_extract
from utils.normalize_utils import to_float
from utils.text_utils import clip_text, normalize_text
from utils.hads_validation import number_supported, parse_oxidation_state, text_supported

EMPTY = {'none','null','nan','unknown','not mentioned','n/a','na',''}

METRIC_VALUE_FIELDS = [
    'H_adsorption_energy_value','H_adsorption_free_energy_value',
    'OH_adsorption_energy_value','OH_adsorption_free_energy_value',
    'H2O_adsorption_energy_value','deprotonation_energy_value',
    'proton_transfer_barrier','Volmer_barrier','water_dissociation_barrier',
]
METRIC_EV_FIELDS = [f + '_eV' for f in METRIC_VALUE_FIELDS]

DESCRIPTOR_FIELDS = [
    'ads_site_type','site_label','active_atom','neighbor_atoms','coordination_number',
    'local_geometry','vacancy_nearby','hydroxylated_surface','oxidation_state',
    'oxidation_state_raw','oxidation_state_element','oxidation_state_value',
    'bader_charge','d_band_center','charge_transfer_direction',
    'bridge_structure_nearby','ELF_descriptor','work_function','PZC',
    'hbond_acceptor_site','H2O_binding_mode','interfacial_water_role',
    'hydrogen_bond_network','strong_HB_water_signal','weak_HB_water_signal',
]
NUMERIC_DESCRIPTOR_FIELDS = {'coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC'}
LIST_DESCRIPTOR_FIELDS = {'neighbor_atoms'}
KEY_DESCRIPTOR_FIELDS = ['coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC','vacancy_nearby','hydroxylated_surface','bridge_structure_nearby']

REPAIR_KEYWORDS = [
    'd-band center','d band center','Bader charge','charge transfer','coordination number',
    'coordinated','coordination environment','oxidation state','valence state','XPS',
    'XANES','EXAFS','DOS','PDOS','density of states','work function','PZC',
    'vacancy','hydroxylated surface','hydroxyl','bridge structure','bridge oxygen','active site',
    'adsorption energy','Delta E_H','Delta G_H','H adsorption','H*',
]


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    return None if s.lower() in EMPTY else s


def clean_listlike(x: Any) -> List[str]:
    if x is None:
        return []
    arr = x if isinstance(x, list) else str(x).replace('|',';').replace(',',';').split(';')
    out, seen = [], set()
    for a in arr:
        t = clean_scalar(a)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def uniq(xs: Iterable[Any]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        if isinstance(x, list):
            vals = x
        else:
            vals = [x]
        for y in vals:
            t = clean_scalar(y)
            if t and t not in seen:
                seen.add(t); out.append(t)
    return out


def num(x: Any) -> Optional[float]:
    return to_float(x)


def has_metric(row: Dict[str, Any], h_only: bool = True) -> bool:
    fields = ['H_adsorption_energy_value_eV','H_adsorption_free_energy_value_eV'] if h_only else METRIC_EV_FIELDS
    return any(num(row.get(f)) is not None for f in fields)


def descriptor_missing(row: Dict[str, Any], f: str) -> bool:
    if f in LIST_DESCRIPTOR_FIELDS:
        return len(clean_listlike(row.get(f))) == 0
    if f in NUMERIC_DESCRIPTOR_FIELDS:
        return num(row.get(f)) is None
    return clean_scalar(row.get(f)) is None


def descriptor_value(row: Dict[str, Any], f: str) -> Any:
    if f in LIST_DESCRIPTOR_FIELDS:
        vals = clean_listlike(row.get(f))
        return vals if vals else None
    if f in NUMERIC_DESCRIPTOR_FIELDS:
        return num(row.get(f))
    return clean_scalar(row.get(f))


def descriptor_presence_count(row: Dict[str, Any]) -> int:
    return sum(1 for f in DESCRIPTOR_FIELDS if descriptor_value(row, f) is not None)


def evidence_chunk_ids(evs: Any) -> set:
    ids = set()
    for e in normalize_evidence_list(evs):
        cid = e.get('chunk_id')
        if cid is not None:
            ids.add(str(cid))
    return ids


def merge_evidence(*lists_: Any, limit: int = 60) -> List[Dict[str, Any]]:
    merged, seen = [], set()
    for evs in lists_:
        for e in normalize_evidence_list(evs):
            quote = str(e.get('quote') or e.get('text') or '')
            key = (e.get('chunk_id'), quote[:200], e.get('page_start'))
            if key in seen:
                continue
            seen.add(key)
            merged.append(e)
    return merged[:limit]


def norm_for_match(x: Any) -> str:
    s = normalize_text(x).lower()
    s = re.sub(r'[^a-z0-9+\-]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def tokens_for_match(x: Any) -> set:
    s = norm_for_match(x)
    return {t for t in s.split() if len(t) >= 2 and t not in {'site','surface','adsorption','adsorbed','configuration','model','the','and','for'}}


def site_identity_text(site: Dict[str, Any]) -> str:
    return ' '.join(str(site.get(k) or '') for k in [
        'site_local_id','site_label','ads_site_type','active_atom','neighbor_atoms',
        'local_geometry','oxidation_state','oxidation_state_raw','vacancy_nearby',
        'hydroxylated_surface','bridge_structure_nearby','descriptor_claim'
    ])


def ads_identity_text(row: Dict[str, Any]) -> str:
    return ' '.join(str(row.get(k) or '') for k in [
        'site_local_id','adsorbate','adsorption_configuration','reference_surface',
        'adsorption_strength_trend','proton_transfer_pathway','rate_determining_step',
        'adsorption_descriptor_claim','coverage','coadsorbates','active_atom','site_label'
    ])


def site_ads_fuzzy_score(site: Dict[str, Any], row: Dict[str, Any]) -> float:
    score = 0.0
    stoks = tokens_for_match(site_identity_text(site))
    atoks = tokens_for_match(ads_identity_text(row))
    if stoks and atoks:
        score += min(0.25, 0.05 * len(stoks & atoks))
    active = clean_scalar(site.get('active_atom'))
    if active and active.lower() in norm_for_match(ads_identity_text(row)):
        score += 0.20
    slabel = clean_scalar(site.get('site_label'))
    if slabel and norm_for_match(slabel) and norm_for_match(slabel) in norm_for_match(ads_identity_text(row)):
        score += 0.25
    stype = clean_scalar(site.get('ads_site_type'))
    if stype and norm_for_match(stype) and norm_for_match(stype) in norm_for_match(ads_identity_text(row)):
        score += 0.15
    cfg = norm_for_match(row.get('adsorption_configuration'))
    if active and cfg and active.lower() in cfg:
        score += 0.15
    sid = evidence_chunk_ids(site.get('evidence', []))
    aid = evidence_chunk_ids(row.get('evidence_all', []))
    if sid and aid:
        overlap = len(sid & aid)
        if overlap:
            score += min(0.30, 0.12 * overlap)
    score += min(0.15, 0.02 * descriptor_presence_count(site))
    return round(score, 4)


def site_group_key(rec: Dict[str, Any], ordinal: int) -> str:
    sid = clean_scalar(rec.get('site_local_id'))
    if sid:
        return sid
    parts = [clean_scalar(rec.get(k)) for k in ['site_label','ads_site_type','active_atom','oxidation_state','oxidation_state_value','vacancy_nearby','hydroxylated_surface','bridge_structure_nearby']]
    cn = num(rec.get('coordination_number'))
    if cn is not None:
        parts.append(f'CN{cn:g}')
    parts = [p for p in parts if p]
    return 'site::' + '__'.join(parts) if parts else f'site::fallback_{ordinal:03d}'


def site_strength(r: Dict[str, Any]) -> float:
    s = 0.0
    for f in DESCRIPTOR_FIELDS:
        if descriptor_value(r, f) is not None:
            s += 0.08 if f in NUMERIC_DESCRIPTOR_FIELDS else 0.05
    s += min(0.10, 0.01 * len(normalize_evidence_list(r.get('evidence', []))))
    conf = num(r.get('confidence_score'))
    if conf is not None:
        s += 0.05 * conf
    return round(s, 4)


def best_descriptor(rows: List[Dict[str, Any]], field: str) -> Any:
    best, sc = None, -1
    for r in rows:
        v = descriptor_value(r, field)
        if v is not None and site_strength(r) > sc:
            best, sc = v, site_strength(r)
    return best


def merge_site_rows(rows: List[Dict[str, Any]], fallback_id: str) -> Dict[str, Any]:
    base = max(rows, key=site_strength) if rows else {}
    out = {
        'paper_id': clean_scalar(base.get('paper_id')),
        'system_local_id': clean_scalar(base.get('system_local_id')),
        'site_local_id': clean_scalar(base.get('site_local_id')) or fallback_id,
    }
    for f in DESCRIPTOR_FIELDS:
        if f == 'neighbor_atoms':
            out[f] = uniq([x for r in rows for x in clean_listlike(r.get(f))])
        else:
            out[f] = best_descriptor(rows, f)
    out['evidence'] = merge_evidence(*[r.get('evidence', []) for r in rows])
    out['site_confidence_score'] = max([num(r.get('confidence_score')) or 0 for r in rows] + [0])
    out['site_system_match_score'] = max([num(r.get('system_match_score')) or 0 for r in rows] + [0])
    out['site_extraction_source'] = clean_scalar(base.get('extraction_source'))
    return out


def grouped_merged_sites(site_path: Path) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    raw = defaultdict(list)
    for r in read_jsonl(site_path):
        key = (str(r.get('paper_id','')).strip(), str(r.get('system_local_id','')).strip())
        if key[0] and key[1]:
            raw[key].append(r)
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for key, rows in raw.items():
        groups = defaultdict(list)
        for i, r in enumerate(rows, 1):
            groups[site_group_key(r, i)].append(r)
        out[key] = [merge_site_rows(grp, f'{key[1]}_repair_site_{i:03d}') for i, grp in enumerate(groups.values(), 1)]
    return out


def choose_repair_site(row: Dict[str, Any], candidates: List[Dict[str, Any]], min_fuzzy_score: float) -> Tuple[Optional[Dict[str, Any]], str, float]:
    if not candidates:
        return None, 'no_candidate', 0.0
    sid = clean_scalar(row.get('site_local_id'))
    if sid:
        for s in candidates:
            if clean_scalar(s.get('site_local_id')) == sid:
                return s, 'repair_exact_site_id', 1.0
    informative = [s for s in candidates if descriptor_presence_count(s) > 0]
    if len(informative) == 1:
        return informative[0], 'repair_unique_system_site', 0.82
    if informative:
        scored = sorted(((site_ads_fuzzy_score(s, row), s) for s in informative), key=lambda x: x[0], reverse=True)
        best_score, best_site = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= min_fuzzy_score and (best_score - second >= 0.04 or best_score >= 0.45):
            return best_site, 'repair_fuzzy_site_match', round(min(0.78, max(0.55, best_score)), 4)
    return None, 'unlinked', 0.0


def fill_from_site(row: Dict[str, Any], site: Dict[str, Any], method: str, conf: float, overwrite: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    repaired = []
    out = dict(row)
    for f in DESCRIPTOR_FIELDS:
        val = descriptor_value(site, f)
        if val is None:
            continue
        if overwrite or descriptor_missing(out, f):
            out[f] = val
            repaired.append(f)
    if repaired:
        out['evidence_all'] = merge_evidence(out.get('evidence_all', []), site.get('evidence', []))
        out['descriptor_repair_method'] = method
        out['descriptor_repair_confidence'] = conf
        out['descriptor_repair_source'] = 'site_descriptor_candidates'
        out['descriptor_repaired_fields'] = repaired
        old_method = clean_scalar(out.get('descriptor_link_method'))
        if old_method in {None, 'none', 'unlinked'} or method in {'repair_exact_site_id','repair_unique_system_site'}:
            out['descriptor_link_method'] = method.replace('repair_', '')
            out['descriptor_link_confidence'] = conf
        out['descriptor_link_is_soft'] = int((clean_scalar(out.get('descriptor_link_method')) or '').find('fuzzy') >= 0 or (clean_scalar(out.get('descriptor_link_method')) or '').find('unique') >= 0)
        if num(out.get('link_score')) is not None:
            out['link_score'] = round(min(1.0, float(out.get('link_score') or 0) + 0.04 * conf), 4)
    return out, repaired


def load_chunks(parsed_root: Path, paper_id: str) -> List[Dict[str, Any]]:
    fp = parsed_root / 'papers' / paper_id / 'chunks.jsonl'
    if not fp.exists():
        return []
    return read_jsonl(fp)


def local_descriptor_search(parsed_root: Path, row: Dict[str, Any], topk: int = 18) -> List[Dict[str, Any]]:
    pid = clean_scalar(row.get('paper_id'))
    if not pid:
        return []
    material_terms = ' '.join(str(row.get(k) or '') for k in ['material_name','composition','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration','site_label','active_atom'])
    mtoks = [t.lower() for t in re.split(r'\W+', material_terms) if len(t) >= 2]
    hits = []
    for ch in load_chunks(parsed_root, pid):
        txt = normalize_text(ch.get('text',''))
        low = txt.lower()
        score = 0
        for kw in REPAIR_KEYWORDS:
            if kw.lower() in low:
                score += 3 if kw.lower() in {'d-band center','bader charge','coordination number','oxidation state','valence state'} else 1
        score += sum(1 for t in mtoks[:30] if t in low[:5000])
        ctype = normalize_text(ch.get('chunk_type') or '')
        if ctype in {'table','caption'}:
            score += 3
        if score > 0:
            ev = {'paper_id': ch.get('paper_id'), 'chunk_id': ch.get('chunk_id'), 'chunk_type': ch.get('chunk_type'), 'section_path': ch.get('section_path'), 'page_start': ch.get('page_start'), 'page_end': ch.get('page_end'), 'source': ch.get('source'), 'quote': txt[:3500], 'score': score}
            hits.append(ev)
    hits.sort(key=lambda x: x.get('score', 0), reverse=True)
    return hits[:topk]


def build_llm_repair_prompt(row: Dict[str, Any], evidence: List[Dict[str, Any]]) -> str:
    context = {k: row.get(k) for k in [
        'paper_id','system_local_id','material_name','composition','surface_facet','dopant','defect_type',
        'bridge_structure','M_O_X_configuration','site_local_id','site_label','active_atom','adsorbate',
        'H_adsorption_energy_value_eV','H_adsorption_free_energy_value_eV','adsorption_configuration',
    ]}
    ev_txt = '\n\n'.join(f"[EVIDENCE {i+1} | chunk_id={e.get('chunk_id')} | type={e.get('chunk_type')} | page={e.get('page_start')}]\n{clip_text(e.get('quote',''), 3000)}" for i, e in enumerate(evidence))
    return f"""
You are repairing missing SITE / MECHANISM descriptors for one adsorption-energy record.
Use ONLY the evidence. Return STRICT JSON only.

TARGET_RECORD:
{json.dumps(context, ensure_ascii=False)}

EVIDENCE:
{ev_txt}

Extract only descriptors explicitly linked to TARGET_RECORD's material/system/site. Do not infer.
Return JSON with keys:
{{
  "descriptors": {{
    "ads_site_type": null,
    "site_label": null,
    "active_atom": null,
    "neighbor_atoms": [],
    "coordination_number": null,
    "local_geometry": null,
    "vacancy_nearby": null,
    "hydroxylated_surface": null,
    "oxidation_state_raw": null,
    "oxidation_state_element": null,
    "oxidation_state_value": null,
    "bader_charge": null,
    "d_band_center": null,
    "charge_transfer_direction": null,
    "bridge_structure_nearby": null,
    "ELF_descriptor": null,
    "work_function": null,
    "PZC": null
  }},
  "confidence_score": 0.0,
  "evidence": []
}}
Rules:
- Numeric values must appear in evidence. Do not use page numbers, figure/table numbers, potentials, currents, temperatures, or unrelated reference systems.
- oxidation_state_value must be the active metal oxidation state as a number; keep oxidation_state_raw as written.
- Use null if not explicitly supported.
""".strip()


def apply_llm_descriptors(row: Dict[str, Any], obj: Dict[str, Any], evidence_text: str, evidence_list: List[Dict[str, Any]], overwrite: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    desc = obj.get('descriptors', {}) if isinstance(obj, dict) else {}
    if not isinstance(desc, dict):
        return row, []
    out = dict(row)
    repaired = []
    # Parse oxidation state if raw is given without numeric value.
    if desc.get('oxidation_state_raw') and desc.get('oxidation_state_value') is None:
        raw, el, val = parse_oxidation_state(desc.get('oxidation_state_raw'), desc.get('active_atom') or row.get('active_atom'))
        desc['oxidation_state_raw'] = raw
        desc['oxidation_state_element'] = desc.get('oxidation_state_element') or el
        desc['oxidation_state_value'] = val
    for f in DESCRIPTOR_FIELDS:
        if f not in desc:
            continue
        v = desc.get(f)
        if v in (None, '', [], {}):
            continue
        if f in LIST_DESCRIPTOR_FIELDS:
            v = clean_listlike(v)
            if not v:
                continue
            supported = any(text_supported(x, evidence_text) for x in v)
        elif f in NUMERIC_DESCRIPTOR_FIELDS:
            v = num(v)
            if v is None:
                continue
            supported = number_supported(v, evidence_text)
        else:
            v = clean_scalar(v)
            if not v:
                continue
            supported = text_supported(v, evidence_text)
        if not supported:
            continue
        if overwrite or descriptor_missing(out, f):
            out[f] = v
            repaired.append(f)
    if repaired:
        out['evidence_all'] = merge_evidence(out.get('evidence_all', []), evidence_list, normalize_evidence_list(obj.get('evidence', [])))
        out['descriptor_repair_method'] = 'llm_repair'
        out['descriptor_repair_confidence'] = num(obj.get('confidence_score')) or 0.65
        out['descriptor_repair_source'] = 'parsed_text_llm_repair'
        out['descriptor_repaired_fields'] = repaired
        out['descriptor_link_method'] = clean_scalar(out.get('descriptor_link_method')) or 'llm_repair'
        out['descriptor_link_confidence'] = max(num(out.get('descriptor_link_confidence')) or 0, num(out.get('descriptor_repair_confidence')) or 0)
        out['descriptor_link_is_soft'] = 1
    return out, repaired


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--canonical', required=True, help='Step3e canonical JSONL')
    ap.add_argument('--sites', required=True, help='Step3c site descriptor JSONL')
    ap.add_argument('--out', required=True)
    ap.add_argument('--summary', default='')
    ap.add_argument('--h-only', action='store_true', default=True, help='Repair rows with H adsorption metrics only (default).')
    ap.add_argument('--all-metrics', action='store_true', help='Repair rows with any adsorption/deprotonation metric.')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing descriptor values (not recommended).')
    ap.add_argument('--min-fuzzy-score', type=float, default=0.26)
    ap.add_argument('--llm-repair', action='store_true', help='Use model to repair records still missing key descriptors.')
    ap.add_argument('--parsed-root', default='pdf-files-parsed')
    ap.add_argument('--api-bases', default=os.environ.get('API_BASES','') or os.environ.get('SCNET_BASE_URL',''))
    ap.add_argument('--model-id', default=os.environ.get('MODEL_ID',''))
    ap.add_argument('--timeout', type=int, default=int(os.environ.get('TIMEOUT','360')))
    ap.add_argument('--max-retries', type=int, default=int(os.environ.get('MAX_RETRIES','5')))
    ap.add_argument('--max-tokens', type=int, default=2200)
    ap.add_argument('--max-llm-records', type=int, default=0, help='0 means no cap when --llm-repair is enabled.')
    ap.add_argument('--llm-topk', type=int, default=18)
    args = ap.parse_args()

    canonical_rows = read_jsonl(Path(args.canonical))
    sites_by_key = grouped_merged_sites(Path(args.sites))
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        outp.unlink()
    summary = defaultdict(int)
    llm = None
    if args.llm_repair:
        bases = [x.strip() for x in args.api_bases.split(',') if x.strip()]
        if not bases or not args.model_id:
            raise SystemExit('--llm-repair requires --api-bases and --model-id or env API_BASES/MODEL_ID')
        llm = MultiEndpointClient(bases, args.model_id, timeout=args.timeout, max_retries=args.max_retries)

    llm_used = 0
    parsed_root = Path(args.parsed_root)
    h_only = not args.all_metrics

    for i, row in enumerate(canonical_rows, 1):
        out = dict(row)
        if has_metric(out, h_only=h_only):
            key = (str(out.get('paper_id','')).strip(), str(out.get('system_local_id','')).strip())
            candidates = sites_by_key.get(key, [])
            site, method, conf = choose_repair_site(out, candidates, args.min_fuzzy_score)
            if site:
                out, repaired = fill_from_site(out, site, method, conf, overwrite=args.overwrite)
                if repaired:
                    summary['deterministic_repaired_rows'] += 1
                    summary[f'method_{method}'] += 1
                    for f in repaired:
                        summary[f'field_{f}'] += 1
            # Optional LLM repair for rows that still miss key descriptors.
            still_missing = [f for f in KEY_DESCRIPTOR_FIELDS if descriptor_missing(out, f)]
            if llm is not None and still_missing and (args.max_llm_records <= 0 or llm_used < args.max_llm_records):
                evidence = local_descriptor_search(parsed_root, out, topk=args.llm_topk)
                if evidence:
                    try:
                        prompt = build_llm_repair_prompt(out, evidence)
                        obj = safe_json_extract(llm.completions(prompt, temperature=0.0, max_tokens=args.max_tokens))
                        llm_used += 1
                        evidence_text = normalize_text(' '.join(str(e.get('quote') or '') for e in evidence))
                        if obj:
                            out, repaired2 = apply_llm_descriptors(out, obj, evidence_text, evidence, overwrite=args.overwrite)
                            if repaired2:
                                summary['llm_repaired_rows'] += 1
                                for f in repaired2:
                                    summary[f'llm_field_{f}'] += 1
                    except Exception as e:
                        summary['llm_errors'] += 1
                        out['descriptor_repair_error'] = str(e)[:300]
        append_jsonl(out, outp)
        summary['rows_processed'] += 1
    summary['llm_calls'] = llm_used
    summary['rows_written'] = summary['rows_processed']
    summary_path = Path(args.summary) if args.summary else outp.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(dict(summary), ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[DONE] repaired canonical records -> {outp}')
    print(f'[DONE] summary -> {summary_path}')
    print(json.dumps(dict(summary), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
