#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3e: link systems, site descriptors, and adsorption/deprotonation metrics into canonical records."""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.evidence_utils import normalize_evidence_list
from utils.io_utils import append_jsonl, load_progress, read_jsonl, save_progress
from utils.normalize_utils import to_ev, clean_unit, to_float
from utils.text_utils import normalize_text

EMPTY = {'none','null','nan','unknown','not mentioned','n/a','na',''}

SYSTEM_FIELDS = ['material_name','material_class','composition','surface_facet','termination','phase','support','dopant','interface_type','defect_type','bridge_structure','M_O_X_configuration','model_type']
SITE_FIELDS = ['ads_site_type','site_label','active_atom','neighbor_atoms','coordination_number','local_geometry','vacancy_nearby','hydroxylated_surface','oxidation_state','oxidation_state_raw','oxidation_state_element','oxidation_state_value','bader_charge','d_band_center','charge_transfer_direction','bridge_structure_nearby','ELF_descriptor','work_function','PZC','hbond_acceptor_site','H2O_binding_mode','interfacial_water_role','hydrogen_bond_network','strong_HB_water_signal','weak_HB_water_signal','descriptor_claim','system_match_score','qc_numeric_support','qc_field_support','extraction_source']
ADS_FIELDS = ['adsorbate','H_adsorption_energy_value','H_adsorption_energy_unit','H_adsorption_free_energy_value','H_adsorption_free_energy_unit','OH_adsorption_energy_value','OH_adsorption_energy_unit','OH_adsorption_free_energy_value','OH_adsorption_free_energy_unit','H2O_adsorption_energy_value','H2O_adsorption_energy_unit','deprotonation_energy_value','deprotonation_energy_unit','proton_transfer_barrier','proton_transfer_barrier_unit','Volmer_barrier','Volmer_barrier_unit','water_dissociation_barrier','water_dissociation_barrier_unit','coverage','coadsorbates','solvation_model','spin_state','DFT_functional','U_value','dispersion_correction','calculation_model','adsorption_configuration','reference_surface','adsorption_strength_trend','proton_transfer_pathway','rate_determining_step','descriptor_claim','system_match_score','qc_numeric_support','qc_field_support','extraction_source']
METRIC_VALUE_FIELDS = ['H_adsorption_energy_value','H_adsorption_free_energy_value','OH_adsorption_energy_value','OH_adsorption_free_energy_value','H2O_adsorption_energy_value','deprotonation_energy_value','proton_transfer_barrier','Volmer_barrier','water_dissociation_barrier']
UNIT_FIELDS = {f: f.replace('_value','_unit') for f in METRIC_VALUE_FIELDS if f + '_unit' in ADS_FIELDS}
# Special mappings because field names differ.
UNIT_FIELDS.update({'H2O_adsorption_energy_value': 'H2O_adsorption_energy_unit', 'proton_transfer_barrier': 'proton_transfer_barrier_unit', 'Volmer_barrier': 'Volmer_barrier_unit', 'water_dissociation_barrier': 'water_dissociation_barrier_unit'})

# Site descriptors that should be preferentially carried into metric-bearing
# canonical records. These are the fields that most often suffer from
# under-linking when the paper reports site descriptors and adsorption metrics
# in different paragraphs/tables.
DESCRIPTOR_FIELDS = [
    'ads_site_type','site_label','active_atom','neighbor_atoms','coordination_number',
    'local_geometry','vacancy_nearby','hydroxylated_surface','oxidation_state',
    'oxidation_state_raw','oxidation_state_element','oxidation_state_value',
    'bader_charge','d_band_center','charge_transfer_direction',
    'bridge_structure_nearby','ELF_descriptor','work_function','PZC',
    'hbond_acceptor_site','H2O_binding_mode','interfacial_water_role',
    'hydrogen_bond_network','strong_HB_water_signal','weak_HB_water_signal',
    'descriptor_claim',
]
NUMERIC_DESCRIPTOR_FIELDS = {'coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC','system_match_score'}


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


def uniq(xs: List[Any]) -> List[str]:
    out, seen = [], set()
    for x in xs:
        t = clean_scalar(x)
        if t and t not in seen:
            seen.add(t); out.append(t)
    return out


def num(x: Any) -> Optional[float]:
    return to_float(x)


def safe_num01(x: Any) -> Optional[float]:
    v = num(x)
    return v if v is not None and 0 <= v <= 1 else None


def load_triage_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for r in read_jsonl(path):
        pid = str(r.get('paper_id','')).strip()
        if pid and int(pd.to_numeric(r.get('should_extract'), errors='coerce') or 0) == 1:
            out[pid] = r
    return out


def load_grouped(path: Path, key_cols: Tuple[str,...]) -> Dict[Tuple[str,...], List[Dict[str, Any]]]:
    out = defaultdict(list)
    for r in read_jsonl(path):
        key = tuple(str(r.get(c,'')).strip() for c in key_cols)
        out[key].append(r)
    return out


def merge_evidence(*lists_: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged, seen = [], set()
    for evs in lists_:
        for e in normalize_evidence_list(evs):
            key = (e.get('chunk_id'), e.get('quote')[:180], e.get('page_start'))
            if key in seen: continue
            seen.add(key); merged.append(e)
    return merged[:40]



def evidence_chunk_ids(evs: Any) -> set:
    ids = set()
    for e in normalize_evidence_list(evs):
        cid = e.get('chunk_id')
        if cid is not None:
            ids.add(str(cid))
    return ids


def descriptor_presence_count(row: Dict[str, Any]) -> int:
    n = 0
    for f in DESCRIPTOR_FIELDS:
        if f == 'neighbor_atoms':
            if clean_listlike(row.get(f)):
                n += 1
        elif f in NUMERIC_DESCRIPTOR_FIELDS:
            if num(row.get(f)) is not None:
                n += 1
        elif clean_scalar(row.get(f)) is not None:
            n += 1
    return n


def has_ads_metric(row: Dict[str, Any]) -> bool:
    return any(num(row.get(f)) is not None for f in METRIC_VALUE_FIELDS)


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


def ads_identity_text(ads: Dict[str, Any]) -> str:
    return ' '.join(str(ads.get(k) or '') for k in [
        'site_local_id','adsorbate','adsorption_configuration','reference_surface',
        'adsorption_strength_trend','proton_transfer_pathway','rate_determining_step',
        'descriptor_claim','coverage','coadsorbates'
    ])


def site_ads_fuzzy_score(site: Dict[str, Any], ads: Dict[str, Any]) -> float:
    """Score whether a site descriptor row and an adsorption metric row describe the same local site.

    The score is intentionally conservative. Exact site IDs are handled outside. Here we combine
    text overlap, active-atom/configuration agreement, and evidence chunk overlap. The returned
    score is used only inside the same paper_id + system_local_id group.
    """
    score = 0.0
    stxt = site_identity_text(site)
    atxt = ads_identity_text(ads)
    stoks = tokens_for_match(stxt)
    atoks = tokens_for_match(atxt)
    if stoks and atoks:
        inter = len(stoks & atoks)
        score += min(0.25, 0.05 * inter)
    active = clean_scalar(site.get('active_atom'))
    if active and active.lower() in norm_for_match(atxt):
        score += 0.20
    slabel = clean_scalar(site.get('site_label'))
    if slabel and norm_for_match(slabel) and norm_for_match(slabel) in norm_for_match(atxt):
        score += 0.25
    stype = clean_scalar(site.get('ads_site_type'))
    if stype and norm_for_match(stype) and norm_for_match(stype) in norm_for_match(atxt):
        score += 0.15
    # Active atom + generic adsorption configuration often indicates a valid match.
    cfg = norm_for_match(ads.get('adsorption_configuration'))
    if active and cfg and active.lower() in cfg:
        score += 0.15
    # Evidence overlap is a strong signal when the same chunk produced both rows.
    sid = evidence_chunk_ids(site.get('evidence', []))
    aid = evidence_chunk_ids(ads.get('evidence', []))
    if sid and aid:
        overlap = len(sid & aid)
        if overlap:
            score += min(0.30, 0.12 * overlap)
    # Prefer sites that actually contain useful descriptors.
    score += min(0.15, 0.02 * descriptor_presence_count(site))
    return round(score, 4)


def best_site_for_ads(ads: Dict[str, Any], merged_sites: List[Dict[str, Any]], min_fuzzy_score: float = 0.26) -> Tuple[Optional[Dict[str, Any]], str, float]:
    """Link an adsorption row to the best site descriptor row with traceable method labels."""
    sid = clean_scalar(ads.get('site_local_id'))
    if sid:
        for s in merged_sites:
            if clean_scalar(s.get('site_local_id')) == sid:
                s = dict(s)
                s['descriptor_link_method'] = 'exact_site_id'
                s['descriptor_link_confidence'] = 1.0
                return s, 'exact_site_id', 1.0
    informative_sites = [s for s in merged_sites if descriptor_presence_count(s) > 0]
    if len(informative_sites) == 1:
        s = dict(informative_sites[0])
        s['descriptor_link_method'] = 'unique_system_site'
        s['descriptor_link_confidence'] = 0.82
        return s, 'unique_system_site', 0.82
    if informative_sites:
        scored = sorted(((site_ads_fuzzy_score(s, ads), s) for s in informative_sites), key=lambda x: x[0], reverse=True)
        best_score, best_site = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        # Require an absolute score and a small margin over the second best site.
        if best_score >= min_fuzzy_score and (best_score - second >= 0.04 or best_score >= 0.45):
            s = dict(best_site)
            conf = min(0.78, max(0.55, best_score))
            s['descriptor_link_method'] = 'fuzzy_site_match'
            s['descriptor_link_confidence'] = round(conf, 4)
            return s, 'fuzzy_site_match', round(conf, 4)
    return None, 'unlinked', 0.0


def record_conf(row: Dict[str, Any]) -> float:
    v = safe_num01(row.get('confidence_score'))
    return float(v) if v is not None else 0.0


def site_strength(r: Dict[str, Any]) -> float:
    s = 0.0
    for fld in ['site_label','ads_site_type','active_atom','local_geometry','vacancy_nearby','hydroxylated_surface','oxidation_state','charge_transfer_direction','bridge_structure_nearby','ELF_descriptor','hbond_acceptor_site','H2O_binding_mode','interfacial_water_role','hydrogen_bond_network','descriptor_claim']:
        if clean_scalar(r.get(fld)): s += 0.05
    for fld in ['coordination_number','bader_charge','d_band_center','work_function','PZC']:
        if num(r.get(fld)) is not None: s += 0.08
    if clean_listlike(r.get('neighbor_atoms')): s += 0.04
    s += min(0.08, 0.01 * len(normalize_evidence_list(r.get('evidence', []))))
    s += record_conf(r) * 0.05
    return round(s,4)


def ads_strength(r: Dict[str, Any]) -> float:
    s = 0.0
    for fld in METRIC_VALUE_FIELDS:
        if num(r.get(fld)) is not None: s += 0.12
    for fld in ['DFT_functional','coverage','solvation_model','spin_state','dispersion_correction','calculation_model','adsorption_configuration','reference_surface','adsorption_strength_trend','proton_transfer_pathway','rate_determining_step','descriptor_claim','system_match_score','qc_numeric_support','qc_field_support','extraction_source']:
        if clean_scalar(r.get(fld)): s += 0.04
    if clean_listlike(r.get('coadsorbates')): s += 0.03
    if num(r.get('U_value')) is not None: s += 0.03
    s += min(0.08, 0.01 * len(normalize_evidence_list(r.get('evidence', []))))
    s += record_conf(r) * 0.05
    return round(s,4)


def best_scalar(rows: List[Dict[str, Any]], field: str, strength_fn) -> Optional[str]:
    best, sc = None, -1.0
    for r in rows:
        val = clean_scalar(r.get(field))
        if val is not None and strength_fn(r) > sc:
            best, sc = val, strength_fn(r)
    return best


def best_num(rows: List[Dict[str, Any]], field: str, strength_fn) -> Optional[float]:
    best, sc = None, -1.0
    for r in rows:
        val = num(r.get(field))
        if val is not None and strength_fn(r) > sc:
            best, sc = val, strength_fn(r)
    return best


def site_group_key(rec: Dict[str, Any], ordinal: int) -> str:
    sid = clean_scalar(rec.get('site_local_id'))
    if sid: return sid
    parts = [clean_scalar(rec.get(k)) for k in ['site_label','ads_site_type','active_atom','oxidation_state','oxidation_state_value','vacancy_nearby','hydroxylated_surface','bridge_structure_nearby']]
    cn = num(rec.get('coordination_number'))
    if cn is not None: parts.append(f'CN{cn:g}')
    parts = [p for p in parts if p]
    return 'site::' + '__'.join(parts) if parts else f'site::fallback_{ordinal:03d}'


def merge_site_group(rows: List[Dict[str, Any]], fallback_site_id: str) -> Dict[str, Any]:
    base = max(rows, key=site_strength)
    merged = {'paper_id': clean_scalar(base.get('paper_id')), 'system_local_id': clean_scalar(base.get('system_local_id')), 'site_local_id': clean_scalar(base.get('site_local_id')) or fallback_site_id}
    for fld in SITE_FIELDS:
        if fld == 'neighbor_atoms':
            merged[fld] = uniq([x for r in rows for x in clean_listlike(r.get(fld))])
        elif fld in ['coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC','system_match_score']:
            merged[fld] = best_num(rows, fld, site_strength)
        else:
            merged[fld] = best_scalar(rows, fld, site_strength)
    merged['confidence_score'] = max([record_conf(r) for r in rows] + [0.0])
    merged['confidence_score_raw'] = best_scalar(rows, 'confidence_score_raw', site_strength)
    merged['confidence_score_status'] = best_scalar(rows, 'confidence_score_status', site_strength)
    merged['evidence'] = merge_evidence(*[r.get('evidence', []) for r in rows])
    merged['site_merge_count'] = len(rows)
    return merged


def ads_group_key(rec: Dict[str, Any], ordinal: int) -> str:
    parts = [clean_scalar(rec.get('site_local_id')) or 'site::unlinked', clean_scalar(rec.get('adsorbate')) or 'adsorbate::unknown']
    for fld in METRIC_VALUE_FIELDS:
        v = num(rec.get(fld))
        if v is not None: parts.append(f'{fld}:{v:.6f}')
    for fld in ['DFT_functional','coverage','reference_surface','adsorption_configuration','proton_transfer_pathway']:
        val = clean_scalar(rec.get(fld))
        if val: parts.append(f'{fld}:{val}')
    if len(parts) > 2:
        return 'ads::' + '__'.join(parts)
    return f'ads::{parts[0]}::fallback_{ordinal:03d}'


def merge_ads_group(rows: List[Dict[str, Any]], fallback_site_id: Optional[str]) -> Dict[str, Any]:
    base = max(rows, key=ads_strength)
    merged: Dict[str, Any] = {'paper_id': clean_scalar(base.get('paper_id')), 'system_local_id': clean_scalar(base.get('system_local_id')), 'site_local_id': clean_scalar(base.get('site_local_id')) or fallback_site_id}
    for fld in ADS_FIELDS:
        if fld == 'coadsorbates':
            merged[fld] = uniq([x for r in rows for x in clean_listlike(r.get(fld))])
        elif fld in METRIC_VALUE_FIELDS or fld in ['U_value','system_match_score']:
            merged[fld] = best_num(rows, fld, ads_strength)
        elif fld.endswith('_unit'):
            merged[fld] = clean_unit(best_scalar(rows, fld, ads_strength))
        else:
            merged[fld] = best_scalar(rows, fld, ads_strength)
    if not merged.get('adsorbate'):
        merged['adsorbate'] = 'H*'
    for fld in METRIC_VALUE_FIELDS:
        unit = UNIT_FIELDS.get(fld)
        if unit:
            merged[fld + '_eV'] = to_ev(merged.get(fld), merged.get(unit))
    merged['confidence_score'] = max([record_conf(r) for r in rows] + [0.0])
    merged['confidence_score_raw'] = best_scalar(rows, 'confidence_score_raw', ads_strength)
    merged['confidence_score_status'] = best_scalar(rows, 'confidence_score_status', ads_strength)
    merged['evidence'] = merge_evidence(*[r.get('evidence', []) for r in rows])
    merged['adsorption_merge_count'] = len(rows)
    return merged


def linked_site_for_ads(ads: Dict[str, Any], merged_sites: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # Backward-compatible wrapper. New code should use best_site_for_ads() so
    # that link method/confidence are propagated.
    site, _, _ = best_site_for_ads(ads, merged_sites)
    return site


def link_score(system_row: Dict[str, Any], site_row: Optional[Dict[str, Any]], ads_row: Optional[Dict[str, Any]]) -> float:
    score = 0.15
    if system_row: score += 0.20
    if site_row: score += 0.20
    if ads_row: score += 0.20
    if ads_row and any(num(ads_row.get(f)) is not None for f in METRIC_VALUE_FIELDS): score += 0.15
    if site_row and any(num(site_row.get(f)) is not None for f in ['d_band_center','bader_charge','work_function','PZC']): score += 0.05
    if site_row and clean_scalar(site_row.get('hydrogen_bond_network')): score += 0.03
    if system_row and clean_scalar(system_row.get('bridge_structure')): score += 0.02
    return round(min(1.0, score), 4)


def build_canonical(triage: Dict[str, Any], system: Dict[str, Any], site: Optional[Dict[str, Any]], ads: Optional[Dict[str, Any]], idx: int, descriptor_link_method: Optional[str] = None, descriptor_link_confidence: Optional[float] = None) -> Dict[str, Any]:
    site = site or {}
    ads = ads or {}
    rec: Dict[str, Any] = {
        'paper_id': clean_scalar(system.get('paper_id')) or clean_scalar(triage.get('paper_id')),
        'canonical_id': f"{clean_scalar(triage.get('paper_id'))}_{idx:04d}",
        'reaction': clean_scalar(triage.get('reaction')),
        'bucket': clean_scalar(triage.get('bucket')),
        'relpath': clean_scalar(triage.get('relpath')),
        'reaction_family': clean_scalar(triage.get('reaction_family')),
        'paper_type': clean_scalar(triage.get('paper_type')),
        'target_relevance': num(triage.get('target_relevance')),
        'system_signal': num(triage.get('system_signal')),
        'metric_signal': num(triage.get('metric_signal')),
        'mechanism_signal': num(triage.get('mechanism_signal')),
        'priority_hint': clean_scalar(triage.get('priority_hint')),
        'system_local_id': clean_scalar(system.get('system_local_id')),
        'site_local_id': clean_scalar(site.get('site_local_id')) or clean_scalar(ads.get('site_local_id')),
    }
    for fld in SYSTEM_FIELDS:
        rec[fld] = clean_scalar(system.get(fld)) if fld != 'neighbor_atoms' else clean_listlike(system.get(fld))
    for fld in SITE_FIELDS:
        out_name = 'site_descriptor_claim' if fld == 'descriptor_claim' else fld
        if fld == 'neighbor_atoms': rec[out_name] = clean_listlike(site.get(fld))
        elif fld in ['coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC','system_match_score']: rec[out_name] = num(site.get(fld))
        else: rec[out_name] = clean_scalar(site.get(fld))
    for fld in ADS_FIELDS:
        out_name = 'adsorption_descriptor_claim' if fld == 'descriptor_claim' else fld
        if fld == 'coadsorbates': rec[out_name] = clean_listlike(ads.get(fld))
        elif fld in METRIC_VALUE_FIELDS or fld in ['U_value','system_match_score']: rec[out_name] = num(ads.get(fld))
        elif fld.endswith('_unit'): rec[out_name] = clean_unit(ads.get(fld))
        else: rec[out_name] = clean_scalar(ads.get(fld))
    for fld in METRIC_VALUE_FIELDS:
        unit = UNIT_FIELDS.get(fld)
        if unit:
            rec[fld + '_eV'] = to_ev(rec.get(fld), rec.get(unit))
    rec['system_confidence_score'] = record_conf(system)
    rec['site_confidence_score'] = record_conf(site)
    rec['adsorption_confidence_score'] = record_conf(ads)
    # Quality-control metadata propagated from extraction steps.
    rec['site_system_match_score'] = num(site.get('system_match_score'))
    rec['adsorption_system_match_score'] = num(ads.get('system_match_score'))
    rec['site_qc_numeric_support'] = site.get('qc_numeric_support')
    rec['adsorption_qc_numeric_support'] = ads.get('qc_numeric_support')
    rec['site_qc_field_support'] = site.get('qc_field_support')
    rec['adsorption_qc_field_support'] = ads.get('qc_field_support')
    rec['site_extraction_source'] = clean_scalar(site.get('extraction_source'))
    rec['adsorption_extraction_source'] = clean_scalar(ads.get('extraction_source'))
    method = descriptor_link_method or clean_scalar(site.get('descriptor_link_method')) or ('site_only' if site and not ads else 'none')
    conf = descriptor_link_confidence
    if conf is None:
        conf = num(site.get('descriptor_link_confidence'))
    rec['descriptor_link_method'] = method
    rec['descriptor_link_confidence'] = conf
    rec['descriptor_link_is_soft'] = int(method in {'unique_system_site','fuzzy_site_match','repair_unique_system_site','repair_fuzzy_site_match','llm_repair'})
    rec['link_score'] = link_score(system, site if site else None, ads if ads else None)
    if conf is not None:
        # Keep score compatible with older scripts but gently reward successful descriptor linking.
        rec['link_score'] = round(min(1.0, float(rec['link_score'] or 0) + 0.05 * float(conf)), 4)
    rec['evidence_all'] = merge_evidence(system.get('evidence', []), site.get('evidence', []), ads.get('evidence', []), triage.get('evidence', []))
    return rec


def canonical_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    return tuple(rec.get(k) for k in ['paper_id','system_local_id','site_local_id','adsorbate','H_adsorption_energy_value_eV','H_adsorption_free_energy_value_eV','OH_adsorption_energy_value_eV','H2O_adsorption_energy_value_eV','deprotonation_energy_value_eV','Volmer_barrier_eV','proton_transfer_barrier_eV','DFT_functional','coverage','adsorption_configuration'])


def merge_duplicate_canonicals(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups = defaultdict(list)
    for r in rows:
        groups[canonical_key(r)].append(r)
    out = []
    for _, grp in groups.items():
        base = max(grp, key=lambda r: (num(r.get('link_score')) or 0, len(normalize_evidence_list(r.get('evidence_all', [])))))
        if len(grp) == 1:
            out.append(base); continue
        merged = dict(base)
        merged['neighbor_atoms'] = uniq([x for g in grp for x in clean_listlike(g.get('neighbor_atoms'))])
        merged['coadsorbates'] = uniq([x for g in grp for x in clean_listlike(g.get('coadsorbates'))])
        merged['evidence_all'] = merge_evidence(*[g.get('evidence_all', []) for g in grp])
        merged['link_score'] = max(num(g.get('link_score')) or 0 for g in grp)
        out.append(merged)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--triage', required=True); ap.add_argument('--systems', required=True); ap.add_argument('--sites', required=True); ap.add_argument('--adsorption', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--progress', default='outputs/progress_step3e_hads_link.json'); ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    triage_map = load_triage_map(Path(args.triage))
    systems_by_paper = load_grouped(Path(args.systems), ('paper_id',))
    sites_by_key = load_grouped(Path(args.sites), ('paper_id','system_local_id'))
    ads_by_key = load_grouped(Path(args.adsorption), ('paper_id','system_local_id'))
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress)
    done = set(load_progress(progress_path).get('done', [])) if progress_path.exists() and not args.force else set()
    if args.force and outp.exists(): outp.unlink()
    prog = {'done': sorted(done)}
    total = len(triage_map)
    processed = 0

    for paper_id, triage_row in triage_map.items():
        if paper_id in done: continue
        canonical_rows: List[Dict[str, Any]] = []
        idx = 1
        for sys_row in systems_by_paper.get((paper_id,), []):
            sys_id = str(sys_row.get('system_local_id','')).strip()
            if not sys_id: continue
            raw_sites = sites_by_key.get((paper_id, sys_id), [])
            raw_ads = ads_by_key.get((paper_id, sys_id), [])
            # Merge sites.
            site_groups = defaultdict(list)
            for i, s in enumerate(raw_sites, 1): site_groups[site_group_key(s, i)].append(s)
            merged_sites = [merge_site_group(grp, f'{sys_id}_site_{i:03d}') for i, grp in enumerate(site_groups.values(), 1)]
            # Merge ads rows.
            if len(merged_sites) == 1:
                for a in raw_ads:
                    if not clean_scalar(a.get('site_local_id')):
                        a['site_local_id'] = merged_sites[0].get('site_local_id')
            ads_groups = defaultdict(list)
            for i, a in enumerate(raw_ads, 1): ads_groups[ads_group_key(a, i)].append(a)
            merged_ads = [merge_ads_group(grp, clean_scalar(grp[0].get('site_local_id'))) for grp in ads_groups.values()]

            used_site_ids = set()
            if merged_ads:
                for a in merged_ads:
                    s, link_method, link_conf = best_site_for_ads(a, merged_sites)
                    if s: used_site_ids.add(clean_scalar(s.get('site_local_id')))
                    canonical_rows.append(build_canonical(triage_row, sys_row, s, a, idx, link_method, link_conf)); idx += 1
            if merged_sites:
                # Preserve site-only records for descriptors even when no adsorption metric was linked.
                for s in merged_sites:
                    sid = clean_scalar(s.get('site_local_id'))
                    if sid not in used_site_ids:
                        canonical_rows.append(build_canonical(triage_row, sys_row, s, None, idx)); idx += 1
            if not merged_ads and not merged_sites:
                canonical_rows.append(build_canonical(triage_row, sys_row, None, None, idx)); idx += 1
        canonical_rows = merge_duplicate_canonicals(canonical_rows)
        for r in canonical_rows:
            append_jsonl(r, outp)
        done.add(paper_id); prog['done'] = sorted(done); save_progress(prog, progress_path)
        processed += 1
        print(f'[OK] {paper_id} canonical_records={len(canonical_rows)} progress={processed}/{total}', flush=True)
    print(f'[DONE] wrote {args.out}', flush=True)

if __name__ == '__main__':
    main()
