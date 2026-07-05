#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4: normalize canonical H adsorption / deprotonation records into analysis tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.io_utils import ensure_dir, read_jsonl
from utils.normalize_utils import to_ev, clean_unit, to_float
from utils.text_utils import normalize_text
from utils.hads_validation import is_bridge_oxygen_motif

METRIC_VALUE_FIELDS = ['H_adsorption_energy_value','H_adsorption_free_energy_value','OH_adsorption_energy_value','OH_adsorption_free_energy_value','H2O_adsorption_energy_value','deprotonation_energy_value','proton_transfer_barrier','Volmer_barrier','water_dissociation_barrier']
UNIT_FIELDS = {
    'H_adsorption_energy_value': 'H_adsorption_energy_unit',
    'H_adsorption_free_energy_value': 'H_adsorption_free_energy_unit',
    'OH_adsorption_energy_value': 'OH_adsorption_energy_unit',
    'OH_adsorption_free_energy_value': 'OH_adsorption_free_energy_unit',
    'H2O_adsorption_energy_value': 'H2O_adsorption_energy_unit',
    'deprotonation_energy_value': 'deprotonation_energy_unit',
    'proton_transfer_barrier': 'proton_transfer_barrier_unit',
    'Volmer_barrier': 'Volmer_barrier_unit',
    'water_dissociation_barrier': 'water_dissociation_barrier_unit',
}
LIST_FIELDS = {'neighbor_atoms','coadsorbates','evidence_all','site_qc_numeric_support','adsorption_qc_numeric_support','site_qc_field_support','adsorption_qc_field_support','descriptor_repaired_fields'}

SYSTEM_COLS = ['paper_id','system_local_id','material_name','material_class','composition','surface_facet','termination','phase','support','dopant','interface_type','defect_type','bridge_structure','M_O_X_configuration','model_type','system_confidence_score']
SITE_COLS = ['paper_id','system_local_id','site_local_id','ads_site_type','site_label','active_atom','h_binding_anchor_atom','h_binding_site_label','neighbor_atoms','o_site_coordination_number','o_site_bader_charge','o_site_charge_transfer','o_h_bond_length','o_site_descriptor_claim','neighbor_metal_element','neighbor_metal_site_label','metal_coordination_number','metal_oxidation_state','metal_oxidation_state_raw','metal_oxidation_state_element','metal_bader_charge','metal_d_band_center','metal_o_distance','band_gap','descriptor_atom_scope','descriptor_assignment_confidence','descriptor_assignment_note','coordination_number','local_geometry','vacancy_nearby','hydroxylated_surface','oxidation_state','oxidation_state_raw','oxidation_state_element','oxidation_state_value','bader_charge','d_band_center','charge_transfer_direction','bridge_structure_nearby','ELF_descriptor','work_function','PZC','hbond_acceptor_site','H2O_binding_mode','interfacial_water_role','hydrogen_bond_network','strong_HB_water_signal','weak_HB_water_signal','site_descriptor_claim','site_confidence_score','site_system_match_score','site_extraction_source','site_qc_numeric_support','site_qc_field_support','descriptor_link_method','descriptor_link_confidence','descriptor_link_is_soft','descriptor_repair_method','descriptor_repair_confidence','descriptor_repair_source','descriptor_repaired_fields']
ADS_COLS = ['paper_id','system_local_id','site_local_id','adsorbate'] + METRIC_VALUE_FIELDS + [UNIT_FIELDS[f] for f in METRIC_VALUE_FIELDS] + [f + '_eV' for f in METRIC_VALUE_FIELDS] + ['coverage','coadsorbates','solvation_model','spin_state','DFT_functional','U_value','dispersion_correction','calculation_model','adsorption_configuration','reference_surface','adsorption_strength_trend','proton_transfer_pathway','rate_determining_step','adsorption_descriptor_claim','adsorption_confidence_score','adsorption_system_match_score','adsorption_extraction_source','adsorption_qc_numeric_support','adsorption_qc_field_support']
FEATURE_COLS = ['paper_id','canonical_id','reaction_family','paper_type','priority_hint','material_name','material_class','composition','surface_facet','support','dopant','interface_type','defect_type','bridge_structure','M_O_X_configuration','model_type','ads_site_type','site_label','active_atom','h_binding_anchor_atom','h_binding_site_label','neighbor_atoms','o_site_coordination_number','o_site_bader_charge','o_site_charge_transfer','o_h_bond_length','o_site_descriptor_claim','neighbor_metal_element','neighbor_metal_site_label','metal_coordination_number','metal_oxidation_state','metal_oxidation_state_raw','metal_oxidation_state_element','metal_bader_charge','metal_d_band_center','metal_o_distance','band_gap','descriptor_atom_scope','descriptor_assignment_confidence','descriptor_assignment_note','coordination_number','vacancy_nearby','hydroxylated_surface','oxidation_state','oxidation_state_raw','oxidation_state_element','oxidation_state_value','bader_charge','d_band_center','charge_transfer_direction','bridge_structure_nearby','ELF_descriptor','work_function','PZC','hbond_acceptor_site','H2O_binding_mode','interfacial_water_role','hydrogen_bond_network','strong_HB_water_signal','weak_HB_water_signal','adsorbate'] + [f + '_eV' for f in METRIC_VALUE_FIELDS] + ['coverage','coadsorbates','solvation_model','DFT_functional','U_value','dispersion_correction','calculation_model','adsorption_configuration','reference_surface','adsorption_strength_trend','proton_transfer_pathway','rate_determining_step','site_system_match_score','adsorption_system_match_score','site_extraction_source','adsorption_extraction_source','descriptor_link_method','descriptor_link_confidence','descriptor_link_is_soft','descriptor_repair_method','descriptor_repair_confidence','descriptor_repair_source','descriptor_repaired_fields','site_qc_numeric_support','adsorption_qc_numeric_support','site_qc_field_support','adsorption_qc_field_support','qc_has_h_binding_site_descriptor','qc_has_neighbor_metal_descriptor','qc_descriptor_atom_resolved','link_score','target_relevance','system_signal','metric_signal','mechanism_signal']


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    if not s or s.lower() in {'none','null','nan','unknown','not mentioned','n/a','na'}:
        return None
    return s


def list_to_json(x: Any) -> str:
    if x is None:
        return '[]'
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    else:
        arr = [p.strip() for p in str(x).replace('|',';').split(';') if p.strip()]
    return json.dumps(arr, ensure_ascii=False)



def support_count(x: Any, status: str = 'supported') -> int:
    if x is None:
        return 0
    if isinstance(x, dict):
        return sum(1 for v in x.values() if str(v) == status)
    try:
        obj = json.loads(x) if isinstance(x, str) and x.strip().startswith('{') else None
        if isinstance(obj, dict):
            return sum(1 for v in obj.values() if str(v) == status)
    except Exception:
        pass
    return 0


def support_has_dropped(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, dict):
        return int(any('dropped' in str(v) or 'unsupported' in str(v) for v in x.values()))
    try:
        obj = json.loads(x) if isinstance(x, str) and x.strip().startswith('{') else None
        if isinstance(obj, dict):
            return int(any('dropped' in str(v) or 'unsupported' in str(v) for v in obj.values()))
    except Exception:
        pass
    return 0

def normalize_record(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    for fld in METRIC_VALUE_FIELDS:
        unit_f = UNIT_FIELDS[fld]
        out[fld] = to_float(out.get(fld))
        out[unit_f] = clean_unit(out.get(unit_f))
        out[fld + '_eV'] = to_ev(out.get(fld), out.get(unit_f))
    for fld in ['coordination_number','oxidation_state_value','bader_charge','d_band_center','work_function','PZC','U_value','o_site_coordination_number','o_site_bader_charge','o_h_bond_length','metal_coordination_number','metal_oxidation_state','metal_bader_charge','metal_d_band_center','metal_o_distance','band_gap','link_score','target_relevance','system_signal','metric_signal','mechanism_signal','system_confidence_score','site_confidence_score','adsorption_confidence_score','site_system_match_score','adsorption_system_match_score','descriptor_link_confidence','descriptor_repair_confidence','descriptor_link_is_soft']:
        out[fld] = to_float(out.get(fld))
    for fld in LIST_FIELDS:
        out[fld] = out.get(fld, [])
    # Binary convenience flags.
    out['vacancy_flag_yes'] = int(clean_scalar(out.get('vacancy_nearby')) is not None or (clean_scalar(out.get('defect_type')) or '').lower().find('vacancy') >= 0)
    out['hydroxyl_flag_yes'] = int(clean_scalar(out.get('hydroxylated_surface')) is not None or (clean_scalar(out.get('termination')) or '').lower().find('oh') >= 0)
    out['bridge_oxygen_flag_yes'] = int(any(is_bridge_oxygen_motif(clean_scalar(out.get(k))) for k in ['bridge_structure','bridge_structure_nearby','M_O_X_configuration']))
    out['interfacial_water_flag_yes'] = int(any(clean_scalar(out.get(k)) for k in ['interfacial_water_role','hydrogen_bond_network','H2O_binding_mode','strong_HB_water_signal','weak_HB_water_signal']))
    out['has_H_metric'] = int(any(out.get(f + '_eV') is not None for f in ['H_adsorption_energy_value','H_adsorption_free_energy_value']))
    out['has_OH_H2O_metric'] = int(any(out.get(f + '_eV') is not None for f in ['OH_adsorption_energy_value','OH_adsorption_free_energy_value','H2O_adsorption_energy_value']))
    out['has_deprot_barrier_metric'] = int(any(out.get(f + '_eV') is not None for f in ['deprotonation_energy_value','proton_transfer_barrier','Volmer_barrier','water_dissociation_barrier']))
    out['qc_numeric_supported_count'] = support_count(out.get('adsorption_qc_numeric_support')) + support_count(out.get('site_qc_numeric_support'))
    out['qc_has_dropped_numeric'] = int(support_has_dropped(out.get('adsorption_qc_numeric_support')) or support_has_dropped(out.get('site_qc_numeric_support')))
    out['qc_system_match_ok'] = int((out.get('adsorption_system_match_score') is None or float(out.get('adsorption_system_match_score') or 0) >= 0.15) and (out.get('site_system_match_score') is None or float(out.get('site_system_match_score') or 0) >= 0.10))
    out['qc_keep_for_model'] = int((out['has_H_metric'] or out['has_OH_H2O_metric'] or out['has_deprot_barrier_metric']) and clean_scalar(out.get('material_name')) is not None and out['qc_system_match_ok'] == 1)
    out['qc_has_site'] = int(clean_scalar(out.get('site_local_id')) is not None or clean_scalar(out.get('site_label')) is not None or clean_scalar(out.get('active_atom')) is not None)
    out['qc_has_electronic_descriptor'] = int(any(out.get(k) is not None for k in ['oxidation_state_value','bader_charge','d_band_center','work_function','PZC','o_site_coordination_number','o_site_bader_charge','o_h_bond_length','metal_coordination_number','metal_oxidation_state','metal_bader_charge','metal_d_band_center','metal_o_distance','band_gap']) or clean_scalar(out.get('ELF_descriptor')) is not None)
    out['qc_has_mechanism_descriptor'] = int(out['bridge_oxygen_flag_yes'] or out['interfacial_water_flag_yes'] or clean_scalar(out.get('proton_transfer_pathway')) is not None)
    # DFT priority: evidence-driven, not a physical score.
    score = 0.0
    score += 2.0 * out['has_H_metric']
    score += 1.5 * out['has_OH_H2O_metric']
    score += 2.0 * out['has_deprot_barrier_metric']
    score += 1.2 * out['bridge_oxygen_flag_yes']
    score += 1.0 * out['interfacial_water_flag_yes']
    score += 1.0 * out['qc_has_electronic_descriptor']
    score += 0.8 * out['qc_has_site']
    score += min(1.0, float(out.get('link_score') or 0.0))
    out['DFT_priority_score'] = round(score, 3)
    return out


def to_dataframe(rows: List[Dict[str, Any]], cols: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols].copy()
    for c in df.columns:
        if c in LIST_FIELDS or c in {'neighbor_atoms','coadsorbates','evidence_all'}:
            df[c] = df[c].apply(list_to_json)
    return df


def dedup_table(df: pd.DataFrame, subset: List[str]) -> pd.DataFrame:
    if df.empty:
        return df
    keep = [c for c in subset if c in df.columns]
    if keep:
        return df.drop_duplicates(subset=keep, keep='first').reset_index(drop=True)
    return df.drop_duplicates().reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', required=True)
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    raw = read_jsonl(args.inp)
    rows = [normalize_record(r) for r in raw]

    paper_cols = ['paper_id','reaction_family','paper_type','priority_hint','target_relevance','system_signal','metric_signal','mechanism_signal','relpath','bucket']
    paper_table = to_dataframe(rows, paper_cols)
    paper_table = dedup_table(paper_table, ['paper_id'])

    system_table = to_dataframe(rows, SYSTEM_COLS)
    system_table = dedup_table(system_table, ['paper_id','system_local_id'])

    site_table = to_dataframe(rows, SITE_COLS)
    site_table = dedup_table(site_table, ['paper_id','system_local_id','site_local_id'])

    ads_table = to_dataframe(rows, ADS_COLS)
    ads_table = dedup_table(ads_table, ['paper_id','system_local_id','site_local_id','adsorbate'] + [f + '_eV' for f in METRIC_VALUE_FIELDS])

    extra_feature_cols = FEATURE_COLS + ['vacancy_flag_yes','hydroxyl_flag_yes','bridge_oxygen_flag_yes','interfacial_water_flag_yes','has_H_metric','has_OH_H2O_metric','has_deprot_barrier_metric','qc_keep_for_model','qc_has_site','qc_has_electronic_descriptor','qc_has_mechanism_descriptor','qc_numeric_supported_count','qc_has_dropped_numeric','qc_system_match_ok','DFT_priority_score']
    model_feature_table = to_dataframe(rows, extra_feature_cols)
    model_feature_table = dedup_table(model_feature_table, ['paper_id','canonical_id'])

    # Candidate table focused on actionable DFT follow-up.
    cand_cols = ['paper_id','canonical_id','material_name','surface_facet','dopant','defect_type','bridge_structure','M_O_X_configuration','active_atom','site_label','h_binding_anchor_atom','h_binding_site_label','neighbor_metal_element','adsorbate','H_adsorption_free_energy_value_eV','H_adsorption_energy_value_eV','OH_adsorption_energy_value_eV','H2O_adsorption_energy_value_eV','deprotonation_energy_value_eV','Volmer_barrier_eV','proton_transfer_barrier_eV','water_dissociation_barrier_eV','oxidation_state_value','d_band_center','bader_charge','o_site_coordination_number','o_site_bader_charge','o_h_bond_length','metal_coordination_number','metal_oxidation_state','metal_bader_charge','metal_d_band_center','metal_o_distance','band_gap','work_function','PZC','hydrogen_bond_network','proton_transfer_pathway','DFT_priority_score','link_score','descriptor_link_method','descriptor_link_confidence','descriptor_repair_method','descriptor_repair_confidence','qc_keep_for_model','qc_numeric_supported_count','qc_has_dropped_numeric','qc_system_match_ok','adsorption_system_match_score']
    dft_priority = to_dataframe(rows, cand_cols).sort_values(['DFT_priority_score','link_score'], ascending=False)

    system_summary = model_feature_table.groupby(['material_class','dopant','defect_type','bridge_structure'], dropna=False).agg(
        n_records=('paper_id','count'),
        n_papers=('paper_id','nunique'),
        mean_DFT_priority=('DFT_priority_score','mean'),
        n_H_metrics=('has_H_metric','sum'),
        n_deprot_metrics=('has_deprot_barrier_metric','sum'),
    ).reset_index() if not model_feature_table.empty else pd.DataFrame()

    qc_summary = pd.DataFrame([{
        'n_canonical_records': len(model_feature_table),
        'n_papers': model_feature_table['paper_id'].nunique() if not model_feature_table.empty else 0,
        'n_systems': len(system_table),
        'n_sites': len(site_table),
        'n_adsorption_rows': len(ads_table),
        'n_qc_keep_for_model': int(pd.to_numeric(model_feature_table.get('qc_keep_for_model', pd.Series(dtype=int)), errors='coerce').fillna(0).sum()) if not model_feature_table.empty else 0,
        'n_has_H_metric': int(pd.to_numeric(model_feature_table.get('has_H_metric', pd.Series(dtype=int)), errors='coerce').fillna(0).sum()) if not model_feature_table.empty else 0,
        'n_has_deprot_barrier_metric': int(pd.to_numeric(model_feature_table.get('has_deprot_barrier_metric', pd.Series(dtype=int)), errors='coerce').fillna(0).sum()) if not model_feature_table.empty else 0,
        'n_qc_system_match_ok': int(pd.to_numeric(model_feature_table.get('qc_system_match_ok', pd.Series(dtype=int)), errors='coerce').fillna(0).sum()) if not model_feature_table.empty else 0,
        'n_qc_has_dropped_numeric': int(pd.to_numeric(model_feature_table.get('qc_has_dropped_numeric', pd.Series(dtype=int)), errors='coerce').fillna(0).sum()) if not model_feature_table.empty else 0,
    }])

    paper_table.to_csv(outdir / 'paper_table.csv', index=False)
    system_table.to_csv(outdir / 'system_table.csv', index=False)
    site_table.to_csv(outdir / 'site_table.csv', index=False)
    ads_table.to_csv(outdir / 'adsorption_table.csv', index=False)
    model_feature_table.to_csv(outdir / 'model_feature_table.csv', index=False)
    system_summary.to_csv(outdir / 'system_summary_table.csv', index=False)
    qc_summary.to_csv(outdir / 'qc_database_summary.csv', index=False)
    dft_priority.to_csv(outdir / 'dft_priority_candidates.csv', index=False)

    print(f'[DONE] normalized database -> {outdir}')
    print(qc_summary.to_string(index=False))

if __name__ == '__main__':
    main()
