#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 3a: triage papers for H adsorption / deprotonation feature extraction.

Outputs one row per paper with four scores:
- target_relevance: H adsorption / deprotonation / electrocatalysis relevance
- system_signal: concrete catalyst/surface/site information
- metric_signal: numerical DFT/adsorption/barrier/descriptor evidence
- mechanism_signal: proton transfer, interfacial water, electronic mechanism clues
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.io_utils import read_jsonl, write_jsonl
from utils.text_utils import normalize_text, clip_text

# --- domain patterns -------------------------------------------------------
RE_REVIEW = re.compile(r"\b(review|perspective|outlook|mini-review|tutorial review)\b", re.I)
RE_REACTION = re.compile(r"\b(HER|hydrogen evolution reaction|HOR|hydrogen oxidation reaction|hydrogen adsorption|alkaline hydrogen oxidation|water splitting|electrocatalysis|photocatalysis)\b", re.I)
RE_HADS = re.compile(r"\b(hydrogen adsorption|H adsorption|H\*|HBE|hydrogen binding energy|adsorbed hydrogen|Delta\s*G[_\- ]?H\*?|DG[_\- ]?H\*?|Delta\s*E[_\- ]?H|DE[_\- ]?H|free energy of H adsorption)\b", re.I)
RE_DEPROT = re.compile(r"\b(deprotonation|proton transfer|protonation|Volmer step|Heyrovsky step|Tafel step|PCET|proton[- ]coupled electron transfer|water dissociation|hydroxyl transfer)\b", re.I)
RE_OH_H2O = re.compile(r"\b(OH adsorption|OHBE|OH\*|hydroxyl binding|H2O adsorption|water adsorption|interfacial water|hydrogen[- ]bond network|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water|EDL|electrical double layer)\b", re.I)
RE_SYSTEM = re.compile(r"\b(catalyst|electrocatalyst|surface|facet|slab|support|substrate|dopant|doped|vacancy|defect|interface|heterostructure|single[- ]atom|SAC|nanoparticle|nanosheet|nanotube|oxide|sulfide|nitride|carbide|phosphide|hydroxide|alloy|carbon[- ]coated)\b", re.I)
RE_SITE = re.compile(r"\b(active site|adsorption site|top site|bridge site|hollow site|metal[- ]top|O[- ]top|oxygen[- ]top|coordinat(?:e|ion)|undercoordinated|M[-–—]?O[-–—]?[A-Z][a-z]?|Ni[-–—]?O[-–—]?[CP]|bridge oxygen|bridging oxygen)\b", re.I)
RE_METRIC = re.compile(r"\b(eV|kJ\s*mol|kcal\s*mol|adsorption energy|free energy|reaction barrier|activation barrier|energy barrier|overpotential|exchange current|mass activity|Tafel|j0|ECSA|binding energy)\b", re.I)
RE_NUMERIC_ENERGY = re.compile(r"[-+−–]?\d+(?:\.\d+)?\s*(?:eV|kJ\s*mol[-−–]?[1l]?|kcal\s*mol[-−–]?[1l]?)", re.I)
RE_ELECTRONIC = re.compile(r"\b(d[- ]?band center|PDOS|DOS|Bader charge|charge transfer|charge density difference|electron density difference|ELF|electron localization function|work function|PZC|point of zero charge|oxidation state|valence state|XPS|XAS|EXAFS|XANES)\b", re.I)
RE_THEORY = re.compile(r"\b(DFT|density functional theory|first[- ]principles|calculated|computed|simulation|NEB|transition state|VASP|Quantum ESPRESSO|DMol3|PBE|RPBE|SCAN|HSE06|PBE\+U|DFT\+U)\b", re.I)
RE_EXP = re.compile(r"\b(experimental|measured|in[- ]situ|operando|SEIRAS|infrared|RDE|polarization|cyclic voltammetry|XPS|XAS|TEM|XRD)\b", re.I)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = pd.to_numeric(x, errors='coerce')
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def first_present_num(row: Dict[str, Any], keys: List[str], default: float = 0.0) -> float:
    for k in keys:
        if k in row:
            v = safe_float(row.get(k), default=None)  # type: ignore[arg-type]
            if v is not None:
                return float(v)
    return default


def load_chunks_for_paper(parsed_root: Path, paper_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(parsed_root / 'papers' / paper_id / 'chunks.jsonl')


def aggregate_text(chunks: List[Dict[str, Any]], max_chars: int = 35000) -> str:
    priority = {'table': 0, 'caption': 1, 'text': 2}
    chunks = sorted(chunks, key=lambda c: (priority.get(normalize_text(c.get('chunk_type')), 9), normalize_text(c.get('chunk_id'))))
    parts: List[str] = []
    used = 0
    for ch in chunks:
        txt = normalize_text(ch.get('text', ''))
        if not txt:
            continue
        block = txt + '\n'
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return ''.join(parts)


def infer_reaction_family(text: str) -> str:
    low = text.lower()
    if re.search(r'\bHOR\b|hydrogen oxidation', text, re.I):
        return 'HOR'
    if re.search(r'\bHER\b|hydrogen evolution', text, re.I):
        return 'HER'
    if 'water splitting' in low:
        return 'water_splitting'
    if 'hydrogen adsorption' in low or 'h adsorption' in low:
        return 'H_adsorption'
    return 'unknown'


def infer_paper_type(meta_type: Optional[str], bucket: str, text: str) -> str:
    mt = normalize_text(meta_type).lower()
    if mt in {'experimental', 'theoretical', 'exp_theo', 'review', 'other'}:
        return mt
    if RE_REVIEW.search(text) or 'review' in normalize_text(bucket).lower():
        return 'review'
    has_exp = bool(RE_EXP.search(text))
    has_theo = bool(RE_THEORY.search(text))
    if has_exp and has_theo:
        return 'exp_theo'
    if has_theo:
        return 'theoretical'
    if has_exp:
        return 'experimental'
    return 'other'


def score_target_relevance(text: str) -> Tuple[float, Dict[str, int]]:
    counts = {
        'reaction': len(RE_REACTION.findall(text)),
        'h_adsorption': len(RE_HADS.findall(text)),
        'deprotonation': len(RE_DEPROT.findall(text)),
        'oh_water': len(RE_OH_H2O.findall(text)),
        'theory': len(RE_THEORY.findall(text)),
    }
    score = 0.0
    score += min(0.24, 0.05 * counts['reaction'])
    score += min(0.32, 0.08 * counts['h_adsorption'])
    score += min(0.22, 0.06 * counts['deprotonation'])
    score += min(0.14, 0.04 * counts['oh_water'])
    score += min(0.08, 0.02 * counts['theory'])
    return round(min(1.0, score), 4), counts


def score_system_signal(text: str) -> Tuple[float, Dict[str, int]]:
    counts = {
        'system': len(RE_SYSTEM.findall(text)),
        'site': len(RE_SITE.findall(text)),
        'surface_formula': len(re.findall(r'\b[A-Z][a-z]?(?:[A-Z][a-z]?)*(?:O\d?|S\d?|N\d?|P\d?|C\d?)+(?:\([0-9]{3}\))?\b', text)),
    }
    score = 0.0
    score += min(0.42, 0.04 * counts['system'])
    score += min(0.36, 0.07 * counts['site'])
    score += min(0.22, 0.02 * counts['surface_formula'])
    return round(min(1.0, score), 4), counts


def score_metric_signal(text: str) -> Tuple[float, Dict[str, int]]:
    counts = {
        'metric': len(RE_METRIC.findall(text)),
        'numeric_energy': len(RE_NUMERIC_ENERGY.findall(text)),
        'h_energy': len(RE_HADS.findall(text)),
        'oh_water': len(RE_OH_H2O.findall(text)),
        'electronic': len(RE_ELECTRONIC.findall(text)),
    }
    score = 0.0
    score += min(0.26, 0.04 * counts['metric'])
    score += min(0.28, 0.05 * counts['numeric_energy'])
    score += min(0.18, 0.04 * counts['h_energy'])
    score += min(0.12, 0.03 * counts['oh_water'])
    score += min(0.16, 0.03 * counts['electronic'])
    return round(min(1.0, score), 4), counts


def score_mechanism_signal(text: str) -> Tuple[float, Dict[str, int]]:
    counts = {
        'deprotonation': len(RE_DEPROT.findall(text)),
        'oh_water': len(RE_OH_H2O.findall(text)),
        'electronic': len(RE_ELECTRONIC.findall(text)),
        'site': len(RE_SITE.findall(text)),
    }
    score = 0.0
    score += min(0.30, 0.07 * counts['deprotonation'])
    score += min(0.25, 0.05 * counts['oh_water'])
    score += min(0.30, 0.04 * counts['electronic'])
    score += min(0.15, 0.03 * counts['site'])
    return round(min(1.0, score), 4), counts


def select_evidence(chunks: List[Dict[str, Any]], max_items: int = 10) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for ch in chunks:
        txt = normalize_text(ch.get('text'))
        if not txt:
            continue
        s = 0
        for pat, weight in [(RE_HADS, 5), (RE_DEPROT, 5), (RE_OH_H2O, 4), (RE_ELECTRONIC, 4), (RE_SITE, 3), (RE_SYSTEM, 2), (RE_METRIC, 2), (RE_THEORY, 1)]:
            if pat.search(txt):
                s += weight
        if RE_NUMERIC_ENERGY.search(txt):
            s += 3
        if ch.get('chunk_type') == 'table':
            s += 4
        elif ch.get('chunk_type') == 'caption':
            s += 2
        if s > 0:
            scored.append((s, ch))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: List[Dict[str, Any]] = []
    seen = set()
    for score, ch in scored:
        cid = normalize_text(ch.get('chunk_id'))
        quote = clip_text(ch.get('text', ''), 450)
        key = (cid, quote[:160])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'chunk_id': cid,
            'chunk_type': normalize_text(ch.get('chunk_type')),
            'source': normalize_text(ch.get('source')),
            'section_path': normalize_text(ch.get('section_path')),
            'page_start': ch.get('page_start'),
            'page_end': ch.get('page_end'),
            'quote': quote,
            'score': score,
        })
        if len(out) >= max_items:
            break
    return out


def dedup_df_by_paper_id(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if 'paper_id' not in df.columns:
        return df
    df = df.copy()
    df['paper_id'] = df['paper_id'].astype(str).str.strip()
    df = df[df['paper_id'].ne('')].copy()
    dup_n = int(df['paper_id'].duplicated().sum())
    if dup_n > 0:
        print(f'[WARN] {name} has duplicate paper_id rows: {dup_n}, keeping last occurrence')
    return df.drop_duplicates(subset=['paper_id'], keep='last')


def triage_one(row: Dict[str, Any], parsed_root: Path, tags_map: Dict[str, Dict[str, Any]], qc_map: Dict[str, Dict[str, Any]], min_relevance: float, min_system_signal: float, min_metric_signal: float) -> Dict[str, Any]:
    paper_id = normalize_text(row.get('paper_id'))
    reaction = normalize_text(row.get('reaction')) or 'UNKNOWN'
    bucket = normalize_text(row.get('bucket')) or 'UNKNOWN'
    relpath = normalize_text(row.get('relpath'))
    chunks = load_chunks_for_paper(parsed_root, paper_id)
    if not chunks:
        return {'paper_id': paper_id, 'reaction': reaction, 'bucket': bucket, 'relpath': relpath, 'paper_type': 'other', 'reaction_family': 'unknown', 'target_relevance': 0.0, 'system_signal': 0.0, 'metric_signal': 0.0, 'mechanism_signal': 0.0, 'has_target_systems': 0, 'has_target_metrics': 0, 'has_mechanism_clues': 0, 'should_extract': 0, 'priority_hint': 'skip', 'reason': 'no_chunks', 'evidence': []}
    text_all = aggregate_text(chunks)
    tag_row = tags_map.get(paper_id, {})
    qc_row = qc_map.get(paper_id, {})
    reaction_family = normalize_text(tag_row.get('reaction_family')) or infer_reaction_family(text_all)
    paper_type = infer_paper_type(tag_row.get('paper_type'), bucket, text_all)

    target_relevance, rel_counts = score_target_relevance(text_all)
    system_signal, sys_counts = score_system_signal(text_all)
    metric_signal, met_counts = score_metric_signal(text_all)
    mechanism_signal, mech_counts = score_mechanism_signal(text_all)

    n_tables = safe_float(qc_row.get('n_tables'), 0)
    if n_tables > 0:
        metric_signal = round(min(1.0, metric_signal + 0.04), 4)
    # Optional parser cues from Step1 if present.
    for keys, add_to in [
        (['cue_has_h_adsorption', 'cue_H_adsorption', 'has_H_adsorption'], 'target'),
        (['cue_has_deprotonation', 'cue_deprotonation', 'has_deprotonation'], 'mechanism'),
        (['cue_has_interfacial_water', 'cue_water', 'has_interfacial_water'], 'mechanism'),
        (['cue_has_dft', 'cue_DFT', 'has_DFT'], 'metric'),
    ]:
        if first_present_num(qc_row, keys, 0.0) > 0:
            if add_to == 'target':
                target_relevance = round(min(1.0, target_relevance + 0.05), 4)
            elif add_to == 'mechanism':
                mechanism_signal = round(min(1.0, mechanism_signal + 0.05), 4)
            elif add_to == 'metric':
                metric_signal = round(min(1.0, metric_signal + 0.04), 4)

    has_target_systems = int(system_signal >= min_system_signal)
    has_target_metrics = int(metric_signal >= min_metric_signal)
    has_mechanism_clues = int(mechanism_signal >= 0.18)

    if paper_type == 'review':
        # Reviews can be useful for background but not direct record extraction.
        should_extract = 0
        priority_hint = 'review_only'
        reason = 'review'
    else:
        should_extract = int(target_relevance >= min_relevance and (has_target_systems or has_target_metrics or has_mechanism_clues))
        if should_extract:
            if has_target_systems and has_target_metrics and has_mechanism_clues:
                priority_hint = 'high'
            elif has_target_metrics and has_mechanism_clues:
                priority_hint = 'mechanism_metrics'
            elif has_target_metrics:
                priority_hint = 'metrics_first'
            elif has_mechanism_clues:
                priority_hint = 'mechanism_first'
            else:
                priority_hint = 'systems_first'
            reason = 'pass'
        else:
            priority_hint = 'low'
            reason = 'below_threshold'

    evidence = select_evidence(chunks, max_items=10)
    return {
        'paper_id': paper_id,
        'reaction': reaction,
        'bucket': bucket,
        'relpath': relpath,
        'paper_type': paper_type,
        'reaction_family': reaction_family,
        'target_relevance': target_relevance,
        'system_signal': system_signal,
        'metric_signal': metric_signal,
        'mechanism_signal': mechanism_signal,
        'has_target_systems': has_target_systems,
        'has_target_metrics': has_target_metrics,
        'has_mechanism_clues': has_mechanism_clues,
        'should_extract': should_extract,
        'priority_hint': priority_hint,
        'reason': reason,
        'n_chunks': len(chunks),
        'evidence': evidence,
        'debug_counts': {'relevance': rel_counts, 'systems': sys_counts, 'metrics': met_counts, 'mechanism': mech_counts},
    }


def build_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {'n_rows': 0}
    df = pd.DataFrame(rows)
    return {
        'n_rows': int(len(df)),
        'n_should_extract': int(pd.to_numeric(df['should_extract'], errors='coerce').fillna(0).sum()),
        'n_has_target_systems': int(pd.to_numeric(df['has_target_systems'], errors='coerce').fillna(0).sum()),
        'n_has_target_metrics': int(pd.to_numeric(df['has_target_metrics'], errors='coerce').fillna(0).sum()),
        'n_has_mechanism_clues': int(pd.to_numeric(df['has_mechanism_clues'], errors='coerce').fillna(0).sum()),
        'paper_type_counts': df['paper_type'].fillna('unknown').astype(str).value_counts().to_dict(),
        'reaction_family_counts': df['reaction_family'].fillna('unknown').astype(str).value_counts().to_dict(),
        'priority_hint_counts': df['priority_hint'].fillna('unknown').astype(str).value_counts().to_dict(),
        'mean_target_relevance': round(float(pd.to_numeric(df['target_relevance'], errors='coerce').mean()), 4),
        'mean_system_signal': round(float(pd.to_numeric(df['system_signal'], errors='coerce').mean()), 4),
        'mean_metric_signal': round(float(pd.to_numeric(df['metric_signal'], errors='coerce').mean()), 4),
        'mean_mechanism_signal': round(float(pd.to_numeric(df['mechanism_signal'], errors='coerce').mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--parsed-root', required=True)
    ap.add_argument('--paper-tags', required=True)
    ap.add_argument('--qc-report', required=True)
    ap.add_argument('--out-jsonl', required=True)
    ap.add_argument('--out-csv', required=True)
    ap.add_argument('--out-summary', required=True)
    ap.add_argument('--min-relevance', type=float, default=0.18)
    ap.add_argument('--min-system-signal', type=float, default=0.16)
    ap.add_argument('--min-metric-signal', type=float, default=0.12)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    paper_tags = dedup_df_by_paper_id(pd.read_csv(args.paper_tags), 'paper_tags') if Path(args.paper_tags).exists() else pd.DataFrame()
    qc_report = dedup_df_by_paper_id(pd.read_csv(args.qc_report), 'qc_report') if Path(args.qc_report).exists() else pd.DataFrame()
    tags_map = paper_tags.set_index('paper_id').to_dict(orient='index') if not paper_tags.empty and 'paper_id' in paper_tags.columns else {}
    qc_map = qc_report.set_index('paper_id').to_dict(orient='index') if not qc_report.empty and 'paper_id' in qc_report.columns else {}
    rows = []
    parsed_root = Path(args.parsed_root)
    for _, row in manifest.iterrows():
        rows.append(triage_one(row.to_dict(), parsed_root, tags_map, qc_map, args.min_relevance, args.min_system_signal, args.min_metric_signal))
    write_jsonl(rows, Path(args.out_jsonl))
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    Path(args.out_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_summary).write_text(json.dumps(build_summary(rows), ensure_ascii=False, indent=2), encoding='utf-8')
    print('[DONE] triage ->', args.out_jsonl)


if __name__ == '__main__':
    main()
