#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1 enhanced parser wrapper for H adsorption / deprotonation literature mining.

It runs the legacy PDF parser and then adds domain-specific metadata files:
- paper_domain_tags.csv
- section_manifest.csv
- table_header_semantics.csv
- enhanced_parse_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from utils.io_utils import ensure_dir, read_jsonl
from utils.text_utils import infer_section_type, match_table_semantics, normalize_text


KEYWORDS = {
    'H_adsorption': re.compile(r"\b(hydrogen adsorption|H adsorption|H\*|HBE|hydrogen binding energy|Delta\s*G[_\- ]?H|DG[_\- ]?H|Delta\s*E[_\- ]?H|DE[_\- ]?H|free energy of H adsorption)\b", re.I),
    'OH_water': re.compile(r"\b(OH adsorption|OHBE|OH\*|H2O adsorption|water adsorption|water dissociation|interfacial water|hydrogen[- ]bond network|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water|EDL)\b", re.I),
    'deprotonation': re.compile(r"\b(deprotonation|proton transfer|Volmer step|Heyrovsky step|Tafel step|PCET|proton[- ]coupled electron transfer|reaction barrier|activation barrier|NEB)\b", re.I),
    'electronic': re.compile(r"\b(d[- ]?band center|PDOS|DOS|Bader charge|charge density difference|electron density difference|ELF|electron localization function|work function|PZC|point of zero charge|oxidation state)\b", re.I),
    'system': re.compile(r"\b(catalyst|electrocatalyst|surface|facet|slab|support|dopant|doped|vacancy|defect|interface|single[- ]atom|nanoparticle|oxide|sulfide|nitride|carbide|phosphide|hydroxide|alloy|carbon[- ]coated|bridge oxygen|bridging oxygen|M[-–—]?O[-–—]?[A-Z][a-z]?)\b", re.I),
    'DFT': re.compile(r"\b(DFT|density functional theory|first[- ]principles|VASP|Quantum ESPRESSO|DMol3|PBE|RPBE|SCAN|HSE06|PBE\+U|DFT\+U|D3|D2|vdW|solvation)\b", re.I),
}


def run_legacy_parser(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(args.legacy_parser),
        '--root', args.root,
        '--out', args.out,
        '--grobid-url', args.grobid_url,
        '--table-engines', args.table_engines,
        '--workers', str(args.workers),
    ]
    if args.force:
        cmd.append('--force')
    print('[INFO] running legacy parser:', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def load_chunks(paper_dir: Path) -> List[Dict[str, Any]]:
    return read_jsonl(paper_dir / 'chunks.jsonl')


def sample_chunks_for_tagging(chunks: List[Dict[str, Any]], max_total: int = 140) -> List[Dict[str, Any]]:
    scored = []
    for i, c in enumerate(chunks):
        txt = normalize_text(c.get('text', ''))
        if not txt:
            continue
        ctype = normalize_text(c.get('chunk_type', 'text'))
        sec = infer_section_type(c.get('section_path', ''), txt)
        score = 0
        if ctype == 'table':
            score += 10
        elif ctype == 'caption':
            score += 6
        if sec == 'abstract':
            score += 6
        elif sec == 'results_discussion':
            score += 7
        elif sec == 'experimental':
            score += 5
        elif sec == 'conclusion':
            score += 3
        for pat in KEYWORDS.values():
            if pat.search(txt):
                score += 4
        if score > 0:
            scored.append((score, i, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [x[2] for x in scored[:max_total]]


def count_hits(text: str) -> Dict[str, int]:
    return {k: len(p.findall(text)) for k, p in KEYWORDS.items()}


def infer_reaction_family(text: str) -> str:
    if re.search(r'\bHOR\b|hydrogen oxidation', text, re.I):
        return 'HOR'
    if re.search(r'\bHER\b|hydrogen evolution', text, re.I):
        return 'HER'
    if re.search(r'water splitting', text, re.I):
        return 'water_splitting'
    if KEYWORDS['H_adsorption'].search(text):
        return 'H_adsorption'
    return 'unknown'


def infer_paper_type(text: str, bucket: str = '') -> str:
    low = f'{bucket} {text}'.lower()
    if re.search(r'\b(review|perspective|outlook|mini-review)\b', low):
        return 'review'
    has_dft = KEYWORDS['DFT'].search(text) is not None
    has_exp = re.search(r'\b(experimental|in[- ]situ|operando|RDE|XPS|XAS|TEM|XRD|SEIRAS|polarization|cyclic voltammetry)\b', text, re.I) is not None
    if has_dft and has_exp:
        return 'exp_theo'
    if has_dft:
        return 'theoretical'
    if has_exp:
        return 'experimental'
    return 'other'


def process_one_paper(paper_dir: Path, manifest_row: Dict[str, Any] | None = None) -> Dict[str, Any]:
    paper_id = paper_dir.name
    chunks = load_chunks(paper_dir)
    sampled = sample_chunks_for_tagging(chunks)
    text = normalize_text(' '.join(str(c.get('text', '')) for c in sampled))
    hits = count_hits(text)
    relpath = normalize_text((manifest_row or {}).get('relpath', ''))
    bucket = normalize_text((manifest_row or {}).get('bucket', ''))
    domain_score = 0.0
    domain_score += min(0.30, hits['H_adsorption'] * 0.06)
    domain_score += min(0.22, hits['deprotonation'] * 0.05)
    domain_score += min(0.18, hits['OH_water'] * 0.04)
    domain_score += min(0.16, hits['electronic'] * 0.03)
    domain_score += min(0.08, hits['system'] * 0.01)
    domain_score += min(0.06, hits['DFT'] * 0.01)
    domain_score = round(min(1.0, domain_score), 4)
    return {
        'paper_id': paper_id,
        'relpath': relpath,
        'bucket': bucket,
        'reaction_family': infer_reaction_family(text + ' ' + relpath),
        'paper_type': infer_paper_type(text, bucket),
        'domain_score': domain_score,
        'cue_has_H_adsorption': int(hits['H_adsorption'] > 0),
        'cue_has_OH_water': int(hits['OH_water'] > 0),
        'cue_has_deprotonation': int(hits['deprotonation'] > 0),
        'cue_has_electronic_descriptor': int(hits['electronic'] > 0),
        'cue_has_DFT': int(hits['DFT'] > 0),
        'cue_has_system': int(hits['system'] > 0),
        'hit_counts_json': json.dumps(hits, ensure_ascii=False),
        'n_chunks': len(chunks),
        'n_sampled_chunks': len(sampled),
    }


def build_section_manifest(out_root: Path) -> pd.DataFrame:
    rows = []
    for chunks_path in sorted((out_root / 'papers').glob('*/chunks.jsonl')):
        paper_id = chunks_path.parent.name
        for c in read_jsonl(chunks_path):
            text = normalize_text(c.get('text', ''))
            rows.append({
                'paper_id': paper_id,
                'chunk_id': c.get('chunk_id'),
                'chunk_type': c.get('chunk_type'),
                'section_path': c.get('section_path'),
                'section_type': infer_section_type(c.get('section_path', ''), text),
                'page_start': c.get('page_start'),
                'page_end': c.get('page_end'),
                'n_chars': len(text),
            })
    return pd.DataFrame(rows)


def build_table_semantics(out_root: Path) -> pd.DataFrame:
    rows = []
    for chunks_path in sorted((out_root / 'papers').glob('*/chunks.jsonl')):
        paper_id = chunks_path.parent.name
        for c in read_jsonl(chunks_path):
            if normalize_text(c.get('chunk_type')) != 'table':
                continue
            text = normalize_text(c.get('text', ''))
            sem = match_table_semantics(text)
            rows.append({
                'paper_id': paper_id,
                'chunk_id': c.get('chunk_id'),
                'page_start': c.get('page_start'),
                'page_end': c.get('page_end'),
                **{f'table_sem_{k}': v for k, v in sem.items()},
                'preview': text[:500],
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--grobid-url', default='http://127.0.0.1:8070')
    ap.add_argument('--table-engines', default='camelot')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--legacy-parser', default='./1_parse_corpus.py')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--skip-legacy', action='store_true', help='Only rebuild HADS metadata from existing parsed output')
    args = ap.parse_args()

    out_root = Path(args.out)
    ensure_dir(out_root)
    if not args.skip_legacy:
        run_legacy_parser(args)

    manifest_path = out_root / 'manifest.csv'
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
    else:
        manifest = pd.DataFrame({'paper_id': [p.name for p in sorted((out_root / 'papers').glob('*')) if p.is_dir()]})
    manifest_map = manifest.set_index('paper_id').to_dict(orient='index') if 'paper_id' in manifest.columns and not manifest.empty else {}

    paper_dirs = [p for p in sorted((out_root / 'papers').glob('*')) if p.is_dir()]
    rows: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_one_paper, p, manifest_map.get(p.name, {})): p for p in paper_dirs}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception as e:
                p = futs[fut]
                rows.append({'paper_id': p.name, 'domain_score': 0.0, 'paper_type': 'other', 'reaction_family': 'unknown', 'error': str(e)})
    tags = pd.DataFrame(rows).sort_values('paper_id') if rows else pd.DataFrame()
    tags.to_csv(out_root / 'paper_domain_tags.csv', index=False)

    section_df = build_section_manifest(out_root)
    section_df.to_csv(out_root / 'section_manifest.csv', index=False)
    table_df = build_table_semantics(out_root)
    table_df.to_csv(out_root / 'table_header_semantics.csv', index=False)

    summary = {
        'n_papers': int(len(tags)),
        'paper_type_counts': tags.get('paper_type', pd.Series(dtype=str)).fillna('unknown').astype(str).value_counts().to_dict() if not tags.empty else {},
        'reaction_family_counts': tags.get('reaction_family', pd.Series(dtype=str)).fillna('unknown').astype(str).value_counts().to_dict() if not tags.empty else {},
        'mean_domain_score': float(pd.to_numeric(tags.get('domain_score', pd.Series(dtype=float)), errors='coerce').mean()) if not tags.empty else 0.0,
        'n_sections': int(len(section_df)),
        'n_tables': int(len(table_df)),
    }
    (out_root / 'enhanced_parse_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[DONE] enhanced HADS parse metadata -> {out_root}')


if __name__ == '__main__':
    main()
