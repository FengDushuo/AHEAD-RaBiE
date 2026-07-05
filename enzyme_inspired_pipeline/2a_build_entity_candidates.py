#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 2a: build entity candidates for H adsorption / deprotonation literature mining.

This script creates an
entity view for Qdrant using sentences that contain catalyst, surface, adsorption,
electronic-structure, interfacial-water, proton-transfer, or DFT-model signals.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from utils.io_utils import write_jsonl
from utils.text_utils import infer_section_type, normalize_text, split_sentences

ENTITY_PATTERNS = [
    ('material_system', re.compile(r"\b(?:catalyst|electrocatalyst|photocatalyst|surface|substrate|support|slab|nanoparticle|nanosheet|nanotube|oxide|sulfide|nitride|carbide|phosphide|hydroxide|alloy|single[- ]atom|SAC|MOF|carbon[- ]coated|graphene|BNNT|boron nitride)\b|\b[A-Z][a-z]?(?:[A-Z][a-z]?)*(?:O\d?|S\d?|N\d?|P\d?|C\d?)*(?:\([0-9]{3}\))?\b", re.I)),
    ('surface_facet', re.compile(r"\b(?:facet|surface|plane|termination|terminated|\([0-9]{3}\)|\([0-9]{2,4}\))\b", re.I)),
    ('active_site', re.compile(r"\b(?:active site|adsorption site|top site|bridge site|hollow site|metal[- ]top|O[- ]top|oxygen[- ]top|interface site|coordinatively unsaturated|undercoordinated)\b", re.I)),
    ('dopant_defect', re.compile(r"\b(?:dopant|doped|P[- ]doped|N[- ]doped|S[- ]doped|heteroatom|oxygen vacancy|sulfur vacancy|nitrogen vacancy|vacancy|defect|edge site|hydroxylated|OH[- ]terminated)\b", re.I)),
    ('bridge_structure', re.compile(r"\b(?:bridging oxygen|bridge oxygen|M[-–—]?O[-–—]?[A-Z][a-z]?|Ni[-–—]?O[-–—]?[CP]|Fe[-–—]?O[-–—]?[CP]|Co[-–—]?O[-–—]?[CP]|M[-–—]?O[-–—]?C|M[-–—]?O[-–—]?P|P[-–—]?O[-–—]?Ni)\b", re.I)),
    ('h_adsorption_metric', re.compile(r"\b(?:hydrogen adsorption|H adsorption|H\*|HBE|hydrogen binding energy|Delta\s*G[_\- ]?H|Delta\s*E[_\- ]?H|DG[_\- ]?H|DE[_\- ]?H|adsorption free energy of hydrogen|free energy of H adsorption)\b", re.I)),
    ('oh_water_metric', re.compile(r"\b(?:OH adsorption|OHBE|OH\*|hydroxyl binding|H2O adsorption|water adsorption|water dissociation|interfacial water|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water|hydrogen[- ]bond network|H[- ]bond network)\b", re.I)),
    ('proton_transfer', re.compile(r"\b(?:deprotonation|protonation|proton transfer|Volmer step|Heyrovsky step|Tafel step|PCET|proton[- ]coupled electron transfer|reaction barrier|activation barrier|NEB|transition state)\b", re.I)),
    ('electronic_descriptor', re.compile(r"\b(?:d[- ]?band center|PDOS|DOS|Bader charge|charge transfer|charge density difference|electron density difference|ELF|electron localization function|work function|PZC|point of zero charge|oxidation state|valence state)\b", re.I)),
    ('dft_method', re.compile(r"\b(?:DFT|density functional theory|first[- ]principles|VASP|Quantum ESPRESSO|DMol3|PBE|RPBE|SCAN|HSE06|PBE\+U|DFT\+U|D3|D2|vdW|solvation|VASPsol|implicit solvent|explicit water|coverage)\b", re.I)),
]

PRIORITY_TYPES = {
    'h_adsorption_metric', 'oh_water_metric', 'proton_transfer', 'electronic_descriptor',
    'bridge_structure', 'dopant_defect', 'active_site', 'dft_method'
}


def infer_entity_type(sent: str) -> str:
    for name, pat in ENTITY_PATTERNS:
        if pat.search(sent):
            return name
    return 'other'


def extract_candidate_phrases(text: str, section_type: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sent in split_sentences(text):
        sent = normalize_text(sent)
        if not sent or len(sent) < 18:
            continue
        etype = infer_entity_type(sent)
        if etype != 'other' or section_type in {'table', 'caption', 'results_discussion', 'experimental', 'abstract'}:
            priority = 2 if etype in PRIORITY_TYPES else 1 if etype != 'other' else 0
            if priority > 0 or len(sent) <= 380:
                out.append({'entity_type': etype, 'text': sent[:700], 'priority': priority})
    return out


def load_chunks(parsed_root: Path):
    for fn in sorted(parsed_root.glob('papers/*/chunks.jsonl')):
        paper_id = fn.parent.name
        with fn.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                row['paper_id'] = str(row.get('paper_id') or paper_id)
                yield row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--parsed-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-per-paper', type=int, default=220)
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    rows: List[Dict[str, Any]] = []
    per_paper: Dict[str, int] = {}
    seen = set()
    for ch in load_chunks(parsed_root):
        paper_id = str(ch.get('paper_id', '')).strip()
        if not paper_id:
            continue
        if per_paper.get(paper_id, 0) >= args.max_per_paper:
            continue
        text = normalize_text(ch.get('text', ''))
        if not text:
            continue
        ctype = normalize_text(ch.get('chunk_type') or 'text')
        sec = infer_section_type(ch.get('section_path', ''), text)
        if ctype in {'table', 'caption'}:
            # Tables/captions are highly informative, preserve compact full block too.
            etype = infer_entity_type(text)
            if etype != 'other' or len(text) >= 40:
                key = (paper_id, ch.get('chunk_id'), text[:200])
                if key not in seen:
                    seen.add(key)
                    rows.append({
                        'paper_id': paper_id,
                        'chunk_id': ch.get('chunk_id'),
                        'chunk_type': ctype,
                        'section_type': sec,
                        'page_start': ch.get('page_start'),
                        'page_end': ch.get('page_end'),
                        'entity_type': etype,
                        'priority': 2 if etype in PRIORITY_TYPES else 1,
                        'text': text[:900],
                    })
                    per_paper[paper_id] = per_paper.get(paper_id, 0) + 1
        for cand in extract_candidate_phrases(text, sec):
            if per_paper.get(paper_id, 0) >= args.max_per_paper:
                break
            key = (paper_id, cand['entity_type'], cand['text'][:240])
            if key in seen:
                continue
            seen.add(key)
            rows.append({
                'paper_id': paper_id,
                'chunk_id': ch.get('chunk_id'),
                'chunk_type': ctype,
                'section_type': sec,
                'page_start': ch.get('page_start'),
                'page_end': ch.get('page_end'),
                **cand,
            })
            per_paper[paper_id] = per_paper.get(paper_id, 0) + 1
    # Higher-priority entities first improves retrieval if rows are later sampled.
    rows.sort(key=lambda r: (r.get('paper_id', ''), -int(r.get('priority', 0)), r.get('entity_type', ''), r.get('chunk_id') or ''))
    write_jsonl(rows, Path(args.out))
    print(f'[DONE] entity candidates -> {args.out} rows={len(rows)} papers={len(per_paper)}')


if __name__ == '__main__':
    main()
