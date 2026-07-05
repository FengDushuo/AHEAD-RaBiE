#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3b: extract catalyst/material/surface systems for H adsorption and deprotonation mining."""
from __future__ import annotations

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.evidence_utils import dedup_evidence, normalize_evidence_list, pack_evidence
from utils.io_utils import append_jsonl, load_progress, read_jsonl, save_progress
from utils.llm_client import MultiEndpointClient, safe_json_extract
from utils.extraction_verifier import verify_extraction_json
from utils.text_utils import clip_text, normalize_text
from utils.hads_validation import clean_material_name, clean_bridge, is_bad_material_name, packed_evidence_text, field_support_summary

SEARCH_LOCK = threading.Lock()

QUERY_MAIN = "catalyst material surface facet support dopant defect vacancy bridge oxygen M-O-C M-O-P hydrogen adsorption deprotonation Volmer DFT"
QUERY_TABLE = "table catalyst surface facet dopant defect support adsorption energy DFT bridge oxygen"
QUERY_ENTITY = "material system catalyst surface active site dopant vacancy bridge structure"

SYSTEM_PROMPT = """
You are extracting MATERIAL / SURFACE SYSTEM records for hydrogen adsorption and deprotonation literature mining.
Use ONLY the provided evidence. Output STRICT JSON only.
If unsupported, return null or [].

Return JSON with keys: paper_id, systems[].
Each system in systems[] must contain:
system_local_id, material_name, material_class, composition, surface_facet, termination, phase, support, dopant, interface_type, defect_type, bridge_structure, M_O_X_configuration, model_type, confidence_score, evidence[].

Rules:
- Extract catalyst/surface models from the target article only, not methods, descriptors, equations, characterization names, or background citations.
- material_name must be a concrete material/catalyst/model such as Ni@PC, Ni(111), Pt/C, CeO2(111), M-O-P interface, not PBE, DFT, HBE, KOH, H2O, Volmer, Figure, or a generic phrase.
- Keep bridge structures such as Ni-O-P, Ni-O-C, M-O-C/P, bridging oxygen, and metal-support interface explicitly, but do not call H-O-H, C-O-C, O-H or H2O a bridge_structure.
- Return one record per distinct catalyst/surface/model.
- Every non-null field must be directly supported by one of the evidence blocks.
- Do not invent unsupported fields; use null when not stated.
- confidence_score must be float in [0,1] or null.
""".strip()

# Common system / structure patterns.
RE_FACET = re.compile(r"\([0-9]{3,4}\)|\b(?:facet|surface|plane)\s*(?:of\s*)?\(?([0-9]{3,4})\)?", re.I)
RE_DOPANT = re.compile(r"\b([A-Z][a-z]?)\s*[- ]?(?:doped|doping)|\b(?:doped with|dopant(?:s)?[:=]?)\s*([A-Z][a-z]?)", re.I)
RE_DEFECT = re.compile(r"\b(oxygen vacancy|sulfur vacancy|nitrogen vacancy|carbon vacancy|vacancy|defect|edge defect|unsaturated site)\b", re.I)
RE_SUPPORT = re.compile(r"\b(?:on|supported on|loaded on|anchored on|confined in|coated by|coated with)\s+([A-Z][A-Za-z0-9@/\-() ]{1,50}?)(?:\s+surface|\s+support|[,.;&)]|$)", re.I)
RE_INTERFACE = re.compile(r"\b(heterointerface|interface|metal[- ]support interaction|core[- ]shell|carbon[- ]coated|encapsulation|heterojunction)\b", re.I)
RE_BRIDGE = re.compile(r"\b(bridging oxygen|bridge oxygen|(?:Ni|Fe|Co|Cu|Zn|Pt|Pd|Ru|Rh|Ir|Ce|Ti|Mo|W|Mn|Cr|Zr|La|Ga|In|Sn)[-–—]?O[-–—]?[A-Z][a-z]?|M[-–—]?O[-–—]?C/P|M[-–—]?O[-–—]?[A-Z][a-z]?|P[-–—]?O[-–—]?Ni|C[-–—]?O[-–—]?Ni)\b", re.I)
RE_MODEL = re.compile(r"\b(slab model|cluster model|supercell|periodic slab|periodic DFT|nanoparticle|nanosheet|nanotube|single[- ]atom model|core[- ]shell model)\b", re.I)
RE_PHASE = re.compile(r"\b(rocksalt|rutile|anatase|cubic|hexagonal|orthorhombic|monoclinic|metallic|amorphous|crystalline|graphitic)\b", re.I)
RE_TERMINATION = re.compile(r"\b([A-Z][a-z]?[- ]terminated|OH[- ]terminated|O[- ]terminated|metal[- ]terminated|hydroxylated)\b", re.I)
RE_MATERIAL = re.compile(r"\b(?:[A-Z][a-z]?(?:[A-Z][a-z]?|\d|O|S|N|P|C|Fe|Co|Ni|Cu|Zn|Mo|W|Pt|Pd|Ru|Rh|Ir|Ce|Ti|V|Mn|Cr|Zr|La|Ga|In|Sn|Se|Te|@|/|-){1,24})(?:\([0-9]{3}\))?\b")
RE_MENTION_CUE = re.compile(r"\b(catalyst|surface|facet|slab|support|dopant|doped|vacancy|defect|interface|hydrogen adsorption|H adsorption|H\*|Volmer|DFT|bridge oxygen|bridging oxygen|M[-–—]?O)\b", re.I)

KNOWN_NON_MATERIAL = {"DFT", "HBE", "HER", "HOR", "EDL", "PZC", "XPS", "XAS", "TEM", "XRD", "DOS", "PDOS", "ELF", "RHE", "KOH", "H2O", "OH", "CO", "H2", "Ar", "N2"}


def clean_scalar(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = normalize_text(x)
    if not s or s.lower() in {'none', 'null', 'nan', 'unknown', 'not mentioned', 'n/a', 'na'}:
        return None
    return s


def to_num(x: Any) -> Optional[float]:
    try:
        v = pd.to_numeric(x, errors='coerce')
        return None if pd.isna(v) else float(v)
    except Exception:
        return None


def strict_confidence_score(x: Any) -> Tuple[Optional[float], Optional[str], str]:
    if x is None:
        return None, None, 'missing'
    raw = str(x).strip()
    if raw == '':
        return None, None, 'missing'
    v = to_num(x)
    if v is None:
        return None, raw, 'non_numeric'
    if 0 <= v <= 1:
        return v, raw, 'ok'
    return None, raw, 'out_of_range'


def normalize_material_class(x: Any, context: str = '') -> Optional[str]:
    s = normalize_text(x).lower() + ' ' + normalize_text(context).lower()
    if not s.strip():
        return None
    pairs = [
        ('oxide', ['oxide', 'o2', 'o3']), ('sulfide', ['sulfide']), ('nitride', ['nitride']),
        ('carbide', ['carbide']), ('phosphide', ['phosphide']), ('hydroxide', ['hydroxide']),
        ('alloy', ['alloy']), ('single-atom catalyst', ['single atom', 'single-atom', 'sac']),
        ('carbon-coated metal', ['carbon-coated', 'carbon coated', '@c', '@pc']),
        ('boron nitride', ['boron nitride', 'bnnt']), ('interface', ['interface', 'heterojunction', 'heterointerface']),
        ('metal', ['metallic', 'metal nanoparticle']),
    ]
    for canon, pats in pairs:
        if any(p in s for p in pats):
            return canon
    return clean_scalar(x)


def is_local_qdrant_prefix(prefix: str, shard_ids: List[int]) -> bool:
    return any(Path(f'{prefix}{sid}').exists() for sid in shard_ids)


def load_triage_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = {}
    for r in read_jsonl(path):
        pid = str(r.get('paper_id', '')).strip()
        flag = int(pd.to_numeric(r.get('should_extract'), errors='coerce') or 0)
        if pid and flag == 1:
            rows[pid] = r
    return rows


def load_chunks(parsed_root: Path, paper_id: str) -> List[Dict[str, Any]]:
    return read_jsonl(parsed_root / 'papers' / paper_id / 'chunks.jsonl')


def evidence_from_chunk(ch: Dict[str, Any], score: float = 0.0) -> Dict[str, Any]:
    return {
        'paper_id': ch.get('paper_id'), 'chunk_id': ch.get('chunk_id'), 'chunk_type': ch.get('chunk_type'),
        'section_path': ch.get('section_path'), 'page_start': ch.get('page_start'), 'page_end': ch.get('page_end'),
        'source': ch.get('source'), 'text': normalize_text(ch.get('text', '')), 'score': score,
    }


def local_keyword_search(parsed_root: Path, paper_id: str, keywords: List[str], allow_types: List[str], topk: int) -> List[Dict[str, Any]]:
    rows = []
    for ch in load_chunks(parsed_root, paper_id):
        ctype = normalize_text(ch.get('chunk_type') or 'text')
        if allow_types and ctype not in allow_types:
            continue
        txt = normalize_text(ch.get('text', ''))
        low = txt.lower()
        score = sum(1 for kw in keywords if kw.lower() in low)
        if RE_MENTION_CUE.search(txt):
            score += 2
        if ctype == 'table':
            score += 2
        if score > 0:
            rows.append(evidence_from_chunk(ch, float(score)))
    rows.sort(key=lambda r: float(r.get('score') or 0), reverse=True)
    return rows[:topk]


def qdrant_search_or_fallback(parsed_root: Path, qdrant_prefix: str, collection_prefix: str, shard_ids: List[int], qvec: Optional[List[float]], paper_id: str, allow_types: List[str], topk: int, query_text: str, use_lock: bool) -> List[Dict[str, Any]]:
    """High-recall search: combine semantic Qdrant hits with local keyword hits.

    The previous version returned Qdrant hits immediately, which could miss exact
    table/caption mentions. Combining both greatly improves recall for material,
    facet, bridge, vacancy, and energy-table evidence.
    """
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
    kws = [w for w in re.split(r'\s+', query_text) if len(w) >= 3][:40]
    try:
        rows.extend(local_keyword_search(parsed_root, paper_id, kws, allow_types, topk=max(topk, 12)))
    except Exception:
        pass
    rows = dedup_evidence(rows)
    rows.sort(key=lambda r: float(r.get('score') or 0), reverse=True)
    return rows[:max(topk, min(len(rows), topk + 8))]


def rank_hit(ev: Dict[str, Any]) -> int:
    txt = normalize_text(ev.get('text') or ev.get('quote')).lower()
    score = 0
    for kw in ['catalyst', 'surface', 'facet', 'slab', 'dopant', 'doped', 'vacancy', 'defect', 'interface', 'bridge oxygen', 'm-o', 'h adsorption', 'hydrogen adsorption', 'dft']:
        if kw in txt:
            score += 1
    if ev.get('chunk_type') == 'table':
        score += 3
    return score


def trim_evidence(evs_main, evs_table, evs_entity, args):
    evs_main = sorted(dedup_evidence(evs_main), key=rank_hit, reverse=True)[:max(8, args.max_items_main + 3)]
    evs_table = sorted(dedup_evidence(evs_table), key=rank_hit, reverse=True)[:max(8, args.max_items_table + 3)]
    evs_entity = sorted(dedup_evidence(evs_entity), key=rank_hit, reverse=True)[:min(max(4, args.max_items_entity), 8)]
    return evs_main, evs_table, evs_entity


def extract_first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    if not m:
        return None
    for g in m.groups():
        if g:
            return clean_scalar(g)
    return clean_scalar(m.group(0))


def extract_facet(text: str) -> Optional[str]:
    m = RE_FACET.search(text)
    if not m:
        return None
    raw = m.group(0)
    nums = re.findall(r'[0-9]{3,4}', raw)
    return f"({nums[0]})" if nums else clean_scalar(raw)


def extract_materials(text: str) -> List[str]:
    mats: List[str] = []
    # High-value explicit systems with @, slash, doped, or facet.
    for m in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:@|/|-)[A-Z][A-Za-z0-9/\-@]*\b(?:\([0-9]{3}\))?", text):
        cm = clean_material_name(m.group(0), text)
        if cm:
            mats.append(cm)
    for m in RE_MATERIAL.finditer(text):
        token = m.group(0).strip()
        if token in KNOWN_NON_MATERIAL or is_bad_material_name(token):
            continue
        if len(token) < 2:
            continue
        # Avoid ordinary capitalized words.
        if not re.search(r'[0-9@/()]|[A-Z][a-z]?[A-Z]|O|S|N|P|C|Fe|Co|Ni|Cu|Zn|Mo|W|Pt|Pd|Ru|Rh|Ir|Ce|Ti|Mn|Cr', token):
            continue
        cm = clean_material_name(token, text)
        if cm:
            mats.append(cm)
    out, seen = [], set()
    for t in mats:
        key = t.lower()
        if key not in seen:
            seen.add(key); out.append(t)
    return out[:12]


def heuristic_systems(evs_main, evs_table, evs_entity, paper_id: str) -> List[Dict[str, Any]]:
    evs_all = evs_table + evs_entity + evs_main
    text = normalize_text(' '.join(str(e.get('text') or e.get('quote') or '') for e in evs_all))
    mats = extract_materials(text)
    if not mats and RE_MENTION_CUE.search(text):
        mats = ['reported catalyst/surface']
    rows: List[Dict[str, Any]] = []
    for mat in mats[:8]:
        pos = text.find(mat) if mat != 'reported catalyst/surface' else 0
        snippet = text[max(0, pos-350):pos+500]
        support = extract_first(RE_SUPPORT, snippet) or extract_first(RE_SUPPORT, text)
        dop = extract_first(RE_DOPANT, snippet) or extract_first(RE_DOPANT, text)
        bridge = clean_bridge(extract_first(RE_BRIDGE, snippet), snippet) or clean_bridge(extract_first(RE_BRIDGE, text), text)
        rec = {
            'paper_id': paper_id,
            'system_local_id': f'{paper_id}_sys_{len(rows)+1:03d}',
            'material_name': mat,
            'material_class': normalize_material_class(mat, snippet),
            'composition': mat if mat != 'reported catalyst/surface' else None,
            'surface_facet': extract_facet(snippet) or extract_facet(text),
            'termination': extract_first(RE_TERMINATION, snippet),
            'phase': extract_first(RE_PHASE, snippet),
            'support': support,
            'dopant': dop,
            'interface_type': extract_first(RE_INTERFACE, snippet) or extract_first(RE_INTERFACE, text),
            'defect_type': extract_first(RE_DEFECT, snippet) or extract_first(RE_DEFECT, text),
            'bridge_structure': bridge,
            'M_O_X_configuration': bridge,
            'model_type': extract_first(RE_MODEL, snippet) or extract_first(RE_MODEL, text),
            'confidence_score': None,
            'qc_field_support': field_support_summary({'material_name': mat, 'bridge_structure': bridge}, text, ['material_name','bridge_structure']),
            'extraction_source': 'heuristic',
            'evidence': normalize_evidence_list([
                {'chunk_id': e.get('chunk_id'), 'quote': str(e.get('text') or e.get('quote') or '')[:500], 'source': e.get('source'), 'chunk_type': e.get('chunk_type'), 'page_start': e.get('page_start'), 'page_end': e.get('page_end'), 'section_path': e.get('section_path')}
                for e in evs_all[:8]
            ]),
        }
        rows.append(rec)
    # de-duplicate
    out, seen = [], set()
    for r in rows:
        key = tuple((r.get(k) or '').lower() for k in ['material_name', 'surface_facet', 'dopant', 'defect_type', 'bridge_structure', 'support'])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def postprocess_systems(obj: Dict[str, Any], paper_id: str, evidence_txt: str = "") -> Dict[str, Any]:
    systems = obj.get('systems', []) if isinstance(obj, dict) else []
    if not isinstance(systems, list):
        systems = []
    out: List[Dict[str, Any]] = []
    seen = set()
    for i, s in enumerate(systems, 1):
        if not isinstance(s, dict):
            continue
        conf_num, conf_raw, conf_status = strict_confidence_score(s.get('confidence_score'))
        bridge = clean_bridge(s.get('bridge_structure'), evidence_txt) or clean_bridge(s.get('M_O_X_configuration'), evidence_txt)
        rec = {
            'paper_id': paper_id,
            'system_local_id': clean_scalar(s.get('system_local_id')) or f'{paper_id}_sys_{i:03d}',
            'material_name': clean_material_name(s.get('material_name'), evidence_txt),
            'material_class': normalize_material_class(s.get('material_class'), str(s)),
            'composition': clean_material_name(s.get('composition'), evidence_txt) or clean_material_name(s.get('material_name'), evidence_txt),
            'surface_facet': clean_scalar(s.get('surface_facet')),
            'termination': clean_scalar(s.get('termination')),
            'phase': clean_scalar(s.get('phase')),
            'support': clean_scalar(s.get('support')),
            'dopant': clean_scalar(s.get('dopant')),
            'interface_type': clean_scalar(s.get('interface_type')),
            'defect_type': clean_scalar(s.get('defect_type')),
            'bridge_structure': bridge,
            'M_O_X_configuration': clean_bridge(s.get('M_O_X_configuration'), evidence_txt) or bridge,
            'model_type': clean_scalar(s.get('model_type')),
            'confidence_score': conf_num,
            'confidence_score_raw': conf_raw,
            'confidence_score_status': conf_status,
            'evidence': normalize_evidence_list(s.get('evidence', [])),
            'qc_field_support': field_support_summary({'material_name': clean_material_name(s.get('material_name'), evidence_txt), 'bridge_structure': bridge}, evidence_txt, ['material_name','bridge_structure']),
            'extraction_source': 'llm',
        }
        if not any(rec.get(k) for k in ['material_name', 'composition', 'surface_facet', 'dopant', 'defect_type', 'bridge_structure', 'support']):
            continue
        key = tuple((rec.get(k) or '').lower() for k in ['material_name', 'composition', 'surface_facet', 'dopant', 'defect_type', 'bridge_structure', 'support'])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    return {'paper_id': paper_id, 'systems': out}


def build_prompt(paper_id: str, evs_main, evs_table, evs_entity, args) -> str:
    return SYSTEM_PROMPT + '\n\n' + (
        f'paper_id={paper_id}\n\n'
        f'EVIDENCE_TABLE:\n{pack_evidence(evs_table, max_items=args.max_items_table, total_chars=args.table_budget_chars, max_quote_chars=args.max_table_chars)}\n\n'
        f'EVIDENCE_ENTITY:\n{pack_evidence(evs_entity, max_items=args.max_items_entity, total_chars=args.entity_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        f'EVIDENCE_MAIN:\n{pack_evidence(evs_main, max_items=args.max_items_main, total_chars=args.main_budget_chars, max_quote_chars=args.max_text_chars)}\n\n'
        'TASK: Extract concrete MATERIAL / SURFACE SYSTEMS only. Include bridge_structure and M_O_X_configuration when present.\n'
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--triage', required=True)
    ap.add_argument('--parsed-root', required=True)
    ap.add_argument('--api-bases', default='')
    ap.add_argument('--model-id', default='')
    ap.add_argument('--embed-model', required=True)
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--qdrant-prefix', required=True)
    ap.add_argument('--collection-prefix', required=True)
    ap.add_argument('--shard-ids', default='0,1,2')
    ap.add_argument('--topk-main', type=int, default=14)
    ap.add_argument('--topk-table', type=int, default=20)
    ap.add_argument('--topk-entity', type=int, default=6)
    ap.add_argument('--main-budget-chars', type=int, default=8500)
    ap.add_argument('--table-budget-chars', type=int, default=5500)
    ap.add_argument('--entity-budget-chars', type=int, default=1500)
    ap.add_argument('--max-items-main', type=int, default=10)
    ap.add_argument('--max-items-table', type=int, default=8)
    ap.add_argument('--max-items-entity', type=int, default=4)
    ap.add_argument('--max-text-chars', type=int, default=850)
    ap.add_argument('--max-table-chars', type=int, default=1500)
    ap.add_argument('--max-caption-chars', type=int, default=1200)
    ap.add_argument('--workers', type=int, default=1)
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--max-retries', type=int, default=3)
    ap.add_argument('--verify-llm', action='store_true', help='Run a second LLM pass to verify extracted JSON against evidence before deterministic QC.')
    ap.add_argument('--verifier-api-bases', default='', help='Optional API bases for verifier; defaults to --api-bases.')
    ap.add_argument('--verifier-model-id', default='', help='Optional verifier model; defaults to --model-id.')
    ap.add_argument('--verify-max-tokens', type=int, default=2600, help='Max tokens for second-pass verification.')
    ap.add_argument('--max-tokens', type=int, default=1500)
    ap.add_argument('--out', required=True)
    ap.add_argument('--progress', default='outputs/progress_hads_step3b_systems.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    parsed_root = Path(args.parsed_root)
    triage_rows = load_triage_rows(Path(args.triage))
    total = len(triage_rows)
    print(f'[INFO] HADS Step3b papers_to_process={total}', flush=True)
    shard_ids = [int(x) for x in args.shard_ids.split(',') if x.strip()]
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    progress_path = Path(args.progress)
    done = set(load_progress(progress_path).get('done', [])) if progress_path.exists() and not args.force else set()
    if args.force and outp.exists():
        outp.unlink()

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

    def process_row(row: Dict[str, Any]):
        paper_id = str(row.get('paper_id', ''))
        if paper_id in done:
            return 'skip', paper_id, 0, None
        try:
            evs_main = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_main_text_shard', shard_ids, qvecs['main'], paper_id, ['text'], args.topk_main, QUERY_MAIN, use_qdrant_lock)
            evs_table = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_table_caption_shard', shard_ids, qvecs['table'], paper_id, ['caption', 'table'], args.topk_table, QUERY_TABLE, use_qdrant_lock)
            evs_entity = qdrant_search_or_fallback(parsed_root, args.qdrant_prefix, f'{args.collection_prefix}_entity_shard', shard_ids, qvecs['entity'], paper_id, ['text', 'caption', 'table'], args.topk_entity, QUERY_ENTITY, use_qdrant_lock)
            evs_main, evs_table, evs_entity = trim_evidence(evs_main, evs_table, evs_entity, args)
            evidence_txt = packed_evidence_text(evs_main, evs_table, evs_entity)
            obj = None
            if llm is not None and (evs_main or evs_table or evs_entity):
                try:
                    obj = safe_json_extract(llm.completions(build_prompt(paper_id, evs_main, evs_table, evs_entity, args), temperature=0.0, max_tokens=args.max_tokens))
                except Exception as e:
                    print(f'[WARN] LLM failed for {paper_id}: {e}', flush=True)
            if obj is not None and verifier_llm is not None:
                vobj = verify_extraction_json(verifier_llm, paper_id, 'systems', obj, evidence_txt, target_context={'paper_id': paper_id}, max_tokens=args.verify_max_tokens)
                if vobj is not None:
                    obj = vobj
            if obj is None:
                obj = {'paper_id': paper_id, 'systems': heuristic_systems(evs_main, evs_table, evs_entity, paper_id)}
            obj = postprocess_systems(obj, paper_id, evidence_txt)
            for s in obj['systems']:
                append_jsonl(s, outp)
            return 'ok', paper_id, len(obj['systems']), None
        except Exception as e:
            return 'err', paper_id, 0, str(e)

    prog = {'done': sorted(done)}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(process_row, row): row.get('paper_id') for row in triage_rows.values()}
        for fut in as_completed(futs):
            status, paper_id, n_rows, err = fut.result()
            with progress_lock:
                counter['done'] += 1
                if status == 'ok':
                    counter['ok'] += 1; done.add(paper_id); prog['done'] = sorted(done); save_progress(prog, progress_path)
                    print(f'[OK] {paper_id} systems={n_rows} progress={counter["done"]}/{total}', flush=True)
                elif status == 'skip':
                    counter['skip'] += 1; print(f'[SKIP] {paper_id} progress={counter["done"]}/{total}', flush=True)
                else:
                    counter['err'] += 1; print(f'[ERR] {paper_id} error={err} progress={counter["done"]}/{total}', flush=True)
    print(f'[DONE] wrote {args.out}', flush=True)


if __name__ == '__main__':
    main()
