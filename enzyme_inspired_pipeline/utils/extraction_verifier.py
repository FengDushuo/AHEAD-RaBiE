#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LLM-based second-pass verification for evidence-grounded extraction.

The verifier is deliberately conservative: it should remove unsupported fields
rather than adding new speculative content. It is used after the first extraction
pass and before deterministic post-processing/QC.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from utils.llm_client import safe_json_extract
from utils.text_utils import clip_text, normalize_text


def verify_extraction_json(
    llm: Any,
    paper_id: str,
    task_name: str,
    raw_obj: Dict[str, Any],
    evidence_text: str,
    target_context: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2400,
) -> Optional[Dict[str, Any]]:
    """Verify/correct extracted JSON against evidence using an LLM.

    Returns a JSON dict with the same top-level schema when successful, otherwise None.
    """
    if llm is None or not isinstance(raw_obj, dict):
        return None
    evidence = clip_text(normalize_text(evidence_text), 22000)
    raw_json = json.dumps(raw_obj, ensure_ascii=False)
    ctx_json = json.dumps(target_context or {}, ensure_ascii=False)
    prompt = f"""
You are a strict scientific extraction verifier for H adsorption / deprotonation literature mining.
Verify the extracted JSON against the EVIDENCE only.

paper_id: {paper_id}
task_name: {task_name}
TARGET_CONTEXT: {ctx_json}

Rules:
1. Keep the same top-level JSON schema as INPUT_JSON. For systems use systems[]; for sites/metrics use records[].
2. Remove duplicate records.
3. For every non-null field, require direct textual/table support in EVIDENCE.
4. If a field is unsupported, cross-system, copied from a reference catalyst, or likely a figure/page/potential/current/temperature, set it to null.
5. Do not invent new numeric values. Only keep numbers that appear in EVIDENCE near the stated descriptor/metric.
6. Keep ΔE_H and ΔG_H separate; keep barriers separate from adsorption energies.
7. Preserve exact units when stated. If a unit is not stated near the value, set the unit to null.
8. If site/system linkage is uncertain, keep the record but set site_local_id or weak fields to null.
9. Add a short field verification_status to each record: "verified", "partially_verified", or "weak_evidence".
10. Return STRICT JSON only.

EVIDENCE:
{evidence}

INPUT_JSON:
{raw_json}

Return the corrected JSON object now.
""".strip()
    try:
        txt = llm.completions(prompt, temperature=0.0, max_tokens=max_tokens)
        obj = safe_json_extract(txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
