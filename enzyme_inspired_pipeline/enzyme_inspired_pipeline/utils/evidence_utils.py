#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List


def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8", errors="ignore")).hexdigest()


def normalize_evidence_list(evs: Any, max_quote_chars: int = 500) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for e in evs if isinstance(evs, list) else []:
        if not isinstance(e, dict):
            continue
        quote = re.sub(r"\s+", " ", str(e.get("quote", "") or e.get("text", "") or "").strip())[:max_quote_chars]
        out.append(
            {
                "chunk_id": str(e.get("chunk_id", "") or ""),
                "quote": quote,
                "source": e.get("source"),
                "chunk_type": e.get("chunk_type"),
                "page_start": e.get("page_start"),
                "page_end": e.get("page_end"),
                "section_path": e.get("section_path"),
            }
        )
    seen, uniq = set(), []
    for e in out:
        key = (e.get("chunk_id"), e.get("quote"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq[:30]


def dedup_evidence(evs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for e in evs:
        key = (e.get("chunk_id"), sha1_text(str(e.get("text") or e.get("quote") or ""))[:12])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def evidence_stats(evs: List[Dict[str, Any]]) -> Dict[str, Any]:
    evs = normalize_evidence_list(evs)
    c_table = sum(1 for e in evs if str(e.get("chunk_type", "")).lower() == "table")
    c_caption = sum(1 for e in evs if str(e.get("chunk_type", "")).lower() == "caption")
    pages = {e.get("page_start") for e in evs if e.get("page_start") is not None}
    total_chars = sum(len(str(e.get("quote", ""))) for e in evs)
    numeric = sum(1 for e in evs if re.search(r"\d", str(e.get("quote", ""))))
    return {
        "evidence_count": len(evs),
        "evidence_table_count": c_table,
        "evidence_caption_count": c_caption,
        "evidence_text_count": len(evs) - c_table - c_caption,
        "evidence_unique_pages": len(pages),
        "evidence_total_chars": total_chars,
        "evidence_numeric_snippets": numeric,
    }


def pack_evidence(
    evs: List[Dict[str, Any]],
    max_items: int = 8,
    total_chars: int = 4000,
    max_quote_chars: int = 850,
) -> str:
    parts: List[str] = []
    used = 0
    for i, e in enumerate(evs[:max_items], start=1):
        quote = re.sub(r"\s+", " ", str(e.get("text") or e.get("quote") or "").strip())[:max_quote_chars]
        block = (
            f"[E{i}] chunk_id={e.get('chunk_id', '')} type={e.get('chunk_type', '')} source={e.get('source', '')} "
            f"page_start={e.get('page_start')} page_end={e.get('page_end')} section={e.get('section_path', '')}\n{quote}\n"
        )
        if used + len(block) > total_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)
