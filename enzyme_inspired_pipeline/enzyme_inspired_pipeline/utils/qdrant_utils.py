#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def open_qdrant_local(path: str) -> QdrantClient:
    return QdrantClient(path=path)


def build_filter(
    paper_id: Optional[str] = None,
    allow_types: Optional[List[str]] = None,
    extra_must: Optional[List[Any]] = None,
) -> rest.Filter:
    must = list(extra_must or [])
    if paper_id:
        must.append(rest.FieldCondition(key="paper_id", match=rest.MatchValue(value=paper_id)))
    if allow_types:
        must.append(rest.FieldCondition(key="chunk_type", match=rest.MatchAny(any=allow_types)))
    return rest.Filter(must=must)


def _list_collection_names(client: QdrantClient) -> List[str]:
    try:
        return [c.name for c in client.get_collections().collections]
    except Exception:
        return []


def _resolve_collection_name(client: QdrantClient, collection_prefix: str, sid: int) -> str:
    names = set(_list_collection_names(client))
    candidates = [
        f"{collection_prefix}{sid}",
        f"{collection_prefix}_shard{sid}",
        f"{collection_prefix}_{sid}",
        collection_prefix,
    ]
    for c in candidates:
        if c in names:
            return c
    return f"{collection_prefix}{sid}"


def _query_collection(
    client: QdrantClient,
    collection_name: str,
    qvec: List[float],
    flt: rest.Filter,
    topk: int,
):
    try:
        res = client.query_points(
            collection_name=collection_name,
            query=qvec,
            limit=topk,
            query_filter=flt,
            with_payload=True,
            with_vectors=False,
        )
        return getattr(res, "points", []) or []
    except Exception:
        # compatibility fallback for older client versions
        try:
            return client.search(
                collection_name=collection_name,
                query_vector=qvec,
                limit=topk,
                query_filter=flt,
                with_payload=True,
                with_vectors=False,
            )
        except Exception:
            return []


def search_multishard(
    qdrant_prefix: str,
    collection_prefix: str,
    shard_ids: List[int],
    qvec: List[float],
    paper_id: Optional[str],
    allow_types: List[str],
    topk: int = 20,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for sid in shard_ids:
        client = open_qdrant_local(f"{qdrant_prefix}{sid}")
        collection = _resolve_collection_name(client, collection_prefix, sid)
        flt = build_filter(paper_id=paper_id, allow_types=allow_types)
        pts = _query_collection(client, collection, qvec, flt, topk)
        for p in pts:
            payload = getattr(p, "payload", None) or {}
            out.append(
                {
                    "score": float(getattr(p, "score", None)) if getattr(p, "score", None) is not None else None,
                    "chunk_id": str(payload.get("chunk_id", "")),
                    "chunk_type": str(payload.get("chunk_type", "")),
                    "source": str(payload.get("source", "")),
                    "section_path": str(payload.get("section_path", "")),
                    "section_type": str(payload.get("section_type", "")),
                    "entity_type": str(payload.get("entity_type", "")),
                    "page_start": payload.get("page_start"),
                    "page_end": payload.get("page_end"),
                    "text": str(payload.get("text", "")),
                    "paper_id": str(payload.get("paper_id", "")),
                }
            )
    out.sort(key=lambda x: (x.get("score") is None, -(x.get("score") or 0.0)))
    return out
