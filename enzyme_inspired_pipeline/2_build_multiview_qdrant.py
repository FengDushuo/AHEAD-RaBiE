#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams
except Exception:  # allow --help / static checks without qdrant-client
    QdrantClient = None
    Distance = PointStruct = VectorParams = None
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

from utils.io_utils import ensure_dir, write_json
from utils.text_utils import infer_section_type, normalize_text


def stable_int_id(text: str) -> int:
    """Generate a stable integer point id for local Qdrant."""
    h = hashlib.sha1((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:16], 16)


def collection_exists(client: QdrantClient, name: str) -> bool:
    try:
        return any(c.name == name for c in client.get_collections().collections)
    except Exception:
        return False


def iter_chunk_rows(parsed_root: str) -> Iterable[Dict[str, Any]]:
    for fn in Path(parsed_root).glob("papers/*/chunks.jsonl"):
        paper_id = fn.parent.name
        with fn.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                text = normalize_text(row.get("text", ""))
                row["paper_id"] = str(row.get("paper_id") or paper_id)
                row["text"] = text
                row["chunk_type"] = str(row.get("chunk_type", "text"))
                row["section_type"] = infer_section_type(row.get("section_path", ""), row.get("text", ""))
                yield row


def iter_entity_rows(parsed_root: str) -> Iterable[Dict[str, Any]]:
    fp = Path(parsed_root) / "entity_candidates.jsonl"
    if not fp.exists():
        return
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            row["paper_id"] = str(row.get("paper_id", "")).strip()
            row["text"] = normalize_text(row.get("text", ""))
            yield row


def load_rows(parsed_root: str, view: str, min_chars: int = 80) -> pd.DataFrame:
    rows = []

    if view == "main_text":
        for r in iter_chunk_rows(parsed_root):
            if r["chunk_type"] == "text" and len(r["text"]) >= min_chars:
                rows.append(r)

    elif view == "table_caption":
        for r in iter_chunk_rows(parsed_root):
            if r["chunk_type"] in {"table", "caption"} and len(r["text"]) >= min_chars:
                rows.append(r)

    elif view == "entity":
        for r in iter_entity_rows(parsed_root):
            if len(str(r.get("text", "")).strip()) >= 20:
                rows.append(r)

    else:
        raise ValueError(f"Unknown view={view}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    df["_gid"] = np.arange(len(df), dtype=np.int64)
    return df


def build_embedding_text(row: Dict[str, Any], view: str) -> str:
    if view == "entity":
        return (
            f"paper_id: {row.get('paper_id', '')} | "
            f"entity_type: {row.get('entity_type', '')} | "
            f"section: {row.get('section_type', '')}\n"
            f"{row.get('text', '')}"
        )

    meta = [
        f"paper_id: {row.get('paper_id', '')}",
        f"reaction: {row.get('reaction', '')}",
        f"bucket: {row.get('bucket', '')}",
        f"chunk_type: {row.get('chunk_type', '')}",
        f"section: {row.get('section_type', '')}",
    ]
    meta = [m for m in meta if m.split(": ", 1)[1]]
    return " | ".join(meta) + "\n" + str(row.get("text", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parsed-root", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--qdrant-path", required=True)
    ap.add_argument("--collection-prefix", required=True)
    ap.add_argument("--view", required=True, choices=["main_text", "table_caption", "entity"])
    ap.add_argument("--min-chars", type=int, default=80)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--shard-num", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if QdrantClient is None or PointStruct is None or VectorParams is None or Distance is None:
        raise RuntimeError('qdrant-client is required for Step2. Install with: pip install qdrant-client')
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers is required for Step2. Install with: pip install sentence-transformers')

    df = load_rows(args.parsed_root, args.view, min_chars=args.min_chars)
    if len(df) == 0:
        print("[DONE] no rows for view", args.view)
        return

    df = df[df["_gid"] % args.shard_num == args.shard_id].reset_index(drop=True)
    if len(df) == 0:
        print("[DONE] shard empty")
        return

    df["_embed_text"] = [build_embedding_text(r, args.view) for r in df.to_dict(orient="records")]

    model = SentenceTransformer(args.model, device=args.device)
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass

    vec0 = model.encode([df["_embed_text"].iloc[0]], normalize_embeddings=True, batch_size=1)
    dim = int(len(vec0[0]))

    qpath = Path(args.qdrant_path)
    ensure_dir(qpath)

    collection = f"{args.collection_prefix}_{args.view}_shard{args.shard_id}"
    client = QdrantClient(path=str(qpath))

    if args.force and collection_exists(client, collection):
        client.delete_collection(collection_name=collection)

    if not collection_exists(client, collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    embs = model.encode(
        df["_embed_text"].tolist(),
        batch_size=args.batch,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = []
    for vec, row in zip(embs, df.to_dict(orient="records")):
        payload = {k: row.get(k) for k in row.keys() if k not in {"_gid", "_embed_text"}}
        pid = str(row.get("candidate_id") if args.view == "entity" else row.get("chunk_id"))
        point_id = stable_int_id(f"{args.view}|{pid}")
        points.append(
            PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points)

    write_json(
        {
            "collection": collection,
            "view": args.view,
            "rows": len(df),
            "embed_dim": dim,
            "shard_id": args.shard_id,
            "shard_num": args.shard_num,
        },
        qpath / f".stats_{collection}.json",
    )
    print("[DONE] built", collection, "rows=", len(df))


if __name__ == "__main__":
    main()