#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge previous canonical HADS records with add-on ΔE_H descriptor records.

The previous corpus is not modified. New records are appended after light de-duplication.
Duplicate detection is conservative and based on DOI/title-unavailable canonical fields:
paper_id/canonical_id exact duplicates and a content key of material/site/target energy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from utils.io_utils import read_jsonl
from utils.normalize_utils import to_float
from utils.text_utils import normalize_text


def clean(x: Any) -> str:
    return normalize_text(x).lower()


def numkey(x: Any) -> str:
    v = to_float(x)
    return "" if v is None else f"{v:.6f}"


def content_key(r: Dict[str, Any]) -> Tuple[str, ...]:
    return (
        clean(r.get("paper_id")),
        clean(r.get("material_name")),
        clean(r.get("surface_facet")),
        clean(r.get("site_label")),
        clean(r.get("active_atom")),
        clean(r.get("adsorption_configuration")),
        numkey(r.get("H_adsorption_energy_value_eV")),
        numkey(r.get("H_adsorption_energy_value")),
        clean(r.get("DFT_functional")),
        clean(r.get("coverage")),
    )


def has_deltae(r: Dict[str, Any]) -> bool:
    return to_float(r.get("H_adsorption_energy_value_eV")) is not None or to_float(r.get("H_adsorption_energy_value")) is not None


def ensure_unique_ids(records: List[Dict[str, Any]], source_batch: str) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for i, r in enumerate(records, 1):
        rr = dict(r)
        pid = normalize_text(rr.get("paper_id")) or f"{source_batch}_paper_unknown"
        cid = normalize_text(rr.get("canonical_id")) or f"{pid}_{source_batch}_{i:05d}"
        if cid in seen:
            cid = f"{cid}_{source_batch}_{i:05d}"
        seen.add(cid)
        rr["paper_id"] = pid
        rr["canonical_id"] = cid
        rr["source_batch"] = rr.get("source_batch") or source_batch
        out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="Previous canonical JSONL. This file is read-only.")
    ap.add_argument("--new", required=True, help="Add-on canonical JSONL from 3g.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="")
    ap.add_argument("--source-batch-old", default="previous")
    ap.add_argument("--source-batch-new", default="pdf_files_add_deltaeh")
    ap.add_argument("--keep-new-without-deltae", action="store_true", help="Normally add-on records without ΔE_H are skipped.")
    args = ap.parse_args()

    old = ensure_unique_ids(read_jsonl(Path(args.old)), args.source_batch_old)
    new_raw = ensure_unique_ids(read_jsonl(Path(args.new)), args.source_batch_new)
    new = [r for r in new_raw if args.keep_new_without_deltae or has_deltae(r)]

    old_id_keys = {clean(r.get("paper_id")) + "::" + clean(r.get("canonical_id")) for r in old}
    old_content = {content_key(r) for r in old if has_deltae(r)}

    merged = list(old)
    skipped_id = 0
    skipped_content = 0
    added = 0
    for r in new:
        id_key = clean(r.get("paper_id")) + "::" + clean(r.get("canonical_id"))
        if id_key in old_id_keys:
            skipped_id += 1
            continue
        ck = content_key(r)
        if ck in old_content:
            skipped_content += 1
            continue
        merged.append(r)
        old_id_keys.add(id_key)
        old_content.add(ck)
        added += 1

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "old_records": len(old),
        "new_raw_records": len(new_raw),
        "new_records_with_deltaE": len(new),
        "new_added": added,
        "new_skipped_exact_id": skipped_id,
        "new_skipped_content_duplicate": skipped_content,
        "merged_records": len(merged),
        "old_file": args.old,
        "new_file": args.new,
        "out_file": args.out,
    }
    summary_path = Path(args.summary) if args.summary else outp.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
