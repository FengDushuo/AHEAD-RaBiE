#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
1_parse_corpus.py

Rigorous PDF parsing pipeline for H adsorption / deprotonation / proton-transfer literature mining.
This version is tuned for downstream extraction of catalyst systems, active sites, H/OH/H2O adsorption energies, deprotonation/proton-transfer barriers, electronic descriptors, and interfacial-water mechanisms.

Key upgrades
------------
1) Stronger QC and manifest fields for downstream filtering.
2) Header/footer noise suppression via repeated-line detection.
3) Basic duplicate-block suppression within each PDF.
4) Better caption merging safety when bbox is missing.
5) Table metadata enriched with page / bbox / preview text.
6) Fused chunks include text_source_rank and parser diagnostics.
7) Summary report for corpus-level parser health.
8) Resume remains supported.

Output structure
----------------
<out>/manifest.csv
<out>/qc_report.csv
<out>/summary_report.json
<out>/errors.csv (if any)
<out>/papers/<paper_id>/...

Notes
-----
- No OCR is performed here. Scanned PDFs are flagged for manual handling.
- GROBID is optional and used when available.
"""

import os
import re
import json
import time
import math
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict

import fitz  # pymupdf
import pandas as pd
import requests
from tqdm import tqdm
from lxml import etree
from concurrent.futures import ProcessPoolExecutor, as_completed


FIG_PAT = re.compile(r"^(fig\.?|figure)\s*\d+", re.IGNORECASE)
TAB_PAT = re.compile(r"^(tab\.?|table)\s*\d+", re.IGNORECASE)
DOI_PAT = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.I)
YEAR_PAT = re.compile(r"\b(19|20)\d{2}\b")
PH_PAT = re.compile(r"\bpH\b", re.I)
HADS_PAT = re.compile(r"hydrogen adsorption|H adsorption|HBE|hydrogen binding energy|Delta\s*G[_\- ]?H|DG[_\- ]?H|Delta\s*E[_\- ]?H|DE[_\- ]?H|H\*", re.I)
DEPROT_PAT = re.compile(r"deprotonation|proton transfer|Volmer step|Heyrovsky step|Tafel step|PCET|water dissociation", re.I)
OH_WATER_PAT = re.compile(r"OH adsorption|OHBE|H2O adsorption|water adsorption|interfacial water|hydrogen[- ]bond network|strongly hydrogen[- ]bonded water|weakly hydrogen[- ]bonded water", re.I)
SITE_PAT = re.compile(r"active\s+site|adsorption\s+site|top site|bridge site|hollow site|interface site|vacancy|defect|dopant|surface|facet|bridge oxygen|bridging oxygen", re.I)
ELEC_PAT = re.compile(r"d[- ]?band center|Bader charge|charge density difference|electron density difference|ELF|electron localization function|work function|PZC|PDOS|DOS", re.I)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sha1_of_file(fp: Path) -> str:
    h = hashlib.sha1()
    with fp.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def guess_labels(root: Path, pdf_path: Path) -> Tuple[str, str, str]:
    rel = pdf_path.relative_to(root).as_posix()
    parts = rel.split("/")
    reaction = parts[0] if len(parts) > 0 else "UNKNOWN"
    bucket = parts[1] if len(parts) > 1 else "UNKNOWN"
    return reaction, bucket, rel


def normalize_text(s: str) -> str:
    s = s or ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def detect_scanned(pdf_path: Path, max_pages: int = 5) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    pages = list(range(min(n, max_pages)))
    chars = []
    for i in pages:
        try:
            t = doc.load_page(i).get_text("text").strip()
        except Exception:
            t = ""
        chars.append(len(t))
    avg_chars = sum(chars) / max(1, len(chars))
    is_scanned = avg_chars < 50
    return {"n_pages": n, "avg_chars_first_pages": avg_chars, "is_scanned": is_scanned}


def collect_repeated_margin_lines(doc: fitz.Document, sample_pages: int = 8) -> Dict[str, set]:
    """Detect repeated short lines near top/bottom margins as probable headers/footers."""
    top_counter = Counter()
    bottom_counter = Counter()
    n = min(doc.page_count, sample_pages)
    for pno in range(n):
        page = doc.load_page(pno)
        h = page.rect.height
        try:
            blocks = page.get_text("blocks")
        except Exception:
            continue
        for b in blocks:
            x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
            t = normalize_text(text)
            if len(t) < 4 or len(t) > 120:
                continue
            if y1 < 0.08 * h:
                top_counter[t] += 1
            if y0 > 0.92 * h:
                bottom_counter[t] += 1
    repeated_top = {t for t, c in top_counter.items() if c >= max(2, math.ceil(n * 0.4))}
    repeated_bottom = {t for t, c in bottom_counter.items() if c >= max(2, math.ceil(n * 0.4))}
    return {"top": repeated_top, "bottom": repeated_bottom}



def extract_layout(pdf_path: Path, out_layout_jsonl: Path) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    total_chars = 0
    mupdf_err = 0
    duplicate_blocks_removed = 0
    margin_blocks_removed = 0

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        tmp = out_layout_jsonl.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"page": 1, "bbox": None, "text": f"[OPEN_FAIL] {e}", "order": 0, "col": 0}, ensure_ascii=False) + "\n")
        tmp.replace(out_layout_jsonl)
        return {
            "layout_blocks": 1,
            "layout_chars": 0,
            "mupdf_err": 1,
            "duplicate_blocks_removed": 0,
            "margin_blocks_removed": 0,
        }

    repeated_margin = collect_repeated_margin_lines(doc)
    global_seen = set()

    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        try:
            blocks = page.get_text("blocks")
            clean_blocks = []
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], b[4]
                t = normalize_text(text)
                if len(t) < 10:
                    continue
                clean_blocks.append((x0, y0, x1, y1, t))

            h = page.rect.height
            filtered = []
            for x0, y0, x1, y1, t in clean_blocks:
                if t in repeated_margin["top"] and y1 < 0.12 * h:
                    margin_blocks_removed += 1
                    continue
                if t in repeated_margin["bottom"] and y0 > 0.88 * h:
                    margin_blocks_removed += 1
                    continue
                if y1 < 0.04 * h or y0 > 0.965 * h:
                    margin_blocks_removed += 1
                    continue
                sig = (pno, re.sub(r"\W+", "", t.lower())[:180])
                if sig in global_seen:
                    duplicate_blocks_removed += 1
                    continue
                global_seen.add(sig)
                filtered.append((x0, y0, x1, y1, t))

            xs = sorted([b[0] for b in filtered])
            col_cut = None
            if len(xs) >= 10:
                w = page.rect.width
                mid = w * 0.5
                left = [x for x in xs if x < mid]
                right = [x for x in xs if x >= mid]
                if len(left) > 3 and len(right) > 3:
                    col_cut = mid

            def sort_key(b):
                x0, y0, x1, y1, _t = b
                col = 1 if (col_cut is not None and x0 >= col_cut) else 0
                return (col, y0, x0)

            filtered.sort(key=sort_key)

            for order, (x0, y0, x1, y1, t) in enumerate(filtered):
                total_chars += len(t)
                records.append({
                    "page": pno + 1,
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": t,
                    "order": order,
                    "col": 1 if (col_cut is not None and x0 >= col_cut) else 0,
                    "parser_text_source": "blocks",
                })

        except Exception:
            mupdf_err = 1
            try:
                t = normalize_text(page.get_text("text") or "")
            except Exception:
                t = ""
            total_chars += len(t)
            records.append({
                "page": pno + 1,
                "bbox": None,
                "text": t,
                "order": 0,
                "col": 0,
                "parser_text_source": "page_fallback",
            })

    tmp = out_layout_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(out_layout_jsonl)

    return {
        "layout_blocks": len(records),
        "layout_chars": total_chars,
        "mupdf_err": mupdf_err,
        "duplicate_blocks_removed": duplicate_blocks_removed,
        "margin_blocks_removed": margin_blocks_removed,
    }


def extract_captions(layout_jsonl: Path, out_captions_jsonl: Path) -> Dict[str, Any]:
    blocks = []
    with layout_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            blocks.append(json.loads(line))

    captions = []
    i = 0
    while i < len(blocks):
        b = blocks[i]
        t = normalize_text(b.get("text", ""))
        kind = None
        if FIG_PAT.match(t):
            kind = "figure"
        elif TAB_PAT.match(t):
            kind = "table"
        if kind is None:
            i += 1
            continue

        page = b["page"]
        texts = [t]
        bbox = b.get("bbox")[:] if isinstance(b.get("bbox"), list) else None
        j = i + 1
        while j < len(blocks):
            nb = blocks[j]
            if nb["page"] != page:
                break
            nt = normalize_text(nb.get("text", ""))
            if FIG_PAT.match(nt) or TAB_PAT.match(nt):
                break
            if bbox is not None and isinstance(nb.get("bbox"), list):
                if nb["bbox"][1] - bbox[3] > 60:
                    break
                bbox = [
                    min(bbox[0], nb["bbox"][0]),
                    min(bbox[1], nb["bbox"][1]),
                    max(bbox[2], nb["bbox"][2]),
                    max(bbox[3], nb["bbox"][3]),
                ]
            elif len(" ".join(texts)) > 1500:
                break
            texts.append(nt)
            j += 1

        label = texts[0].split(":")[0].strip()
        captions.append({
            "kind": kind,
            "label": label,
            "page": page,
            "bbox": bbox,
            "text": " ".join(texts),
        })
        i = j

    tmp = out_captions_jsonl.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in captions:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(out_captions_jsonl)

    return {"n_captions": len(captions)}


def grobid_fulltext(pdf_path: Path, out_tei: Path, grobid_url: str,
                    timeout: int = 240, retries: int = 2, backoff: float = 2.0) -> Dict[str, Any]:
    endpoint = grobid_url.rstrip("/") + "/api/processFulltextDocument"
    payload = {"input": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(endpoint, files=payload, timeout=timeout)
            if r.status_code == 200 and r.text.strip().startswith("<?xml"):
                tmp = out_tei.with_suffix(".xml.tmp")
                tmp.write_text(r.text, encoding="utf-8")
                tmp.replace(out_tei)
                return {"grobid_ok": True}
            last_err = f"status={r.status_code}, head={r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        time.sleep(backoff * attempt)

    return {"grobid_ok": False, "grobid_msg": last_err}


def parse_tei_sections(tei_path: Path) -> List[Dict[str, Any]]:
    if not tei_path.exists():
        return []
    parser = etree.XMLParser(recover=True)
    root = etree.parse(str(tei_path), parser).getroot()
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}
    body = root.find(".//tei:text/tei:body", namespaces=ns)
    if body is None:
        return []

    def get_text(el) -> str:
        txt = " ".join(el.itertext())
        return normalize_text(txt)

    sections = []

    def walk_div(div, path: List[str]):
        head = div.find("tei:head", namespaces=ns)
        title = get_text(head) if head is not None else None
        new_path = path + ([title] if title else [])
        ps = div.findall("./tei:p", namespaces=ns)
        text = " ".join(get_text(p) for p in ps)
        text = normalize_text(text)
        if text:
            sections.append({"section_path": " > ".join([p for p in new_path if p]), "text": text})
        for child in div.findall("tei:div", namespaces=ns):
            walk_div(child, new_path)

    for div in body.findall("tei:div", namespaces=ns):
        walk_div(div, [])

    if not sections:
        whole = get_text(body)
        if whole:
            sections = [{"section_path": "BODY", "text": whole}]
    return sections


def guess_table_pages_fast(pdf_path: Path, max_pages_scan: int = 100) -> List[int]:
    doc = fitz.open(pdf_path)
    n = doc.page_count
    pages = []
    for i in range(min(n, max_pages_scan)):
        t = doc.load_page(i).get_text("text")
        if re.search(r"\b(Table|TAB\.?)\s*\d+\b", t, re.I):
            pages.append(i + 1)
    if not pages:
        pages = list(range(1, min(n, 6) + 1))
    return pages


def extract_tables(pdf_path: Path, tables_dir: Path, engine: str = "camelot") -> Dict[str, Any]:
    ensure_dir(tables_dir)
    n_tables = 0
    meta = []
    doc = fitz.open(pdf_path)

    if engine == "camelot":
        try:
            import camelot
            pages = guess_table_pages_fast(pdf_path)
            page_str = ",".join(map(str, pages))

            for flavor in ["lattice", "stream"]:
                try:
                    tables = camelot.read_pdf(str(pdf_path), pages=page_str, flavor=flavor)
                    for t in tables:
                        df = t.df
                        if df is None or df.empty:
                            continue
                        if df.shape[0] < 2 and df.shape[1] < 2:
                            continue
                        preview = df.head(30).fillna("").astype(str).to_csv(index=False)
                        if len(normalize_text(preview)) < 20:
                            continue
                        n_tables += 1
                        csv_path = tables_dir / f"table_{n_tables:03d}_{engine}_{flavor}.csv"
                        df.to_csv(csv_path, index=False)

                        bbox = None
                        try:
                            bbox = list(map(float, t._bbox))
                        except Exception:
                            bbox = None

                        rec = {
                            "engine": engine,
                            "flavor": flavor,
                            "csv": csv_path.name,
                            "bbox": bbox,
                            "page": t.page,
                            "n_rows": int(df.shape[0]),
                            "n_cols": int(df.shape[1]),
                            "preview": preview[:4000],
                        }
                        meta.append(rec)
                except Exception:
                    continue
        except Exception as e:
            return {"n_tables": 0, "tables_err": f"camelot import/read failed: {e}"}

    elif engine == "tabula":
        try:
            import tabula
            dfs = tabula.read_pdf(str(pdf_path), pages="all", multiple_tables=True)
            for df in dfs:
                if df is None or df.empty:
                    continue
                n_tables += 1
                csv_path = tables_dir / f"table_{n_tables:03d}_{engine}.csv"
                df.to_csv(csv_path, index=False)
                meta.append({
                    "engine": engine,
                    "csv": csv_path.name,
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                    "preview": df.head(30).fillna("").astype(str).to_csv(index=False)[:4000],
                })
        except Exception as e:
            return {"n_tables": 0, "tables_err": f"tabula failed (need java?): {e}"}

    for m in meta:
        if "page" in m and m.get("bbox") and len(m["bbox"]) == 4:
            try:
                pno = int(m["page"]) - 1
                if pno < 0 or pno >= doc.page_count:
                    continue
                page = doc.load_page(pno)
                x1, y1, x2, y2 = m["bbox"]
                ph = page.rect.height
                clip = fitz.Rect(x1, ph - y2, x2, ph - y1)
                pix = page.get_pixmap(clip=clip, dpi=150)
                png_name = Path(m["csv"]).with_suffix(".png").name
                pix.save(str(tables_dir / png_name))
                m["png"] = png_name
            except Exception:
                continue

    if meta:
        with (tables_dir / "tables_meta.jsonl").open("w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    return {"n_tables": n_tables}


def chunk_text(text: str, max_chars: int = 3600, overlap: int = 300) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    out = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        out.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out


def fuse_chunks(paper_meta: Dict[str, Any],
                tei_path: Path,
                layout_jsonl: Path,
                captions_jsonl: Path,
                tables_dir: Path,
                out_chunks: Path) -> Dict[str, Any]:
    chunks = []
    pid = paper_meta["paper_id"]
    text_chunk_count = 0
    caption_chunk_count = 0
    table_chunk_count = 0

    if captions_jsonl.exists():
        with captions_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                c = json.loads(line)
                caption_chunk_count += 1
                chunks.append({
                    "paper_id": pid,
                    "doc_type": paper_meta.get("doc_type", "main"),
                    "reaction": paper_meta["reaction"],
                    "bucket": paper_meta["bucket"],
                    "relpath": paper_meta["relpath"],
                    "chunk_id": f"{pid}_p{int(c['page']):03d}_caption_{caption_chunk_count:04d}",
                    "chunk_type": "caption",
                    "section_path": "CAPTION",
                    "page_start": int(c["page"]),
                    "page_end": int(c["page"]),
                    "bbox": c.get("bbox"),
                    "source": "layout_caption",
                    "text_source_rank": 1,
                    "text": c.get("text", ""),
                })

    if tables_dir.exists():
        meta_map = {}
        meta_path = tables_dir / "tables_meta.jsonl"
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    meta_map[rec.get("csv")] = rec
        for csv in sorted(tables_dir.glob("table_*.csv")):
            try:
                df = pd.read_csv(csv)
                preview = df.head(80).fillna("").astype(str).to_csv(index=False)
            except Exception:
                preview = csv.read_text(encoding="utf-8", errors="ignore")[:8000]
            table_chunk_count += 1
            m = meta_map.get(csv.name, {})
            chunks.append({
                "paper_id": pid,
                "doc_type": paper_meta.get("doc_type", "main"),
                "reaction": paper_meta["reaction"],
                "bucket": paper_meta["bucket"],
                "relpath": paper_meta["relpath"],
                "chunk_id": f"{pid}_table_{table_chunk_count:04d}",
                "chunk_type": "table",
                "section_path": "TABLE",
                "page_start": m.get("page"),
                "page_end": m.get("page"),
                "bbox": m.get("bbox"),
                "source": "table_extract",
                "text_source_rank": 1,
                "text": preview[:8000],
            })

    sections = parse_tei_sections(tei_path)
    source_name = "tei" if tei_path.exists() and sections else "layout_fallback"
    if not sections and layout_jsonl.exists():
        pages = defaultdict(list)
        with layout_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                b = json.loads(line)
                pages[b["page"]].append((b.get("order", 0), b.get("text", "")))
        for p in sorted(pages):
            text = " ".join(t for _, t in sorted(pages[p], key=lambda x: x[0]))
            sections.append({"section_path": f"PAGE_{p:03d}", "text": text})

    for s in sections:
        spath = s.get("section_path") or "BODY"
        for part in chunk_text(s.get("text", "")):
            if len(part) < 30:
                continue
            text_chunk_count += 1
            chunks.append({
                "paper_id": pid,
                "doc_type": paper_meta.get("doc_type", "main"),
                "reaction": paper_meta["reaction"],
                "bucket": paper_meta["bucket"],
                "relpath": paper_meta["relpath"],
                "chunk_id": f"{pid}_text_{text_chunk_count:04d}",
                "chunk_type": "text",
                "section_path": spath,
                "page_start": None,
                "page_end": None,
                "bbox": None,
                "source": source_name,
                "text_source_rank": 0 if source_name == "tei" else 2,
                "text": part,
            })

    tmp = out_chunks.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    tmp.replace(out_chunks)

    return {
        "n_chunks": len(chunks),
        "n_text_chunks": text_chunk_count,
        "n_caption_chunks": caption_chunk_count,
        "n_table_chunks": table_chunk_count,
        "used_tei": int(source_name == "tei"),
    }


def summarize_chunks(chunks_path: Path) -> Dict[str, Any]:
    n_text = 0
    n_caption = 0
    n_table = 0
    total_chars = 0
    pages_with_table = set()
    cue_counts = Counter()
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            txt = r.get("text", "") or ""
            total_chars += len(txt)
            ctype = r.get("chunk_type")
            if ctype == "text":
                n_text += 1
            elif ctype == "caption":
                n_caption += 1
            elif ctype == "table":
                n_table += 1
                if r.get("page_start") is not None:
                    pages_with_table.add(r.get("page_start"))
            if PH_PAT.search(txt):
                cue_counts["has_pH"] += 1
            if HADS_PAT.search(txt):
                cue_counts["has_H_adsorption"] += 1
            if DEPROT_PAT.search(txt):
                cue_counts["has_deprotonation"] += 1
            if OH_WATER_PAT.search(txt):
                cue_counts["has_interfacial_water"] += 1
            if SITE_PAT.search(txt):
                cue_counts["has_site_system"] += 1
            if ELEC_PAT.search(txt):
                cue_counts["has_electronic_descriptor"] += 1
            if DOI_PAT.search(txt):
                cue_counts["has_doi"] += 1
    return {
        "n_text_chunks": n_text,
        "n_caption_chunks": n_caption,
        "n_table_chunks": n_table,
        "chunk_total_chars": total_chars,
        "pages_with_table": len(pages_with_table),
        **cue_counts,
    }


def process_one(pdf_str: str,
                root_str: str,
                out_str: str,
                grobid_url: str,
                no_grobid: bool,
                table_engines: List[str],
                force: bool,
                grobid_timeout: int,
                grobid_retries: int) -> Dict[str, Any]:
    pdf = Path(pdf_str)
    root = Path(root_str)
    out_root = Path(out_str)
    papers_root = out_root / "papers"

    try:
        reaction, bucket, relpath = guess_labels(root, pdf)
        fid = sha1_of_file(pdf)
        paper_id = fid[:16]
        paper_dir = papers_root / paper_id
        ensure_dir(paper_dir)

        chunks_path = paper_dir / "chunks.jsonl"
        qc_json_path = paper_dir / "paper_qc.json"
        if (not force) and chunks_path.exists() and chunks_path.stat().st_size > 1500 and qc_json_path.exists():
            qc_row = json.loads(qc_json_path.read_text(encoding="utf-8"))
            return {
                "ok": True,
                "skipped": True,
                "paper_id": paper_id,
                "reaction": reaction,
                "bucket": bucket,
                "relpath": relpath,
                "input_pdf": str(pdf),
                "out_dir": str(paper_dir),
                "qc": qc_row,
            }

        in_copy = paper_dir / "input.pdf"
        if (not in_copy.exists()) or force:
            shutil.copy2(pdf, in_copy)

        scaninfo = detect_scanned(in_copy)
        meta = {
            "paper_id": paper_id,
            "sha1": fid,
            "reaction": reaction,
            "bucket": bucket,
            "relpath": relpath,
            "doc_type": "main",
            "parsed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **scaninfo,
        }
        (paper_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        layout_jsonl = paper_dir / "layout.jsonl"
        layout_stat = extract_layout(in_copy, layout_jsonl)

        captions_jsonl = paper_dir / "captions.jsonl"
        cap_stat = extract_captions(layout_jsonl, captions_jsonl)

        tables_dir = paper_dir / "tables"
        tab_stat_all = {"n_tables": 0}
        for eng in table_engines:
            stat = extract_tables(in_copy, tables_dir, engine=eng)
            tab_stat_all["n_tables"] = max(tab_stat_all["n_tables"], stat.get("n_tables", 0))
            if "tables_err" in stat:
                tab_stat_all[f"{eng}_err"] = stat["tables_err"]

        tei_path = paper_dir / "tei.xml"
        grobid_stat = {"grobid_ok": False, "grobid_status": ""}
        if not no_grobid:
            g = grobid_fulltext(in_copy, tei_path, grobid_url, timeout=grobid_timeout, retries=grobid_retries)
            grobid_stat.update(g)
            if not g.get("grobid_ok"):
                grobid_stat["grobid_status"] = "FAIL"
            else:
                grobid_stat["grobid_status"] = "OK"
        else:
            grobid_stat["grobid_status"] = "DISABLED"

        fuse_stat = fuse_chunks(meta, tei_path, layout_jsonl, captions_jsonl, tables_dir, chunks_path)
        chunk_stat = summarize_chunks(chunks_path)

        avg_chars_per_page = round(layout_stat.get("layout_chars", 0) / max(1, meta["n_pages"]), 2)
        parser_warning = []
        if meta["is_scanned"]:
            parser_warning.append("scanned_pdf")
        if grobid_stat.get("grobid_status") == "FAIL":
            parser_warning.append("grobid_fail")
        if chunk_stat.get("has_H_adsorption", 0) == 0:
            parser_warning.append("no_H_adsorption_signal")
        if chunk_stat.get("has_deprotonation", 0) == 0:
            parser_warning.append("no_deprotonation_signal")
        if chunk_stat.get("has_site_system", 0) == 0:
            parser_warning.append("no_site_or_system_signal")

        qc_row = {
            "paper_id": paper_id,
            "reaction": reaction,
            "bucket": bucket,
            "relpath": relpath,
            "n_pages": meta["n_pages"],
            "avg_chars_first_pages": meta["avg_chars_first_pages"],
            "avg_chars_per_page": avg_chars_per_page,
            "is_scanned": meta["is_scanned"],
            "layout_blocks": layout_stat.get("layout_blocks"),
            "layout_chars": layout_stat.get("layout_chars"),
            "duplicate_blocks_removed": layout_stat.get("duplicate_blocks_removed"),
            "margin_blocks_removed": layout_stat.get("margin_blocks_removed"),
            "n_captions": cap_stat.get("n_captions"),
            "n_tables": tab_stat_all.get("n_tables"),
            "grobid_ok": grobid_stat.get("grobid_ok"),
            "grobid_status": grobid_stat.get("grobid_status", ""),
            "n_chunks": fuse_stat.get("n_chunks"),
            "n_text_chunks": chunk_stat.get("n_text_chunks"),
            "n_caption_chunks": chunk_stat.get("n_caption_chunks"),
            "n_table_chunks": chunk_stat.get("n_table_chunks"),
            "chunk_total_chars": chunk_stat.get("chunk_total_chars"),
            "pages_with_table": chunk_stat.get("pages_with_table"),
            "cue_pH": chunk_stat.get("has_pH", 0),
            "cue_H_adsorption": chunk_stat.get("has_H_adsorption", 0),
            "cue_deprotonation": chunk_stat.get("has_deprotonation", 0),
            "cue_interfacial_water": chunk_stat.get("has_interfacial_water", 0),
            "cue_site_system": chunk_stat.get("has_site_system", 0),
            "cue_electronic_descriptor": chunk_stat.get("has_electronic_descriptor", 0),
            "cue_doi": chunk_stat.get("has_doi", 0),
            # backward/forward compatible aliases for downstream triage readers
            "cue_has_pH": chunk_stat.get("has_pH", 0),
            "cue_has_H_adsorption": chunk_stat.get("has_H_adsorption", 0),
            "cue_has_deprotonation": chunk_stat.get("has_deprotonation", 0),
            "cue_has_interfacial_water": chunk_stat.get("has_interfacial_water", 0),
            "cue_has_site_system": chunk_stat.get("has_site_system", 0),
            "cue_has_electronic_descriptor": chunk_stat.get("has_electronic_descriptor", 0),
            "cue_has_doi": chunk_stat.get("has_doi", 0),
            "parser_warning": ";".join(parser_warning),
        }
        qc_json_path.write_text(json.dumps(qc_row, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "ok": True,
            "skipped": False,
            "paper_id": paper_id,
            "reaction": reaction,
            "bucket": bucket,
            "relpath": relpath,
            "input_pdf": str(pdf),
            "out_dir": str(paper_dir),
            "qc": qc_row,
        }

    except Exception as e:
        return {
            "ok": False,
            "skipped": False,
            "paper_id": None,
            "reaction": None,
            "bucket": None,
            "relpath": None,
            "input_pdf": str(pdf),
            "out_dir": None,
            "error": str(e),
            "qc": None,
        }


def build_summary(qc_df: pd.DataFrame, manifest_df: pd.DataFrame, errors_df: pd.DataFrame) -> Dict[str, Any]:
    summary = {
        "n_manifest": int(len(manifest_df)),
        "n_qc": int(len(qc_df)),
        "n_errors": int(len(errors_df)),
    }
    if len(qc_df) == 0:
        return summary

    summary.update({
        "n_scanned": int(pd.to_numeric(qc_df.get("is_scanned", 0), errors="coerce").fillna(0).astype(int).sum()),
        "n_with_tables": int((pd.to_numeric(qc_df.get("n_tables", 0), errors="coerce").fillna(0) > 0).sum()),
        "n_grobid_ok": int((qc_df.get("grobid_status", "") == "OK").sum()),
        "median_pages": float(pd.to_numeric(qc_df.get("n_pages", 0), errors="coerce").median()),
        "median_chunks": float(pd.to_numeric(qc_df.get("n_chunks", 0), errors="coerce").median()),
        "median_layout_chars": float(pd.to_numeric(qc_df.get("layout_chars", 0), errors="coerce").median()),
        "n_parser_warning": int((qc_df.get("parser_warning", "").fillna("") != "").sum()),
    })
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PDF root dir")
    ap.add_argument("--out", required=True, help="Output dir")
    ap.add_argument("--grobid-url", default="http://127.0.0.1:8070")
    ap.add_argument("--no-grobid", action="store_true")
    ap.add_argument("--table-engines", default="camelot", help="comma separated engines")
    ap.add_argument("--ext", default=".pdf")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--grobid-timeout", type=int, default=240)
    ap.add_argument("--grobid-retries", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out).resolve()
    papers_root = out_root / "papers"
    ensure_dir(papers_root)

    engines = [e.strip() for e in args.table_engines.split(",") if e.strip() and e.strip().lower() != "none"]
    pdfs = sorted([p for p in root.rglob(f"*{args.ext}") if p.is_file()])
    print(f"[INFO] found PDFs={len(pdfs)} under {root}")

    manifest_rows = []
    qc_rows = []
    err_rows = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [
            ex.submit(
                process_one,
                str(pdf),
                str(root),
                str(out_root),
                args.grobid_url,
                args.no_grobid,
                engines,
                args.force,
                args.grobid_timeout,
                args.grobid_retries,
            )
            for pdf in pdfs
        ]

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Parsing (parallel)"):
            res = fut.result()
            if res.get("ok"):
                manifest_rows.append({
                    "paper_id": res["paper_id"],
                    "reaction": res["reaction"],
                    "bucket": res["bucket"],
                    "relpath": res["relpath"],
                    "input_pdf": res["input_pdf"],
                    "out_dir": res["out_dir"],
                    "skipped": res.get("skipped", False),
                })
                if res.get("qc"):
                    qc_rows.append(res["qc"])
            else:
                err_rows.append({"input_pdf": res.get("input_pdf"), "error": res.get("error", "unknown")})

    ensure_dir(out_root)
    manifest_df = pd.DataFrame(manifest_rows)
    qc_df = pd.DataFrame(qc_rows)
    errors_df = pd.DataFrame(err_rows)

    manifest_df.to_csv(out_root / "manifest.csv", index=False)
    qc_df.to_csv(out_root / "qc_report.csv", index=False)
    if len(errors_df):
        errors_df.to_csv(out_root / "errors.csv", index=False)

    summary = build_summary(qc_df, manifest_df, errors_df)
    (out_root / "summary_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] PDFs={len(pdfs)} -> {out_root}")
    print(f"[DONE] manifest={out_root/'manifest.csv'} qc={out_root/'qc_report.csv'} summary={out_root/'summary_report.json'}")
    if len(errors_df):
        print(f"[DONE] errors={out_root/'errors.csv'}")


if __name__ == "__main__":
    main()
