#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import gzip
import zipfile
from pathlib import Path


def sha256_file(p: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    print(f"[DOWNLOAD] {url}")
    print(f"          -> {out_path}")
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        total = r.length
        downloaded = 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                print(f"\r          {downloaded/1e6:.1f}MB / {total/1e6:.1f}MB ({pct:.1f}%)", end="")
    print()
    tmp.replace(out_path)


def unpack_file(src: Path, mode: str) -> None:
    if mode == "none":
        return

    if mode == "zip":
        # unzip into the same folder
        with zipfile.ZipFile(src, "r") as z:
            z.extractall(src.parent)
        return

    if mode == "gz":
        # produce decompressed file without .gz suffix
        dst = src.with_suffix("")  # remove ".gz"
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            while True:
                chunk = f_in.read(1024 * 1024)
                if not chunk:
                    break
                f_out.write(chunk)
        return

    raise ValueError(f"Unknown unpack mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.json", help="Path to data manifest json")
    ap.add_argument("--root", default=".", help="Project root directory")
    ap.add_argument("--skip-hash", action="store_true", help="Skip sha256 verification")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest).resolve()
    if not manifest_path.exists():
        print(f"[ERROR] manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = (root / manifest.get("base_dir", "static/data")).resolve()

    for item in manifest["files"]:
        rel_path = item["path"]
        url = item["url"]
        sha256_expected = item.get("sha256", "").strip().lower()
        unpack = item.get("unpack", "none")

        # support "../upload/..." from base_dir
        target = (base_dir / rel_path).resolve()

        # If the url points to a compressed file, keep the filename as provided by URL if needed.
        # Here we download to the target path as described in manifest.
        download(url, target)

        if (not args.skip_hash) and sha256_expected and sha256_expected != "put_sha256_here":
            got = sha256_file(target)
            if got != sha256_expected:
                print(f"[ERROR] sha256 mismatch for {target}", file=sys.stderr)
                print(f" expected: {sha256_expected}", file=sys.stderr)
                print(f"      got: {got}", file=sys.stderr)
                sys.exit(2)
            print(f"[OK] sha256 verified: {target.name}")

        unpack_file(target, unpack)
        if unpack != "none":
            print(f"[OK] unpacked ({unpack}): {target.name}")

    print("[DONE] All data files downloaded.")


if __name__ == "__main__":
    main()
