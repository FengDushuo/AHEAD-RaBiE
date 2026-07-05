#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union

import pandas as pd

PathLike = Union[str, Path]


def _p(path: PathLike) -> Path:
    return path if isinstance(path, Path) else Path(path)


def ensure_dir(p: PathLike) -> Path:
    path = _p(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(p: PathLike) -> Path:
    path = _p(p)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.parent


def atomic_write_text(text: str, path: PathLike) -> None:
    path = _p(path)
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def read_json(path: PathLike, default: Any = None) -> Any:
    path = _p(path)
    if not path.exists():
        return {} if default is None else default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {} if default is None else default


def write_json(obj: Any, path: PathLike, indent: int = 2) -> None:
    atomic_write_text(json.dumps(obj, ensure_ascii=False, indent=indent), path)


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    path = _p(path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def iter_jsonl(path: PathLike) -> Iterator[Dict[str, Any]]:
    path = _p(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def write_jsonl(rows: Iterable[Dict[str, Any]], path: PathLike) -> None:
    path = _p(path)
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(row: Dict[str, Any], path: PathLike) -> None:
    path = _p(path)
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_safe(path: PathLike) -> pd.DataFrame:
    path = _p(path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_progress(path: PathLike) -> Dict[str, Any]:
    obj = read_json(path, default={})
    return obj if isinstance(obj, dict) else {}


def save_progress(obj: Dict[str, Any], path: PathLike) -> None:
    write_json(obj, path)
