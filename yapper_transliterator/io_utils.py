"""
io_utils.py — read the yapper_map into (source, target) pairs.

Accepts the common shapes ``yapper_map.json`` may take:

  * ``{"देवनागरी": "roman", ...}``                       (dict, single target)
  * ``{"देवनागरी": ["roman", "romaan"], ...}``           (dict, list -> first)
  * ``[["देवनागरी", "roman"], ...]``                      (list of pairs)
  * ``[{"src": "...", "tgt": "..."}, ...]``               (list of dicts)
  * one JSON object per line (JSONL) of any of the above row shapes
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple


def _pair_from_row(row) -> Tuple[str, str] | None:
    if isinstance(row, dict):
        src = row.get("src") or row.get("source") or row.get("native")
        tgt = row.get("tgt") or row.get("target") or row.get("roman")
        if src is not None and tgt is not None:
            return str(src), str(tgt if not isinstance(tgt, list) else tgt[0])
        return None
    if isinstance(row, (list, tuple)) and len(row) >= 2:
        return str(row[0]), str(row[1])
    return None


def iter_map_entries(path: str) -> Iterator[Tuple[str, str]]:
    text = Path(path).read_text(encoding="utf-8")
    stripped = text.lstrip()

    # Whole-file JSON (dict or list).
    if stripped[:1] in "{[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            for source, target in data.items():
                if isinstance(target, list):
                    target = target[0] if target else ""
                yield str(source), str(target)
            return
        if isinstance(data, list):
            for row in data:
                pair = _pair_from_row(row)
                if pair:
                    yield pair
            return

    # Fall back to JSONL.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and len(row) == 1 and not (
            "src" in row or "source" in row or "native" in row
        ):
            for source, target in row.items():
                if isinstance(target, list):
                    target = target[0] if target else ""
                yield str(source), str(target)
            continue
        pair = _pair_from_row(row)
        if pair:
            yield pair
