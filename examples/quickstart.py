#!/usr/bin/env python3
"""
quickstart.py — minimal library usage for both backends.

The checkpoint (models/lipi) and dictionary (data/yapper_map.json) ship in the
repo, so this runs as-is from the repo root:

    python examples/quickstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yapper_transliterator import Transliterator

SAMPLES = ["नमस्ते", "क्षमा", "एक्सप्रेस", "मैंने कहा कि भारत महान है"]


def run(backend: str) -> None:
    print(f"\n=== backend: {backend} ===")
    t = Transliterator(backend=backend)          # paths default to models/lipi + data/yapper_map.json
    for text in SAMPLES:
        print(f"  {text!r:40s} -> {t.transliterate_text(text)!r}")


if __name__ == "__main__":
    run("map_lipi")   # map + Lipi fallback (the yapper_transliterator system)
    run("lipi")       # Lipi model only
