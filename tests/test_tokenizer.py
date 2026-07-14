"""
Tokenizer tests — segmentation and normalization. These need no model weights,
so they run in CI on a fresh checkout.

    python -m pytest tests/            # or:  python tests/test_tokenizer.py
"""

from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from yapper_transliterator.tokenizer import (
    ctc_min_frames,
    normalize_source,
    normalize_target,
    required_slots,
    segment_aksharas,
)


def test_normalize_strips_joiners_and_nfd():
    # ZWJ / ZWNJ removed; output is NFD.
    raw = "क\u200dष"
    norm = normalize_source(raw)
    assert "\u200d" not in norm and "\u200c" not in norm
    assert norm == unicodedata.normalize("NFD", norm)


def test_conjunct_is_one_cluster():
    # क्षमा -> [क्ष, मा] : the virama-joined conjunct is a single akshara.
    clusters = segment_aksharas(normalize_source("क्षमा"))
    assert len(clusters) == 2, clusters
    assert "\u094d" in clusters[0]  # first cluster carries the virama


def test_matra_absorbed_into_cluster():
    # मा -> one cluster (base + aa-matra).
    clusters = segment_aksharas(normalize_source("मा"))
    assert len(clusters) == 1, clusters


def test_simple_word_akshara_count():
    # भारत -> भा, र, त  (3 aksharas, before bow/eow markers)
    clusters = segment_aksharas(normalize_source("भारत"))
    assert len(clusters) == 3, clusters


def test_cluster_cap():
    long = "क" + "\u093c" * 40  # base + many nuktas
    clusters = segment_aksharas(normalize_source(long), max_cluster_chars=16)
    assert all(len(c) <= 16 for c in clusters)


def test_normalize_target_casefold():
    assert normalize_target("  NaMaStE  ") == "namaste"
    assert normalize_target("NaMaStE", casefold=False) == "NaMaStE"


def test_ctc_min_frames():
    assert ctc_min_frames("namaste") == 7          # no adjacent repeats
    assert ctc_min_frames("express") == 8          # one repeated pair (ss)
    assert ctc_min_frames("") == 0


def test_required_slots():
    # slots-per-akshara = ceil(ctc_min_frames(target) / (num_aksharas + 2)).
    # नमस्ते -> [न, म, स्ते] = 3 aksharas; +2 markers = 5; ctc_min("namaste")=7.
    assert required_slots("नमस्ते", "namaste") == 2   # ceil(7/5)
    # भारत -> [भा, र, त] = 3; +2 = 5; ctc_min("bharat")=6.
    assert required_slots("भारत", "bharat") == 2      # ceil(6/5)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
