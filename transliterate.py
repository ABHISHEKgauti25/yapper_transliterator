#!/usr/bin/env python3
"""
transliterate.py — command-line Devanagari -> Roman transliteration.

Two backends (see README / ARCHITECTURE):

    map_lipi   yapper_map first, Lipi fallback  (default)
    lipi       Lipi model only

Paths default to ./models/lipi and ./data/yapper_map.json, which ship in the
repo, so the flags below are optional.

Examples
--------
    # single string, map + Lipi fallback
    python transliterate.py --text "नमस्ते दुनिया"

    # model-only, from a file, one line in one line out
    python transliterate.py --input-file in.txt --backend lipi

    # from stdin, JSON output with per-word provenance
    echo "मैंने कहा" | python transliterate.py --json

    # higher-quality (slower) beam decoding
    python transliterate.py --text "एक्सप्रेस" --backend lipi --beam-width 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

# Allow running from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from yapper_transliterator.transliterator import (  # noqa: E402
    BACKENDS,
    DEVANAGARI_SPAN,
    Transliterator,
)
from yapper_transliterator.tokenizer import normalize_source  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Yapper / Lipi transliteration")
    parser.add_argument("--model-dir", default=None,
                        help="dir with lipi.pt + configs (default: ./models/lipi)")
    parser.add_argument("--backend", choices=BACKENDS, default="map_lipi")
    parser.add_argument("--map", default=None,
                        help="yapper_map.json (default: ./data/yapper_map.json)")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text")
    group.add_argument("--input-file")

    parser.add_argument("--beam-width", type=int, default=1)
    parser.add_argument("--beam-token-prune", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON per input line with per-word provenance")
    return parser.parse_args()


def read_inputs(args: argparse.Namespace) -> List[str]:
    if args.text is not None:
        return [args.text]
    if args.input_file is not None:
        return Path(args.input_file).read_text(encoding="utf-8").splitlines()
    return [line.rstrip("\n") for line in sys.stdin]


def main() -> None:
    args = parse_args()

    transliterator = Transliterator(
        model_dir=args.model_dir,
        map_path=args.map,
        backend=args.backend,
        device=args.device,
        beam_width=args.beam_width,
        beam_token_prune=args.beam_token_prune,
        batch_size=args.batch_size,
        threads=args.threads,
        int8=args.int8,
    )

    for line in read_inputs(args):
        result = transliterate_line(transliterator, line, as_json=args.json)
        print(result)


def transliterate_line(transliterator: Transliterator, line: str, as_json: bool) -> str:
    if not as_json:
        return transliterator.transliterate_text(line)

    output = transliterator.transliterate_text(line)
    details = {}
    for match in DEVANAGARI_SPAN.finditer(line):
        word = match.group(0)
        norm = normalize_source(word)
        if transliterator.backend == "map_lipi" and norm in transliterator.exact_map:
            details[word] = {"source": "map", "output": transliterator.exact_map[norm]}
        else:
            details[word] = {"source": "lipi", "output": transliterator._cache.get(norm, word)}
    return json.dumps({"input": line, "output": output, "details": details}, ensure_ascii=False)


if __name__ == "__main__":
    main()
