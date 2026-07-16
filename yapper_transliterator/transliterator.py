"""
transliterator.py — the high-level, two-backend Devanagari -> Roman API.

Backends
--------
``map_lipi``  (default; this is the *yapper_transliterator* system)
    Look each word up in the curated ``yapper_map`` first. On a miss, fall back
    to the Lipi model. Best accuracy — the map handles the common case, Lipi
    generalises to OOV words. Reported in the benchmarks as ``yapper_map_fb_lipi``.

``lipi``
    Ignore the map; every word goes through the Lipi model. Measures what the
    model learned on its own. Reported in the benchmarks as ``lipi (ours)``.

Paths are optional. When ``model_dir`` / ``map_path`` are omitted they default to
``models/lipi`` and ``data/yapper_map.json`` next to the repository root (with a
fall-back to the current working directory), so a populated checkout just works::

    from yapper_transliterator import Transliterator

    t = Transliterator()                     # uses models/lipi + data/yapper_map.json
    t.transliterate_word("नमस्ते")          # -> "namaste"
    t.transliterate_text("मैंने कहा")        # -> "maine kaha"

    # explicit paths still work, e.g. for a checkout laid out differently:
    t = Transliterator(model_dir="/models/lipi", map_path="/data/yapper_map.json")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .data import BatchCollator, load_vocabs
from .decoding import ctc_prefix_beam_search, greedy_decode_batch
from .io_utils import iter_map_entries
from .runtime import load_model, validate_checkpoint_vocab
from .tokenizer import normalize_source, normalize_target

# Devanagari + Vedic extensions + zero-width joiners.
DEVANAGARI_SPAN = re.compile(
    r"[\u0900-\u0963\u0971-\u097f\ua8e0-\ua8ff\u200c\u200d]+"
)

# Devanagari punctuation and digits live inside the letter block. The span
# regex above deliberately excludes them so they don't get pulled into
# word matches, but they'd then echo through in the output as-is (e.g.
# 'है।' -> 'hai।'). Map them to ASCII so downstream text is uniform.
DEVA_PUNCT_TO_ASCII = str.maketrans({
    "\u0964": ".",   # । danda
    "\u0965": ".",   # ॥ double danda
    "\u0970": ".",   # ॰ abbreviation sign
    "\u0966": "0", "\u0967": "1", "\u0968": "2", "\u0969": "3", "\u096a": "4",
    "\u096b": "5", "\u096c": "6", "\u096d": "7", "\u096e": "8", "\u096f": "9",
})

BACKENDS = ("map_lipi", "lipi")

# Repository root is two levels up from this file
# (<repo>/yapper_transliterator/transliterator.py -> <repo>).
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = _REPO_ROOT / "models" / "lipi"
DEFAULT_MAP_PATH = _REPO_ROOT / "data" / "yapper_map.json"


def _first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class Transliterator:
    def __init__(
        self,
        model_dir: Optional[str] = None,
        map_path: Optional[str] = None,
        backend: str = "map_lipi",
        device: str = "cpu",
        beam_width: int = 1,
        beam_token_prune: int = 16,
        batch_size: int = 256,
        threads: int = 0,
        int8: bool = False,
    ) -> None:
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.backend = backend

        # Resolve model_dir: explicit > repo default > cwd default.
        if model_dir is None:
            resolved = _first_existing(
                [DEFAULT_MODEL_DIR, Path.cwd() / "models" / "lipi"]
            )
            if resolved is None:
                raise FileNotFoundError(
                    "No model directory found. Expected a populated 'models/lipi' "
                    f"at {DEFAULT_MODEL_DIR} or ./models/lipi, or pass "
                    "model_dir=... explicitly."
                )
            model_dir = str(resolved)
        self.beam_width = beam_width
        self.beam_token_prune = beam_token_prune
        self.batch_size = batch_size

        if threads > 0:
            torch.set_num_threads(threads)
            try:
                torch.set_num_interop_threads(max(1, min(4, threads)))
            except RuntimeError:
                pass

        self.device = torch.device(device)
        self.source_vocab, self.target_vocab, self.vocab_config = load_vocabs(model_dir)

        checkpoint_path = self._resolve_checkpoint(Path(model_dir))
        self.model, checkpoint = load_model(
            str(checkpoint_path), device=self.device, int8=int8
        )
        validate_checkpoint_vocab(
            checkpoint, self.source_vocab.tokens, self.target_vocab.tokens
        )

        self.collator = BatchCollator(
            self.source_vocab,
            self.target_vocab,
            max_cluster_chars=int(self.vocab_config["max_cluster_chars"]),
        )

        # Exact map (only used by the map_lipi backend).
        casefold = "casefold" in str(
            self.vocab_config.get("normalization", {}).get("target", "")
        )
        self.exact_map: Dict[str, str] = {}
        if backend == "map_lipi":
            if not map_path:
                resolved_map = _first_existing(
                    [DEFAULT_MAP_PATH, Path.cwd() / "data" / "yapper_map.json"]
                )
                if resolved_map is None:
                    raise FileNotFoundError(
                        "backend='map_lipi' needs yapper_map.json. Expected it at "
                        f"{DEFAULT_MAP_PATH} or ./data/yapper_map.json, or pass "
                        "map_path=... explicitly (or use backend='lipi')."
                    )
                map_path = str(resolved_map)
            for source, target in iter_map_entries(map_path):
                self.exact_map[normalize_source(source)] = normalize_target(
                    target, casefold=casefold
                )

        self._cache: Dict[str, str] = {}

    # ── checkpoint discovery ────────────────────────────────────────────────
    @staticmethod
    def _resolve_checkpoint(model_dir: Path) -> Path:
        for name in ("lipi.pt", "best.pt", "last.pt"):
            candidate = model_dir / name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"No checkpoint (lipi.pt / best.pt / last.pt) found in {model_dir}"
        )

    # ── model inference ─────────────────────────────────────────────────────
    def _predict_batch(self, normalized_words: List[str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        with torch.inference_mode():
            for start in range(0, len(normalized_words), self.batch_size):
                chunk = normalized_words[start : start + self.batch_size]
                batch = self.collator([{"src": w, "tgt": ""} for w in chunk]).to(self.device)
                logits, lengths = self.model(batch.source_ids, batch.source_lengths)

                if self.beam_width <= 1:
                    for word, pred in zip(
                        chunk, greedy_decode_batch(logits, lengths, self.target_vocab)
                    ):
                        out[word] = pred
                else:
                    log_probs = logits.float().log_softmax(dim=-1)
                    for row, (word, length) in enumerate(zip(chunk, lengths.tolist())):
                        cands = ctc_prefix_beam_search(
                            log_probs[row, :length],
                            self.target_vocab,
                            beam_width=self.beam_width,
                            token_prune=self.beam_token_prune,
                        )
                        out[word] = cands[0][0] if cands else ""
        return out

    def _model_predict(self, words: Sequence[str]) -> Dict[str, str]:
        """Normalise + dedupe + model-predict, memoised across calls."""
        norm_map = {w: normalize_source(w) for w in words}
        need = list({n for n in norm_map.values() if n and n not in self._cache})
        if need:
            self._cache.update(self._predict_batch(need))
        return {w: self._cache.get(n, w) for w, n in norm_map.items()}

    # ── public API ──────────────────────────────────────────────────────────
    def transliterate_words(self, words: Sequence[str]) -> Dict[str, str]:
        """Return ``{original_word: romanization}`` for each input word."""
        results: Dict[str, str] = {}
        to_model: List[str] = []
        for word in words:
            norm = normalize_source(word)
            if self.backend == "map_lipi" and norm in self.exact_map:
                results[word] = self.exact_map[norm]
            else:
                to_model.append(word)
        if to_model:
            results.update(self._model_predict(to_model))
        return results

    def transliterate_word(self, word: str) -> str:
        return self.transliterate_words([word])[word]

    def transliterate_text(self, text: str) -> str:
        """Replace every Devanagari span in ``text`` with its romanization.

        Devanagari punctuation (danda, double danda, abbreviation sign) and
        Devanagari digits are mapped to their ASCII equivalents before matching,
        so the output contains only ASCII in place of those codepoints.
        """
        text = text.translate(DEVA_PUNCT_TO_ASCII)

        # Prime the cache/map with all spans in one batch.
        spans = [m.group(0) for m in DEVANAGARI_SPAN.finditer(text)]
        resolved = self.transliterate_words(spans) if spans else {}

        def replace(match: re.Match[str]) -> str:
            return resolved.get(match.group(0), match.group(0))

        return DEVANAGARI_SPAN.sub(replace, text)
