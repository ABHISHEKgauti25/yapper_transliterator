"""
data.py — vocabularies and batching for Lipi.

``vocab.json`` (canonical release format) looks like::

    {
      "source": ["<pad>", "<bow>", "<eow>", "<unk>", "क", "ा", "्", ...],
      "target": ["<blank>", "a", "b", "c", ...],
      "config": {
        "max_cluster_chars": 16,
        "slots_per_akshara": 6,
        "normalization": {"source": "nfd_strip_joiners", "target": "casefold"}
      }
    }

List index == token id. Special ids are resolved by *name* (not position), so
the ordering of the lists does not matter as long as the special tokens are
present. ``<blank>`` must exist in ``target`` (CTC assumes it, conventionally at
index 0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .tokenizer import (
    BLANK,
    BOW,
    EOW,
    PAD,
    UNK,
    normalize_source,
    segment_aksharas,
)


class Vocab:
    """A token<->id table with name-resolved special ids."""

    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens: List[str] = list(tokens)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self.tokens)}
        # Resolve common specials if present.
        self.pad_id = self.token_to_id.get(PAD)
        self.unk_id = self.token_to_id.get(UNK)
        self.bow_id = self.token_to_id.get(BOW)
        self.eow_id = self.token_to_id.get(EOW)
        self.blank_id = self.token_to_id.get(BLANK)

    def __len__(self) -> int:
        return len(self.tokens)

    def id_of(self, token: str) -> int:
        if self.unk_id is not None:
            return self.token_to_id.get(token, self.unk_id)
        return self.token_to_id[token]

    def decode(self, ids: Iterable[int]) -> str:
        """Join ids into a string, skipping special ``<...>`` tokens."""
        out = []
        for i in ids:
            tok = self.tokens[i]
            if tok.startswith("<") and tok.endswith(">"):
                continue
            out.append(tok)
        return "".join(out)


def load_vocabs(data_dir: str) -> Tuple[Vocab, Vocab, Dict[str, object]]:
    """Load ``vocab.json`` from ``data_dir`` -> (source_vocab, target_vocab, config)."""
    raw = json.loads((Path(data_dir) / "vocab.json").read_text(encoding="utf-8"))
    source_tokens = raw.get("source") or raw.get("source_tokens")
    target_tokens = raw.get("target") or raw.get("target_tokens")
    if source_tokens is None or target_tokens is None:
        raise ValueError(
            "vocab.json must contain 'source'/'target' (or "
            "'source_tokens'/'target_tokens') lists of tokens."
        )
    config: Dict[str, object] = dict(raw.get("config", {}))
    config.setdefault("max_cluster_chars", 16)
    config.setdefault("slots_per_akshara", 6)
    config.setdefault("normalization", {"source": "nfd_strip_joiners", "target": "casefold"})
    return Vocab(source_tokens), Vocab(target_tokens), config


@dataclass
class Batch:
    source_ids: torch.Tensor       # [B, A_max, L] long
    source_lengths: torch.Tensor   # [B] long (akshara counts, incl. bow/eow)

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            source_ids=self.source_ids.to(device),
            source_lengths=self.source_lengths.to(device),
        )


class BatchCollator:
    """Turns ``[{"src": word, "tgt": ...}, ...]`` into a padded source ``Batch``.

    Only the source side is materialised — that is all inference needs. Each
    word becomes a sequence of aksharas ``[<bow>, cluster_1, ..., <eow>]``; each
    akshara becomes ``L`` codepoint ids, right-padded with ``pad_id``.
    """

    def __init__(
        self,
        source_vocab: Vocab,
        target_vocab: Vocab,
        max_cluster_chars: int = 16,
    ) -> None:
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.max_cluster_chars = int(max_cluster_chars)
        self.pad_id = source_vocab.pad_id if source_vocab.pad_id is not None else 0
        if source_vocab.bow_id is None or source_vocab.eow_id is None:
            raise ValueError("source vocab is missing <bow>/<eow> markers")

    def encode_source(self, word: str) -> List[List[int]]:
        """Return a list of aksharas, each a list of codepoint ids (<= L)."""
        norm = normalize_source(word)
        clusters = segment_aksharas(norm, self.max_cluster_chars)
        aksharas: List[List[int]] = [[self.source_vocab.bow_id]]
        for cluster in clusters:
            aksharas.append([self.source_vocab.id_of(ch) for ch in cluster])
        aksharas.append([self.source_vocab.eow_id])
        return aksharas

    def __call__(self, items: Sequence[Dict[str, str]]) -> Batch:
        encoded = [self.encode_source(item["src"]) for item in items]
        lengths = [len(a) for a in encoded]
        batch_size = len(items)
        akshara_max = max(lengths) if lengths else 1
        L = self.max_cluster_chars

        source_ids = torch.full(
            (batch_size, akshara_max, L), self.pad_id, dtype=torch.long
        )
        for b, aksharas in enumerate(encoded):
            for a, ids in enumerate(aksharas):
                for l, tok in enumerate(ids[:L]):
                    source_ids[b, a, l] = tok

        return Batch(
            source_ids=source_ids,
            source_lengths=torch.tensor(lengths, dtype=torch.long),
        )
