"""
tokenizer.py — source/target normalisation, akshara segmentation, and vocab.

This is the contract between raw text and Lipi's input. Its behaviour must match
the tokenizer used at training time exactly, or codepoint ids will not line up
with the trained ``char_embedding`` table. This module reproduces the training
tokenizer verbatim.

Source normalisation (``normalize_source``)
    strip → Unicode NFD → remove ZWJ (U+200D) / ZWNJ (U+200C).

Akshara segmentation (``segment_aksharas``)
    Self-normalises the input, skips whitespace, and grows one cluster per
    orthographic syllable: a base character absorbs following combining marks
    (Unicode category ``M*``); a virama (U+094D) additionally absorbs the
    following base so conjuncts such as क्ष stay whole; non-Devanagari
    characters are emitted as singletons (they absorb nothing); clusters longer
    than ``max_cluster_chars`` are split into consecutive chunks (nothing is
    dropped).

Target normalisation (``normalize_target``)
    strip → Unicode NFKC → optional case-fold.
"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Sequence


PAD = "<pad>"
UNK = "<unk>"
BOW = "<bow>"
EOW = "<eow>"
BLANK = "<blank>"

VIRAMA = "\u094d"
ZWJ = "\u200d"
ZWNJ = "\u200c"

DEVANAGARI_START = 0x0900
DEVANAGARI_END = 0x097F
DEVANAGARI_EXT_START = 0xA8E0
DEVANAGARI_EXT_END = 0xA8FF

# ``L`` in the architecture doc — the per-akshara codepoint budget. Also carried
# in vocab.json ``config.max_cluster_chars``; this is the default when absent.
MAX_CLUSTER_CHARS = 16


def normalize_source(text: str) -> str:
    """Canonicalise source spelling for model input.

    NFD makes canonically equivalent nukta spellings identical. Joiners are
    removed because they control glyph shaping, not the intended roman form.
    """
    text = unicodedata.normalize("NFD", str(text).strip())
    return text.replace(ZWJ, "").replace(ZWNJ, "")


def normalize_target(text: str, casefold: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(text).strip())
    return text.casefold() if casefold else text


def is_devanagari(ch: str) -> bool:
    cp = ord(ch)
    return (
        DEVANAGARI_START <= cp <= DEVANAGARI_END
        or DEVANAGARI_EXT_START <= cp <= DEVANAGARI_EXT_END
    )


def segment_aksharas(text: str, max_cluster_chars: int = MAX_CLUSTER_CHARS) -> List[str]:
    """Split text into orthographic syllable-like clusters.

    The input is normalised internally, so this is safe to call on raw text. A
    cluster starts at a base character and absorbs combining marks. A virama
    also absorbs the following base character, so conjuncts remain together.
    Non-Devanagari symbols are emitted as separate clusters.
    """
    text = normalize_source(text)
    clusters: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue

        current = [ch]
        i += 1

        if not is_devanagari(ch):
            clusters.append(ch)
            continue

        while i < n:
            nxt = text[i]

            if nxt == VIRAMA:
                current.append(nxt)
                i += 1
                if i < n and text[i] in (ZWJ, ZWNJ):
                    i += 1
                if i < n:
                    current.append(text[i])
                    i += 1
                continue

            category = unicodedata.category(nxt)
            if category.startswith("M"):
                current.append(nxt)
                i += 1
                continue

            break

        cluster = "".join(current)
        if len(cluster) <= max_cluster_chars:
            clusters.append(cluster)
        else:
            for start in range(0, len(cluster), max_cluster_chars):
                clusters.append(cluster[start : start + max_cluster_chars])

    return clusters


def ctc_min_frames(target: str) -> int:
    """Minimum CTC frames, including blanks needed between repeated labels."""
    if not target:
        return 0
    repeats = sum(a == b for a, b in zip(target, target[1:]))
    return len(target) + repeats


def required_slots(source: str, target: str, max_cluster_chars: int = MAX_CLUSTER_CHARS) -> int:
    """Minimum output slots per akshara needed for a (source, target) pair.

    ``ceil(ctc_min_frames(target) / (num_aksharas(source) + 2))``; the ``+2``
    accounts for the ``<bow>`` / ``<eow>`` markers. Used at data-prep time to
    choose ``slots_per_akshara`` and to filter pathological over-length targets.
    """
    # Two extra source positions are added for BOW and EOW.
    source_positions = len(segment_aksharas(source, max_cluster_chars)) + 2
    if source_positions <= 0:
        return 0
    return math.ceil(ctc_min_frames(target) / source_positions)


@dataclass(frozen=True)
class SourceVocabulary:
    tokens: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})
        for required in (PAD, UNK, BOW, EOW):
            if required not in self.token_to_id:
                raise ValueError(f"Missing source special token: {required}")

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def bow_id(self) -> int:
        return self.token_to_id[BOW]

    @property
    def eow_id(self) -> int:
        return self.token_to_id[EOW]

    def encode_word(self, text: str, max_cluster_chars: int) -> List[List[int]]:
        clusters = segment_aksharas(text, max_cluster_chars=max_cluster_chars)
        encoded: List[List[int]] = [[self.bow_id]]
        for cluster in clusters:
            encoded.append([self.token_to_id.get(ch, self.unk_id) for ch in cluster])
        encoded.append([self.eow_id])
        return encoded


@dataclass(frozen=True)
class TargetVocabulary:
    tokens: Sequence[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})
        if not self.tokens or self.tokens[0] != BLANK:
            raise ValueError("Target vocabulary must use <blank> at index 0")

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        missing = sorted({ch for ch in text if ch not in self.token_to_id})
        if missing:
            raise ValueError(f"Target contains characters absent from vocabulary: {missing!r}")
        return [self.token_to_id[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.tokens[i] for i in ids if i != self.blank_id)
