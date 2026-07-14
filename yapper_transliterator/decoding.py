"""
decoding.py — CTC decoders for Lipi.

Two decoders, matching ARCHITECTURE.md §6:

* ``greedy_decode_batch`` — argmax per frame, then collapse (remove consecutive
  repeats, then blanks). Single pass, linear time. This is the fast default.
* ``ctc_prefix_beam_search`` — standard log-space prefix beam decoder for
  higher-quality N-best output. Each prefix tracks blank / non-blank ending
  masses; the top ``token_prune`` tokens per frame are considered (blank always
  included).
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import List, Sequence, Tuple

import torch

NEG_INF = -1e30


def _logsumexp(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a >= b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def collapse_ctc(ids: Sequence[int], blank_id: int) -> List[int]:
    """Remove consecutive duplicates, then drop blanks."""
    out: List[int] = []
    prev = None
    for x in ids:
        if x != prev:
            if x != blank_id:
                out.append(x)
            prev = x
    return out


def greedy_decode_batch(logits, lengths, target_vocab) -> List[str]:
    """Greedy CTC decode a batch of logits ``[B, T, V]`` into Roman strings."""
    blank_id = target_vocab.blank_id if target_vocab.blank_id is not None else 0
    argmax = logits.argmax(dim=-1)  # [B, T]
    lengths = lengths.tolist() if hasattr(lengths, "tolist") else list(lengths)
    results: List[str] = []
    for row, length in zip(argmax, lengths):
        seq = row[: int(length)].tolist()
        collapsed = collapse_ctc(seq, blank_id)
        results.append(target_vocab.decode(collapsed))
    return results


def ctc_prefix_beam_search(
    log_probs: "torch.Tensor",
    target_vocab,
    beam_width: int = 8,
    token_prune: int = 16,
) -> List[Tuple[str, float]]:
    """Prefix beam search over ``log_probs`` of shape ``[T, V]``.

    Returns ``[(string, log_score), ...]`` sorted by descending score.
    """
    blank_id = target_vocab.blank_id if target_vocab.blank_id is not None else 0
    T, V = log_probs.shape
    token_prune = min(token_prune, V)

    # prefix (tuple of ids) -> [p_blank, p_nonblank] in log space
    beams = {(): [0.0, NEG_INF]}

    for t in range(T):
        row = log_probs[t]
        top_tokens = torch.topk(row, token_prune).indices.tolist()
        if blank_id not in top_tokens:
            top_tokens.append(blank_id)
        row = row.tolist()

        next_beams = defaultdict(lambda: [NEG_INF, NEG_INF])

        for prefix, (p_b, p_nb) in beams.items():
            p_total = _logsumexp(p_b, p_nb)
            last = prefix[-1] if prefix else None

            for c in top_tokens:
                lp = row[c]
                if c == blank_id:
                    entry = next_beams[prefix]
                    entry[0] = _logsumexp(entry[0], p_total + lp)
                elif c == last:
                    # repeat with a separating blank -> extends the prefix
                    ext = next_beams[prefix + (c,)]
                    ext[1] = _logsumexp(ext[1], p_b + lp)
                    # repeat with no blank -> stays on the same prefix
                    same = next_beams[prefix]
                    same[1] = _logsumexp(same[1], p_nb + lp)
                else:
                    ext = next_beams[prefix + (c,)]
                    ext[1] = _logsumexp(ext[1], p_total + lp)

        beams = dict(
            sorted(
                next_beams.items(),
                key=lambda kv: _logsumexp(kv[1][0], kv[1][1]),
                reverse=True,
            )[:beam_width]
        )

    scored = [
        (target_vocab.decode(prefix), _logsumexp(p_b, p_nb))
        for prefix, (p_b, p_nb) in beams.items()
    ]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored
