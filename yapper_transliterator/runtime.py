"""
runtime.py — load a Lipi checkpoint and (optionally) quantise it.

Release checkpoint format (``models/lipi/lipi.pt``) is a single dict::

    {
      "config":        { ... LipiConfig.to_dict() ... },
      "state_dict":    { "char_embedding.weight": ..., ... },
      "source_tokens": [ ... ],   # optional, for validation
      "target_tokens": [ ... ],   # optional, for validation
    }

For robustness ``load_model`` also accepts:
  * a raw ``state_dict`` (keys like ``char_embedding.weight``), with config read
    from a sibling ``lipi_config.json`` / ``model_config.json``;
  * a dict whose weights live under ``state_dict`` / ``model_state`` / ``model``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence, Tuple

import torch
from torch import nn

from .lipi import Lipi, LipiConfig


_STATE_KEYS = ("state_dict", "model_state", "model")


def _looks_like_state_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and any(
        isinstance(k, str) and "." in k for k in obj.keys()
    )


def _read_sibling_config(checkpoint_path: Path) -> LipiConfig:
    for name in ("lipi_config.json", "model_config.json"):
        candidate = checkpoint_path.with_name(name)
        if candidate.exists():
            return LipiConfig.from_dict(json.loads(candidate.read_text(encoding="utf-8")))
    raise FileNotFoundError(
        f"No config found. Provide 'config' inside {checkpoint_path.name} or a "
        f"sibling lipi_config.json / model_config.json."
    )


def load_model(
    checkpoint: str,
    device: torch.device | str = "cpu",
    int8: bool = False,
) -> Tuple[nn.Module, Any]:
    """Load a Lipi checkpoint. Returns ``(model, raw_checkpoint_object)``."""
    checkpoint_path = Path(checkpoint)
    raw = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and "config" in raw:
        config = LipiConfig.from_dict(raw["config"])
        state = next((raw[k] for k in _STATE_KEYS if k in raw), None)
        if state is None and _looks_like_state_dict(raw):
            state = raw
    elif _looks_like_state_dict(raw):
        config = _read_sibling_config(checkpoint_path)
        state = raw
    elif isinstance(raw, dict):
        config = _read_sibling_config(checkpoint_path)
        state = next((raw[k] for k in _STATE_KEYS if k in raw), None)
    else:  # pragma: no cover
        raise ValueError(f"Unrecognised checkpoint object of type {type(raw)!r}")

    if state is None:
        raise ValueError(
            f"Could not locate weights in {checkpoint_path.name}. Expected one of "
            f"{_STATE_KEYS} or a raw state_dict."
        )

    model = Lipi(config)
    model.load_state_dict(state)
    model.eval().to(device)

    if int8:
        model = torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.GRU}, dtype=torch.qint8
        )

    return model, raw


def validate_checkpoint_vocab(
    checkpoint: Any,
    source_tokens: Sequence[str],
    target_tokens: Sequence[str],
) -> None:
    """Warn/raise if the vocab embedded in the checkpoint disagrees with vocab.json.

    A no-op (with a note) when the checkpoint carries no embedded vocab.
    """
    if not isinstance(checkpoint, dict):
        return
    ckpt_src = checkpoint.get("source_tokens")
    ckpt_tgt = checkpoint.get("target_tokens")
    if ckpt_src is None and ckpt_tgt is None:
        return
    if ckpt_src is not None and list(ckpt_src) != list(source_tokens):
        raise ValueError(
            "source vocab in checkpoint does not match vocab.json "
            f"({len(ckpt_src)} vs {len(source_tokens)} tokens)."
        )
    if ckpt_tgt is not None and list(ckpt_tgt) != list(target_tokens):
        raise ValueError(
            "target vocab in checkpoint does not match vocab.json "
            f"({len(ckpt_tgt)} vs {len(target_tokens)} tokens)."
        )
