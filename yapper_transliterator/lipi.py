"""
lipi.py — the Lipi model.

Lipi is a compact (~0.5 M parameter), non-autoregressive Devanagari -> Roman
transliterator. It is the out-of-vocabulary (OOV) fallback in the
``yapper_transliterator`` pipeline: the curated ``yapper_map`` resolves the
common case, and Lipi generalises to everything the map has not seen.

Architecture (an "akshara-level codepoint CTC"):

    Devanagari word
      -> akshara segmentation                 [B, A, L]
      -> codepoint embedding + cluster proj    [B, A, d]
      -> learned slot expansion (S per akshara)[B, A*S, d]
      -> bidirectional GRU encoder             [B, A*S, 2h]
      -> CTC head (log-softmax over V_tgt)      [B, A*S, V_tgt]
      -> greedy collapse / prefix beam         Roman string

See ``ARCHITECTURE.md`` for the full design rationale.

NOTE: the ``nn.Module`` submodule names (``char_embedding``,
``cluster_projection``, ``slot_embedding``, ``frame_norm``, ``encoder``,
``output``) are part of the checkpoint contract — renaming them would break
``state_dict`` loading of an existing Lipi checkpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LipiConfig:
    """Serialisable hyperparameter bundle for a Lipi checkpoint.

    Serialised form is exactly ``dataclasses.asdict(config)`` and is what
    ``models/lipi/lipi_config.json`` contains.
    """

    source_vocab_size: int
    target_vocab_size: int
    source_pad_id: int
    max_cluster_chars: int = 16      # L — codepoints per akshara cluster
    slots_per_akshara: int = 6       # S — CTC emission budget per akshara
    char_embedding_dim: int = 32     # d_c
    model_dim: int = 96              # d
    gru_hidden_size: int = 128       # h (per direction)
    gru_layers: int = 2
    dropout: float = 0.10

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "LipiConfig":
        # Tolerate extra keys from older configs.
        fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in fields})


class Lipi(nn.Module):
    """Ordered-codepoint akshara encoder + learned slot expansion + BiGRU CTC."""

    def __init__(self, config: LipiConfig) -> None:
        super().__init__()
        self.config = config
        self.char_embedding = nn.Embedding(
            config.source_vocab_size,
            config.char_embedding_dim,
            padding_idx=config.source_pad_id,
        )
        flattened_dim = config.max_cluster_chars * config.char_embedding_dim
        self.cluster_projection = nn.Sequential(
            nn.Linear(flattened_dim, config.model_dim),
            nn.GELU(),
            nn.LayerNorm(config.model_dim),
            nn.Dropout(config.dropout),
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(config.slots_per_akshara, config.model_dim)
        )
        self.frame_norm = nn.LayerNorm(config.model_dim)
        self.encoder = nn.GRU(
            input_size=config.model_dim,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_layers,
            batch_first=True,
            dropout=config.dropout if config.gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(config.gru_hidden_size * 2),
            nn.Dropout(config.dropout),
            nn.Linear(config.gru_hidden_size * 2, config.target_vocab_size),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_embedding, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, source_ids: torch.Tensor, source_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if source_ids.ndim != 3:
            raise ValueError("source_ids must have shape [batch, akshara, codepoint]")
        batch_size, akshara_count, cluster_chars = source_ids.shape
        if cluster_chars != self.config.max_cluster_chars:
            raise ValueError(
                f"Expected {self.config.max_cluster_chars} codepoints per cluster, "
                f"got {cluster_chars}"
            )

        embedded = self.char_embedding(source_ids)
        flattened = embedded.reshape(batch_size, akshara_count, -1)
        clusters = self.cluster_projection(flattened)

        frames = clusters.unsqueeze(2) + self.slot_embedding.view(
            1, 1, self.config.slots_per_akshara, self.config.model_dim
        )
        frames = frames.reshape(
            batch_size,
            akshara_count * self.config.slots_per_akshara,
            self.config.model_dim,
        )
        frames = self.frame_norm(frames)
        frame_lengths = source_lengths * self.config.slots_per_akshara

        packed = pack_padded_sequence(
            frames,
            frame_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        encoded, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=frames.shape[1],
        )
        logits = self.output(encoded)
        return logits, frame_lengths


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


