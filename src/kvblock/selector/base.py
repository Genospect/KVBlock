"""Base selector records and helpers for the V1 heuristic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding


QuerySummary = Sequence[float] | torch.Tensor | SummaryEncoding | MultiHeadQuerySummary


@dataclass(frozen=True, slots=True)
class ScoredBlock:
    """Inspectable scored block record shared across selector stages."""

    metadata: BlockMetadata
    approx_similarity_score: float = 0.0
    recency_score: float = 0.0
    attn_score: float = 0.0
    priority_score: float = 0.0
    stage_a_score: float = 0.0
    hamming_similarity: float = 0.0
    stage_b_score: float = 0.0
    final_score: float = 0.0

    @property
    def block_id(self) -> BlockId:
        """Convenience accessor for the underlying block identifier."""

        return self.metadata.block_id

    @property
    def summary_similarity(self) -> float:
        """Compatibility alias for the approximate Stage A similarity feature."""

        return self.approx_similarity_score


@dataclass(frozen=True, slots=True)
class FinalSelection:
    """Stage C output broken down by preserved rails and semantic picks."""

    selected_blocks: tuple[ScoredBlock, ...]
    recent_blocks: tuple[ScoredBlock, ...] = field(default_factory=tuple)
    anchor_blocks: tuple[ScoredBlock, ...] = field(default_factory=tuple)
    semantic_blocks: tuple[ScoredBlock, ...] = field(default_factory=tuple)

    @property
    def selected_block_ids(self) -> tuple[BlockId, ...]:
        """Return selected block identifiers in Stage C output order."""

        return tuple(block.block_id for block in self.selected_blocks)


def coerce_query_summary(query_summary: QuerySummary) -> torch.Tensor:
    """Convert a synthetic query summary into a flat float tensor."""

    if isinstance(query_summary, MultiHeadQuerySummary):
        tensor = query_summary.dequantize()
    elif isinstance(query_summary, SummaryEncoding):
        tensor = query_summary.dequantize()
    elif isinstance(query_summary, torch.Tensor):
        tensor = query_summary.detach().to(dtype=torch.float32)
    else:
        tensor = torch.tensor(tuple(query_summary), dtype=torch.float32)

    tensor = tensor.reshape(-1)
    if tensor.numel() == 0:
        raise ValueError("query_summary must not be empty")
    return tensor


def metadata_summary_tensor(metadata: BlockMetadata) -> torch.Tensor:
    """Return the dequantized low-precision summary as an approximate float tensor."""

    return metadata.dequantize_summary()


def coerce_per_head_query_summary(query_summary: QuerySummary) -> torch.Tensor | None:
    """Return per-head query summaries when available."""

    if not isinstance(query_summary, MultiHeadQuerySummary):
        return None
    return query_summary.dequantize_heads()


def metadata_per_head_summary_tensor(metadata: BlockMetadata) -> torch.Tensor | None:
    """Return per-head block summaries when present."""

    if not metadata.per_head_summary_fp8:
        return None
    return metadata.dequantize_per_head_summary()
