"""Dense-representation to V1 block metadata ingest helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from kvblock.kv.block_modes import (
    BlockCandidate,
    BlockModeName,
    block_mode_from_name,
    generate_block_candidates,
)
from kvblock.kv.block_types import BlockId, TokenSpan
from kvblock.kv.metadata import BlockMetadata
from kvblock.kv.qk_aggregation import (
    QKAggregationStrategy,
    aggregate_block_states_for_summary,
    aggregate_query_key_heads,
    qk_aggregation_strategy_from_name,
)
from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding
from kvblock.summaries.fp8_summary import FP8SummaryBuilder
from kvblock.summaries.sign_sketch import generate_sign_sketch


@dataclass(frozen=True, slots=True)
class BlockIngestConfig:
    """Configuration for splitting dense token representations into metadata blocks."""

    block_size: int = 32
    summary_dim: int = 32
    pool_id: int = 0
    precision_tier: str = "fp16"
    representation_name: str = "final_hidden_state"
    qk_aggregation_strategy: QKAggregationStrategy = "mean_pool"
    top_token_count: int = 4
    block_mode: BlockModeName = "fixed"
    overlap_stride: int | None = None
    block_candidates: tuple[BlockCandidate, ...] = ()

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")
        if self.pool_id < 0:
            raise ValueError("pool_id must be >= 0")
        if not self.precision_tier.strip():
            raise ValueError("precision_tier must be non-empty")
        if not self.representation_name.strip():
            raise ValueError("representation_name must be non-empty")
        qk_aggregation_strategy_from_name(self.qk_aggregation_strategy)
        if self.top_token_count <= 0:
            raise ValueError("top_token_count must be > 0")
        block_mode_from_name(self.block_mode)
        if self.overlap_stride is not None and self.overlap_stride <= 0:
            raise ValueError("overlap_stride must be > 0 when set")
        if self.block_candidates:
            ids = [candidate.block_id for candidate in self.block_candidates]
            if len(set(ids)) != len(ids):
                raise ValueError("block_candidates must have unique block_id values")


@dataclass(frozen=True, slots=True)
class BlockIngestResult:
    """Metadata and query summary produced from one dense prefill representation."""

    metadata_blocks: tuple[BlockMetadata, ...]
    query_summary: SummaryEncoding | MultiHeadQuerySummary
    representation_name: str
    token_count: int
    block_size: int
    summary_dim: int
    block_mode: str = "fixed"
    block_candidates: tuple[BlockCandidate, ...] = ()

    def __post_init__(self) -> None:
        if not self.metadata_blocks:
            raise ValueError("metadata_blocks must not be empty")
        if self.token_count <= 0:
            raise ValueError("token_count must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")
        if self.block_candidates and len(self.block_candidates) != len(self.metadata_blocks):
            raise ValueError("block_candidates length must match metadata_blocks")

    @property
    def candidate_by_block_id(self) -> dict[int, BlockCandidate]:
        """Return block candidate sidecar records keyed by integer block id."""

        return {candidate.block_id: candidate for candidate in self.block_candidates}


def split_token_blocks(token_count: int, block_size: int) -> tuple[TokenSpan, ...]:
    """Split a token sequence into deterministic contiguous block spans."""

    if token_count <= 0:
        raise ValueError("token_count must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    spans: list[TokenSpan] = []
    for token_start in range(0, token_count, block_size):
        token_len = min(block_size, token_count - token_start)
        spans.append(TokenSpan(token_start=token_start, token_len=token_len))
    return tuple(spans)


def build_block_metadata_from_representations(
    token_representations: torch.Tensor,
    token_ids: Sequence[int],
    config: BlockIngestConfig | None = None,
    query_representation: torch.Tensor | None = None,
    per_head_token_representations: torch.Tensor | None = None,
    per_head_query_representation: torch.Tensor | None = None,
) -> BlockIngestResult:
    """Build V1 metadata from dense per-token model representations.

    The first bridge summarizes the selected dense token representation stream
    with the same low-precision summary/sketch path used by synthetic selector
    tests. Local HF sources may be hidden-state streams or K/V-adjacent key
    streams; the metadata schema stays source-agnostic in this pass.
    """

    resolved = config or BlockIngestConfig()
    representations = _normalize_representations(token_representations)
    query_vector = (
        representations[-1]
        if query_representation is None
        else _normalize_query_representation(query_representation)
    )
    per_head_representations = (
        None
        if per_head_token_representations is None
        else _normalize_per_head_representations(per_head_token_representations)
    )
    per_head_query = (
        None
        if per_head_query_representation is None
        else _normalize_per_head_query_representation(per_head_query_representation)
    )
    if (
        resolved.qk_aggregation_strategy in {"max_pool", "norm_weighted_mean"}
        and (per_head_representations is None or per_head_query is None)
    ):
        raise ValueError(
            f"{resolved.qk_aggregation_strategy} requires per-head query/key representations"
        )
    if representations.shape[0] != len(token_ids):
        raise ValueError("token_representations rows must match token_ids length")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    if query_vector.numel() != representations.shape[1]:
        raise ValueError("query_representation dim must match token representation dim")
    if per_head_representations is not None:
        if per_head_representations.shape[1] != len(token_ids):
            raise ValueError("per_head_token_representations tokens must match token_ids")
        if per_head_representations.shape[2] != representations.shape[1]:
            raise ValueError("per-head feature dim must match token representation dim")
    if per_head_query is not None:
        if per_head_representations is None:
            raise ValueError("per_head_token_representations are required with per_head_query_representation")
        if per_head_query.shape[0] != per_head_representations.shape[0]:
            raise ValueError("per-head query count must match per-head token representations")
        if per_head_query.shape[1] != per_head_representations.shape[2]:
            raise ValueError("per-head query feature dim must match per-head token representations")
        representations, query_vector = aggregate_query_key_heads(
            per_head_token_representations=per_head_representations,
            per_head_query_representation=per_head_query,
            strategy=resolved.qk_aggregation_strategy,
        )
    elif resolved.qk_aggregation_strategy in {"top_token_mean", "block_max"}:
        # These strategies alter block-level token pooling only, so they can run
        # on any source with an explicit query vector. Query/key experiments use
        # this with per-head data, but hidden-state smoke tests can still inspect
        # the behavior without changing selector code.
        pass

    builder = FP8SummaryBuilder(summary_dim=resolved.summary_dim)
    metadata_blocks: list[BlockMetadata] = []
    candidates = (
        resolved.block_candidates
        if resolved.block_candidates
        else generate_block_candidates(
            token_count=len(token_ids),
            mode=resolved.block_mode,
            default_block_size=resolved.block_size,
            overlap_stride=resolved.overlap_stride,
        )
    )
    if any(candidate.token_end > len(token_ids) for candidate in candidates):
        raise ValueError("block_candidates must fit within token_ids")
    for candidate in candidates:
        block_states = representations[candidate.token_start : candidate.token_end]
        summary_states = aggregate_block_states_for_summary(
            block_states,
            query_vector,
            strategy=resolved.qk_aggregation_strategy,
            top_token_count=resolved.top_token_count,
        )
        encoding = builder.build(summary_states)
        per_head_encodings = (
            ()
            if per_head_representations is None
            else tuple(
                builder.build(head_states[candidate.token_start : candidate.token_end])
                for head_states in per_head_representations
            )
        )
        metadata_blocks.append(
            BlockMetadata(
                block_id=BlockId(candidate.block_id),
                pool_id=resolved.pool_id,
                token_start=candidate.token_start,
                token_len=candidate.token_len,
                precision_tier=resolved.precision_tier,
                summary_fp8=encoding.values,
                summary_scale=encoding.scale,
                sign_sketch=generate_sign_sketch(encoding),
                summary_norm=encoding.summary_norm,
                # Dense prefill has no sparse access history yet. The token-end
                # proxy keeps recency rails deterministic for selector smoke runs.
                last_access_step=candidate.token_end - 1,
                rope_bucket=candidate.token_start // candidate.block_size,
                per_head_summary_fp8=tuple(
                    head_encoding.values for head_encoding in per_head_encodings
                ),
                per_head_summary_scale=tuple(
                    head_encoding.scale for head_encoding in per_head_encodings
                ),
                per_head_summary_norm=tuple(
                    head_encoding.summary_norm for head_encoding in per_head_encodings
                ),
            )
        )

    pooled_query_summary = builder.build(query_vector)
    query_summary: SummaryEncoding | MultiHeadQuerySummary
    if per_head_query is None:
        query_summary = pooled_query_summary
    else:
        query_summary = MultiHeadQuerySummary(
            pooled=pooled_query_summary,
            per_head=tuple(builder.build(head_query) for head_query in per_head_query),
        )
    return BlockIngestResult(
        metadata_blocks=tuple(metadata_blocks),
        query_summary=query_summary,
        representation_name=resolved.representation_name,
        token_count=len(token_ids),
        block_size=resolved.block_size,
        summary_dim=resolved.summary_dim,
        block_mode=resolved.block_mode,
        block_candidates=candidates,
    )


def _normalize_representations(token_representations: torch.Tensor) -> torch.Tensor:
    if not isinstance(token_representations, torch.Tensor):
        raise TypeError("token_representations must be a torch.Tensor")
    if token_representations.ndim == 3 and token_representations.shape[0] == 1:
        token_representations = token_representations.squeeze(0)
    if token_representations.ndim != 2:
        raise ValueError("token_representations must have shape [tokens, features]")
    if token_representations.shape[0] == 0:
        raise ValueError("token_representations must not be empty")
    if token_representations.shape[1] == 0:
        raise ValueError("token_representations feature dim must be > 0")
    return token_representations.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _normalize_query_representation(query_representation: torch.Tensor) -> torch.Tensor:
    if not isinstance(query_representation, torch.Tensor):
        raise TypeError("query_representation must be a torch.Tensor")
    if query_representation.ndim != 1:
        raise ValueError("query_representation must have shape [features]")
    if query_representation.numel() == 0:
        raise ValueError("query_representation must not be empty")
    return query_representation.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _normalize_per_head_representations(
    per_head_token_representations: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(per_head_token_representations, torch.Tensor):
        raise TypeError("per_head_token_representations must be a torch.Tensor")
    if per_head_token_representations.ndim != 3:
        raise ValueError(
            "per_head_token_representations must have shape [heads, tokens, features]"
        )
    if 0 in per_head_token_representations.shape:
        raise ValueError("per_head_token_representations dimensions must be non-empty")
    return per_head_token_representations.detach().to(
        dtype=torch.float32,
        device="cpu",
    ).contiguous()


def _normalize_per_head_query_representation(
    per_head_query_representation: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(per_head_query_representation, torch.Tensor):
        raise TypeError("per_head_query_representation must be a torch.Tensor")
    if per_head_query_representation.ndim != 2:
        raise ValueError("per_head_query_representation must have shape [heads, features]")
    if 0 in per_head_query_representation.shape:
        raise ValueError("per_head_query_representation dimensions must be non-empty")
    return per_head_query_representation.detach().to(
        dtype=torch.float32,
        device="cpu",
    ).contiguous()
