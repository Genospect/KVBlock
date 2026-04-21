"""Dense-oracle scaffolding for backend-agnostic selector evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import QuerySummary
from kvblock.selector.stage_a import approx_cosine_similarity


@dataclass(frozen=True, slots=True)
class DenseReferenceBlock:
    """One block in a dense-oracle reference ranking."""

    block_id: BlockId
    importance_score: float


@dataclass(frozen=True, slots=True)
class DenseReferenceBlockSet:
    """Dense reference block set for one query or decode step."""

    block_ids: tuple[BlockId, ...]
    ranked_blocks: tuple[DenseReferenceBlock, ...] = field(default_factory=tuple)
    step_id: str | int | None = None

    @property
    def size(self) -> int:
        """Return the number of referenced dense blocks."""

        return len(self.block_ids)


@dataclass(frozen=True, slots=True)
class SparseSelectedBlockSet:
    """Sparse block set emitted by the selector."""

    block_ids: tuple[BlockId, ...]
    step_id: str | int | None = None

    @property
    def size(self) -> int:
        """Return the number of sparse-selected blocks."""

        return len(self.block_ids)


@dataclass(frozen=True, slots=True)
class BlockSetComparison:
    """Comparison between dense reference and sparse selection."""

    dense_reference: DenseReferenceBlockSet
    sparse_selection: SparseSelectedBlockSet
    overlap_block_ids: tuple[BlockId, ...]
    missed_important_block_ids: tuple[BlockId, ...]
    extra_selected_block_ids: tuple[BlockId, ...]
    recall_rate: float
    precision_rate: float

    @property
    def overlap_count(self) -> int:
        """Return the number of overlapping blocks."""

        return len(self.overlap_block_ids)


class DenseOracle(Protocol):
    """Protocol for oracle providers that emit dense reference block sets."""

    def reference_blocks(
        self,
        query_summary: QuerySummary,
        metadata_blocks: Sequence[BlockMetadata],
        *,
        step_id: str | int | None = None,
    ) -> DenseReferenceBlockSet:
        """Return the dense reference block set for one query."""


@dataclass(frozen=True, slots=True)
class SyntheticDenseOracleConfig:
    """Config for the synthetic dense-oracle approximation."""

    top_k: int = 8
    summary_weight: float = 0.8
    attn_weight: float = 0.15
    recency_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.summary_weight < 0 or self.attn_weight < 0 or self.recency_weight < 0:
            raise ValueError("Synthetic oracle weights must be >= 0")
        if self.summary_weight + self.attn_weight + self.recency_weight <= 0:
            raise ValueError("Synthetic oracle must have at least one positive weight")


class SyntheticDenseOracle:
    """Synthetic dense oracle built from metadata and query summaries.

    This is not a real dense-attention oracle. It is a deterministic reference
    signal that can be replaced later with a runtime-backed dense oracle while
    preserving the comparison API.
    """

    def __init__(self, config: SyntheticDenseOracleConfig | None = None) -> None:
        self.config = config or SyntheticDenseOracleConfig()

    def reference_blocks(
        self,
        query_summary: QuerySummary,
        metadata_blocks: Sequence[BlockMetadata],
        *,
        step_id: str | int | None = None,
    ) -> DenseReferenceBlockSet:
        if not metadata_blocks:
            return DenseReferenceBlockSet(block_ids=(), ranked_blocks=(), step_id=step_id)

        max_step = max(block.last_access_step for block in metadata_blocks)
        max_attn = max(block.attn_ema for block in metadata_blocks)
        ranked: list[DenseReferenceBlock] = []

        for block in metadata_blocks:
            recency = _oracle_recency_score(block.last_access_step, max_step=max_step)
            attn = 0.0 if max_attn <= 0 else block.attn_ema / max_attn
            score = (
                self.config.summary_weight * approx_cosine_similarity(query_summary, block)
                + self.config.attn_weight * attn
                + self.config.recency_weight * recency
            )
            ranked.append(DenseReferenceBlock(block_id=block.block_id, importance_score=score))

        ranked.sort(key=lambda item: (item.importance_score, int(item.block_id)), reverse=True)
        selected = tuple(ranked[: self.config.top_k])
        return DenseReferenceBlockSet(
            block_ids=tuple(item.block_id for item in selected),
            ranked_blocks=selected,
            step_id=step_id,
        )


def compare_block_sets(
    dense_reference: DenseReferenceBlockSet,
    sparse_selection: SparseSelectedBlockSet,
) -> BlockSetComparison:
    """Compare sparse selection against a dense reference block set."""

    dense_ids = set(dense_reference.block_ids)
    sparse_ids = set(sparse_selection.block_ids)

    overlap = tuple(
        block_id for block_id in dense_reference.block_ids if block_id in sparse_ids
    )
    missed = tuple(
        block_id for block_id in dense_reference.block_ids if block_id not in sparse_ids
    )
    extra = tuple(
        block_id for block_id in sparse_selection.block_ids if block_id not in dense_ids
    )

    recall = 1.0 if not dense_reference.block_ids else len(overlap) / len(dense_reference.block_ids)
    precision = (
        1.0 if not sparse_selection.block_ids else len(overlap) / len(sparse_selection.block_ids)
    )

    return BlockSetComparison(
        dense_reference=dense_reference,
        sparse_selection=sparse_selection,
        overlap_block_ids=overlap,
        missed_important_block_ids=missed,
        extra_selected_block_ids=extra,
        recall_rate=recall,
        precision_rate=precision,
    )


def sparse_selected_block_set(
    block_ids: Sequence[BlockId | int], *, step_id: str | int | None = None
) -> SparseSelectedBlockSet:
    """Build a sparse selected block set from ids."""

    normalized = tuple(
        block_id if isinstance(block_id, BlockId) else BlockId(int(block_id))
        for block_id in block_ids
    )
    return SparseSelectedBlockSet(block_ids=normalized, step_id=step_id)


def dense_reference_block_set(
    block_ids: Sequence[BlockId | int], *, step_id: str | int | None = None
) -> DenseReferenceBlockSet:
    """Build a dense reference block set from ids."""

    normalized = tuple(
        block_id if isinstance(block_id, BlockId) else BlockId(int(block_id))
        for block_id in block_ids
    )
    ranked = tuple(
        DenseReferenceBlock(block_id=block_id, importance_score=float(len(normalized) - index))
        for index, block_id in enumerate(normalized)
    )
    return DenseReferenceBlockSet(block_ids=normalized, ranked_blocks=ranked, step_id=step_id)


def _oracle_recency_score(last_access_step: int, *, max_step: int) -> float:
    if last_access_step < 0:
        return 0.0
    distance = max_step - last_access_step
    if distance < 0:
        distance = 0
    return 1.0 / (1.0 + float(distance))

