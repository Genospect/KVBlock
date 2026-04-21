"""Benchmark-friendly metric helpers for selector analysis."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from kvblock.selector.oracle import BlockSetComparison

if TYPE_CHECKING:
    from kvblock.benchmark.selector_microbench import SelectorMicrobenchRow


@dataclass(frozen=True, slots=True)
class LogicalSelectionMetrics:
    """Logical selection-cost metrics for one sparse decision."""

    selected_page_count: int
    dense_page_count: int
    selected_pages_per_token: float
    logical_kv_read_bytes_per_token: float
    estimated_alpha: float


@dataclass(frozen=True, slots=True)
class SelectorQualityMetrics:
    """Precision/recall style selector-quality metrics."""

    selector_recall_rate: float
    selector_precision_rate: float
    overlap_count: int
    missed_important_count: int
    extra_selected_count: int


@dataclass(frozen=True, slots=True)
class FallbackFrequencySummary:
    """Aggregate fallback counts and rates."""

    total_runs: int
    counts: dict[str, int]
    rates: dict[str, float]


@dataclass(frozen=True, slots=True)
class SelectionSplitSummary:
    """Aggregate semantic-vs-rail selection split."""

    avg_semantic_selected_blocks: float
    avg_rail_preserved_blocks: float
    avg_semantic_fraction: float
    avg_rail_fraction: float


@dataclass(frozen=True, slots=True)
class ConfidenceSummary:
    """Aggregate confidence metrics across multiple runs."""

    avg_raw_margin: float
    avg_normalized_margin: float | None
    avg_selected_mass: float | None
    avg_normalized_mass: float | None
    confident_fraction: float


def logical_selection_metrics(
    *,
    selected_page_count: int,
    dense_page_count: int,
    page_bytes: int,
    output_tokens: int = 1,
) -> LogicalSelectionMetrics:
    """Compute logical bytes/token, pages/token, and alpha."""

    if selected_page_count < 0:
        raise ValueError("selected_page_count must be >= 0")
    if dense_page_count <= 0:
        raise ValueError("dense_page_count must be > 0")
    if page_bytes <= 0:
        raise ValueError("page_bytes must be > 0")
    if output_tokens <= 0:
        raise ValueError("output_tokens must be > 0")

    return LogicalSelectionMetrics(
        selected_page_count=selected_page_count,
        dense_page_count=dense_page_count,
        selected_pages_per_token=selected_page_count / output_tokens,
        logical_kv_read_bytes_per_token=(selected_page_count * page_bytes) / output_tokens,
        estimated_alpha=selected_page_count / dense_page_count,
    )


def selector_quality_metrics(comparison: BlockSetComparison) -> SelectorQualityMetrics:
    """Convert a block-set comparison into simple quality metrics."""

    return SelectorQualityMetrics(
        selector_recall_rate=comparison.recall_rate,
        selector_precision_rate=comparison.precision_rate,
        overlap_count=comparison.overlap_count,
        missed_important_count=len(comparison.missed_important_block_ids),
        extra_selected_count=len(comparison.extra_selected_block_ids),
    )


def aggregate_fallback_frequency(
    rows: Sequence[SelectorMicrobenchRow],
) -> FallbackFrequencySummary:
    """Aggregate fallback mode counts and rates from microbench rows."""

    counter = Counter(row.fallback_mode for row in rows)
    total_runs = len(rows)
    rates = (
        {mode: count / total_runs for mode, count in counter.items()} if total_runs else {}
    )
    return FallbackFrequencySummary(
        total_runs=total_runs,
        counts=dict(counter),
        rates=rates,
    )


def average_selection_split(rows: Sequence[SelectorMicrobenchRow]) -> SelectionSplitSummary:
    """Compute average semantic-vs-rail selected block counts and fractions."""

    if not rows:
        return SelectionSplitSummary(
            avg_semantic_selected_blocks=0.0,
            avg_rail_preserved_blocks=0.0,
            avg_semantic_fraction=0.0,
            avg_rail_fraction=0.0,
        )

    semantic_avg = sum(row.semantic_selected_block_count for row in rows) / len(rows)
    rail_avg = sum(row.rail_preserved_block_count for row in rows) / len(rows)

    semantic_fractions: list[float] = []
    rail_fractions: list[float] = []
    for row in rows:
        total = row.final_selected_block_count
        if total <= 0:
            semantic_fractions.append(0.0)
            rail_fractions.append(0.0)
        else:
            semantic_fractions.append(row.semantic_selected_block_count / total)
            rail_fractions.append(row.rail_preserved_block_count / total)

    return SelectionSplitSummary(
        avg_semantic_selected_blocks=semantic_avg,
        avg_rail_preserved_blocks=rail_avg,
        avg_semantic_fraction=sum(semantic_fractions) / len(semantic_fractions),
        avg_rail_fraction=sum(rail_fractions) / len(rail_fractions),
    )


def summarize_confidence(rows: Sequence[SelectorMicrobenchRow]) -> ConfidenceSummary:
    """Aggregate confidence metrics from microbench rows."""

    if not rows:
        return ConfidenceSummary(
            avg_raw_margin=0.0,
            avg_normalized_margin=None,
            avg_selected_mass=None,
            avg_normalized_mass=None,
            confident_fraction=0.0,
        )

    avg_raw_margin = sum(row.raw_margin for row in rows) / len(rows)
    avg_normalized_margin = _optional_average(
        [row.normalized_margin for row in rows if row.normalized_margin is not None]
    )
    avg_selected_mass = _optional_average(
        [row.selected_mass for row in rows if row.selected_mass is not None]
    )
    avg_normalized_mass = _optional_average(
        [row.normalized_mass for row in rows if row.normalized_mass is not None]
    )
    confident_fraction = sum(1 for row in rows if row.is_confident) / len(rows)

    return ConfidenceSummary(
        avg_raw_margin=avg_raw_margin,
        avg_normalized_margin=avg_normalized_margin,
        avg_selected_mass=avg_selected_mass,
        avg_normalized_mass=avg_normalized_mass,
        confident_fraction=confident_fraction,
    )


def _optional_average(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
