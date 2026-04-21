"""Flattened analysis records for selector microbenchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from kvblock.benchmark.metrics import (
    aggregate_fallback_frequency,
    summarize_confidence,
)
from kvblock.benchmark.selector_microbench import (
    SelectorMicrobenchCaseResult,
    SelectorMicrobenchRow,
)


@dataclass(frozen=True, slots=True)
class SelectorAnalysisRunRow:
    """Compact per-run analysis record derived from one microbench row."""

    case_id: str
    workload_profile: str
    query_index: int
    step_id: str
    seed: int
    oracle_enabled: bool
    num_blocks: int
    block_size: int
    summary_dim: int
    shortlist_m: int
    semantic_k: int
    keep_recent_blocks: int
    keep_anchor_blocks: int
    confidence_margin: float
    normalized_margin_threshold: float | None
    selector_latency_sec: float
    stage_a_candidate_count: int
    stage_a_shortlist_size: int
    stage_b_refinement_count: int
    final_selected_block_count: int
    semantic_selected_block_count: int
    rail_preserved_block_count: int
    selected_to_semantic_k_ratio: float
    fallback_mode: str
    fallback_action: str
    fallback_reason_code: str
    raw_margin: float
    normalized_margin: float | None
    selected_mass: float | None
    normalized_mass: float | None
    is_confident: bool
    trace_size_bytes: int
    oracle_recall_rate: float | None
    oracle_precision_rate: float | None
    oracle_overlap_count: int | None
    oracle_missed_important_count: int | None
    oracle_extra_selected_count: int | None
    oracle_missed_important_block_ids: tuple[int, ...]
    oracle_extra_selected_block_ids: tuple[int, ...]

    @classmethod
    def from_microbench_row(cls, row: SelectorMicrobenchRow) -> SelectorAnalysisRunRow:
        """Project one microbench row into a compact, stable analysis row."""

        return cls(
            case_id=row.case_id,
            workload_profile=row.population_profile,
            query_index=row.query_index,
            step_id=row.step_id,
            seed=row.seed,
            oracle_enabled=row.oracle_recall_rate is not None,
            num_blocks=row.num_blocks,
            block_size=row.block_size,
            summary_dim=row.summary_dim,
            shortlist_m=row.shortlist_size,
            semantic_k=row.semantic_top_k,
            keep_recent_blocks=row.keep_recent_blocks,
            keep_anchor_blocks=row.keep_anchor_blocks,
            confidence_margin=row.confidence_margin,
            normalized_margin_threshold=row.normalized_margin_threshold,
            selector_latency_sec=row.selector_latency_sec,
            stage_a_candidate_count=row.stage_a_candidate_count,
            stage_a_shortlist_size=row.stage_a_shortlist_size,
            stage_b_refinement_count=row.stage_b_refinement_count,
            final_selected_block_count=row.final_selected_block_count,
            semantic_selected_block_count=row.semantic_selected_block_count,
            rail_preserved_block_count=row.rail_preserved_block_count,
            selected_to_semantic_k_ratio=(
                row.final_selected_block_count / row.semantic_top_k
            ),
            fallback_mode=row.fallback_mode,
            fallback_action=row.fallback_action,
            fallback_reason_code=row.fallback_reason_code,
            raw_margin=row.raw_margin,
            normalized_margin=row.normalized_margin,
            selected_mass=row.selected_mass,
            normalized_mass=row.normalized_mass,
            is_confident=row.is_confident,
            trace_size_bytes=row.trace_size_bytes,
            oracle_recall_rate=row.oracle_recall_rate,
            oracle_precision_rate=row.oracle_precision_rate,
            oracle_overlap_count=row.oracle_overlap_count,
            oracle_missed_important_count=row.oracle_missed_important_count,
            oracle_extra_selected_count=row.oracle_extra_selected_count,
            oracle_missed_important_block_ids=row.oracle_missed_important_block_ids,
            oracle_extra_selected_block_ids=row.oracle_extra_selected_block_ids,
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the stable field order used for serialization."""

        return tuple(cls.__dataclass_fields__.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectorAnalysisAggregateRow:
    """Compact per-case aggregate analysis record."""

    case_id: str
    workload_profile: str
    seed: int
    oracle_enabled: bool
    query_count: int
    num_blocks: int
    block_size: int
    summary_dim: int
    shortlist_m: int
    semantic_k: int
    keep_recent_blocks: int
    keep_anchor_blocks: int
    confidence_margin: float
    normalized_margin_threshold: float | None
    mean_selector_latency_sec: float
    mean_final_selected_block_count: float
    mean_semantic_selected_block_count: float
    mean_rail_preserved_block_count: float
    mean_selected_to_semantic_k_ratio: float
    avg_raw_margin: float
    avg_normalized_margin: float | None
    avg_selected_mass: float | None
    avg_normalized_mass: float | None
    confident_fraction: float
    sparse_count: int
    sparse_rate: float
    widen_k_count: int
    widen_k_rate: float
    add_recent_count: int
    add_recent_rate: float
    dense_count: int
    dense_rate: float
    mean_oracle_recall_rate: float | None
    mean_oracle_precision_rate: float | None
    mean_oracle_overlap_count: float | None
    low_oracle_recall_sparse_count: int | None
    low_oracle_recall_widen_k_count: int | None
    low_oracle_recall_add_recent_count: int | None
    low_oracle_recall_dense_count: int | None

    @classmethod
    def from_case_result(
        cls, result: SelectorMicrobenchCaseResult
    ) -> SelectorAnalysisAggregateRow:
        """Aggregate one microbenchmark case result into a compact record."""

        rows = result.rows
        fallback = aggregate_fallback_frequency(rows)
        confidence = summarize_confidence(rows)
        oracle_summary = result.oracle_summary

        return cls(
            case_id=result.spec.case_id,
            workload_profile=result.spec.population_profile,
            seed=result.spec.seed,
            oracle_enabled=result.spec.oracle_enabled,
            query_count=len(rows),
            num_blocks=result.spec.num_blocks,
            block_size=result.spec.block_size,
            summary_dim=result.spec.summary_dim,
            shortlist_m=result.spec.shortlist_size,
            semantic_k=result.spec.semantic_top_k,
            keep_recent_blocks=result.spec.keep_recent_blocks,
            keep_anchor_blocks=result.spec.keep_anchor_blocks,
            confidence_margin=result.spec.confidence_margin,
            normalized_margin_threshold=result.spec.normalized_margin_threshold,
            mean_selector_latency_sec=_mean(
                [row.selector_latency_sec for row in rows]
            ),
            mean_final_selected_block_count=_mean(
                [float(row.final_selected_block_count) for row in rows]
            ),
            mean_semantic_selected_block_count=_mean(
                [float(row.semantic_selected_block_count) for row in rows]
            ),
            mean_rail_preserved_block_count=_mean(
                [float(row.rail_preserved_block_count) for row in rows]
            ),
            mean_selected_to_semantic_k_ratio=_mean(
                [
                    row.final_selected_block_count / row.semantic_top_k
                    for row in rows
                ]
            ),
            avg_raw_margin=confidence.avg_raw_margin,
            avg_normalized_margin=confidence.avg_normalized_margin,
            avg_selected_mass=confidence.avg_selected_mass,
            avg_normalized_mass=confidence.avg_normalized_mass,
            confident_fraction=confidence.confident_fraction,
            sparse_count=_mode_value(fallback.counts, "sparse"),
            sparse_rate=_mode_value(fallback.rates, "sparse"),
            widen_k_count=_mode_value(fallback.counts, "widen_k"),
            widen_k_rate=_mode_value(fallback.rates, "widen_k"),
            add_recent_count=_mode_value(fallback.counts, "add_recent"),
            add_recent_rate=_mode_value(fallback.rates, "add_recent"),
            dense_count=_mode_value(fallback.counts, "dense"),
            dense_rate=_mode_value(fallback.rates, "dense"),
            mean_oracle_recall_rate=(
                None if oracle_summary is None else oracle_summary.mean_recall_rate
            ),
            mean_oracle_precision_rate=(
                None if oracle_summary is None else oracle_summary.mean_precision_rate
            ),
            mean_oracle_overlap_count=(
                None if oracle_summary is None else oracle_summary.mean_overlap_count
            ),
            low_oracle_recall_sparse_count=(
                None
                if oracle_summary is None
                else _mode_value(
                    oracle_summary.low_recall_fallback_frequency_by_mode, "sparse"
                )
            ),
            low_oracle_recall_widen_k_count=(
                None
                if oracle_summary is None
                else _mode_value(
                    oracle_summary.low_recall_fallback_frequency_by_mode, "widen_k"
                )
            ),
            low_oracle_recall_add_recent_count=(
                None
                if oracle_summary is None
                else _mode_value(
                    oracle_summary.low_recall_fallback_frequency_by_mode, "add_recent"
                )
            ),
            low_oracle_recall_dense_count=(
                None
                if oracle_summary is None
                else _mode_value(
                    oracle_summary.low_recall_fallback_frequency_by_mode, "dense"
                )
            ),
        )

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return the stable field order used for serialization."""

        return tuple(cls.__dataclass_fields__.keys())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        return asdict(self)


def flatten_microbench_run_rows(
    results: Sequence[SelectorMicrobenchCaseResult],
) -> list[SelectorAnalysisRunRow]:
    """Flatten multiple microbenchmark case results into per-run rows."""

    return [
        SelectorAnalysisRunRow.from_microbench_row(row)
        for result in results
        for row in result.rows
    ]


def flatten_microbench_case_aggregate_rows(
    results: Sequence[SelectorMicrobenchCaseResult],
) -> list[SelectorAnalysisAggregateRow]:
    """Flatten multiple microbenchmark case results into per-case aggregates."""

    return [SelectorAnalysisAggregateRow.from_case_result(result) for result in results]


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mode_value(mapping: dict[str, int] | dict[str, float], mode: str) -> int | float:
    return mapping.get(mode, 0)


STANDARD_RUN_METRIC_FIELDS = SelectorAnalysisRunRow.field_names()
STANDARD_AGGREGATE_METRIC_FIELDS = SelectorAnalysisAggregateRow.field_names()
