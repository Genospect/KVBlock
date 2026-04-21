"""Focused query/key aggregation benchmark for real-block selector runs.

Current local results favor ``block_max`` as the general query/key aggregation
baseline, with ``top_token_mean`` retained as an identifier-heavy needle
override for targeted experiments.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from kvblock.benchmark.real_block_representation_sweep import (
    PromptRetrievalCase,
    RetrievalQuality,
    default_prompt_retrieval_cases,
    retrieval_quality_for_result,
)
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.kv.qk_aggregation import (
    QKAggregationStrategy,
    qk_aggregation_strategies_from_names,
)
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector

DEFAULT_QK_AGGREGATION_STRATEGIES: tuple[QKAggregationStrategy, ...] = (
    "mean_pool",
    "max_pool",
    "norm_weighted_mean",
    "top_token_mean",
    "block_max",
)


@dataclass(frozen=True, slots=True)
class QKAggregationRunRow:
    """One model/prompt/aggregation run."""

    model_name: str
    prompt_name: str
    prompt_file: str
    representation_source: str
    representation_name: str
    qk_aggregation_strategy: str
    rail_setting: str
    keep_recent_blocks: int
    keep_anchor_blocks: int
    tokens: int
    blocks: int
    selected_ids: tuple[int, ...]
    selected_reasons: dict[int, str]
    selected_count: int
    selected_to_semantic_k_ratio: float
    selector_latency_sec: float
    total_latency_sec: float
    prefill_latency_sec: float
    metadata_latency_sec: float
    inspection_latency_sec: float
    fallback_mode: str
    raw_margin: float
    retrieval_quality: RetrievalQuality

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row."""

        payload = asdict(self)
        payload["retrieval_quality"] = self.retrieval_quality.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class QKAggregationSummary:
    """Aggregate quality/latency grouped by aggregation strategy."""

    qk_aggregation_strategy: str
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float
    recall_delta_vs_mean_pool: float | None = None
    precision_delta_vs_mean_pool: float | None = None
    selected_to_k_delta_vs_mean_pool: float | None = None
    latency_delta_vs_mean_pool_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class QKAggregationPromptBreakdown:
    """Aggregate strategy metrics grouped by prompt."""

    prompt_name: str
    qk_aggregation_strategy: str
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly prompt row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class QKAggregationBenchmarkResult:
    """Full query/key aggregation benchmark result."""

    rows: tuple[QKAggregationRunRow, ...]
    aggregate_summaries: tuple[QKAggregationSummary, ...]
    ranked_summaries: tuple[QKAggregationSummary, ...]
    prompt_breakdowns: tuple[QKAggregationPromptBreakdown, ...]
    model_load_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""

        return {
            "rows": [row.to_dict() for row in self.rows],
            "aggregate_summaries": [
                summary.to_dict() for summary in self.aggregate_summaries
            ],
            "ranked_summaries": [
                summary.to_dict() for summary in self.ranked_summaries
            ],
            "prompt_breakdowns": [
                row.to_dict() for row in self.prompt_breakdowns
            ],
            "model_load_seconds": dict(self.model_load_seconds),
        }


def run_qk_aggregation_benchmark(
    *,
    model_names: Sequence[str],
    prompt_cases: Sequence[PromptRetrievalCase] | None = None,
    qk_aggregation_strategies: Sequence[QKAggregationStrategy] = DEFAULT_QK_AGGREGATION_STRATEGIES,
    representation_source: RepresentationSource = "query_mean_last_layer",
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
) -> QKAggregationBenchmarkResult:
    """Run the existing selector across query/key aggregation strategies."""

    if not model_names:
        raise ValueError("model_names must not be empty")
    if not qk_aggregation_strategies:
        raise ValueError("qk_aggregation_strategies must not be empty")

    cases = tuple(prompt_cases or default_prompt_retrieval_cases())
    strategies = tuple(qk_aggregation_strategies_from_names(qk_aggregation_strategies))
    base_config = selector_config or RealBlockSelectorConfig(
        block_size=16,
        shortlist_m=16,
        semantic_k=4,
        confidence_margin=0.0,
        keep_recent_blocks=1,
        keep_anchor_blocks=0,
        preview_chars=160,
        include_block_text=True,
        representation_source=representation_source,
    )
    load_kwargs = dict(load_config_kwargs or {})

    rows: list[QKAggregationRunRow] = []
    model_load_seconds: dict[str, float] = {}
    for model_name in model_names:
        runtime = LocalHfRuntime(
            RuntimeLoadConfig(model_name=model_name, **load_kwargs),
            capture_config=HiddenStateCaptureConfig(
                representation_source=representation_source,
            ),
        )
        started_at = perf_counter()
        runtime.load_model()
        model_load_seconds[model_name] = perf_counter() - started_at

        for strategy in strategies:
            for prompt_case in cases:
                prompt = prompt_case.path.read_text(encoding="utf-8")
                result = run_real_block_selector(
                    runtime,
                    prompt,
                    replace(
                        base_config,
                        representation_source=representation_source,
                        qk_aggregation_strategy=strategy,
                        prompt_id=prompt_case.name,
                        prompt_name=prompt_case.name,
                        relevant_text_fragments=prompt_case.target_fragments,
                        include_block_text=True,
                    ),
                )
                rows.append(
                    _row_from_result(
                        model_name=model_name,
                        prompt_case=prompt_case,
                        representation_source=representation_source,
                        strategy=strategy,
                        config=base_config,
                        result=result,
                    )
                )

    row_tuple = tuple(rows)
    aggregates = _build_aggregate_summaries(row_tuple)
    return QKAggregationBenchmarkResult(
        rows=row_tuple,
        aggregate_summaries=aggregates,
        ranked_summaries=tuple(sorted(aggregates, key=_summary_rank_key)),
        prompt_breakdowns=_build_prompt_breakdowns(row_tuple),
        model_load_seconds=model_load_seconds,
    )


def write_qk_aggregation_benchmark_outputs(
    result: QKAggregationBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text reports for the query/key aggregation benchmark."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_qk_aggregation_report(result), encoding="utf-8")


def format_qk_aggregation_report(result: QKAggregationBenchmarkResult) -> str:
    """Format a compact query/key aggregation benchmark report."""

    lines = [
        "QUERY/KEY AGGREGATION BENCHMARK",
        f"model_load_seconds={result.model_load_seconds}",
        "",
        "RANKED AGGREGATES",
    ]
    for summary in result.ranked_summaries:
        lines.append(_format_summary(summary))

    lines.append("")
    lines.append("PER-PROMPT BREAKDOWN")
    for row in result.prompt_breakdowns:
        lines.append(
            f"{row.prompt_name} | {row.qk_aggregation_strategy} | "
            f"runs={row.run_count} "
            f"mean_recall={_fmt_optional(row.mean_recall)} "
            f"mean_precision={_fmt_optional(row.mean_precision)} "
            f"mean_selected/K={row.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_selector={row.mean_selector_latency_sec:.6f}s"
        )

    lines.append("")
    lines.append("RUNS")
    for row in result.rows:
        quality = row.retrieval_quality
        lines.append(
            f"{row.model_name} | {row.prompt_name} | {row.qk_aggregation_strategy} | "
            f"selected={list(row.selected_ids)} | "
            f"recall={_fmt_optional(quality.target_recall)} "
            f"precision={_fmt_optional(quality.selected_precision)} "
            f"selected/K={row.selected_to_semantic_k_ratio:.3f} "
            f"selector={row.selector_latency_sec:.6f}s"
        )
    return "\n".join(lines)


def _row_from_result(
    *,
    model_name: str,
    prompt_case: PromptRetrievalCase,
    representation_source: str,
    strategy: QKAggregationStrategy,
    config: RealBlockSelectorConfig,
    result: Any,
) -> QKAggregationRunRow:
    quality = retrieval_quality_for_result(
        selected_ids=result.selected_block_ids,
        block_text_by_id={
            block.block_id: block.block_text or block.preview_text
            for block in result.block_inspections
        },
        target_fragments=prompt_case.target_fragments,
    )
    return QKAggregationRunRow(
        model_name=model_name,
        prompt_name=prompt_case.name,
        prompt_file=str(prompt_case.path),
        representation_source=representation_source,
        representation_name=result.run_summary.representation_name,
        qk_aggregation_strategy=strategy,
        rail_setting=config.rail_setting or "configured",
        keep_recent_blocks=config.keep_recent_blocks,
        keep_anchor_blocks=config.keep_anchor_blocks,
        tokens=result.run_summary.token_count,
        blocks=result.run_summary.block_count,
        selected_ids=result.selected_block_ids,
        selected_reasons={
            block.block_id: block.selected_reason
            for block in result.selected_block_inspections
        },
        selected_count=len(result.selected_block_ids),
        selected_to_semantic_k_ratio=result.selected_to_semantic_k_ratio,
        selector_latency_sec=result.latency.selector_sec,
        total_latency_sec=result.latency.total_sec,
        prefill_latency_sec=result.latency.prefill_sec,
        metadata_latency_sec=result.latency.metadata_sec,
        inspection_latency_sec=result.latency.inspection_sec,
        fallback_mode=result.fallback_mode,
        raw_margin=result.confidence.raw_margin,
        retrieval_quality=quality,
    )


def _build_aggregate_summaries(
    rows: Sequence[QKAggregationRunRow],
) -> tuple[QKAggregationSummary, ...]:
    grouped: dict[str, list[QKAggregationRunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.qk_aggregation_strategy].append(row)

    summaries: list[QKAggregationSummary] = []
    for strategy, group in sorted(grouped.items()):
        summaries.append(
            QKAggregationSummary(
                qk_aggregation_strategy=strategy,
                run_count=len(group),
                mean_recall=_mean_optional(
                    row.retrieval_quality.target_recall for row in group
                ),
                mean_precision=_mean_optional(
                    row.retrieval_quality.selected_precision for row in group
                ),
                mean_selected_count=_mean(row.selected_count for row in group),
                mean_selected_to_semantic_k_ratio=_mean(
                    row.selected_to_semantic_k_ratio for row in group
                ),
                mean_selector_latency_sec=_mean(
                    row.selector_latency_sec for row in group
                ),
            )
        )

    baseline = next(
        (summary for summary in summaries if summary.qk_aggregation_strategy == "mean_pool"),
        None,
    )
    if baseline is None:
        return tuple(summaries)
    return tuple(
        replace(
            summary,
            recall_delta_vs_mean_pool=_optional_delta(
                summary.mean_recall,
                baseline.mean_recall,
            ),
            precision_delta_vs_mean_pool=_optional_delta(
                summary.mean_precision,
                baseline.mean_precision,
            ),
            selected_to_k_delta_vs_mean_pool=(
                summary.mean_selected_to_semantic_k_ratio
                - baseline.mean_selected_to_semantic_k_ratio
            ),
            latency_delta_vs_mean_pool_sec=(
                summary.mean_selector_latency_sec
                - baseline.mean_selector_latency_sec
            ),
        )
        for summary in summaries
    )


def _build_prompt_breakdowns(
    rows: Sequence[QKAggregationRunRow],
) -> tuple[QKAggregationPromptBreakdown, ...]:
    grouped: dict[tuple[str, str], list[QKAggregationRunRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.prompt_name, row.qk_aggregation_strategy)].append(row)

    breakdowns: list[QKAggregationPromptBreakdown] = []
    for (prompt_name, strategy), group in sorted(grouped.items()):
        breakdowns.append(
            QKAggregationPromptBreakdown(
                prompt_name=prompt_name,
                qk_aggregation_strategy=strategy,
                run_count=len(group),
                mean_recall=_mean_optional(
                    row.retrieval_quality.target_recall for row in group
                ),
                mean_precision=_mean_optional(
                    row.retrieval_quality.selected_precision for row in group
                ),
                mean_selected_to_semantic_k_ratio=_mean(
                    row.selected_to_semantic_k_ratio for row in group
                ),
                mean_selector_latency_sec=_mean(
                    row.selector_latency_sec for row in group
                ),
            )
        )
    return tuple(breakdowns)


def _summary_rank_key(summary: QKAggregationSummary) -> tuple[float, float, float, float, str]:
    recall = -1.0 if summary.mean_recall is None else summary.mean_recall
    precision = -1.0 if summary.mean_precision is None else summary.mean_precision
    return (
        -recall,
        -precision,
        summary.mean_selected_to_semantic_k_ratio,
        summary.mean_selector_latency_sec,
        summary.qk_aggregation_strategy,
    )


def _format_summary(summary: QKAggregationSummary) -> str:
    return (
        f"{summary.qk_aggregation_strategy} | runs={summary.run_count} "
        f"mean_recall={_fmt_optional(summary.mean_recall)} "
        f"mean_precision={_fmt_optional(summary.mean_precision)} "
        f"mean_selected={summary.mean_selected_count:.3f} "
        f"mean_selected/K={summary.mean_selected_to_semantic_k_ratio:.3f} "
        f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
        f"d_recall={_fmt_optional(summary.recall_delta_vs_mean_pool)} "
        f"d_precision={_fmt_optional(summary.precision_delta_vs_mean_pool)} "
        f"d_selected/K={_fmt_optional(summary.selected_to_k_delta_vs_mean_pool)} "
        f"d_selector={_fmt_optional(summary.latency_delta_vs_mean_pool_sec)}s"
    )


def _mean(values: Iterable[float | int]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    materialized = tuple(float(value) for value in values if value is not None)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
