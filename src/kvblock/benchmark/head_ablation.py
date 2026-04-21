"""Offline per-head ablation benchmark over real-block diagnostics.

This is an exploratory diagnostic path. Per-head routing and static head
specialization did not replace pooled ``mean_heads`` routing in the current V1
baseline, but the reports remain useful for future analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Literal, Sequence

from kvblock.benchmark.real_block_representation_sweep import (
    PromptRetrievalCase,
    default_prompt_retrieval_cases,
    retrieval_quality_for_result,
)
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector

RankingKind = Literal["pooled", "single_head", "leave_one_out"]


@dataclass(frozen=True, slots=True)
class OfflineHeadScoreBlock:
    """Minimal block score record needed for offline head ablation."""

    block_id: int
    head_scores: tuple[float, ...]
    preview_text: str = ""
    token_start: int = 0
    token_end: int = 0

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise ValueError("block_id must be >= 0")
        if not self.head_scores:
            raise ValueError("head_scores must not be empty")


@dataclass(frozen=True, slots=True)
class HeadRankingMetrics:
    """Recall/precision result for one offline ranking policy."""

    kind: RankingKind
    head_index: int | None
    selected_block_ids: tuple[int, ...]
    expected_block_ids: tuple[int, ...]
    recall: float | None
    precision: float | None
    selected_to_k_ratio: float
    mean_selected_score: float
    recall_delta_vs_pooled: float | None = None
    precision_delta_vs_pooled: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metric row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadAblationPromptResult:
    """All offline head-ablation rows for one model/prompt pair."""

    model_name: str
    prompt_name: str
    prompt_file: str
    representation_source: str
    representation_name: str
    block_count: int
    head_count: int
    top_k: int
    expected_block_ids: tuple[int, ...]
    pooled_baseline: HeadRankingMetrics
    single_head_results: tuple[HeadRankingMetrics, ...]
    leave_one_out_results: tuple[HeadRankingMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly prompt result."""

        return {
            "model_name": self.model_name,
            "prompt_name": self.prompt_name,
            "prompt_file": self.prompt_file,
            "representation_source": self.representation_source,
            "representation_name": self.representation_name,
            "block_count": self.block_count,
            "head_count": self.head_count,
            "top_k": self.top_k,
            "expected_block_ids": list(self.expected_block_ids),
            "pooled_baseline": self.pooled_baseline.to_dict(),
            "single_head_results": [
                item.to_dict() for item in self.single_head_results
            ],
            "leave_one_out_results": [
                item.to_dict() for item in self.leave_one_out_results
            ],
        }


@dataclass(frozen=True, slots=True)
class AggregateHeadRow:
    """Aggregate row for a head across model/prompt results."""

    head_index: int
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_k_ratio: float
    mean_recall_delta_vs_pooled: float | None = None
    mean_precision_delta_vs_pooled: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PromptSpecialistHead:
    """Best-performing single head for a prompt family."""

    prompt_name: str
    head_index: int
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_recall_delta_vs_pooled: float | None
    mean_precision_delta_vs_pooled: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly specialist row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class PooledBaselineSummary:
    """Aggregate pooled head-score baseline summary."""

    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_k_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly pooled baseline summary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadAblationBenchmarkResult:
    """Full offline head-ablation benchmark result."""

    prompt_results: tuple[HeadAblationPromptResult, ...]
    best_single_heads_overall: tuple[AggregateHeadRow, ...]
    removal_hurts_most: tuple[AggregateHeadRow, ...]
    removal_improves_most: tuple[AggregateHeadRow, ...]
    prompt_specialist_heads: tuple[PromptSpecialistHead, ...]
    pooled_baseline_summary: PooledBaselineSummary
    model_load_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark payload."""

        return {
            "prompt_results": [
                item.to_dict() for item in self.prompt_results
            ],
            "best_single_heads_overall": [
                item.to_dict() for item in self.best_single_heads_overall
            ],
            "removal_hurts_most": [
                item.to_dict() for item in self.removal_hurts_most
            ],
            "removal_improves_most": [
                item.to_dict() for item in self.removal_improves_most
            ],
            "prompt_specialist_heads": [
                item.to_dict() for item in self.prompt_specialist_heads
            ],
            "pooled_baseline_summary": self.pooled_baseline_summary.to_dict(),
            "model_load_seconds": dict(self.model_load_seconds),
        }


def evaluate_prompt_head_ablation(
    *,
    model_name: str,
    prompt_name: str,
    prompt_file: str,
    representation_source: str,
    representation_name: str,
    blocks: Sequence[OfflineHeadScoreBlock],
    target_fragments: Sequence[str],
    top_k: int,
) -> HeadAblationPromptResult:
    """Evaluate pooled, single-head, and leave-one-out rankings for one prompt."""

    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if not blocks:
        raise ValueError("blocks must not be empty")
    head_count = len(blocks[0].head_scores)
    if any(len(block.head_scores) != head_count for block in blocks):
        raise ValueError("all blocks must have the same head count")

    expected_block_ids = _expected_blocks_from_fragments(blocks, target_fragments)
    pooled_scores = {
        block.block_id: _mean(block.head_scores)
        for block in blocks
    }
    pooled = _evaluate_scores(
        scores_by_block=pooled_scores,
        expected_block_ids=expected_block_ids,
        top_k=top_k,
        kind="pooled",
        head_index=None,
    )

    single_head_results = tuple(
        _with_delta(
            _evaluate_scores(
                scores_by_block={
                    block.block_id: block.head_scores[head_index]
                    for block in blocks
                },
                expected_block_ids=expected_block_ids,
                top_k=top_k,
                kind="single_head",
                head_index=head_index,
            ),
            pooled,
        )
        for head_index in range(head_count)
    )
    leave_one_out_results = tuple(
        _with_delta(
            _evaluate_scores(
                scores_by_block={
                    block.block_id: _leave_one_out_mean(block.head_scores, head_index)
                    for block in blocks
                },
                expected_block_ids=expected_block_ids,
                top_k=top_k,
                kind="leave_one_out",
                head_index=head_index,
            ),
            pooled,
        )
        for head_index in range(head_count)
    )

    return HeadAblationPromptResult(
        model_name=model_name,
        prompt_name=prompt_name,
        prompt_file=prompt_file,
        representation_source=representation_source,
        representation_name=representation_name,
        block_count=len(blocks),
        head_count=head_count,
        top_k=top_k,
        expected_block_ids=expected_block_ids,
        pooled_baseline=pooled,
        single_head_results=single_head_results,
        leave_one_out_results=leave_one_out_results,
    )


def run_head_ablation_benchmark(
    *,
    model_names: Sequence[str],
    prompt_cases: Sequence[PromptRetrievalCase] | None = None,
    representation_source: RepresentationSource = "query_mean_last_layer",
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    top_n: int = 8,
) -> HeadAblationBenchmarkResult:
    """Run dense prefill once per prompt and evaluate offline per-head rankings."""

    if not model_names:
        raise ValueError("model_names must not be empty")
    if top_n <= 0:
        raise ValueError("top_n must be > 0")

    cases = tuple(prompt_cases or default_prompt_retrieval_cases())
    base_config = selector_config or RealBlockSelectorConfig(
        block_size=16,
        shortlist_m=16,
        semantic_k=4,
        confidence_margin=0.0,
        keep_recent_blocks=0,
        keep_anchor_blocks=0,
        preview_chars=160,
        include_block_text=True,
        emit_head_diagnostics=True,
        top_heads=top_n,
        representation_source=representation_source,
    )
    load_kwargs = dict(load_config_kwargs or {})

    prompt_results: list[HeadAblationPromptResult] = []
    model_load_seconds: dict[str, float] = {}
    for model_name in model_names:
        runtime = LocalHfRuntime(RuntimeLoadConfig(model_name=model_name, **load_kwargs))
        started_at = perf_counter()
        runtime.load_model()
        model_load_seconds[model_name] = perf_counter() - started_at
        runtime.capture_config = HiddenStateCaptureConfig(
            representation_source=representation_source,
        )
        for prompt_case in cases:
            prompt = prompt_case.path.read_text(encoding="utf-8")
            result = run_real_block_selector(
                runtime,
                prompt,
                replace(
                    base_config,
                    representation_source=representation_source,
                    prompt_name=prompt_case.name,
                    emit_head_diagnostics=True,
                    top_heads=top_n,
                ),
            )
            blocks = _blocks_from_real_result(result)
            prompt_results.append(
                evaluate_prompt_head_ablation(
                    model_name=model_name,
                    prompt_name=prompt_case.name,
                    prompt_file=str(prompt_case.path),
                    representation_source=representation_source,
                    representation_name=result.run_summary.representation_name,
                    blocks=blocks,
                    target_fragments=prompt_case.target_fragments,
                    top_k=base_config.semantic_k,
                )
            )

    return summarize_head_ablation_results(
        tuple(prompt_results),
        top_n=top_n,
        model_load_seconds=model_load_seconds,
    )


def summarize_head_ablation_results(
    prompt_results: Sequence[HeadAblationPromptResult],
    *,
    top_n: int = 8,
    model_load_seconds: dict[str, float] | None = None,
) -> HeadAblationBenchmarkResult:
    """Aggregate prompt-level head-ablation rows into report tables."""

    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    prompt_tuple = tuple(prompt_results)
    single_rows = [row for result in prompt_tuple for row in result.single_head_results]
    loo_rows = [row for result in prompt_tuple for row in result.leave_one_out_results]
    pooled_rows = [result.pooled_baseline for result in prompt_tuple]

    best_single = sorted(
        _aggregate_by_head(single_rows),
        key=_single_head_sort_key,
    )[:top_n]
    leave_one_out = _aggregate_by_head(loo_rows)
    removal_hurts = sorted(
        leave_one_out,
        key=_removal_hurts_sort_key,
    )[:top_n]
    removal_improves = sorted(
        leave_one_out,
        key=_removal_improves_sort_key,
    )[:top_n]

    return HeadAblationBenchmarkResult(
        prompt_results=prompt_tuple,
        best_single_heads_overall=tuple(best_single),
        removal_hurts_most=tuple(removal_hurts),
        removal_improves_most=tuple(removal_improves),
        prompt_specialist_heads=_prompt_specialists(prompt_tuple, top_n=top_n),
        pooled_baseline_summary=PooledBaselineSummary(
            run_count=len(pooled_rows),
            mean_recall=_mean_optional(row.recall for row in pooled_rows),
            mean_precision=_mean_optional(row.precision for row in pooled_rows),
            mean_selected_to_k_ratio=_mean(row.selected_to_k_ratio for row in pooled_rows),
        ),
        model_load_seconds=dict(model_load_seconds or {}),
    )


def load_blocks_from_head_diagnostic_json(path: str | Path) -> tuple[OfflineHeadScoreBlock, ...]:
    """Load full per-block diagnostic records from a CLI diagnostic JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    diagnostics = payload.get("head_diagnostics")
    if diagnostics is None:
        raise ValueError("diagnostic JSON must contain full head_diagnostics records")
    return tuple(
        OfflineHeadScoreBlock(
            block_id=int(record["block_id"]),
            head_scores=tuple(float(value) for value in record["head_scores"]),
            preview_text=str(record.get("preview_text", "")),
            token_start=int(record.get("token_start", 0)),
            token_end=int(record.get("token_end", 0)),
        )
        for record in diagnostics
    )


def write_head_ablation_outputs(
    result: HeadAblationBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text report outputs."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_head_ablation_report(result), encoding="utf-8")


def format_head_ablation_report(result: HeadAblationBenchmarkResult) -> str:
    """Format compact tables for offline head-ablation results."""

    lines = [
        "OFFLINE HEAD-ABLATION BENCHMARK",
        f"model_load_seconds={result.model_load_seconds}",
        "",
        "POOLED BASELINE",
        _format_pooled_baseline(result.pooled_baseline_summary),
        "",
        "BEST SINGLE HEADS OVERALL",
    ]
    lines.extend(_format_aggregate_head_row(row) for row in result.best_single_heads_overall)
    lines.append("")
    lines.append("HEADS WHOSE REMOVAL HURTS POOLED MOST")
    lines.extend(_format_delta_head_row(row) for row in result.removal_hurts_most)
    lines.append("")
    lines.append("HEADS WHOSE REMOVAL IMPROVES POOLED MOST")
    lines.extend(_format_delta_head_row(row) for row in result.removal_improves_most)
    lines.append("")
    lines.append("PROMPT-SPECIALIST HEADS")
    lines.extend(_format_prompt_specialist(row) for row in result.prompt_specialist_heads)
    lines.append("")
    lines.append("PROMPT RESULTS")
    for prompt_result in result.prompt_results:
        best = sorted(prompt_result.single_head_results, key=_ranking_row_sort_key)[0]
        most_hurt = sorted(
            prompt_result.leave_one_out_results,
            key=_ranking_removal_hurts_sort_key,
        )[0]
        most_help = sorted(
            prompt_result.leave_one_out_results,
            key=_ranking_removal_improves_sort_key,
        )[0]
        lines.append(
            f"{prompt_result.model_name} | {prompt_result.prompt_name} | "
            f"pooled recall={_fmt_optional(prompt_result.pooled_baseline.recall)} "
            f"precision={_fmt_optional(prompt_result.pooled_baseline.precision)} | "
            f"best_single=h{best.head_index} "
            f"recall={_fmt_optional(best.recall)} precision={_fmt_optional(best.precision)} | "
            f"remove_hurts=h{most_hurt.head_index} "
            f"d_recall={_fmt_optional(most_hurt.recall_delta_vs_pooled)} "
            f"d_precision={_fmt_optional(most_hurt.precision_delta_vs_pooled)} | "
            f"remove_improves=h{most_help.head_index} "
            f"d_recall={_fmt_optional(most_help.recall_delta_vs_pooled)} "
            f"d_precision={_fmt_optional(most_help.precision_delta_vs_pooled)}"
        )
    return "\n".join(lines)


def _blocks_from_real_result(result: Any) -> tuple[OfflineHeadScoreBlock, ...]:
    diagnostics = tuple(getattr(result, "head_diagnostics", ()) or ())
    if not diagnostics:
        raise ValueError("real-block result does not contain head diagnostics")
    return tuple(
        OfflineHeadScoreBlock(
            block_id=int(record.block_id),
            head_scores=tuple(float(score) for score in record.head_scores),
            preview_text=record.preview_text,
            token_start=record.token_start,
            token_end=record.token_end,
        )
        for record in diagnostics
    )


def _expected_blocks_from_fragments(
    blocks: Sequence[OfflineHeadScoreBlock],
    target_fragments: Sequence[str],
) -> tuple[int, ...]:
    quality = retrieval_quality_for_result(
        selected_ids=(),
        block_text_by_id={block.block_id: block.preview_text for block in blocks},
        target_fragments=target_fragments,
    )
    return quality.expected_block_ids


def _evaluate_scores(
    *,
    scores_by_block: dict[int, float],
    expected_block_ids: Sequence[int],
    top_k: int,
    kind: RankingKind,
    head_index: int | None,
) -> HeadRankingMetrics:
    ranked = sorted(scores_by_block.items(), key=lambda item: (-item[1], item[0]))
    selected = tuple(block_id for block_id, _score in ranked[:top_k])
    expected = tuple(int(block_id) for block_id in expected_block_ids)
    expected_set = set(expected)
    selected_set = set(selected)
    hits = selected_set & expected_set
    recall = None if not expected else len(hits) / len(expected)
    precision = None if not selected else len(hits) / len(selected)
    selected_scores = [scores_by_block[block_id] for block_id in selected]
    return HeadRankingMetrics(
        kind=kind,
        head_index=head_index,
        selected_block_ids=selected,
        expected_block_ids=expected,
        recall=recall,
        precision=precision,
        selected_to_k_ratio=len(selected) / top_k,
        mean_selected_score=_mean(selected_scores),
    )


def _with_delta(row: HeadRankingMetrics, pooled: HeadRankingMetrics) -> HeadRankingMetrics:
    return replace(
        row,
        recall_delta_vs_pooled=_optional_delta(row.recall, pooled.recall),
        precision_delta_vs_pooled=_optional_delta(row.precision, pooled.precision),
    )


def _aggregate_by_head(rows: Sequence[HeadRankingMetrics]) -> tuple[AggregateHeadRow, ...]:
    grouped: dict[int, list[HeadRankingMetrics]] = {}
    for row in rows:
        if row.head_index is None:
            continue
        grouped.setdefault(row.head_index, []).append(row)

    aggregates: list[AggregateHeadRow] = []
    for head_index, group in sorted(grouped.items()):
        aggregates.append(
            AggregateHeadRow(
                head_index=head_index,
                run_count=len(group),
                mean_recall=_mean_optional(row.recall for row in group),
                mean_precision=_mean_optional(row.precision for row in group),
                mean_selected_to_k_ratio=_mean(row.selected_to_k_ratio for row in group),
                mean_recall_delta_vs_pooled=_mean_optional(
                    row.recall_delta_vs_pooled for row in group
                ),
                mean_precision_delta_vs_pooled=_mean_optional(
                    row.precision_delta_vs_pooled for row in group
                ),
            )
        )
    return tuple(aggregates)


def _prompt_specialists(
    prompt_results: Sequence[HeadAblationPromptResult],
    *,
    top_n: int,
) -> tuple[PromptSpecialistHead, ...]:
    grouped: dict[str, list[HeadRankingMetrics]] = {}
    for result in prompt_results:
        grouped.setdefault(result.prompt_name, []).extend(result.single_head_results)

    specialists: list[PromptSpecialistHead] = []
    for prompt_name, rows in sorted(grouped.items()):
        best = sorted(_aggregate_by_head(rows), key=_single_head_sort_key)[:top_n]
        for row in best:
            specialists.append(
                PromptSpecialistHead(
                    prompt_name=prompt_name,
                    head_index=row.head_index,
                    run_count=row.run_count,
                    mean_recall=row.mean_recall,
                    mean_precision=row.mean_precision,
                    mean_recall_delta_vs_pooled=row.mean_recall_delta_vs_pooled,
                    mean_precision_delta_vs_pooled=row.mean_precision_delta_vs_pooled,
                )
            )
    return tuple(specialists)


def _leave_one_out_mean(values: Sequence[float], excluded_index: int) -> float:
    if len(values) == 1:
        return 0.0
    return _mean(
        value for index, value in enumerate(values) if index != excluded_index
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


def _single_head_sort_key(row: AggregateHeadRow) -> tuple[float, float, float, int]:
    recall = -1.0 if row.mean_recall is None else row.mean_recall
    precision = -1.0 if row.mean_precision is None else row.mean_precision
    return (-recall, -precision, row.mean_selected_to_k_ratio, row.head_index)


def _removal_hurts_sort_key(row: AggregateHeadRow) -> tuple[float, float, int]:
    recall_delta = (
        0.0 if row.mean_recall_delta_vs_pooled is None else row.mean_recall_delta_vs_pooled
    )
    precision_delta = (
        0.0
        if row.mean_precision_delta_vs_pooled is None
        else row.mean_precision_delta_vs_pooled
    )
    return (recall_delta, precision_delta, row.head_index)


def _removal_improves_sort_key(row: AggregateHeadRow) -> tuple[float, float, int]:
    recall_delta = (
        0.0 if row.mean_recall_delta_vs_pooled is None else row.mean_recall_delta_vs_pooled
    )
    precision_delta = (
        0.0
        if row.mean_precision_delta_vs_pooled is None
        else row.mean_precision_delta_vs_pooled
    )
    return (-recall_delta, -precision_delta, row.head_index)


def _ranking_row_sort_key(row: HeadRankingMetrics) -> tuple[float, float, float, int]:
    recall = -1.0 if row.recall is None else row.recall
    precision = -1.0 if row.precision is None else row.precision
    head_index = -1 if row.head_index is None else row.head_index
    return (-recall, -precision, row.selected_to_k_ratio, head_index)


def _ranking_removal_hurts_sort_key(
    row: HeadRankingMetrics,
) -> tuple[float, float, int]:
    recall_delta = 0.0 if row.recall_delta_vs_pooled is None else row.recall_delta_vs_pooled
    precision_delta = (
        0.0 if row.precision_delta_vs_pooled is None else row.precision_delta_vs_pooled
    )
    head_index = -1 if row.head_index is None else row.head_index
    return (recall_delta, precision_delta, head_index)


def _ranking_removal_improves_sort_key(
    row: HeadRankingMetrics,
) -> tuple[float, float, int]:
    recall_delta = 0.0 if row.recall_delta_vs_pooled is None else row.recall_delta_vs_pooled
    precision_delta = (
        0.0 if row.precision_delta_vs_pooled is None else row.precision_delta_vs_pooled
    )
    head_index = -1 if row.head_index is None else row.head_index
    return (-recall_delta, -precision_delta, head_index)


def _format_pooled_baseline(summary: PooledBaselineSummary) -> str:
    return (
        f"runs={summary.run_count} "
        f"mean_recall={_fmt_optional(summary.mean_recall)} "
        f"mean_precision={_fmt_optional(summary.mean_precision)} "
        f"mean_selected/K={summary.mean_selected_to_k_ratio:.3f}"
    )


def _format_aggregate_head_row(row: AggregateHeadRow) -> str:
    return (
        f"h{row.head_index} | runs={row.run_count} "
        f"mean_recall={_fmt_optional(row.mean_recall)} "
        f"mean_precision={_fmt_optional(row.mean_precision)} "
        f"mean_selected/K={row.mean_selected_to_k_ratio:.3f} "
        f"d_recall={_fmt_optional(row.mean_recall_delta_vs_pooled)} "
        f"d_precision={_fmt_optional(row.mean_precision_delta_vs_pooled)}"
    )


def _format_delta_head_row(row: AggregateHeadRow) -> str:
    return (
        f"remove h{row.head_index} | runs={row.run_count} "
        f"mean_recall={_fmt_optional(row.mean_recall)} "
        f"mean_precision={_fmt_optional(row.mean_precision)} "
        f"d_recall={_fmt_optional(row.mean_recall_delta_vs_pooled)} "
        f"d_precision={_fmt_optional(row.mean_precision_delta_vs_pooled)}"
    )


def _format_prompt_specialist(row: PromptSpecialistHead) -> str:
    return (
        f"{row.prompt_name} | h{row.head_index} | runs={row.run_count} "
        f"mean_recall={_fmt_optional(row.mean_recall)} "
        f"mean_precision={_fmt_optional(row.mean_precision)} "
        f"d_recall={_fmt_optional(row.mean_recall_delta_vs_pooled)} "
        f"d_precision={_fmt_optional(row.mean_precision_delta_vs_pooled)}"
    )


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
