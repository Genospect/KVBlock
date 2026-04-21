"""Constrained static head-weight validation for query/key routing.

This validation is archived as evidence: tested static schemes did not beat
pooled query/key routing strongly enough to become V1 defaults.
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
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector


@dataclass(frozen=True, slots=True)
class StaticHeadWeightScheme:
    """One explicit static head-weight policy to validate against pooled routing."""

    name: str
    head_scoring_mode: str
    description: str
    base_weight: float = 1.0
    boosts: tuple[tuple[int, float], ...] = ()
    head_only: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.head_scoring_mode not in {"mean_heads", "weighted_head_mean"}:
            raise ValueError(
                "static validation only supports mean_heads and weighted_head_mean"
            )
        if self.base_weight < 0:
            raise ValueError("base_weight must be >= 0")
        if self.head_only is not None and self.head_only < 0:
            raise ValueError("head_only must be >= 0")
        for head_index, weight in self.boosts:
            if head_index < 0:
                raise ValueError("boost head indices must be >= 0")
            if weight < 0:
                raise ValueError("boost weights must be >= 0")

    def weights_for_head_count(self, head_count: int) -> tuple[float, ...]:
        """Return the concrete weight vector for a model head count."""

        if head_count <= 0:
            raise ValueError("head_count must be > 0")
        if self.head_scoring_mode == "mean_heads":
            return ()
        if self.head_only is not None:
            if self.head_only >= head_count:
                raise ValueError(
                    f"scheme {self.name!r} requires head {self.head_only}, "
                    f"but model exposes {head_count} heads"
                )
            weights = [0.0] * head_count
            weights[self.head_only] = 1.0
            return tuple(weights)

        weights = [self.base_weight] * head_count
        for head_index, weight in self.boosts:
            if head_index >= head_count:
                raise ValueError(
                    f"scheme {self.name!r} requires head {head_index}, "
                    f"but model exposes {head_count} heads"
                )
            weights[head_index] = weight
        if sum(weights) <= 0:
            raise ValueError(f"scheme {self.name!r} produced all-zero weights")
        return tuple(weights)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly scheme record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StaticHeadWeightRunRow:
    """One prompt/model/scheme validation run."""

    model_name: str
    prompt_name: str
    prompt_file: str
    representation_source: str
    scheme_name: str
    scheme_description: str
    head_scoring_mode: str
    head_weights: tuple[float, ...]
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
        """Return a JSON-friendly run row."""

        payload = asdict(self)
        payload["retrieval_quality"] = self.retrieval_quality.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class StaticHeadWeightAggregate:
    """Aggregate quality/latency row for one static scheme."""

    scheme_name: str
    scheme_description: str
    head_scoring_mode: str
    head_weights: tuple[float, ...]
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float
    recall_delta_vs_pooled: float | None = None
    precision_delta_vs_pooled: float | None = None
    selected_to_k_delta_vs_pooled: float | None = None
    selector_latency_delta_vs_pooled_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StaticHeadWeightPromptBreakdown:
    """Aggregate scheme metrics grouped by prompt family."""

    prompt_name: str
    scheme_name: str
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly prompt-breakdown row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class StaticHeadWeightValidationResult:
    """Full static head-weight validation benchmark result."""

    rows: tuple[StaticHeadWeightRunRow, ...]
    aggregate_summaries: tuple[StaticHeadWeightAggregate, ...]
    prompt_breakdowns: tuple[StaticHeadWeightPromptBreakdown, ...]
    ranked_summaries: tuple[StaticHeadWeightAggregate, ...]
    schemes: tuple[StaticHeadWeightScheme, ...]
    model_load_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""

        return {
            "rows": [row.to_dict() for row in self.rows],
            "aggregate_summaries": [
                row.to_dict() for row in self.aggregate_summaries
            ],
            "prompt_breakdowns": [
                row.to_dict() for row in self.prompt_breakdowns
            ],
            "ranked_summaries": [
                row.to_dict() for row in self.ranked_summaries
            ],
            "schemes": [scheme.to_dict() for scheme in self.schemes],
            "model_load_seconds": dict(self.model_load_seconds),
        }


def default_static_head_weight_schemes() -> tuple[StaticHeadWeightScheme, ...]:
    """Return the constrained evidence-backed schemes for GPT-2-class heads."""

    return (
        StaticHeadWeightScheme(
            name="pooled_mean_heads",
            head_scoring_mode="mean_heads",
            description="Current pooled query/key summary baseline.",
        ),
        StaticHeadWeightScheme(
            name="head9_only",
            head_scoring_mode="weighted_head_mean",
            description="Use only head 9, the strongest offline single-head candidate.",
            head_only=9,
        ),
        StaticHeadWeightScheme(
            name="head9_heavy",
            head_scoring_mode="weighted_head_mean",
            description="Uniform weights with head 9 modestly upweighted.",
            boosts=((9, 3.0),),
        ),
        StaticHeadWeightScheme(
            name="retrieval_mix",
            head_scoring_mode="weighted_head_mean",
            description=(
                "Upweight head 9 plus heads that surfaced in long-reference/needle "
                "single-head diagnostics."
            ),
            boosts=((1, 2.0), (4, 2.0), (5, 2.0), (9, 3.0), (11, 2.0)),
        ),
        StaticHeadWeightScheme(
            name="code_mix",
            head_scoring_mode="weighted_head_mean",
            description="Upweight code-context specialists head 2 and head 9.",
            boosts=((2, 3.0), (9, 3.0)),
        ),
    )


def schemes_from_names(
    names: Sequence[str],
    *,
    available: Sequence[StaticHeadWeightScheme] | None = None,
) -> tuple[StaticHeadWeightScheme, ...]:
    """Resolve comma-separated scheme names into explicit scheme records."""

    if not names:
        raise ValueError("scheme names must not be empty")
    by_name = {scheme.name: scheme for scheme in (available or default_static_head_weight_schemes())}
    schemes: list[StaticHeadWeightScheme] = []
    for name in names:
        scheme_name = name.strip()
        if not scheme_name:
            continue
        try:
            schemes.append(by_name[scheme_name])
        except KeyError as exc:
            valid = ", ".join(sorted(by_name))
            raise ValueError(f"unknown static head scheme {scheme_name!r}; valid: {valid}") from exc
    if not schemes:
        raise ValueError("scheme names must not be empty")
    return tuple(schemes)


def run_static_head_weight_validation(
    *,
    model_names: Sequence[str],
    prompt_cases: Sequence[PromptRetrievalCase] | None = None,
    schemes: Sequence[StaticHeadWeightScheme] | None = None,
    representation_source: RepresentationSource = "query_mean_last_layer",
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    head_count: int | None = None,
) -> StaticHeadWeightValidationResult:
    """Run constrained static head-weight schemes through the live selector path."""

    if not model_names:
        raise ValueError("model_names must not be empty")
    cases = tuple(prompt_cases or default_prompt_retrieval_cases())
    scheme_tuple = tuple(schemes or default_static_head_weight_schemes())
    if not scheme_tuple:
        raise ValueError("schemes must not be empty")
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

    rows: list[StaticHeadWeightRunRow] = []
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
        model_head_count = head_count or _infer_runtime_head_count(runtime)

        for prompt_case in cases:
            prompt = prompt_case.path.read_text(encoding="utf-8")
            for scheme in scheme_tuple:
                concrete_weights = scheme.weights_for_head_count(model_head_count)
                result = run_real_block_selector(
                    runtime,
                    prompt,
                    replace(
                        base_config,
                        representation_source=representation_source,
                        prompt_name=prompt_case.name,
                        head_scoring_mode=scheme.head_scoring_mode,
                        head_weights=concrete_weights,
                        include_block_text=True,
                    ),
                )
                rows.append(
                    _row_from_result(
                        model_name=model_name,
                        prompt_case=prompt_case,
                        representation_source=representation_source,
                        scheme=scheme,
                        head_weights=concrete_weights,
                        result=result,
                    )
                )

    return summarize_static_head_weight_rows(
        tuple(rows),
        schemes=scheme_tuple,
        model_load_seconds=model_load_seconds,
    )


def summarize_static_head_weight_rows(
    rows: Sequence[StaticHeadWeightRunRow],
    *,
    schemes: Sequence[StaticHeadWeightScheme] | None = None,
    model_load_seconds: dict[str, float] | None = None,
) -> StaticHeadWeightValidationResult:
    """Aggregate validation run rows into scheme and prompt tables."""

    row_tuple = tuple(rows)
    aggregates = _build_aggregate_summaries(row_tuple)
    ranked = tuple(sorted(aggregates, key=_aggregate_rank_key))
    return StaticHeadWeightValidationResult(
        rows=row_tuple,
        aggregate_summaries=aggregates,
        prompt_breakdowns=_build_prompt_breakdowns(row_tuple),
        ranked_summaries=ranked,
        schemes=tuple(schemes or default_static_head_weight_schemes()),
        model_load_seconds=dict(model_load_seconds or {}),
    )


def write_static_head_weight_validation_outputs(
    result: StaticHeadWeightValidationResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text validation outputs."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_static_head_weight_report(result), encoding="utf-8")


def format_static_head_weight_report(result: StaticHeadWeightValidationResult) -> str:
    """Format compact static-head validation tables."""

    lines = [
        "STATIC HEAD-WEIGHT VALIDATION",
        f"model_load_seconds={result.model_load_seconds}",
        "",
        "SCHEMES",
    ]
    for scheme in result.schemes:
        lines.append(f"{scheme.name} | {scheme.head_scoring_mode} | {scheme.description}")

    lines.append("")
    lines.append("RANKED AGGREGATES")
    for row in result.ranked_summaries:
        lines.append(_format_aggregate(row))

    lines.append("")
    lines.append("PER-PROMPT BREAKDOWN")
    for row in result.prompt_breakdowns:
        lines.append(
            f"{row.prompt_name} | {row.scheme_name} | runs={row.run_count} "
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
            f"{row.model_name} | {row.prompt_name} | {row.scheme_name} | "
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
    scheme: StaticHeadWeightScheme,
    head_weights: tuple[float, ...],
    result: Any,
) -> StaticHeadWeightRunRow:
    quality = retrieval_quality_for_result(
        selected_ids=result.selected_block_ids,
        block_text_by_id={
            block.block_id: block.block_text or block.preview_text
            for block in result.block_inspections
        },
        target_fragments=prompt_case.target_fragments,
    )
    return StaticHeadWeightRunRow(
        model_name=model_name,
        prompt_name=prompt_case.name,
        prompt_file=str(prompt_case.path),
        representation_source=representation_source,
        scheme_name=scheme.name,
        scheme_description=scheme.description,
        head_scoring_mode=scheme.head_scoring_mode,
        head_weights=head_weights,
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


def _infer_runtime_head_count(runtime: LocalHfRuntime) -> int:
    model = getattr(runtime, "_model", None)
    config = getattr(model, "config", None)
    head_count = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", None)
    if head_count is None or int(head_count) <= 0:
        raise RuntimeError(
            "static head-weight validation requires a GPT-2-class model config "
            "with n_head/num_attention_heads; pass head_count explicitly in tests"
        )
    return int(head_count)


def _build_aggregate_summaries(
    rows: Sequence[StaticHeadWeightRunRow],
) -> tuple[StaticHeadWeightAggregate, ...]:
    grouped: dict[str, list[StaticHeadWeightRunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.scheme_name].append(row)

    aggregates: list[StaticHeadWeightAggregate] = []
    for scheme_name, group in grouped.items():
        first = group[0]
        aggregates.append(
            StaticHeadWeightAggregate(
                scheme_name=scheme_name,
                scheme_description=first.scheme_description,
                head_scoring_mode=first.head_scoring_mode,
                head_weights=first.head_weights,
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

    pooled = next(
        (row for row in aggregates if row.scheme_name == "pooled_mean_heads"),
        None,
    )
    if pooled is None:
        return tuple(sorted(aggregates, key=lambda item: item.scheme_name))
    return tuple(
        sorted(
            (
                replace(
                    row,
                    recall_delta_vs_pooled=_optional_delta(
                        row.mean_recall,
                        pooled.mean_recall,
                    ),
                    precision_delta_vs_pooled=_optional_delta(
                        row.mean_precision,
                        pooled.mean_precision,
                    ),
                    selected_to_k_delta_vs_pooled=(
                        row.mean_selected_to_semantic_k_ratio
                        - pooled.mean_selected_to_semantic_k_ratio
                    ),
                    selector_latency_delta_vs_pooled_sec=(
                        row.mean_selector_latency_sec - pooled.mean_selector_latency_sec
                    ),
                )
                for row in aggregates
            ),
            key=lambda item: item.scheme_name,
        )
    )


def _build_prompt_breakdowns(
    rows: Sequence[StaticHeadWeightRunRow],
) -> tuple[StaticHeadWeightPromptBreakdown, ...]:
    grouped: dict[tuple[str, str], list[StaticHeadWeightRunRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.prompt_name, row.scheme_name)].append(row)

    breakdowns: list[StaticHeadWeightPromptBreakdown] = []
    for (prompt_name, scheme_name), group in sorted(grouped.items()):
        breakdowns.append(
            StaticHeadWeightPromptBreakdown(
                prompt_name=prompt_name,
                scheme_name=scheme_name,
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


def _aggregate_rank_key(
    row: StaticHeadWeightAggregate,
) -> tuple[float, float, float, float, str]:
    recall = -1.0 if row.mean_recall is None else row.mean_recall
    precision = -1.0 if row.mean_precision is None else row.mean_precision
    return (
        -recall,
        -precision,
        row.mean_selected_to_semantic_k_ratio,
        row.mean_selector_latency_sec,
        row.scheme_name,
    )


def _format_aggregate(row: StaticHeadWeightAggregate) -> str:
    return (
        f"{row.scheme_name} | runs={row.run_count} "
        f"mean_recall={_fmt_optional(row.mean_recall)} "
        f"mean_precision={_fmt_optional(row.mean_precision)} "
        f"mean_selected={row.mean_selected_count:.3f} "
        f"mean_selected/K={row.mean_selected_to_semantic_k_ratio:.3f} "
        f"mean_selector={row.mean_selector_latency_sec:.6f}s "
        f"d_recall={_fmt_optional(row.recall_delta_vs_pooled)} "
        f"d_precision={_fmt_optional(row.precision_delta_vs_pooled)} "
        f"d_selected/K={_fmt_optional(row.selected_to_k_delta_vs_pooled)} "
        f"d_selector={_fmt_optional(row.selector_latency_delta_vs_pooled_sec)}s"
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
