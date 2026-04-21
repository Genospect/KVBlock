"""Real-block representation-source sweep for local Hugging Face models.

This module preserves representation-sweep history. The current attention-native
baseline is ``query_mean_last_layer``; hidden-state bridge variants remain useful
for comparison but are not the current recommended routing source.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence, cast

from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.head_diagnostics import summarize_head_diagnostics
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector

DEFAULT_REPRESENTATION_SOURCES: tuple[RepresentationSource, ...] = (
    "final_hidden",
    "hidden_layer_index",
    "middle_hidden",
    "avg_last4_hidden",
    "avg_mid4_hidden",
)

VALID_REPRESENTATION_SOURCES: tuple[RepresentationSource, ...] = (
    "final_hidden",
    "hidden_layer_index",
    "middle_hidden",
    "avg_last4_hidden",
    "avg_mid4_hidden",
    "key_mean_last_layer",
    "key_mean_mid_layer",
    "key_avg_last4",
    "query_mean_last_layer",
    "query_mean_mid_layer",
    "query_avg_last4",
)

VALID_HEAD_SCORING_MODES: tuple[str, ...] = (
    "mean_heads",
    "max_head_score",
    "topk_head_mean",
    "weighted_head_mean",
)


@dataclass(frozen=True, slots=True)
class RailSetting:
    """Recent/anchor rail settings used to isolate representation quality."""

    name: str
    keep_recent_blocks: int
    keep_anchor_blocks: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.keep_recent_blocks < 0:
            raise ValueError("keep_recent_blocks must be >= 0")
        if self.keep_anchor_blocks < 0:
            raise ValueError("keep_anchor_blocks must be >= 0")

    @property
    def label(self) -> str:
        """Return a compact report label for this rail setting."""

        return (
            f"{self.name}(recent={self.keep_recent_blocks},"
            f"anchors={self.keep_anchor_blocks})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly rail setting record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadScoringSetting:
    """Per-head Stage-A scoring setting for representation sweeps."""

    mode: str = "mean_heads"
    head_top_k: int = 2
    head_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in VALID_HEAD_SCORING_MODES:
            valid = ", ".join(VALID_HEAD_SCORING_MODES)
            raise ValueError(f"unknown head scoring mode {self.mode!r}; valid: {valid}")
        if self.head_top_k <= 0:
            raise ValueError("head_top_k must be > 0")
        if any(weight < 0 for weight in self.head_weights):
            raise ValueError("head_weights must be >= 0")
        if self.head_weights and sum(self.head_weights) <= 0:
            raise ValueError("head_weights must contain at least one positive value")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly scoring setting."""

        return asdict(self)


DEFAULT_RAIL_PRESETS: dict[str, RailSetting] = {
    "default": RailSetting("default", keep_recent_blocks=4, keep_anchor_blocks=2),
    "no_rails": RailSetting("no_rails", keep_recent_blocks=0, keep_anchor_blocks=0),
    "recent_only": RailSetting(
        "recent_only",
        keep_recent_blocks=4,
        keep_anchor_blocks=0,
    ),
    "anchor_only": RailSetting(
        "anchor_only",
        keep_recent_blocks=0,
        keep_anchor_blocks=2,
    ),
    "reduced": RailSetting("reduced", keep_recent_blocks=1, keep_anchor_blocks=0),
}


@dataclass(frozen=True, slots=True)
class PromptRetrievalCase:
    """Prompt file plus answer-bearing text fragments for retrieval scoring."""

    name: str
    path: Path
    target_fragments: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.target_fragments:
            raise ValueError("target_fragments must not be empty")
        if any(not fragment.strip() for fragment in self.target_fragments):
            raise ValueError("target_fragments must be non-empty")


@dataclass(frozen=True, slots=True)
class RetrievalQuality:
    """Prompt-specific selector quality against expected answer blocks."""

    expected_block_ids: tuple[int, ...]
    selected_expected_block_ids: tuple[int, ...]
    missed_expected_block_ids: tuple[int, ...]
    extra_selected_block_ids: tuple[int, ...]
    target_recall: float | None
    selected_precision: float | None
    target_hit: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly quality record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationSweepRunRow:
    """One model/prompt/representation-source sweep result."""

    model_name: str
    prompt_name: str
    prompt_file: str
    representation_source: str
    layer_index: int | None
    representation_name: str
    rail_setting: str
    keep_recent_blocks: int
    keep_anchor_blocks: int
    head_scoring_mode: str
    head_top_k: int
    head_weights: tuple[float, ...]
    tokens: int
    blocks: int
    selected_ids: tuple[int, ...]
    selected_reasons: dict[int, str]
    selected_scores: dict[int, float | None]
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
    head_diagnostic_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row record."""

        payload = asdict(self)
        payload["retrieval_quality"] = self.retrieval_quality.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class LayerDifferenceSummary:
    """Comparison of representation sources for one model/prompt pair."""

    model_name: str
    prompt_name: str
    rail_setting: str
    keep_recent_blocks: int
    keep_anchor_blocks: int
    head_scoring_mode: str
    baseline_source: str
    representation_source: str
    selected_jaccard_vs_baseline: float
    recall_delta_vs_baseline: float | None
    precision_delta_vs_baseline: float | None
    selector_latency_delta_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly comparison record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationAggregateSummary:
    """Aggregate representation quality grouped by model, source, and rails."""

    model_name: str
    representation_source: str
    rail_setting: str
    keep_recent_blocks: int
    keep_anchor_blocks: int
    head_scoring_mode: str
    head_top_k: int
    head_weights: tuple[float, ...]
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationRankingSummary:
    """Ranked aggregate source result for one model/rail setting."""

    rank: int
    model_name: str
    representation_source: str
    rail_setting: str
    keep_recent_blocks: int
    keep_anchor_blocks: int
    head_scoring_mode: str
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly ranking record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationSweepResult:
    """Full representation sweep result and derived layer comparisons."""

    rows: tuple[RepresentationSweepRunRow, ...]
    layer_differences: tuple[LayerDifferenceSummary, ...]
    aggregate_summaries: tuple[RepresentationAggregateSummary, ...]
    model_load_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result payload."""

        return {
            "rows": [row.to_dict() for row in self.rows],
            "layer_differences": [diff.to_dict() for diff in self.layer_differences],
            "aggregate_summaries": [
                aggregate.to_dict() for aggregate in self.aggregate_summaries
            ],
            "ranking_summaries": [
                ranking.to_dict()
                for ranking in rank_aggregate_summaries(self.aggregate_summaries)
            ],
            "model_load_seconds": dict(self.model_load_seconds),
        }


def default_prompt_retrieval_cases() -> tuple[PromptRetrievalCase, ...]:
    """Return the five local prompt cases used by current real-block experiments."""

    return (
        PromptRetrievalCase(
            name="long_reference",
            path=Path("prompts/long_reference.txt"),
            target_fragments=("AES-256", "GCM"),
        ),
        PromptRetrievalCase(
            name="needle",
            path=Path("prompts/needle.txt"),
            target_fragments=("ZXQ-4917-BETA",),
        ),
        PromptRetrievalCase(
            name="repeated_reference",
            path=Path("prompts/repeated_reference.txt"),
            target_fragments=("22 ng/mL",),
        ),
        PromptRetrievalCase(
            name="code_context",
            path=Path("prompts/code_context.txt"),
            target_fragments=(
                "def calculate",
                "_total(items)",
                "total += item",
                '["price"]',
            ),
        ),
        PromptRetrievalCase(
            name="adversarial_semantic",
            path=Path("prompts/adverserial_semantic.txt"),
            target_fragments=("AES-256-GCM",),
        ),
    )


def default_rail_settings() -> tuple[RailSetting, ...]:
    """Return rail-ablation presets for representation-quality sweeps."""

    return tuple(DEFAULT_RAIL_PRESETS.values())


def rail_settings_from_presets(names: Sequence[str]) -> tuple[RailSetting, ...]:
    """Resolve rail preset names into concrete settings."""

    if not names:
        raise ValueError("rail preset names must not be empty")
    settings: list[RailSetting] = []
    for name in names:
        preset_name = name.strip()
        if not preset_name:
            continue
        try:
            settings.append(DEFAULT_RAIL_PRESETS[preset_name])
        except KeyError as exc:
            valid = ", ".join(sorted(DEFAULT_RAIL_PRESETS))
            raise ValueError(f"unknown rail preset {preset_name!r}; valid: {valid}") from exc
    if not settings:
        raise ValueError("rail preset names must not be empty")
    return tuple(settings)


def representation_sources_from_names(
    names: Sequence[str],
) -> tuple[RepresentationSource, ...]:
    """Resolve representation source names with a clear validation error."""

    if not names:
        raise ValueError("representation source names must not be empty")
    valid = set(VALID_REPRESENTATION_SOURCES)
    sources: list[RepresentationSource] = []
    for name in names:
        source_name = name.strip()
        if not source_name:
            continue
        if source_name not in valid:
            valid_names = ", ".join(VALID_REPRESENTATION_SOURCES)
            raise ValueError(
                f"unknown representation source {source_name!r}; valid: {valid_names}"
            )
        sources.append(cast(RepresentationSource, source_name))
    if not sources:
        raise ValueError("representation source names must not be empty")
    return tuple(sources)


def head_scoring_settings_from_names(
    names: Sequence[str],
    *,
    head_top_k: int = 2,
    head_weights: Sequence[float] = (),
) -> tuple[HeadScoringSetting, ...]:
    """Resolve head scoring mode names into concrete settings."""

    if not names:
        raise ValueError("head scoring mode names must not be empty")
    settings: list[HeadScoringSetting] = []
    for name in names:
        mode = name.strip()
        if not mode:
            continue
        settings.append(
            HeadScoringSetting(
                mode=mode,
                head_top_k=head_top_k,
                head_weights=tuple(float(weight) for weight in head_weights),
            )
        )
    if not settings:
        raise ValueError("head scoring mode names must not be empty")
    return tuple(settings)


def run_representation_sweep(
    *,
    model_names: Sequence[str],
    prompt_cases: Sequence[PromptRetrievalCase] | None = None,
    representation_sources: Sequence[RepresentationSource] = DEFAULT_REPRESENTATION_SOURCES,
    hidden_layer_index: int = 1,
    rail_settings: Sequence[RailSetting] | None = None,
    head_scoring_settings: Sequence[HeadScoringSetting] | None = None,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    include_head_diagnostics: bool = False,
    diagnostic_top_heads: int = 5,
) -> RepresentationSweepResult:
    """Run the real-block selector across models, prompts, and representation sources."""

    if not model_names:
        raise ValueError("model_names must not be empty")
    if not representation_sources:
        raise ValueError("representation_sources must not be empty")

    cases = tuple(prompt_cases or default_prompt_retrieval_cases())
    config = selector_config or RealBlockSelectorConfig(
        block_size=16,
        shortlist_m=16,
        semantic_k=4,
        confidence_margin=0.0,
        preview_chars=120,
        include_block_text=True,
    )
    rails = tuple(
        rail_settings
        or (
            RailSetting(
                "configured",
                keep_recent_blocks=config.keep_recent_blocks,
                keep_anchor_blocks=config.keep_anchor_blocks,
            ),
        )
    )
    if not rails:
        raise ValueError("rail_settings must not be empty")
    head_settings = tuple(head_scoring_settings or (HeadScoringSetting(),))
    if not head_settings:
        raise ValueError("head_scoring_settings must not be empty")
    load_kwargs = dict(load_config_kwargs or {})

    rows: list[RepresentationSweepRunRow] = []
    model_load_seconds: dict[str, float] = {}
    for model_name in model_names:
        runtime = LocalHfRuntime(RuntimeLoadConfig(model_name=model_name, **load_kwargs))
        started_at = perf_counter()
        runtime.load_model()
        model_load_seconds[model_name] = perf_counter() - started_at

        for rail_setting in rails:
            for head_setting in head_settings:
                rail_config = _config_for_rail(
                    config,
                    rail_setting,
                    head_setting,
                    include_head_diagnostics=include_head_diagnostics,
                    diagnostic_top_heads=diagnostic_top_heads,
                )
                for source in representation_sources:
                    runtime.capture_config = HiddenStateCaptureConfig(
                        representation_source=source,
                        layer_index=hidden_layer_index,
                    )
                    for prompt_case in cases:
                        row = _run_one_case(
                            runtime=runtime,
                            model_name=model_name,
                            prompt_case=prompt_case,
                            source=source,
                            hidden_layer_index=hidden_layer_index,
                            rail_setting=rail_setting,
                            head_setting=head_setting,
                            config=rail_config,
                        )
                        rows.append(row)

    row_tuple = tuple(rows)
    return RepresentationSweepResult(
        rows=row_tuple,
        layer_differences=_build_layer_differences(row_tuple),
        aggregate_summaries=_build_aggregate_summaries(row_tuple),
        model_load_seconds=model_load_seconds,
    )


def write_representation_sweep_outputs(
    result: RepresentationSweepResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and compact text reports for a representation sweep."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_representation_sweep_report(result), encoding="utf-8")


def format_representation_sweep_report(result: RepresentationSweepResult) -> str:
    """Format a compact report focused on ids, quality, latency, and layer deltas."""

    lines = [
        "REAL-BLOCK REPRESENTATION SWEEP",
        f"model_load_seconds={result.model_load_seconds}",
        "",
        "RUNS",
    ]
    for row in result.rows:
        quality = row.retrieval_quality
        lines.append(
            f"{row.model_name} | {row.prompt_name} | {row.representation_source} "
            f"({row.representation_name}) | rail={row.rail_setting}"
            f"(recent={row.keep_recent_blocks},anchors={row.keep_anchor_blocks}) | "
            f"head_mode={row.head_scoring_mode} | "
            f"selected={list(row.selected_ids)} | "
            f"selected/K={row.selected_to_semantic_k_ratio:.3f} | "
            f"recall={_fmt_optional(quality.target_recall)} "
            f"precision={_fmt_optional(quality.selected_precision)} "
            f"hit={quality.target_hit} | selector={row.selector_latency_sec:.6f}s "
            f"total={row.total_latency_sec:.6f}s"
        )
        lines.append(
            "  reasons="
            + ", ".join(
                f"{block_id}:{reason}:{_fmt_optional(row.selected_scores[block_id])}"
                for block_id, reason in row.selected_reasons.items()
            )
        )
        if row.head_diagnostic_summary is not None:
            lines.append(
                "  selected_top_heads="
                + _fmt_head_summary(row.head_diagnostic_summary)
            )
    lines.append("")
    lines.append("AGGREGATES BY MODEL / REPRESENTATION / RAILS")
    for aggregate in result.aggregate_summaries:
        lines.append(
            f"{aggregate.model_name} | {aggregate.representation_source} | "
            f"rail={aggregate.rail_setting}"
            f"(recent={aggregate.keep_recent_blocks},"
            f"anchors={aggregate.keep_anchor_blocks}) | "
            f"head_mode={aggregate.head_scoring_mode} | "
            f"runs={aggregate.run_count} "
            f"mean_recall={_fmt_optional(aggregate.mean_recall)} "
            f"mean_precision={_fmt_optional(aggregate.mean_precision)} "
            f"mean_selected={aggregate.mean_selected_count:.3f} "
            f"mean_selected/K={aggregate.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_selector={aggregate.mean_selector_latency_sec:.6f}s"
        )
    lines.append("")
    lines.append("SUMMARY RANKING BY MODEL / RAILS")
    for ranking in rank_aggregate_summaries(result.aggregate_summaries):
        lines.append(
            f"{ranking.model_name} | rail={ranking.rail_setting}"
            f"(recent={ranking.keep_recent_blocks},"
            f"anchors={ranking.keep_anchor_blocks}) | "
            f"rank={ranking.rank} | {ranking.representation_source} | "
            f"head_mode={ranking.head_scoring_mode} | "
            f"mean_recall={_fmt_optional(ranking.mean_recall)} "
            f"mean_precision={_fmt_optional(ranking.mean_precision)} "
            f"mean_selected/K={ranking.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_selector={ranking.mean_selector_latency_sec:.6f}s"
        )
    lines.append("")
    lines.append("LAYER DIFFERENCES VS final_hidden")
    for diff in result.layer_differences:
        lines.append(
            f"{diff.model_name} | {diff.prompt_name} | rail={diff.rail_setting} | "
            f"{diff.representation_source} | head_mode={diff.head_scoring_mode} | "
            f"jaccard={diff.selected_jaccard_vs_baseline:.3f} "
            f"recall_delta={_fmt_optional(diff.recall_delta_vs_baseline)} "
            f"precision_delta={_fmt_optional(diff.precision_delta_vs_baseline)} "
            f"selector_delta={diff.selector_latency_delta_sec:.6f}s"
        )
    return "\n".join(lines)


def _run_one_case(
    *,
    runtime: LocalHfRuntime,
    model_name: str,
    prompt_case: PromptRetrievalCase,
    source: RepresentationSource,
    hidden_layer_index: int,
    rail_setting: RailSetting,
    head_setting: HeadScoringSetting,
    config: RealBlockSelectorConfig,
) -> RepresentationSweepRunRow:
    prompt = prompt_case.path.read_text(encoding="utf-8")
    result = run_real_block_selector(
        runtime,
        prompt,
        replace(
            config,
            representation_source=source,
            prompt_name=prompt_case.name,
        ),
    )
    quality = retrieval_quality_for_result(
        selected_ids=result.selected_block_ids,
        block_text_by_id={
            block.block_id: block.block_text or block.preview_text
            for block in result.block_inspections
        },
        target_fragments=prompt_case.target_fragments,
    )
    head_diagnostic_summary = _head_diagnostic_summary_for_row(
        result,
        quality=quality,
        prompt_name=prompt_case.name,
        top_n=config.top_heads,
    )
    return RepresentationSweepRunRow(
        model_name=model_name,
        prompt_name=prompt_case.name,
        prompt_file=str(prompt_case.path),
        representation_source=source,
        layer_index=hidden_layer_index if source == "hidden_layer_index" else None,
        representation_name=result.run_summary.representation_name,
        rail_setting=rail_setting.name,
        keep_recent_blocks=rail_setting.keep_recent_blocks,
        keep_anchor_blocks=rail_setting.keep_anchor_blocks,
        head_scoring_mode=head_setting.mode,
        head_top_k=head_setting.head_top_k,
        head_weights=head_setting.head_weights,
        tokens=result.run_summary.token_count,
        blocks=result.run_summary.block_count,
        selected_ids=result.selected_block_ids,
        selected_reasons={
            block.block_id: block.selected_reason
            for block in result.selected_block_inspections
        },
        selected_scores={
            block.block_id: block.final_score
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
        head_diagnostic_summary=head_diagnostic_summary,
    )


def _head_diagnostic_summary_for_row(
    result: Any,
    *,
    quality: RetrievalQuality,
    prompt_name: str,
    top_n: int,
) -> dict[str, Any] | None:
    diagnostics = tuple(getattr(result, "head_diagnostics", ()) or ())
    if diagnostics:
        return summarize_head_diagnostics(
            diagnostics,
            expected_block_ids=quality.expected_block_ids,
            prompt_name=prompt_name,
            top_n=top_n,
        ).to_dict()
    existing_summary = getattr(result, "head_diagnostic_summary", None)
    if existing_summary is None:
        return None
    return existing_summary.to_dict()


def _config_for_rail(
    config: RealBlockSelectorConfig,
    rail_setting: RailSetting,
    head_setting: HeadScoringSetting,
    include_head_diagnostics: bool = False,
    diagnostic_top_heads: int = 5,
) -> RealBlockSelectorConfig:
    """Return a selector config with only diagnostic rail settings changed."""

    return replace(
        config,
        keep_recent_blocks=rail_setting.keep_recent_blocks,
        keep_anchor_blocks=rail_setting.keep_anchor_blocks,
        head_scoring_mode=head_setting.mode,
        head_top_k=head_setting.head_top_k,
        head_weights=head_setting.head_weights,
        emit_head_diagnostics=include_head_diagnostics,
        top_heads=diagnostic_top_heads,
        rail_setting=rail_setting.name,
    )


def retrieval_quality_for_result(
    *,
    selected_ids: Sequence[int],
    block_text_by_id: dict[int, str],
    target_fragments: Sequence[str],
) -> RetrievalQuality:
    """Score selected blocks against prompt-specific answer-bearing fragments."""

    expected = tuple(
        block_id
        for block_id, text in sorted(block_text_by_id.items())
        if any(fragment in text for fragment in target_fragments)
    )
    selected = tuple(int(block_id) for block_id in selected_ids)
    expected_set = set(expected)
    selected_set = set(selected)
    selected_expected = tuple(block_id for block_id in selected if block_id in expected_set)
    missed = tuple(block_id for block_id in expected if block_id not in selected_set)
    extra = tuple(block_id for block_id in selected if block_id not in expected_set)
    recall = None if not expected else len(selected_expected) / len(expected)
    precision = None if not selected else len(selected_expected) / len(selected)
    return RetrievalQuality(
        expected_block_ids=expected,
        selected_expected_block_ids=selected_expected,
        missed_expected_block_ids=missed,
        extra_selected_block_ids=extra,
        target_recall=recall,
        selected_precision=precision,
        target_hit=bool(selected_expected),
    )


def _build_layer_differences(
    rows: Sequence[RepresentationSweepRunRow],
) -> tuple[LayerDifferenceSummary, ...]:
    grouped: dict[
        tuple[str, str, str, str],
        dict[str, RepresentationSweepRunRow],
    ] = defaultdict(dict)
    for row in rows:
        grouped[(row.model_name, row.prompt_name, row.rail_setting, row.head_scoring_mode)][
            row.representation_source
        ] = row

    differences: list[LayerDifferenceSummary] = []
    for (model_name, prompt_name, rail_setting, head_scoring_mode), by_source in grouped.items():
        baseline = by_source.get("final_hidden")
        if baseline is None:
            continue
        for source, row in by_source.items():
            if source == "final_hidden":
                continue
            differences.append(
                LayerDifferenceSummary(
                    model_name=model_name,
                    prompt_name=prompt_name,
                    rail_setting=rail_setting,
                    keep_recent_blocks=row.keep_recent_blocks,
                    keep_anchor_blocks=row.keep_anchor_blocks,
                    head_scoring_mode=head_scoring_mode,
                    baseline_source="final_hidden",
                    representation_source=source,
                    selected_jaccard_vs_baseline=_jaccard(
                        baseline.selected_ids, row.selected_ids
                    ),
                    recall_delta_vs_baseline=_optional_delta(
                        row.retrieval_quality.target_recall,
                        baseline.retrieval_quality.target_recall,
                    ),
                    precision_delta_vs_baseline=_optional_delta(
                        row.retrieval_quality.selected_precision,
                        baseline.retrieval_quality.selected_precision,
                    ),
                    selector_latency_delta_sec=(
                        row.selector_latency_sec - baseline.selector_latency_sec
                    ),
                )
            )
    return tuple(differences)


def _build_aggregate_summaries(
    rows: Sequence[RepresentationSweepRunRow],
) -> tuple[RepresentationAggregateSummary, ...]:
    grouped: dict[tuple[str, str, str, str], list[RepresentationSweepRunRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.model_name,
                row.representation_source,
                row.rail_setting,
                row.head_scoring_mode,
            )
        ].append(row)

    aggregates: list[RepresentationAggregateSummary] = []
    for (model_name, source, rail_setting, head_scoring_mode), group in sorted(grouped.items()):
        first = group[0]
        aggregates.append(
            RepresentationAggregateSummary(
                model_name=model_name,
                representation_source=source,
                rail_setting=rail_setting,
                keep_recent_blocks=first.keep_recent_blocks,
                keep_anchor_blocks=first.keep_anchor_blocks,
                head_scoring_mode=head_scoring_mode,
                head_top_k=first.head_top_k,
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
    return tuple(aggregates)


def rank_aggregate_summaries(
    aggregates: Sequence[RepresentationAggregateSummary],
) -> tuple[RepresentationRankingSummary, ...]:
    """Rank aggregate rows by quality first, then sparsity and latency."""

    grouped: dict[tuple[str, str], list[RepresentationAggregateSummary]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[(aggregate.model_name, aggregate.rail_setting)].append(aggregate)

    rankings: list[RepresentationRankingSummary] = []
    for (_model_name, _rail_setting), group in sorted(grouped.items()):
        sorted_group = sorted(group, key=_aggregate_ranking_key)
        for index, aggregate in enumerate(sorted_group, start=1):
            rankings.append(
                RepresentationRankingSummary(
                    rank=index,
                    model_name=aggregate.model_name,
                    representation_source=aggregate.representation_source,
                    rail_setting=aggregate.rail_setting,
                    keep_recent_blocks=aggregate.keep_recent_blocks,
                    keep_anchor_blocks=aggregate.keep_anchor_blocks,
                    head_scoring_mode=aggregate.head_scoring_mode,
                    run_count=aggregate.run_count,
                    mean_recall=aggregate.mean_recall,
                    mean_precision=aggregate.mean_precision,
                    mean_selected_to_semantic_k_ratio=(
                        aggregate.mean_selected_to_semantic_k_ratio
                    ),
                    mean_selector_latency_sec=aggregate.mean_selector_latency_sec,
                )
            )
    return tuple(rankings)


def _aggregate_ranking_key(
    aggregate: RepresentationAggregateSummary,
) -> tuple[float, float, float, float, str]:
    recall = -1.0 if aggregate.mean_recall is None else aggregate.mean_recall
    precision = -1.0 if aggregate.mean_precision is None else aggregate.mean_precision
    return (
        -recall,
        -precision,
        aggregate.mean_selected_to_semantic_k_ratio,
        aggregate.mean_selector_latency_sec,
        aggregate.representation_source,
    )


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def _optional_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


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


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_head_summary(summary: dict[str, Any]) -> str:
    frequencies = summary.get("selected_top_head_counts") or ()
    if not frequencies:
        return "n/a"
    return ", ".join(
        f"h{item['head_index']}:{item['count']}({item['fraction']:.2f})"
        for item in frequencies
    )
