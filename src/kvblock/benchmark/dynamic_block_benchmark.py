"""Focused dynamic-block and multi-scale benchmark for real-block selector runs.

This module is benchmark-oriented. Current local findings keep ``fixed_40`` as
the active baseline; multi-scale, suppression, and coarse-to-fine variants are
preserved here as exploratory history rather than selector defaults.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Iterable, Literal, Sequence, cast

import torch

from kvblock.benchmark.candidate_suppression import (
    RankedCandidateSpan,
    SuppressionMode,
    SuppressionResult,
    suppress_ranked_candidates,
    suppression_modes_from_names,
)
from kvblock.benchmark.real_block_representation_sweep import (
    PromptRetrievalCase,
    RetrievalQuality,
    default_prompt_retrieval_cases,
)
from kvblock.kv.block_modes import (
    BlockCandidate,
    BlockModeName,
    block_modes_from_names,
    coarse_to_fine_spec,
    generate_child_block_candidates,
    generate_block_candidates,
    is_coarse_to_fine_mode,
    is_mixed_global_refine_mode,
    is_parent_retention_coarse_to_fine_mode,
    mixed_global_refine_spec,
    retain_parent_and_child_candidates,
)
from kvblock.kv.qk_aggregation import (
    QKAggregationStrategy,
    qk_aggregation_strategy_from_name,
)
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import (
    HiddenStateCaptureConfig,
    RepresentationSource,
    is_query_only_source,
)
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector

DEFAULT_DYNAMIC_BLOCK_MODES: tuple[BlockModeName, ...] = (
    "fixed_16",
    "fixed_24",
    "fixed_32",
    "fixed_40",
    "multiscale_16_32",
    "multiscale_16_24_32",
    "overlap_16_stride_8",
)

DEFAULT_DYNAMIC_PROMPT_NAMES: tuple[str, ...] = (
    "needle",
    "long_reference",
    "code_context",
    "repeated_reference",
)

RerankMode = Literal["none", "semantic_plus_tokenmax", "dense_qk_token_refine"]
RefineScoreMode = Literal["raw_topn_mean", "cosine_topn_mean", "softmax_mass"]
StageCPolicyMode = Literal["refined_only", "semantic_refined_mix"]

VALID_RERANK_MODES: tuple[RerankMode, ...] = (
    "none",
    "semantic_plus_tokenmax",
    "dense_qk_token_refine",
)
VALID_REFINE_SCORE_MODES: tuple[RefineScoreMode, ...] = (
    "raw_topn_mean",
    "cosine_topn_mean",
    "softmax_mass",
)
VALID_STAGE_C_POLICIES: tuple[StageCPolicyMode, ...] = (
    "refined_only",
    "semantic_refined_mix",
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "an",
        "and",
        "among",
        "are",
        "answer",
        "because",
        "before",
        "being",
        "between",
        "both",
        "but",
        "did",
        "for",
        "context",
        "could",
        "dataset",
        "does",
        "from",
        "how",
        "have",
        "here",
        "into",
        "is",
        "in",
        "of",
        "on",
        "or",
        "to",
        "input",
        "was",
        "were",
        "who",
        "the",
        "length",
        "has",
        "had",
        "his",
        "her",
        "its",
        "our",
        "out",
        "not",
        "many",
        "more",
        "most",
        "only",
        "other",
        "over",
        "passage",
        "question",
        "sample_id",
        "same",
        "some",
        "such",
        "than",
        "that",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "through",
        "under",
        "what",
        "when",
        "where",
        "which",
        "while",
        "whose",
        "with",
        "would",
        "year",
    }
)


@dataclass(frozen=True, slots=True)
class DynamicBlockRunRow:
    """One model/prompt/block-mode run for segmentation experiments."""

    model_name: str
    prompt_name: str
    prompt_file: str
    representation_source: str
    representation_name: str
    qk_aggregation_strategy: str
    rerank_mode: str
    rerank_weight: float
    block_mode: str
    suppression_mode: str
    suppression_threshold: float
    keep_recent_blocks: int
    keep_anchor_blocks: int
    tokens: int
    candidate_block_count: int
    candidate_count_before_suppression: int
    candidate_count_after_suppression: int
    neighbor_expansion: int
    halo_radius: int
    max_selected_blocks: int | None
    semantic_selected_ids: tuple[int, ...]
    selected_ids: tuple[int, ...]
    selected_candidate_ids: tuple[str, ...]
    selected_spans: tuple[str, ...]
    selected_block_sizes: tuple[int, ...]
    selected_candidate_roles: tuple[str, ...]
    suppression_decisions: tuple[dict[str, Any], ...]
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
    coarse_candidate_count: int = 0
    fine_candidate_count_after_drilldown: int = 0
    retained_parent_count: int = 0
    mixed_fallback_used: bool = False
    coarse_selected_candidate_ids: tuple[str, ...] = ()
    coarse_selected_spans: tuple[str, ...] = ()
    block_inspection_records: tuple[dict[str, Any], ...] = ()
    refine_top_n_tokens: int = 4
    refine_score_mode: str = "raw_topn_mean"
    stage_c_policy: str = "refined_only"
    scaffold_excluded_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row record."""

        payload = asdict(self)
        payload["retrieval_quality"] = self.retrieval_quality.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class DynamicBlockAggregateSummary:
    """Aggregate quality and overhead for one block mode."""

    block_mode: str
    suppression_mode: str
    suppression_threshold: float
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_candidate_block_count: float
    mean_candidate_count_after_suppression: float
    mean_selector_latency_sec: float
    mean_coarse_candidate_count: float = 0.0
    mean_fine_candidate_count_after_drilldown: float = 0.0
    mean_retained_parent_count: float = 0.0
    recall_delta_vs_best_fixed: float | None = None
    precision_delta_vs_best_fixed: float | None = None
    selected_to_k_delta_vs_best_fixed: float | None = None
    latency_delta_vs_best_fixed_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DynamicBlockPromptBreakdown:
    """Aggregate block-mode metrics grouped by prompt."""

    prompt_name: str
    block_mode: str
    qk_aggregation_strategy: str
    suppression_mode: str
    suppression_threshold: float
    run_count: int
    mean_recall: float | None
    mean_precision: float | None
    mean_selected_to_semantic_k_ratio: float
    mean_candidate_block_count: float
    mean_candidate_count_after_suppression: float
    mean_selector_latency_sec: float
    mean_coarse_candidate_count: float = 0.0
    mean_fine_candidate_count_after_drilldown: float = 0.0
    mean_retained_parent_count: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly prompt-breakdown record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class DynamicBlockBenchmarkResult:
    """Full dynamic-block benchmark result."""

    rows: tuple[DynamicBlockRunRow, ...]
    aggregate_summaries: tuple[DynamicBlockAggregateSummary, ...]
    ranked_summaries: tuple[DynamicBlockAggregateSummary, ...]
    prompt_breakdowns: tuple[DynamicBlockPromptBreakdown, ...]
    model_load_seconds: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark payload."""

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


def default_dynamic_prompt_cases(
    prompt_names: Sequence[str] = DEFAULT_DYNAMIC_PROMPT_NAMES,
) -> tuple[PromptRetrievalCase, ...]:
    """Return the prompt subset used by dynamic block-size experiments."""

    requested = tuple(name.strip() for name in prompt_names if name.strip())
    if not requested:
        raise ValueError("prompt_names must not be empty")
    by_name = {case.name: case for case in default_prompt_retrieval_cases()}
    missing = [name for name in requested if name not in by_name]
    if missing:
        valid = ", ".join(sorted(by_name))
        raise ValueError(f"unknown prompt names {missing!r}; valid: {valid}")
    return tuple(by_name[name] for name in requested)


def query_prompt_override_for_representation(
    prompt: str,
    *,
    representation_source: str,
) -> str | None:
    """Return a question-only prompt for query-only benchmark modes."""

    return _query_prompt_override(prompt, representation_source=representation_source)


def rerank_mode_from_name(name: str) -> RerankMode:
    """Validate and return one benchmark rerank mode."""

    normalized = name.strip()
    if normalized not in VALID_RERANK_MODES:
        valid = ", ".join(VALID_RERANK_MODES)
        raise ValueError(f"unknown rerank mode {name!r}; valid: {valid}")
    return cast(RerankMode, normalized)


def refine_score_mode_from_name(name: str) -> RefineScoreMode:
    """Validate and return one dense-QK refinement score mode."""

    normalized = name.strip()
    if normalized not in VALID_REFINE_SCORE_MODES:
        valid = ", ".join(VALID_REFINE_SCORE_MODES)
        raise ValueError(f"unknown refine score mode {name!r}; valid: {valid}")
    return cast(RefineScoreMode, normalized)


def stage_c_policy_from_name(name: str) -> StageCPolicyMode:
    """Validate and return one benchmark Stage-C selection policy."""

    normalized = name.strip()
    if normalized not in VALID_STAGE_C_POLICIES:
        valid = ", ".join(VALID_STAGE_C_POLICIES)
        raise ValueError(f"unknown Stage-C policy {name!r}; valid: {valid}")
    return cast(StageCPolicyMode, normalized)


def run_dynamic_block_benchmark(
    *,
    model_names: Sequence[str],
    prompt_cases: Sequence[PromptRetrievalCase] | None = None,
    block_modes: Sequence[BlockModeName] = DEFAULT_DYNAMIC_BLOCK_MODES,
    representation_source: RepresentationSource = "query_mean_last_layer",
    qk_aggregation_strategy: QKAggregationStrategy = "block_max",
    needle_qk_aggregation_strategy: QKAggregationStrategy | None = None,
    suppression_modes: Sequence[SuppressionMode] = ("none",),
    suppression_threshold: float = 0.75,
    coarse_top_k: int = 2,
    mixed_refine_parent_k: int = 8,
    mixed_global_anchor_k: int = 8,
    mixed_fallback_margin: float = 0.05,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    include_block_inspections: bool = False,
    rerank_mode: RerankMode = "none",
    rerank_weight: float = 0.3,
    refine_top_n_tokens: int = 4,
    refine_score_mode: RefineScoreMode = "raw_topn_mean",
    stage_c_policy: StageCPolicyMode = "refined_only",
    exclude_scaffold_blocks: bool = False,
    neighbor_expansion: int = 0,
    halo_radius: int = 0,
    max_selected_blocks: int | None = None,
) -> DynamicBlockBenchmarkResult:
    """Run real-block selector over fixed and multi-scale block candidates."""

    if not model_names:
        raise ValueError("model_names must not be empty")
    if not block_modes:
        raise ValueError("block_modes must not be empty")
    resolved_modes = block_modes_from_names(block_modes)
    default_strategy = qk_aggregation_strategy_from_name(qk_aggregation_strategy)
    resolved_suppression_modes = suppression_modes_from_names(suppression_modes)
    if suppression_threshold < 0.0 or suppression_threshold > 1.0:
        raise ValueError("suppression_threshold must be in [0, 1]")
    resolved_rerank_mode = rerank_mode_from_name(rerank_mode)
    if rerank_weight < 0.0 or rerank_weight > 1.0:
        raise ValueError("rerank_weight must be in [0, 1]")
    if refine_top_n_tokens <= 0:
        raise ValueError("refine_top_n_tokens must be > 0")
    resolved_refine_score_mode = refine_score_mode_from_name(refine_score_mode)
    resolved_stage_c_policy = stage_c_policy_from_name(stage_c_policy)
    if neighbor_expansion < 0:
        raise ValueError("neighbor_expansion must be >= 0")
    if halo_radius < 0:
        raise ValueError("halo_radius must be >= 0")
    if neighbor_expansion > 0 and halo_radius > 0:
        raise ValueError("use either neighbor_expansion or halo_radius, not both")
    if max_selected_blocks is not None and max_selected_blocks <= 0:
        raise ValueError("max_selected_blocks must be > 0 when provided")
    if coarse_top_k <= 0:
        raise ValueError("coarse_top_k must be > 0")
    if mixed_refine_parent_k <= 0:
        raise ValueError("mixed_refine_parent_k must be > 0")
    if mixed_global_anchor_k <= 0:
        raise ValueError("mixed_global_anchor_k must be > 0")
    if mixed_fallback_margin < 0.0:
        raise ValueError("mixed_fallback_margin must be >= 0")
    needle_strategy = (
        None
        if needle_qk_aggregation_strategy is None
        else qk_aggregation_strategy_from_name(needle_qk_aggregation_strategy)
    )
    cases = tuple(prompt_cases or default_dynamic_prompt_cases())
    config = selector_config or RealBlockSelectorConfig(
        block_size=16,
        shortlist_m=16,
        semantic_k=4,
        confidence_margin=0.0,
        keep_recent_blocks=0,
        keep_anchor_blocks=0,
        preview_chars=160,
        include_block_text=True,
        representation_source=representation_source,
    )
    if max_selected_blocks is not None and max_selected_blocks < config.semantic_k:
        raise ValueError("max_selected_blocks must be >= semantic_k")
    load_kwargs = dict(load_config_kwargs or {})

    rows: list[DynamicBlockRunRow] = []
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

        for block_mode in resolved_modes:
            for prompt_case in cases:
                strategy = _strategy_for_prompt(
                    prompt_case.name,
                    default_strategy=default_strategy,
                    needle_strategy=needle_strategy,
                )
                prompt = prompt_case.path.read_text(encoding="utf-8")
                query_prompt = _query_prompt_override(
                    prompt,
                    representation_source=representation_source,
                )
                query_only_candidates = _query_only_context_candidates(
                    runtime,
                    prompt=prompt,
                    block_mode=block_mode,
                    config=config,
                    query_prompt=query_prompt,
                )
                first_pass_result: Any | None = None
                candidate_block_count_override: int | None = None
                mixed_fallback_used = False
                if is_coarse_to_fine_mode(block_mode):
                    fine_result, coarse_result, coarse_selected = (
                        _run_coarse_to_fine_selector(
                            runtime=runtime,
                            prompt=prompt,
                            config=config,
                            block_mode=block_mode,
                            coarse_block_candidates=query_only_candidates,
                            strategy=strategy,
                            representation_source=representation_source,
                            prompt_case=prompt_case,
                            query_prompt=query_prompt,
                            coarse_top_k=coarse_top_k,
                        )
                    )
                    result = fine_result
                    first_pass_result = coarse_result
                    coarse_candidate_count = coarse_result.run_summary.block_count
                    fine_candidate_count_after_drilldown = sum(
                        1
                        for block in result.block_inspections
                        if getattr(block, "candidate_role", "block") == "child"
                    )
                    coarse_selected_candidate_ids = tuple(
                        candidate.candidate_id for candidate in coarse_selected
                    )
                    coarse_selected_spans = tuple(
                        f"{candidate.token_start}:{candidate.token_end}"
                        for candidate in coarse_selected
                    )
                    retained_parent_count = (
                        sum(
                            1
                            for block in result.block_inspections
                            if getattr(block, "candidate_role", "block") == "parent"
                        )
                        if is_parent_retention_coarse_to_fine_mode(block_mode)
                        else 0
                    )
                    candidate_block_count_override = (
                        coarse_candidate_count
                        + fine_candidate_count_after_drilldown
                        + retained_parent_count
                    )
                elif is_mixed_global_refine_mode(block_mode):
                    mixed_result, global_result, global_selected, used_fallback = (
                        _run_mixed_global_refine_selector(
                            runtime=runtime,
                            prompt=prompt,
                            config=config,
                            block_mode=block_mode,
                            global_block_candidates=query_only_candidates,
                            strategy=strategy,
                            representation_source=representation_source,
                            prompt_case=prompt_case,
                            query_prompt=query_prompt,
                            refine_parent_k=mixed_refine_parent_k,
                            global_anchor_k=mixed_global_anchor_k,
                            fallback_margin=mixed_fallback_margin,
                            rerank_mode=resolved_rerank_mode,
                            rerank_weight=rerank_weight,
                            refine_top_n_tokens=refine_top_n_tokens,
                            refine_score_mode=resolved_refine_score_mode,
                            exclude_scaffold_blocks=exclude_scaffold_blocks,
                        )
                    )
                    result = mixed_result
                    mixed_fallback_used = used_fallback
                    first_pass_result = None if used_fallback else global_result
                    coarse_candidate_count = global_result.run_summary.block_count
                    fine_candidate_count_after_drilldown = sum(
                        1
                        for block in result.block_inspections
                        if getattr(block, "candidate_role", "block") == "child"
                    )
                    retained_parent_count = sum(
                        1
                        for block in result.block_inspections
                        if getattr(block, "candidate_role", "block") == "parent"
                    )
                    coarse_selected_candidate_ids = tuple(
                        candidate.candidate_id for candidate in global_selected
                    )
                    coarse_selected_spans = tuple(
                        f"{candidate.token_start}:{candidate.token_end}"
                        for candidate in global_selected
                    )
                    candidate_block_count_override = (
                        None
                        if used_fallback
                        else (
                            coarse_candidate_count
                            + fine_candidate_count_after_drilldown
                            + retained_parent_count
                        )
                    )
                else:
                    result = run_real_block_selector(
                        runtime,
                        prompt,
                        replace(
                            config,
                            block_mode=block_mode,
                            block_candidates=query_only_candidates,
                            qk_aggregation_strategy=strategy,
                            representation_source=representation_source,
                            prompt_id=prompt_case.name,
                            prompt_name=prompt_case.name,
                            query_prompt=query_prompt,
                            relevant_text_fragments=prompt_case.target_fragments,
                            include_block_text=True,
                        ),
                    )
                    coarse_candidate_count = 0
                    fine_candidate_count_after_drilldown = 0
                    retained_parent_count = 0
                    coarse_selected_candidate_ids = ()
                    coarse_selected_spans = ()
                base_ranked_candidates = _ranked_candidates_from_result(result)
                scaffold_excluded_ids = (
                    _scaffold_block_ids_from_result(result)
                    if exclude_scaffold_blocks
                    else ()
                )
                scaffold_excluded_id_set = set(scaffold_excluded_ids)
                original_rank_by_id = {
                    candidate.block_id: candidate.rank
                    for candidate in base_ranked_candidates
                }
                original_stage_c_ranked_block_ids = tuple(
                    int(candidate.block_id)
                    for candidate in base_ranked_candidates
                    if int(candidate.block_id) not in scaffold_excluded_id_set
                )
                ranked_candidates = _rerank_candidates(
                    base_ranked_candidates,
                    result=result,
                    query_prompt=query_prompt or prompt,
                    mode=resolved_rerank_mode,
                    weight=rerank_weight,
                    refine_top_n_tokens=refine_top_n_tokens,
                    refine_score_mode=resolved_refine_score_mode,
                    refine_candidate_limit=config.shortlist_m,
                    excluded_block_ids=scaffold_excluded_ids,
                )
                ranked_score_by_id = {
                    candidate.block_id: candidate.score
                    for candidate in ranked_candidates
                }
                reranked_rank_by_id = {
                    candidate.block_id: candidate.rank
                    for candidate in ranked_candidates
                }
                for suppression_mode in resolved_suppression_modes:
                    suppression = suppress_ranked_candidates(
                        ranked_candidates,
                        mode=suppression_mode,
                        threshold=suppression_threshold,
                    )
                    selector_latency_sec_override = (
                        None
                        if first_pass_result is None
                        else (
                            result.latency.selector_sec
                            + first_pass_result.latency.selector_sec
                        )
                    )
                    total_latency_sec_override = (
                        None
                        if first_pass_result is None
                        else result.latency.total_sec + first_pass_result.latency.total_sec
                    )
                    prefill_latency_sec_override = (
                        None
                        if first_pass_result is None
                        else (
                            result.latency.prefill_sec
                            + first_pass_result.latency.prefill_sec
                        )
                    )
                    metadata_latency_sec_override = (
                        None
                        if first_pass_result is None
                        else (
                            result.latency.metadata_sec
                            + first_pass_result.latency.metadata_sec
                        )
                    )
                    inspection_latency_sec_override = (
                        None
                        if first_pass_result is None
                        else (
                            result.latency.inspection_sec
                            + first_pass_result.latency.inspection_sec
                        )
                    )
                    rows.append(
                        _row_from_result(
                            model_name=model_name,
                            prompt_case=prompt_case,
                            representation_source=representation_source,
                            block_mode=block_mode,
                            strategy=strategy,
                            rerank_mode=resolved_rerank_mode,
                            rerank_weight=rerank_weight,
                            refine_top_n_tokens=refine_top_n_tokens,
                            refine_score_mode=resolved_refine_score_mode,
                            stage_c_policy=resolved_stage_c_policy,
                            scaffold_excluded_count=len(scaffold_excluded_ids),
                            scaffold_excluded_ids=scaffold_excluded_ids,
                            original_ranked_block_ids=original_stage_c_ranked_block_ids,
                            config=config,
                            result=result,
                            suppression=suppression,
                            ranked_score_by_id=ranked_score_by_id,
                            rerank_original_rank_by_id=(
                                original_rank_by_id
                                if resolved_rerank_mode != "none"
                                else None
                            ),
                            rerank_new_rank_by_id=(
                                reranked_rank_by_id
                                if resolved_rerank_mode != "none"
                                else None
                            ),
                            neighbor_expansion=neighbor_expansion,
                            halo_radius=halo_radius,
                            max_selected_blocks=max_selected_blocks,
                            candidate_block_count_override=candidate_block_count_override,
                            selector_latency_sec_override=selector_latency_sec_override,
                            total_latency_sec_override=total_latency_sec_override,
                            prefill_latency_sec_override=prefill_latency_sec_override,
                            metadata_latency_sec_override=metadata_latency_sec_override,
                            inspection_latency_sec_override=inspection_latency_sec_override,
                            coarse_candidate_count=coarse_candidate_count,
                            fine_candidate_count_after_drilldown=(
                                fine_candidate_count_after_drilldown
                            ),
                            retained_parent_count=retained_parent_count,
                            coarse_selected_candidate_ids=coarse_selected_candidate_ids,
                            coarse_selected_spans=coarse_selected_spans,
                            mixed_fallback_used=mixed_fallback_used,
                            include_block_inspections=include_block_inspections,
                        )
                    )

    row_tuple = tuple(rows)
    aggregates = _build_aggregate_summaries(row_tuple)
    return DynamicBlockBenchmarkResult(
        rows=row_tuple,
        aggregate_summaries=aggregates,
        ranked_summaries=tuple(sorted(aggregates, key=_summary_rank_key)),
        prompt_breakdowns=_build_prompt_breakdowns(row_tuple),
        model_load_seconds=model_load_seconds,
    )


def write_dynamic_block_benchmark_outputs(
    result: DynamicBlockBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text reports for dynamic-block experiments."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_dynamic_block_report(result), encoding="utf-8")


def format_dynamic_block_report(result: DynamicBlockBenchmarkResult) -> str:
    """Format a compact dynamic-block benchmark report."""

    lines = [
        "DYNAMIC BLOCK BENCHMARK",
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
            f"{row.prompt_name} | {row.block_mode} | "
            f"suppression={row.suppression_mode}@{row.suppression_threshold:.2f} | "
            f"qk={row.qk_aggregation_strategy} | "
            f"runs={row.run_count} "
            f"mean_recall={_fmt_optional(row.mean_recall)} "
            f"mean_precision={_fmt_optional(row.mean_precision)} "
            f"mean_selected/K={row.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_candidates={row.mean_candidate_block_count:.3f} "
            f"mean_after_suppression={row.mean_candidate_count_after_suppression:.3f} "
            f"mean_coarse={row.mean_coarse_candidate_count:.3f} "
            f"mean_fine={row.mean_fine_candidate_count_after_drilldown:.3f} "
            f"mean_retained_parents={row.mean_retained_parent_count:.3f} "
            f"mean_selector={row.mean_selector_latency_sec:.6f}s"
        )

    lines.append("")
    lines.append("RUNS")
    for row in result.rows:
        quality = row.retrieval_quality
        lines.append(
            f"{row.model_name} | {row.prompt_name} | {row.block_mode} | "
            f"suppression={row.suppression_mode}@{row.suppression_threshold:.2f} | "
            f"qk={row.qk_aggregation_strategy} | "
            f"rerank={row.rerank_mode}@{row.rerank_weight:.2f} "
            f"refine_top_n={row.refine_top_n_tokens} "
            f"refine_mode={row.refine_score_mode} "
            f"stage_c={row.stage_c_policy} "
            f"scaffold_excluded={row.scaffold_excluded_count} | "
            f"neighbor_expansion={row.neighbor_expansion} | "
            f"halo={row.halo_radius} cap={_fmt_int_optional(row.max_selected_blocks)} | "
            f"candidates={row.candidate_block_count} "
            f"ranked={row.candidate_count_before_suppression} "
            f"after={row.candidate_count_after_suppression} | "
            f"coarse={row.coarse_candidate_count} "
            f"fine={row.fine_candidate_count_after_drilldown} "
            f"retained_parents={row.retained_parent_count} | "
            f"mixed_fallback={row.mixed_fallback_used} | "
            f"selected={list(row.selected_candidate_ids)} | "
            f"roles={list(row.selected_candidate_roles)} | "
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
    block_mode: str,
    strategy: QKAggregationStrategy,
    rerank_mode: RerankMode,
    rerank_weight: float,
    refine_top_n_tokens: int,
    refine_score_mode: RefineScoreMode,
    stage_c_policy: StageCPolicyMode,
    scaffold_excluded_count: int,
    config: RealBlockSelectorConfig,
    result: Any,
    suppression: SuppressionResult,
    ranked_score_by_id: dict[int, float] | None = None,
    rerank_original_rank_by_id: dict[int, int] | None = None,
    rerank_new_rank_by_id: dict[int, int] | None = None,
    neighbor_expansion: int = 0,
    halo_radius: int = 0,
    max_selected_blocks: int | None = None,
    candidate_block_count_override: int | None = None,
    selector_latency_sec_override: float | None = None,
    total_latency_sec_override: float | None = None,
    prefill_latency_sec_override: float | None = None,
    metadata_latency_sec_override: float | None = None,
    inspection_latency_sec_override: float | None = None,
    coarse_candidate_count: int = 0,
    fine_candidate_count_after_drilldown: int = 0,
    retained_parent_count: int = 0,
    mixed_fallback_used: bool = False,
    coarse_selected_candidate_ids: tuple[str, ...] = (),
    coarse_selected_spans: tuple[str, ...] = (),
    include_block_inspections: bool = False,
    scaffold_excluded_ids: Sequence[int] = (),
    original_ranked_block_ids: Sequence[int] = (),
) -> DynamicBlockRunRow:
    block_text_by_id = {
        block.block_id: block.block_text or block.preview_text
        for block in result.block_inspections
    }
    block_by_id = {block.block_id: block for block in result.block_inspections}
    excluded_ids = {int(block_id) for block_id in scaffold_excluded_ids}
    refined_selected_ids = _selected_ids_after_suppression(
        result,
        suppression=suppression,
        semantic_k=config.semantic_k,
    )
    semantic_selected_ids = _stage_c_semantic_ids(
        refined_selected_ids=refined_selected_ids,
        original_ranked_block_ids=original_ranked_block_ids,
        semantic_k=config.semantic_k,
        policy=stage_c_policy,
        excluded_block_ids=excluded_ids,
    )
    expansion_radius = halo_radius if halo_radius > 0 else neighbor_expansion
    expansion_cap = max_selected_blocks if halo_radius > 0 else None
    expansion_reason = "halo" if halo_radius > 0 else "neighbor_expansion"
    selected_ids = _expand_selected_ids_by_neighbors(
        semantic_selected_ids,
        block_by_id=block_by_id,
        radius=expansion_radius,
        max_selected_blocks=expansion_cap,
        excluded_block_ids=excluded_ids,
    )
    quality = _fragment_quality_for_result(
        selected_ids=selected_ids,
        block_text_by_id=block_text_by_id,
        block_by_id=block_by_id,
        target_fragments=prompt_case.target_fragments,
    )
    selected = tuple(block_by_id[block_id] for block_id in selected_ids if block_id in block_by_id)
    return DynamicBlockRunRow(
        model_name=model_name,
        prompt_name=prompt_case.name,
        prompt_file=str(prompt_case.path),
        representation_source=representation_source,
        representation_name=result.run_summary.representation_name,
        qk_aggregation_strategy=strategy,
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        block_mode=block_mode,
        suppression_mode=suppression.mode,
        suppression_threshold=suppression.threshold,
        keep_recent_blocks=config.keep_recent_blocks,
        keep_anchor_blocks=config.keep_anchor_blocks,
        tokens=result.run_summary.token_count,
        candidate_block_count=(
            result.run_summary.block_count
            if candidate_block_count_override is None
            else candidate_block_count_override
        ),
        candidate_count_before_suppression=suppression.input_count,
        candidate_count_after_suppression=suppression.output_count,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        semantic_selected_ids=semantic_selected_ids,
        selected_ids=selected_ids,
        selected_candidate_ids=tuple(
            block.candidate_id or str(block.block_id) for block in selected
        ),
        selected_spans=tuple(
            f"{block.token_start}:{block.token_end}" for block in selected
        ),
        selected_block_sizes=tuple(
            int(block.block_size or block.token_count) for block in selected
        ),
        selected_candidate_roles=tuple(
            getattr(block, "candidate_role", "block") for block in selected
        ),
        suppression_decisions=_suppression_records(
            result,
            suppression,
            ranked_score_by_id=ranked_score_by_id,
            rerank_original_rank_by_id=rerank_original_rank_by_id,
            rerank_new_rank_by_id=rerank_new_rank_by_id,
            include_refined_score=rerank_mode != "none",
        ),
        selected_count=len(selected_ids),
        selected_to_semantic_k_ratio=len(selected_ids) / config.semantic_k,
        selector_latency_sec=(
            result.latency.selector_sec
            if selector_latency_sec_override is None
            else selector_latency_sec_override
        ),
        total_latency_sec=(
            result.latency.total_sec
            if total_latency_sec_override is None
            else total_latency_sec_override
        ),
        prefill_latency_sec=(
            result.latency.prefill_sec
            if prefill_latency_sec_override is None
            else prefill_latency_sec_override
        ),
        metadata_latency_sec=(
            result.latency.metadata_sec
            if metadata_latency_sec_override is None
            else metadata_latency_sec_override
        ),
        inspection_latency_sec=(
            result.latency.inspection_sec
            if inspection_latency_sec_override is None
            else inspection_latency_sec_override
        ),
        fallback_mode=result.fallback_mode,
        raw_margin=result.confidence.raw_margin,
        retrieval_quality=quality,
        coarse_candidate_count=coarse_candidate_count,
        fine_candidate_count_after_drilldown=fine_candidate_count_after_drilldown,
        retained_parent_count=retained_parent_count,
        mixed_fallback_used=mixed_fallback_used,
        coarse_selected_candidate_ids=coarse_selected_candidate_ids,
        coarse_selected_spans=coarse_selected_spans,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        stage_c_policy=stage_c_policy,
        scaffold_excluded_count=scaffold_excluded_count,
        block_inspection_records=(
            _block_inspection_records(
                result,
                selected_ids=set(selected_ids),
                semantic_selected_ids=set(semantic_selected_ids),
                expansion_selected_reason=expansion_reason,
            )
            if ranked_score_by_id is None
            else _block_inspection_records(
                result,
                score_by_id=ranked_score_by_id,
                original_rank_by_id=rerank_original_rank_by_id,
                new_rank_by_id=rerank_new_rank_by_id,
                include_refined_score=rerank_mode != "none",
                selected_ids=set(selected_ids),
                semantic_selected_ids=set(semantic_selected_ids),
                selected_reason="rerank" if rerank_mode != "none" else None,
                expansion_selected_reason=expansion_reason,
            )
        )
        if include_block_inspections
        else (),
    )


def _run_coarse_to_fine_selector(
    *,
    runtime: Any,
    prompt: str,
    config: RealBlockSelectorConfig,
    block_mode: BlockModeName,
    coarse_block_candidates: Sequence[BlockCandidate],
    strategy: QKAggregationStrategy,
    representation_source: RepresentationSource,
    prompt_case: PromptRetrievalCase,
    query_prompt: str | None,
    coarse_top_k: int,
) -> tuple[Any, Any, tuple[BlockCandidate, ...]]:
    coarse_size, fine_size = coarse_to_fine_spec(block_mode)
    coarse_result = run_real_block_selector(
        runtime,
        prompt,
        replace(
            config,
            block_size=coarse_size,
            block_mode=f"fixed_{coarse_size}",  # type: ignore[arg-type]
            block_candidates=tuple(coarse_block_candidates),
            qk_aggregation_strategy=strategy,
            representation_source=representation_source,
            prompt_id=prompt_case.name,
            prompt_name=prompt_case.name,
            query_prompt=query_prompt,
            relevant_text_fragments=prompt_case.target_fragments,
            include_block_text=True,
        ),
    )
    coarse_ranked = _ranked_candidates_from_result(coarse_result)
    coarse_block_by_id = _block_candidate_by_id(coarse_result)
    coarse_selected: list[BlockCandidate] = []
    for ranked in coarse_ranked:
        candidate = coarse_block_by_id.get(ranked.block_id)
        if candidate is None:
            continue
        coarse_selected.append(candidate)
        if len(coarse_selected) >= coarse_top_k:
            break
    if not coarse_selected:
        coarse_selected = list(coarse_block_by_id.values())[:coarse_top_k]

    fine_candidates = generate_child_block_candidates(
        token_count=coarse_result.run_summary.token_count,
        parent_candidates=tuple(coarse_selected),
        fine_block_size=fine_size,
        block_mode=block_mode,
    )
    if not fine_candidates:
        raise ValueError(f"{block_mode} produced no fine child candidates")
    final_candidates = (
        retain_parent_and_child_candidates(
            parent_candidates=tuple(coarse_selected),
            child_candidates=fine_candidates,
            block_mode=block_mode,
        )
        if is_parent_retention_coarse_to_fine_mode(block_mode)
        else fine_candidates
    )

    fine_result = run_real_block_selector(
        runtime,
        prompt,
        replace(
            config,
            block_size=fine_size,
            block_mode=block_mode,
            block_candidates=final_candidates,
            qk_aggregation_strategy=strategy,
            representation_source=representation_source,
            prompt_id=prompt_case.name,
            prompt_name=prompt_case.name,
            query_prompt=query_prompt,
            relevant_text_fragments=prompt_case.target_fragments,
            include_block_text=True,
        ),
    )
    return fine_result, coarse_result, tuple(coarse_selected)


def _run_mixed_global_refine_selector(
    *,
    runtime: Any,
    prompt: str,
    config: RealBlockSelectorConfig,
    block_mode: BlockModeName,
    global_block_candidates: Sequence[BlockCandidate],
    strategy: QKAggregationStrategy,
    representation_source: RepresentationSource,
    prompt_case: PromptRetrievalCase,
    query_prompt: str | None,
    refine_parent_k: int,
    global_anchor_k: int,
    fallback_margin: float,
    rerank_mode: RerankMode,
    rerank_weight: float,
    refine_top_n_tokens: int,
    refine_score_mode: RefineScoreMode,
    exclude_scaffold_blocks: bool,
) -> tuple[Any, Any, tuple[BlockCandidate, ...], bool]:
    global_size, fine_size = mixed_global_refine_spec(block_mode)
    global_result = run_real_block_selector(
        runtime,
        prompt,
        replace(
            config,
            block_size=global_size,
            block_mode=f"fixed_{global_size}",  # type: ignore[arg-type]
            block_candidates=tuple(global_block_candidates),
            qk_aggregation_strategy=strategy,
            representation_source=representation_source,
            prompt_id=prompt_case.name,
            prompt_name=prompt_case.name,
            query_prompt=query_prompt,
            relevant_text_fragments=prompt_case.target_fragments,
            include_block_text=True,
        ),
    )
    global_base_ranked = _ranked_candidates_from_result(global_result)
    global_scaffold_excluded_ids = (
        _scaffold_block_ids_from_result(global_result)
        if exclude_scaffold_blocks
        else ()
    )
    global_ranked = _rerank_candidates(
        global_base_ranked,
        result=global_result,
        query_prompt=query_prompt or prompt,
        mode=rerank_mode,
        weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        refine_candidate_limit=config.shortlist_m,
        excluded_block_ids=global_scaffold_excluded_ids,
    )
    global_block_by_id = _block_candidate_by_id(global_result)
    parent_budget = max(refine_parent_k, global_anchor_k)
    global_selected: list[BlockCandidate] = []
    for ranked in global_ranked:
        candidate = global_block_by_id.get(ranked.block_id)
        if candidate is None:
            continue
        global_selected.append(candidate)
        if len(global_selected) >= parent_budget:
            break
    if not global_selected:
        global_selected = list(global_block_by_id.values())[:parent_budget]

    if _is_weak_global_margin(global_result, fallback_margin=fallback_margin):
        return global_result, global_result, tuple(global_selected), True

    refine_parents = tuple(global_selected[:refine_parent_k])
    child_candidates = generate_child_block_candidates(
        token_count=global_result.run_summary.token_count,
        parent_candidates=refine_parents,
        fine_block_size=fine_size,
        block_mode=block_mode,
    )
    if not child_candidates:
        return global_result, global_result, tuple(global_selected), True

    final_candidates = retain_parent_and_child_candidates(
        parent_candidates=tuple(global_selected[:parent_budget]),
        child_candidates=child_candidates,
        block_mode=block_mode,
    )
    mixed_result = run_real_block_selector(
        runtime,
        prompt,
        replace(
            config,
            block_size=fine_size,
            block_mode=block_mode,
            block_candidates=final_candidates,
            qk_aggregation_strategy=strategy,
            representation_source=representation_source,
            prompt_id=prompt_case.name,
            prompt_name=prompt_case.name,
            query_prompt=query_prompt,
            relevant_text_fragments=prompt_case.target_fragments,
            include_block_text=True,
        ),
    )
    return mixed_result, global_result, tuple(global_selected), False


def _is_weak_global_margin(result: Any, *, fallback_margin: float) -> bool:
    if fallback_margin <= 0.0:
        return False
    confidence = getattr(result, "confidence", None)
    raw_margin = float(getattr(confidence, "raw_margin", 0.0) or 0.0)
    return raw_margin < fallback_margin


def _block_candidate_by_id(result: Any) -> dict[int, BlockCandidate]:
    candidates: dict[int, BlockCandidate] = {}
    for block in result.block_inspections:
        candidate_id = getattr(block, "candidate_id", None)
        block_mode = getattr(block, "block_mode", None)
        if candidate_id is None or block_mode is None:
            continue
        block_size = int(getattr(block, "block_size", None) or block.token_count)
        stride = int(getattr(block, "stride", None) or block_size)
        candidates[int(block.block_id)] = BlockCandidate(
            block_id=int(block.block_id),
            candidate_id=candidate_id,
            block_mode=block_mode,
            block_size=block_size,
            stride=stride,
            token_start=int(block.token_start),
            token_len=int(block.token_count),
        )
    return candidates


def _ranked_candidates_from_result(result: Any) -> tuple[RankedCandidateSpan, ...]:
    block_by_id = {block.block_id: block for block in result.block_inspections}
    trace = getattr(result, "trace", None)
    if trace is None:
        return tuple(
            RankedCandidateSpan(
                block_id=block.block_id,
                candidate_id=block.candidate_id or str(block.block_id),
                token_start=block.token_start,
                token_end=block.token_end,
                score=block.final_score or 0.0,
                rank=rank,
                block_size=getattr(block, "block_size", None),
                block_mode=getattr(block, "block_mode", None),
            )
            for rank, block in enumerate(result.block_inspections, start=1)
        )
    scores = tuple(getattr(trace, "stage_b_scores", ()) or ())
    if not scores:
        scores = tuple(getattr(trace, "stage_a_scores", ()) or ())
    ranked_scores = sorted(
        scores,
        key=lambda score: (score.final_score, -score.block_id),
        reverse=True,
    )
    ranked: list[RankedCandidateSpan] = []
    for rank, score in enumerate(ranked_scores, start=1):
        block = block_by_id.get(score.block_id)
        if block is None:
            continue
        ranked.append(
            RankedCandidateSpan(
                block_id=block.block_id,
                candidate_id=block.candidate_id or str(block.block_id),
                token_start=block.token_start,
                token_end=block.token_end,
                score=score.final_score,
                rank=rank,
                block_size=block.block_size,
                block_mode=block.block_mode,
            )
        )
    return tuple(ranked)


def _rerank_candidates(
    ranked_candidates: Sequence[RankedCandidateSpan],
    *,
    result: Any,
    query_prompt: str,
    mode: RerankMode,
    weight: float,
    refine_top_n_tokens: int = 4,
    refine_score_mode: RefineScoreMode = "raw_topn_mean",
    refine_candidate_limit: int | None = None,
    excluded_block_ids: Sequence[int] = (),
) -> tuple[RankedCandidateSpan, ...]:
    """Apply benchmark-only reranking over existing ranked candidates."""

    excluded = {int(block_id) for block_id in excluded_block_ids}
    candidates = tuple(
        candidate
        for candidate in ranked_candidates
        if int(candidate.block_id) not in excluded
    )
    if mode == "none" or not candidates:
        return _with_sequential_ranks(candidates)
    if mode == "dense_qk_token_refine":
        return _dense_qk_refine_candidates(
            candidates,
            result=result,
            top_n_tokens=refine_top_n_tokens,
            score_mode=refine_score_mode,
            candidate_limit=refine_candidate_limit,
        )
    if mode != "semantic_plus_tokenmax":
        raise ValueError(f"unsupported rerank mode: {mode!r}")
    query_tokens = _content_tokens(query_prompt)
    if not query_tokens:
        return candidates
    block_by_id = {block.block_id: block for block in result.block_inspections}
    semantic_scores = [candidate.score for candidate in candidates]
    semantic_min = min(semantic_scores)
    semantic_max = max(semantic_scores)
    semantic_range = semantic_max - semantic_min

    reranked: list[RankedCandidateSpan] = []
    for candidate in candidates:
        semantic = (
            0.5
            if semantic_range == 0
            else (candidate.score - semantic_min) / semantic_range
        )
        block = block_by_id.get(candidate.block_id)
        block_text = (
            ""
            if block is None
            else (
                getattr(block, "block_text", None)
                or getattr(block, "preview_text", "")
            )
        )
        token_score = _tokenmax_score(query_tokens, block_text)
        combined = (1.0 - weight) * semantic + weight * token_score
        reranked.append(
            RankedCandidateSpan(
                block_id=candidate.block_id,
                candidate_id=candidate.candidate_id,
                token_start=candidate.token_start,
                token_end=candidate.token_end,
                score=combined,
                rank=candidate.rank,
                block_size=candidate.block_size,
                block_mode=candidate.block_mode,
            )
        )
    reranked.sort(key=lambda item: (item.score, -item.block_id), reverse=True)
    return tuple(
        RankedCandidateSpan(
            block_id=candidate.block_id,
            candidate_id=candidate.candidate_id,
            token_start=candidate.token_start,
            token_end=candidate.token_end,
            score=candidate.score,
            rank=rank,
            block_size=candidate.block_size,
            block_mode=candidate.block_mode,
        )
        for rank, candidate in enumerate(reranked, start=1)
    )


def _with_sequential_ranks(
    ranked_candidates: Sequence[RankedCandidateSpan],
) -> tuple[RankedCandidateSpan, ...]:
    return tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(ranked_candidates, start=1)
    )


def _dense_qk_refine_candidates(
    ranked_candidates: Sequence[RankedCandidateSpan],
    *,
    result: Any,
    top_n_tokens: int,
    score_mode: RefineScoreMode,
    candidate_limit: int | None,
) -> tuple[RankedCandidateSpan, ...]:
    """Rerank the Stage-A shortlist with exact per-token QK scores."""

    if top_n_tokens <= 0:
        raise ValueError("top_n_tokens must be > 0")
    resolved_score_mode = refine_score_mode_from_name(score_mode)
    candidates = tuple(ranked_candidates)
    if not candidates:
        return ()
    limit = (
        len(candidates)
        if candidate_limit is None
        else min(candidate_limit, len(candidates))
    )
    if limit <= 0:
        return candidates
    token_mass = None
    if resolved_score_mode == "softmax_mass":
        token_heads, query = _dense_qk_tensors_from_result(result)
        token_mass = _dense_qk_token_mass(token_heads, query)
    refined_head = [
        replace(
            candidate,
            score=_dense_qk_token_refine_score(
                candidate,
                result=result,
                top_n_tokens=top_n_tokens,
                score_mode=resolved_score_mode,
                token_mass=token_mass,
            ),
        )
        for candidate in candidates[:limit]
    ]
    refined_head.sort(key=lambda item: (item.score, -item.block_id), reverse=True)
    reranked = tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(refined_head, start=1)
    )
    tail = tuple(
        replace(candidate, rank=rank)
        for rank, candidate in enumerate(
            candidates[limit:],
            start=len(reranked) + 1,
        )
    )
    return reranked + tail


def _dense_qk_token_refine_score(
    candidate: RankedCandidateSpan,
    *,
    result: Any,
    top_n_tokens: int,
    score_mode: RefineScoreMode,
    token_mass: torch.Tensor | None = None,
) -> float:
    if token_mass is None:
        token_heads, query = _dense_qk_tensors_from_result(result)
        token_count = int(token_heads.shape[1])
    else:
        token_heads = None
        query = None
        token_count = int(token_mass.numel())
    start = max(0, min(int(candidate.token_start), token_count))
    end = max(start, min(int(candidate.token_end), token_count))
    if end <= start:
        return float("-inf")
    if score_mode == "softmax_mass":
        if token_mass is None:
            token_mass = _dense_qk_token_mass(token_heads, query)
        return float(token_mass[start:end].sum().item())
    if token_heads is None or query is None:
        raise ValueError("dense_qk token tensors are required for top-n scoring")
    block_heads = token_heads[:, start:end, :]
    if score_mode == "cosine_topn_mean":
        block_heads = torch.nn.functional.normalize(block_heads, p=2, dim=-1)
        query = torch.nn.functional.normalize(query, p=2, dim=-1)
    scale = math.sqrt(float(block_heads.shape[-1]))
    logits = (block_heads * query[:, None, :]).sum(dim=-1)
    if score_mode == "raw_topn_mean":
        logits = logits / scale
    token_scores = logits.mean(dim=0)
    count = min(top_n_tokens, int(token_scores.numel()))
    if count <= 0:
        return float("-inf")
    return float(torch.topk(token_scores, k=count).values.mean().item())


def _dense_qk_tensors_from_result(result: Any) -> tuple[torch.Tensor, torch.Tensor]:
    token_heads = getattr(result, "per_head_token_representations", None)
    query_heads = getattr(result, "per_head_query_representation", None)
    if token_heads is None or query_heads is None:
        raise ValueError("dense_qk_token_refine requires per-head query/key tensors")
    if not isinstance(token_heads, torch.Tensor) or not isinstance(
        query_heads,
        torch.Tensor,
    ):
        raise TypeError("per-head query/key representations must be torch tensors")
    if token_heads.ndim != 3:
        raise ValueError(
            "per_head_token_representations must have shape [heads, tokens, dim]"
        )
    if query_heads.ndim != 2:
        raise ValueError("per_head_query_representation must have shape [heads, dim]")
    if token_heads.shape[0] != query_heads.shape[0]:
        raise ValueError("query/key head counts must match")
    if token_heads.shape[2] != query_heads.shape[1]:
        raise ValueError("query/key head dimensions must match")
    token_heads = token_heads.detach().to(dtype=torch.float32)
    query_heads = query_heads.detach().to(dtype=torch.float32, device=token_heads.device)
    return token_heads, query_heads


def _dense_qk_token_mass(
    per_head_token_representations: torch.Tensor,
    per_head_query_representation: torch.Tensor,
) -> torch.Tensor:
    """Return mean per-head softmax QK mass for each prompt token."""

    token_heads = per_head_token_representations.detach().to(dtype=torch.float32)
    query_heads = per_head_query_representation.detach().to(
        dtype=torch.float32,
        device=token_heads.device,
    )
    scale = math.sqrt(float(token_heads.shape[-1]))
    logits = (token_heads * query_heads[:, None, :]).sum(dim=-1) / scale
    return torch.softmax(logits, dim=-1).mean(dim=0)


def _content_tokens(text: str) -> tuple[str, ...]:
    """Return normalized content-bearing tokens for lexical reranking."""

    seen: set[str] = set()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0).casefold().strip("_-'")
        if not token:
            continue
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def _tokenmax_score(query_tokens: Sequence[str], block_text: str) -> float:
    """Score a block by strongest lexical/entity matches to query tokens."""

    block_tokens = _content_tokens(block_text)
    if not query_tokens or not block_tokens:
        return 0.0
    per_query_scores = [
        max(
            _lexical_token_similarity(query_token, block_token)
            for block_token in block_tokens
        )
        for query_token in query_tokens
    ]
    if not per_query_scores:
        return 0.0
    # Let a few exact entities/dates spike relevance without requiring every
    # question term to appear in the evidence span.
    top_count = min(4, len(per_query_scores))
    top_scores = sorted(per_query_scores, reverse=True)[:top_count]
    return sum(top_scores) / top_count


def _lexical_token_similarity(query_token: str, block_token: str) -> float:
    """Return a deterministic lexical similarity signal in [0, 1]."""

    if query_token == block_token:
        return 1.0
    query_stem = _simple_token_stem(query_token)
    block_stem = _simple_token_stem(block_token)
    if query_stem and query_stem == block_stem:
        return 0.9
    if len(query_token) >= 4 and len(block_token) >= 4:
        if query_token in block_token or block_token in query_token:
            return 0.8
    return 0.0


def _simple_token_stem(token: str) -> str:
    """Normalize common plural/possessive suffixes for exact-name matching."""

    if token.endswith("'s") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and len(token) > 4:
        return token[:-1]
    return token


def _scaffold_block_ids_from_result(result: Any) -> tuple[int, ...]:
    """Return LongBench-style metadata-only blocks to exclude from rerank."""

    return tuple(
        int(block.block_id)
        for block in getattr(result, "block_inspections", ())
        if _is_scaffold_block(block)
    )


def _is_scaffold_block(block: Any) -> bool:
    """Detect prompt wrapper blocks without discarding passage-bearing blocks."""

    text = (
        getattr(block, "block_text", None)
        or getattr(block, "preview_text", "")
        or ""
    )
    normalized = " ".join(str(text).split()).casefold()
    if not normalized:
        return False
    has_passage_content = "passage " in normalized or "context: passage" in normalized
    if normalized.startswith("dataset:") and not has_passage_content:
        return True
    if normalized.startswith("sample_id:") and not has_passage_content:
        return True
    if normalized.startswith("input:"):
        return True
    return normalized in {"context:", "input:", "dataset:", "sample_id:", "length:"}


def _selected_ids_after_suppression(
    result: Any,
    *,
    suppression: SuppressionResult,
    semantic_k: int,
) -> tuple[int, ...]:
    if semantic_k <= 0:
        raise ValueError("semantic_k must be > 0")
    if not suppression.survivor_block_ids:
        return tuple(int(block_id) for block_id in result.selected_block_ids)
    return tuple(int(block_id) for block_id in suppression.survivor_block_ids[:semantic_k])


def _stage_c_semantic_ids(
    *,
    refined_selected_ids: Sequence[int],
    original_ranked_block_ids: Sequence[int],
    semantic_k: int,
    policy: StageCPolicyMode,
    excluded_block_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Return benchmark Stage-C anchors from refined and/or original rankings."""

    if semantic_k <= 0:
        raise ValueError("semantic_k must be > 0")
    resolved_policy = stage_c_policy_from_name(policy)
    excluded = {int(block_id) for block_id in excluded_block_ids}
    refined = _unique_ids(
        block_id
        for block_id in refined_selected_ids
        if int(block_id) not in excluded
    )
    if resolved_policy == "refined_only":
        return refined[:semantic_k]

    original = _unique_ids(
        block_id
        for block_id in original_ranked_block_ids
        if int(block_id) not in excluded
    )
    if not original:
        return refined[:semantic_k]

    selected: list[int] = []
    seen: set[int] = set()

    def add_from(ids: Sequence[int], *, limit: int | None = None) -> None:
        added = 0
        for block_id in ids:
            normalized = int(block_id)
            if normalized in seen or normalized in excluded:
                continue
            selected.append(normalized)
            seen.add(normalized)
            added += 1
            if len(selected) >= semantic_k:
                return
            if limit is not None and added >= limit:
                return

    original_quota = max(1, semantic_k // 2)
    refined_quota = max(0, semantic_k - original_quota)
    add_from(original, limit=original_quota)
    add_from(refined, limit=refined_quota)
    add_from(original)
    add_from(refined)
    return tuple(selected[:semantic_k])


def _unique_ids(ids: Iterable[int]) -> tuple[int, ...]:
    seen: set[int] = set()
    unique: list[int] = []
    for block_id in ids:
        normalized = int(block_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return tuple(unique)


def _expand_selected_ids_by_neighbors(
    selected_ids: Sequence[int],
    *,
    block_by_id: dict[int, Any],
    radius: int,
    max_selected_blocks: int | None = None,
    excluded_block_ids: Iterable[int] = (),
) -> tuple[int, ...]:
    """Expand selected ids with adjacent token-span neighbors for diagnostics."""

    excluded = {int(block_id) for block_id in excluded_block_ids}
    base = tuple(
        int(block_id)
        for block_id in selected_ids
        if int(block_id) not in excluded
    )
    if radius <= 0 or not base:
        return base
    cap = (
        None
        if max_selected_blocks is None
        else max(int(max_selected_blocks), len(base))
    )
    ordered_blocks = sorted(
        block_by_id.values(),
        key=lambda block: (
            int(getattr(block, "token_start", 0)),
            int(getattr(block, "token_end", 0)),
            int(getattr(block, "block_id", 0)),
        ),
    )
    index_by_id = {
        int(block.block_id): index
        for index, block in enumerate(ordered_blocks)
    }
    expanded: list[int] = []
    seen: set[int] = set()

    def append(block_id: int, *, enforce_cap: bool) -> bool:
        if enforce_cap and cap is not None and len(expanded) >= cap:
            return False
        if block_id in excluded:
            return True
        if block_id in block_by_id and block_id not in seen:
            seen.add(block_id)
            expanded.append(block_id)
        return True

    for block_id in base:
        append(block_id, enforce_cap=False)

    for distance in range(1, radius + 1):
        for block_id in base:
            center = index_by_id.get(block_id)
            if center is None:
                continue
            left = center - distance
            right = center + distance
            if left >= 0 and not append(
                int(ordered_blocks[left].block_id),
                enforce_cap=True,
            ):
                return tuple(expanded)
            if right < len(ordered_blocks) and not append(
                int(ordered_blocks[right].block_id),
                enforce_cap=True,
            ):
                return tuple(expanded)
    return tuple(expanded)


def _suppression_records(
    result: Any,
    suppression: SuppressionResult,
    *,
    ranked_score_by_id: dict[int, float] | None = None,
    rerank_original_rank_by_id: dict[int, int] | None = None,
    rerank_new_rank_by_id: dict[int, int] | None = None,
    include_refined_score: bool = False,
) -> tuple[dict[str, Any], ...]:
    block_by_id = {block.block_id: block for block in result.block_inspections}
    records: list[dict[str, Any]] = []
    for decision in suppression.decisions:
        block = block_by_id.get(decision.block_id)
        payload = decision.to_dict()
        original_rank = (
            None
            if rerank_original_rank_by_id is None
            else rerank_original_rank_by_id.get(decision.block_id)
        )
        new_rank = (
            None
            if rerank_new_rank_by_id is None
            else rerank_new_rank_by_id.get(decision.block_id)
        )
        if original_rank is not None:
            payload["rerank_original_rank"] = original_rank
        if new_rank is not None:
            payload["rerank_new_rank"] = new_rank
        if original_rank is not None and new_rank is not None:
            payload["rerank_rank_delta"] = original_rank - new_rank
        if block is not None:
            final_score = (
                ranked_score_by_id[block.block_id]
                if ranked_score_by_id is not None
                and block.block_id in ranked_score_by_id
                else getattr(block, "final_score", None)
            )
            payload.update(
                {
                    "token_start": block.token_start,
                    "token_end": block.token_end,
                    "token_count": block.token_count,
                    "block_size": getattr(block, "block_size", None),
                    "stride": getattr(block, "stride", None),
                    "block_mode": getattr(block, "block_mode", None),
                    "candidate_role": getattr(block, "candidate_role", "block"),
                    "parent_block_id": getattr(block, "parent_block_id", None),
                    "parent_candidate_id": getattr(block, "parent_candidate_id", None),
                    "parent_token_start": getattr(block, "parent_token_start", None),
                    "parent_token_end": getattr(block, "parent_token_end", None),
                    "stage_a_score": getattr(block, "stage_a_score", None),
                    "stage_b_score": getattr(block, "stage_b_score", None),
                    "final_score": final_score,
                    "preview_text": block.preview_text,
                }
            )
            if include_refined_score:
                payload["refined_score"] = final_score
        records.append(payload)
    return tuple(records)


def _block_inspection_records(
    result: Any,
    *,
    score_by_id: dict[int, float] | None = None,
    original_rank_by_id: dict[int, int] | None = None,
    new_rank_by_id: dict[int, int] | None = None,
    include_refined_score: bool = False,
    selected_ids: set[int] | None = None,
    semantic_selected_ids: set[int] | None = None,
    selected_reason: str | None = None,
    expansion_selected_reason: str = "neighbor_expansion",
) -> tuple[dict[str, Any], ...]:
    """Return compact block inspection records for downstream diagnostics."""

    records: list[dict[str, Any]] = []
    for block in result.block_inspections:
        selected = (
            bool(block.selected)
            if selected_ids is None
            else block.block_id in selected_ids
        )
        reason = block.selected_reason
        if (
            semantic_selected_ids is not None
            and selected
            and block.block_id not in semantic_selected_ids
        ):
            reason = expansion_selected_reason
        elif selected_reason is not None and selected and not bool(block.selected):
            reason = selected_reason
        final_score = (
            score_by_id[block.block_id]
            if score_by_id is not None and block.block_id in score_by_id
            else block.final_score
        )
        payload: dict[str, Any] = {
            "block_id": block.block_id,
            "candidate_id": block.candidate_id or str(block.block_id),
            "token_start": block.token_start,
            "token_end": block.token_end,
            "token_count": block.token_count,
            "selected": selected,
            "selected_reason": reason,
            "stage_a_score": block.stage_a_score,
            "stage_b_score": block.stage_b_score,
            "final_score": final_score,
            "preview_text": block.preview_text,
            "block_size": getattr(block, "block_size", None),
            "stride": getattr(block, "stride", None),
            "block_mode": getattr(block, "block_mode", None),
            "candidate_role": getattr(block, "candidate_role", "block"),
        }
        original_rank = (
            None
            if original_rank_by_id is None
            else original_rank_by_id.get(block.block_id)
        )
        new_rank = None if new_rank_by_id is None else new_rank_by_id.get(block.block_id)
        if original_rank is not None:
            payload["rerank_original_rank"] = original_rank
        if new_rank is not None:
            payload["rerank_new_rank"] = new_rank
        if original_rank is not None and new_rank is not None:
            payload["rerank_rank_delta"] = original_rank - new_rank
        if include_refined_score:
            payload["refined_score"] = final_score
        records.append(payload)
    return tuple(records)


def _fragment_quality_for_result(
    *,
    selected_ids: Sequence[int],
    block_text_by_id: dict[int, str],
    block_by_id: dict[int, Any] | None = None,
    target_fragments: Sequence[str],
) -> RetrievalQuality:
    """Score by target-fragment coverage to avoid multi-scale duplicate penalties.

    Target text can straddle a block boundary. Credit adjacent selected spans
    that jointly contain a fragment so fixed-size sweeps do not report false
    misses when evidence is available across selected neighboring blocks.
    """

    selected = tuple(int(block_id) for block_id in selected_ids)
    selected_set = set(selected)
    expected_set: set[int] = set()
    selected_expected_set: set[int] = set()
    hit_fragments: list[str] = []
    for fragment in target_fragments:
        expected_for_fragment = _fragment_block_ids(
            fragment,
            block_ids=tuple(sorted(block_text_by_id)),
            block_text_by_id=block_text_by_id,
            block_by_id=block_by_id,
        )
        selected_for_fragment = _fragment_block_ids(
            fragment,
            block_ids=selected,
            block_text_by_id=block_text_by_id,
            block_by_id=block_by_id,
        )
        expected_set.update(expected_for_fragment)
        selected_expected_set.update(selected_for_fragment)
        if selected_for_fragment:
            hit_fragments.append(fragment)

    expected = tuple(
        block_id
        for block_id in sorted(
            expected_set,
            key=lambda item: _block_sort_key(item, block_by_id=block_by_id),
        )
    )
    selected_expected = tuple(block_id for block_id in selected if block_id in set(expected))
    if selected_expected_set:
        selected_expected = tuple(
            block_id for block_id in selected if block_id in selected_expected_set
        )
    missed = tuple(block_id for block_id in expected if block_id not in selected_set)
    extra = tuple(block_id for block_id in selected if block_id not in set(expected))
    recall = None if not target_fragments else len(tuple(hit_fragments)) / len(target_fragments)
    precision = None if not selected else len(selected_expected) / len(selected)
    return RetrievalQuality(
        expected_block_ids=expected,
        selected_expected_block_ids=selected_expected,
        missed_expected_block_ids=missed,
        extra_selected_block_ids=extra,
        target_recall=recall,
        selected_precision=precision,
        target_hit=bool(tuple(hit_fragments)),
    )


def _fragment_block_ids(
    fragment: str,
    *,
    block_ids: Sequence[int],
    block_text_by_id: dict[int, str],
    block_by_id: dict[int, Any] | None,
) -> tuple[int, ...]:
    """Return block ids that individually or jointly cover ``fragment``."""

    direct = tuple(
        block_id
        for block_id in block_ids
        if fragment in block_text_by_id.get(block_id, "")
    )
    if direct or block_by_id is None:
        return direct

    hit_ids: set[int] = set()
    for chain in _adjacent_block_chains(block_ids, block_by_id=block_by_id):
        hit_ids.update(
            _fragment_block_ids_in_chain(
                fragment,
                chain=chain,
                block_text_by_id=block_text_by_id,
            )
        )
    return tuple(
        block_id
        for block_id in sorted(
            hit_ids,
            key=lambda item: _block_sort_key(item, block_by_id=block_by_id),
        )
    )


def _adjacent_block_chains(
    block_ids: Sequence[int],
    *,
    block_by_id: dict[int, Any],
) -> tuple[tuple[Any, ...], ...]:
    records = [
        block_by_id[block_id]
        for block_id in block_ids
        if block_id in block_by_id
    ]
    records.sort(key=lambda block: (block.token_start, block.token_end, block.block_id))
    chains: list[tuple[Any, ...]] = []
    for record in records:
        chain = [record]
        current_end = record.token_end
        while True:
            next_candidates = [
                candidate
                for candidate in records
                if candidate.token_start == current_end
            ]
            if not next_candidates:
                break
            next_record = min(
                next_candidates,
                key=lambda block: (block.token_end, block.block_id),
            )
            chain.append(next_record)
            current_end = next_record.token_end
        if len(chain) > 1:
            chains.append(tuple(chain))
    return tuple(chains)


def _fragment_block_ids_in_chain(
    fragment: str,
    *,
    chain: Sequence[Any],
    block_text_by_id: dict[int, str],
) -> tuple[int, ...]:
    parts: list[tuple[int, int, int]] = []
    texts: list[str] = []
    cursor = 0
    for block in chain:
        text = block_text_by_id.get(block.block_id, "")
        start = cursor
        cursor += len(text)
        parts.append((block.block_id, start, cursor))
        texts.append(text)
    combined = "".join(texts)
    hit_start = combined.find(fragment)
    if hit_start < 0:
        return ()
    hit_end = hit_start + len(fragment)
    return tuple(
        block_id
        for block_id, start, end in parts
        if max(start, hit_start) < min(end, hit_end)
    )


def _block_sort_key(block_id: int, *, block_by_id: dict[int, Any] | None) -> tuple[int, int, int]:
    if block_by_id is None or block_id not in block_by_id:
        return (0, 0, block_id)
    block = block_by_id[block_id]
    return (int(block.token_start), int(block.token_end), int(block.block_id))


def _build_aggregate_summaries(
    rows: Sequence[DynamicBlockRunRow],
) -> tuple[DynamicBlockAggregateSummary, ...]:
    grouped: dict[tuple[str, str, float], list[DynamicBlockRunRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.block_mode, row.suppression_mode, row.suppression_threshold)].append(row)

    summaries: list[DynamicBlockAggregateSummary] = []
    for (block_mode, suppression_mode, suppression_threshold), group in sorted(grouped.items()):
        summaries.append(
            DynamicBlockAggregateSummary(
                block_mode=block_mode,
                suppression_mode=suppression_mode,
                suppression_threshold=suppression_threshold,
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
                mean_candidate_block_count=_mean(
                    row.candidate_block_count for row in group
                ),
                mean_candidate_count_after_suppression=_mean(
                    row.candidate_count_after_suppression for row in group
                ),
                mean_selector_latency_sec=_mean(
                    row.selector_latency_sec for row in group
                ),
                mean_coarse_candidate_count=_mean(
                    row.coarse_candidate_count for row in group
                ),
                mean_fine_candidate_count_after_drilldown=_mean(
                    row.fine_candidate_count_after_drilldown for row in group
                ),
                mean_retained_parent_count=_mean(
                    row.retained_parent_count for row in group
                ),
            )
        )

    best_fixed = _best_fixed_summary(summaries)
    if best_fixed is None:
        return tuple(summaries)
    return tuple(
        replace(
            summary,
            recall_delta_vs_best_fixed=_optional_delta(
                summary.mean_recall,
                best_fixed.mean_recall,
            ),
            precision_delta_vs_best_fixed=_optional_delta(
                summary.mean_precision,
                best_fixed.mean_precision,
            ),
            selected_to_k_delta_vs_best_fixed=(
                summary.mean_selected_to_semantic_k_ratio
                - best_fixed.mean_selected_to_semantic_k_ratio
            ),
            latency_delta_vs_best_fixed_sec=(
                summary.mean_selector_latency_sec
                - best_fixed.mean_selector_latency_sec
            ),
        )
        for summary in summaries
    )


def _build_prompt_breakdowns(
    rows: Sequence[DynamicBlockRunRow],
) -> tuple[DynamicBlockPromptBreakdown, ...]:
    grouped: dict[tuple[str, str, str, str, float], list[DynamicBlockRunRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row.prompt_name,
                row.block_mode,
                row.qk_aggregation_strategy,
                row.suppression_mode,
                row.suppression_threshold,
            )
        ].append(row)

    breakdowns: list[DynamicBlockPromptBreakdown] = []
    for (
        prompt_name,
        block_mode,
        strategy,
        suppression_mode,
        suppression_threshold,
    ), group in sorted(grouped.items()):
        breakdowns.append(
            DynamicBlockPromptBreakdown(
                prompt_name=prompt_name,
                block_mode=block_mode,
                qk_aggregation_strategy=strategy,
                suppression_mode=suppression_mode,
                suppression_threshold=suppression_threshold,
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
                mean_candidate_block_count=_mean(
                    row.candidate_block_count for row in group
                ),
                mean_candidate_count_after_suppression=_mean(
                    row.candidate_count_after_suppression for row in group
                ),
                mean_selector_latency_sec=_mean(
                    row.selector_latency_sec for row in group
                ),
                mean_coarse_candidate_count=_mean(
                    row.coarse_candidate_count for row in group
                ),
                mean_fine_candidate_count_after_drilldown=_mean(
                    row.fine_candidate_count_after_drilldown for row in group
                ),
                mean_retained_parent_count=_mean(
                    row.retained_parent_count for row in group
                ),
            )
        )
    return tuple(breakdowns)


def _best_fixed_summary(
    summaries: Sequence[DynamicBlockAggregateSummary],
) -> DynamicBlockAggregateSummary | None:
    fixed = [summary for summary in summaries if summary.block_mode.startswith("fixed_")]
    if not fixed:
        return None
    return min(fixed, key=_summary_rank_key)


def _summary_rank_key(
    summary: DynamicBlockAggregateSummary,
) -> tuple[float, float, float, float, str]:
    recall = -1.0 if summary.mean_recall is None else summary.mean_recall
    precision = -1.0 if summary.mean_precision is None else summary.mean_precision
    return (
        -recall,
        -precision,
        summary.mean_selected_to_semantic_k_ratio,
        summary.mean_selector_latency_sec,
        summary.block_mode,
    )


def _strategy_for_prompt(
    prompt_name: str,
    *,
    default_strategy: QKAggregationStrategy,
    needle_strategy: QKAggregationStrategy | None,
) -> QKAggregationStrategy:
    if prompt_name == "needle" and needle_strategy is not None:
        return needle_strategy
    return default_strategy


def _query_prompt_override(
    prompt: str,
    *,
    representation_source: str,
) -> str | None:
    """Extract the LongBench-style input tail for query-only modes."""

    if not is_query_only_source(representation_source):
        return None
    marker = "\nINPUT:\n"
    query = (
        prompt.rsplit(marker, maxsplit=1)[-1].strip()
        if marker in prompt
        else prompt.strip()
    )
    return query or prompt


def _query_only_context_candidates(
    runtime: Any,
    *,
    prompt: str,
    block_mode: BlockModeName,
    config: RealBlockSelectorConfig,
    query_prompt: str | None,
) -> tuple[BlockCandidate, ...]:
    """Generate context-only candidates for query-only LongBench-style prompts."""

    if query_prompt is None or "\nINPUT:\n" not in prompt:
        return config.block_candidates
    context_prompt = prompt.rsplit("\nINPUT:\n", maxsplit=1)[0].strip()
    if not context_prompt:
        return config.block_candidates
    context_tokens = runtime.tokenize(context_prompt).token_count
    candidate_mode = block_mode
    if is_coarse_to_fine_mode(block_mode):
        coarse_size, _fine_size = coarse_to_fine_spec(block_mode)
        candidate_mode = cast(BlockModeName, f"fixed_{coarse_size}")
    elif is_mixed_global_refine_mode(block_mode):
        global_size, _fine_size = mixed_global_refine_spec(block_mode)
        candidate_mode = cast(BlockModeName, f"fixed_{global_size}")
    return generate_block_candidates(
        token_count=context_tokens,
        mode=candidate_mode,
        default_block_size=config.block_size,
        overlap_stride=config.overlap_stride,
    )


def _format_summary(summary: DynamicBlockAggregateSummary) -> str:
    return (
        f"{summary.block_mode} | runs={summary.run_count} "
        f"suppression={summary.suppression_mode}@{summary.suppression_threshold:.2f} "
        f"mean_recall={_fmt_optional(summary.mean_recall)} "
        f"mean_precision={_fmt_optional(summary.mean_precision)} "
        f"mean_selected={summary.mean_selected_count:.3f} "
        f"mean_selected/K={summary.mean_selected_to_semantic_k_ratio:.3f} "
        f"mean_candidates={summary.mean_candidate_block_count:.3f} "
        f"mean_after_suppression={summary.mean_candidate_count_after_suppression:.3f} "
        f"mean_coarse={summary.mean_coarse_candidate_count:.3f} "
        f"mean_fine={summary.mean_fine_candidate_count_after_drilldown:.3f} "
        f"mean_retained_parents={summary.mean_retained_parent_count:.3f} "
        f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
        f"d_recall_vs_best_fixed={_fmt_optional(summary.recall_delta_vs_best_fixed)} "
        f"d_precision_vs_best_fixed={_fmt_optional(summary.precision_delta_vs_best_fixed)} "
        f"d_selected/K_vs_best_fixed={_fmt_optional(summary.selected_to_k_delta_vs_best_fixed)} "
        f"d_selector_vs_best_fixed={_fmt_optional(summary.latency_delta_vs_best_fixed_sec)}s"
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


def _fmt_int_optional(value: int | None) -> str:
    return "n/a" if value is None else str(value)
