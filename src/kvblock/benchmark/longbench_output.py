"""Output-based LongBench QA benchmark over KVBlock-selected context."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import gc
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Literal, Sequence

import torch

from kvblock.benchmark.answer_metrics import normalize_answer, score_qa_answer
from kvblock.benchmark.longbench_adapter import (
    DEFAULT_LONGBENCH_DATASETS,
    DEFAULT_ORACLE_TOP_K,
    DatasetLoader,
    LengthBucket,
    OracleMode,
    load_longbench_records,
    materialize_longbench_prompt_cases,
    parse_length_bucket,
    parse_oracle_mode,
    parse_oracle_top_k,
    run_longbench_selector_benchmark,
)
from kvblock.benchmark.dynamic_block_benchmark import (
    RefineScoreMode,
    RerankMode,
    StageCPolicyMode,
)
from kvblock.kv.block_modes import BlockModeName
from kvblock.kv.qk_aggregation import QKAggregationStrategy
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig
from kvblock.selectors.research_adapter import (
    selected_kv_plan_from_longbench_selector_row,
)

ContextReconstructionMode = Literal["selected_spans", "passage_window"]
ContextPolicy = Literal["selected", "full_context", "answer_oracle"]
OutputPolicy = Literal[
    "manual",
    "length_aware_static",
    "quality_guarded_static",
    "efficiency_guarded_static",
]

_CONTEXT_MARKER = "CONTEXT:\n"
_INPUT_MARKER = "\nINPUT:\n"
_PASSAGE_MARKER_RE = re.compile(r"Passage\s+\d+\s*:")
_ANSWER_ORACLE_UNHELPFUL_ANSWERS = frozenset(("yes", "no", "noanswer"))


@dataclass(frozen=True, slots=True)
class ReconstructedContext:
    """Context text plus token count used for generation."""

    text: str
    token_count: int
    source_spans: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """One context chunk with character bounds relative to the context section."""

    text: str
    text_start: int
    text_end: int


@dataclass(frozen=True, slots=True)
class PassageSpan:
    """One parsed passage section with full-prompt token bounds."""

    text_start: int
    text_end: int
    token_start: int
    token_end: int


@dataclass(frozen=True, slots=True)
class OutputSelection:
    """Selected block ids/spans after optional output-only budget filtering."""

    block_ids: tuple[int, ...]
    spans: tuple[str, ...]
    blocks: tuple[dict[str, Any], ...]
    dropped_count: int


@dataclass(frozen=True, slots=True)
class ResolvedOutputPolicy:
    """Effective output benchmark policy settings."""

    name: OutputPolicy
    max_selected_blocks: int | None
    context_reconstruction: ContextReconstructionMode
    passage_window_tokens: int
    block_modes: tuple[BlockModeName, ...] | None = None
    representation_source: RepresentationSource | None = None
    qk_aggregation_strategy: QKAggregationStrategy | None = None
    mixed_refine_parent_k: int | None = None
    mixed_fallback_margin: float | None = None
    mixed_child_window_radius: int | None = None
    rerank_mode: RerankMode | None = None
    refine_top_n_tokens: int | None = None
    refine_score_mode: RefineScoreMode | None = None
    stage_c_policy: StageCPolicyMode | None = None
    exclude_scaffold_blocks: bool | None = None
    halo_radius: int | None = None
    evidence_window_radius: int | None = None
    selector_shortlist_m: int | None = None
    selector_semantic_k: int | None = None
    selector_confidence_margin: float | None = None


@dataclass(frozen=True, slots=True)
class LongBenchOutputRunRow:
    """One generated-answer evaluation row."""

    dataset: str
    sample_id: str
    model: str
    block_mode: str
    prompt_tokens: int
    longbench_length: int | None
    selected_token_count: int
    selected_token_fraction: float
    reconstructed_context_token_count: int
    reconstructed_context_token_fraction: float
    context_reconstruction: str
    selected_block_count: int
    selection_filter_dropped_count: int
    mixed_fallback_used: bool
    selector_latency_sec: float
    selector_total_latency_sec: float
    generation_latency_sec: float
    total_latency_sec: float
    gold_answers: tuple[str, ...]
    prediction: str
    answer_em: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    selector_recall: float | None
    selector_precision: float | None
    evidence_window_recall: float | None
    evidence_window_precision: float | None
    expected_parent_recall: float | None
    selected_ids: tuple[int, ...]
    selected_spans: tuple[str, ...]
    selected_block_fraction: float = 0.0
    exact_recall: float | None = None
    answer_quality_score: float = 0.0
    dense_sparse_quality_delta: float | None = None
    selected_kv_plan: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchOutputSummary:
    """Aggregate answer-quality and bandwidth proxy metrics."""

    dataset: str
    row_count: int
    mean_answer_em: float
    mean_answer_f1: float
    mean_answer_precision: float
    mean_answer_recall: float
    mean_selected_block_count: float
    mean_selected_block_fraction: float
    mean_selection_filter_dropped_count: float
    mean_selected_token_fraction: float
    mean_selected_tokens: float
    mean_reconstructed_context_token_fraction: float
    mean_reconstructed_context_tokens: float
    mean_selector_latency_sec: float
    mean_generation_latency_sec: float
    mean_evidence_recall: float | None
    mean_evidence_window_recall: float | None
    mean_exact_recall: float | None
    mean_answer_quality_score: float
    mean_dense_sparse_quality_delta: float | None
    mixed_fallback_count: int
    mixed_fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchOutputBenchmarkResult:
    """Full output-based benchmark payload."""

    name: str
    model_names: tuple[str, ...]
    datasets: tuple[str, ...]
    config: dict[str, Any]
    rows: tuple[LongBenchOutputRunRow, ...]
    dataset_summaries: tuple[LongBenchOutputSummary, ...]
    overall_summary: LongBenchOutputSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload."""

        return {
            "name": self.name,
            "model_names": list(self.model_names),
            "datasets": list(self.datasets),
            "config": self.config,
            "rows": [row.to_dict() for row in self.rows],
            "dataset_summaries": [
                summary.to_dict() for summary in self.dataset_summaries
            ],
            "overall_summary": self.overall_summary.to_dict(),
        }


def run_longbench_output_benchmark(
    *,
    name: str,
    model_names: Sequence[str],
    dataset_names: Sequence[str] = DEFAULT_LONGBENCH_DATASETS,
    split: str = "test",
    dataset_repo: str = "THUDM/LongBench",
    limit_per_dataset: int | None = 1,
    length_bucket: LengthBucket | str = "all",
    prompt_cache_dir: str | Path = "results/longbench_output/prompts",
    block_modes: Sequence[BlockModeName] = ("fixed_40",),
    representation_source: RepresentationSource = "query_mean_last_layer",
    qk_aggregation_strategy: QKAggregationStrategy = "block_max",
    needle_qk_aggregation_strategy: QKAggregationStrategy | None = None,
    coarse_top_k: int = 2,
    mixed_refine_parent_k: int = 8,
    mixed_global_anchor_k: int = 8,
    mixed_fallback_margin: float = 0.05,
    mixed_max_children_per_parent: int | None = None,
    mixed_child_window_radius: int = 0,
    rerank_mode: RerankMode = "none",
    rerank_weight: float = 0.3,
    refine_top_n_tokens: int = 4,
    refine_score_mode: RefineScoreMode = "raw_topn_mean",
    stage_c_policy: StageCPolicyMode = "refined_only",
    exclude_scaffold_blocks: bool = False,
    neighbor_expansion: int = 0,
    halo_radius: int = 0,
    max_selected_blocks: int | None = None,
    evidence_window_radius: int = 0,
    oracle_mode: OracleMode = "none",
    oracle_top_k: Sequence[int] = DEFAULT_ORACLE_TOP_K,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    context_policy: ContextPolicy = "selected",
    output_policy: OutputPolicy = "manual",
    context_reconstruction: ContextReconstructionMode = "selected_spans",
    passage_window_tokens: int | None = None,
    passage_header_tokens: int = 24,
    selection_min_blocks: int = 0,
    selection_score_ratio: float | None = None,
    selection_max_total_blocks: int | None = None,
    selection_max_children_per_parent: int | None = None,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> LongBenchOutputBenchmarkResult:
    """Run generation on contexts shortened to KVBlock-selected spans."""

    if not name.strip():
        raise ValueError("name must be non-empty")
    if not model_names:
        raise ValueError("model_names must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")
    if context_policy not in ("selected", "full_context", "answer_oracle"):
        raise ValueError("unsupported context_policy")
    if context_reconstruction not in ("selected_spans", "passage_window"):
        raise ValueError("unsupported context_reconstruction")
    if passage_window_tokens is not None and passage_window_tokens <= 0:
        raise ValueError("passage_window_tokens must be > 0")
    if passage_header_tokens < 0:
        raise ValueError("passage_header_tokens must be >= 0")
    if selection_min_blocks < 0:
        raise ValueError("selection_min_blocks must be >= 0")
    if selection_score_ratio is not None and selection_score_ratio < 0.0:
        raise ValueError("selection_score_ratio must be >= 0 when provided")
    if selection_max_total_blocks is not None and selection_max_total_blocks <= 0:
        raise ValueError("selection_max_total_blocks must be > 0 when provided")
    if (
        selection_max_children_per_parent is not None
        and selection_max_children_per_parent <= 0
    ):
        raise ValueError(
            "selection_max_children_per_parent must be > 0 when provided"
        )

    bucket = (
        parse_length_bucket(length_bucket)
        if isinstance(length_bucket, str)
        else length_bucket
    )
    resolved_output_policy = resolve_output_policy_settings(
        output_policy=output_policy,
        dataset_names=dataset_names,
        length_bucket=bucket,
        max_selected_blocks=max_selected_blocks,
        context_reconstruction=context_reconstruction,
        passage_window_tokens=passage_window_tokens,
    )
    max_selected_blocks = resolved_output_policy.max_selected_blocks
    context_reconstruction = resolved_output_policy.context_reconstruction
    passage_window_tokens = resolved_output_policy.passage_window_tokens
    if resolved_output_policy.block_modes is not None:
        block_modes = resolved_output_policy.block_modes
    if resolved_output_policy.representation_source is not None:
        representation_source = resolved_output_policy.representation_source
    if resolved_output_policy.qk_aggregation_strategy is not None:
        qk_aggregation_strategy = resolved_output_policy.qk_aggregation_strategy
    if resolved_output_policy.mixed_refine_parent_k is not None:
        mixed_refine_parent_k = resolved_output_policy.mixed_refine_parent_k
    if resolved_output_policy.mixed_fallback_margin is not None:
        mixed_fallback_margin = resolved_output_policy.mixed_fallback_margin
    if resolved_output_policy.mixed_child_window_radius is not None:
        mixed_child_window_radius = resolved_output_policy.mixed_child_window_radius
    if resolved_output_policy.rerank_mode is not None:
        rerank_mode = resolved_output_policy.rerank_mode
    if resolved_output_policy.refine_top_n_tokens is not None:
        refine_top_n_tokens = resolved_output_policy.refine_top_n_tokens
    if resolved_output_policy.refine_score_mode is not None:
        refine_score_mode = resolved_output_policy.refine_score_mode
    if resolved_output_policy.stage_c_policy is not None:
        stage_c_policy = resolved_output_policy.stage_c_policy
    if resolved_output_policy.exclude_scaffold_blocks is not None:
        exclude_scaffold_blocks = resolved_output_policy.exclude_scaffold_blocks
    if resolved_output_policy.halo_radius is not None:
        halo_radius = resolved_output_policy.halo_radius
    if resolved_output_policy.evidence_window_radius is not None:
        evidence_window_radius = resolved_output_policy.evidence_window_radius
    selector_config = apply_output_policy_selector_overrides(
        selector_config,
        resolved_output_policy,
    )

    resolved_oracle_mode = parse_oracle_mode(oracle_mode)
    resolved_oracle_top_k = parse_oracle_top_k(oracle_top_k)
    load_kwargs = dict(load_config_kwargs or {})
    if context_policy in ("full_context", "answer_oracle"):
        return _run_direct_context_output_benchmark(
            name=name,
            model_names=model_names,
            dataset_names=dataset_names,
            split=split,
            dataset_repo=dataset_repo,
            limit_per_dataset=limit_per_dataset,
            length_bucket=bucket,
            prompt_cache_dir=prompt_cache_dir,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            context_policy=context_policy,
            output_policy=resolved_output_policy.name,
            context_reconstruction=context_reconstruction,
            passage_window_tokens=passage_window_tokens,
            passage_header_tokens=passage_header_tokens,
            load_kwargs=load_kwargs,
            dataset_loader=dataset_loader,
        )

    selector_result = run_longbench_selector_benchmark(
        model_names=model_names,
        dataset_names=dataset_names,
        split=split,
        dataset_repo=dataset_repo,
        limit_per_dataset=limit_per_dataset,
        length_bucket=bucket,
        prompt_cache_dir=prompt_cache_dir,
        block_modes=block_modes,
        representation_source=representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy,
        needle_qk_aggregation_strategy=needle_qk_aggregation_strategy,
        coarse_top_k=coarse_top_k,
        mixed_refine_parent_k=mixed_refine_parent_k,
        mixed_global_anchor_k=mixed_global_anchor_k,
        mixed_fallback_margin=mixed_fallback_margin,
        mixed_max_children_per_parent=mixed_max_children_per_parent,
        mixed_child_window_radius=mixed_child_window_radius,
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        stage_c_policy=stage_c_policy,
        exclude_scaffold_blocks=exclude_scaffold_blocks,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k=resolved_oracle_top_k,
        load_config_kwargs=load_kwargs,
        selector_config=selector_config,
        dataset_loader=dataset_loader,
    )
    _release_torch_cache()

    answers_by_prompt = {
        sample.prompt_name: sample.answer_labels
        for sample in selector_result.samples
    }
    rows: list[LongBenchOutputRunRow] = []
    for model_name in model_names:
        runtime = LocalHfRuntime(
            RuntimeLoadConfig(model_name=model_name, **load_kwargs),
            capture_config=HiddenStateCaptureConfig(
                representation_source=representation_source,
            ),
        )
        runtime.load_model()
        for selector_row in selector_result.rows:
            if selector_row.model_name != model_name:
                continue
            prompt_text = Path(selector_row.prompt_file).read_text(encoding="utf-8")
            output_selection = filter_output_selection(
                selected_block_ids=selector_row.selected_block_ids,
                selected_spans=selector_row.selected_spans,
                selected_blocks=selector_row.selected_blocks,
                selection_min_blocks=selection_min_blocks,
                selection_score_ratio=selection_score_ratio,
                selection_max_total_blocks=selection_max_total_blocks,
                selection_max_children_per_parent=selection_max_children_per_parent,
            )
            reconstructed_context = reconstruct_selected_context(
                runtime,
                prompt_text=prompt_text,
                selected_spans=output_selection.spans,
                mode=context_reconstruction,
                passage_window_tokens=passage_window_tokens,
                passage_header_tokens=passage_header_tokens,
            )
            question = extract_longbench_question(prompt_text)
            generation_prompt = format_selected_context_prompt(
                question=question,
                selected_context=reconstructed_context.text,
            )
            prediction, generation_latency_sec = generate_answer(
                runtime,
                generation_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            gold_answers = answers_by_prompt.get(selector_row.prompt_name, ())
            answer_scores = score_qa_answer(prediction, gold_answers)
            selected_token_count = _selected_token_count(output_selection.spans)
            selected_token_fraction = (
                0.0
                if selector_row.tokens <= 0
                else selected_token_count / selector_row.tokens
            )
            selector_recall = _filtered_recall(
                selector_row.target_recall,
                expected_ids=selector_row.expected_block_ids,
                selected_ids=output_selection.block_ids,
            )
            selector_precision = _filtered_precision(
                selector_row.selected_precision,
                expected_ids=selector_row.expected_block_ids,
                selected_ids=output_selection.block_ids,
            )
            evidence_window_recall = _filtered_window_recall(
                selector_row.evidence_window_recall,
                expected_ids=selector_row.expected_block_ids,
                selected_ids=output_selection.block_ids,
                radius=selector_row.evidence_window_radius,
            )
            evidence_window_precision = _filtered_window_precision(
                selector_row.evidence_window_precision,
                expected_ids=selector_row.expected_block_ids,
                selected_ids=output_selection.block_ids,
                radius=selector_row.evidence_window_radius,
            )
            selected_plan = selected_kv_plan_from_longbench_selector_row(
                selector_row,
                selected_block_ids=output_selection.block_ids,
                selected_spans=output_selection.spans,
                selected_blocks=output_selection.blocks,
                policy_name=resolved_output_policy.name,
            )
            rows.append(
                LongBenchOutputRunRow(
                    dataset=selector_row.dataset_name,
                    sample_id=selector_row.sample_id,
                    model=selector_row.model_name,
                    block_mode=selector_row.block_mode,
                    prompt_tokens=selector_row.tokens,
                    longbench_length=selector_row.longbench_length,
                    selected_token_count=selected_token_count,
                    selected_token_fraction=selected_token_fraction,
                    reconstructed_context_token_count=(
                        reconstructed_context.token_count
                    ),
                    reconstructed_context_token_fraction=(
                        0.0
                        if selector_row.tokens <= 0
                        else reconstructed_context.token_count / selector_row.tokens
                    ),
                    context_reconstruction=context_reconstruction,
                    selected_block_count=len(output_selection.block_ids),
                    selection_filter_dropped_count=output_selection.dropped_count,
                    mixed_fallback_used=selector_row.mixed_fallback_used,
                    selector_latency_sec=selector_row.selector_latency_sec,
                    selector_total_latency_sec=selector_row.total_latency_sec,
                    generation_latency_sec=generation_latency_sec,
                    total_latency_sec=(
                        selector_row.total_latency_sec + generation_latency_sec
                    ),
                    gold_answers=gold_answers,
                    prediction=prediction,
                    answer_em=answer_scores["em"],
                    answer_f1=answer_scores["f1"],
                    answer_precision=answer_scores["precision"],
                    answer_recall=answer_scores["recall"],
                    selector_recall=selector_recall,
                    selector_precision=selector_precision,
                    evidence_window_recall=evidence_window_recall,
                    evidence_window_precision=evidence_window_precision,
                    expected_parent_recall=(
                        selector_row.expected_parent_recall
                        if output_selection.dropped_count == 0
                        else None
                    ),
                    selected_ids=output_selection.block_ids,
                    selected_spans=output_selection.spans,
                    selected_block_fraction=selected_plan.selected_block_fraction,
                    exact_recall=selector_recall,
                    answer_quality_score=answer_scores["f1"],
                    dense_sparse_quality_delta=None,
                    selected_kv_plan=selected_plan.to_dict(),
                )
            )
    row_tuple = tuple(rows)
    if not row_tuple:
        raise ValueError("output benchmark produced no rows")
    dataset_summaries = build_output_summaries(row_tuple)
    config = _output_config(
        split=split,
        dataset_repo=dataset_repo,
        length_bucket=bucket.to_dict(),
        limit_per_dataset=limit_per_dataset,
        representation_source=representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy,
        block_modes=block_modes,
        coarse_top_k=coarse_top_k,
        mixed_refine_parent_k=mixed_refine_parent_k,
        mixed_global_anchor_k=mixed_global_anchor_k,
        mixed_fallback_margin=mixed_fallback_margin,
        mixed_max_children_per_parent=mixed_max_children_per_parent,
        mixed_child_window_radius=mixed_child_window_radius,
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        stage_c_policy=stage_c_policy,
        exclude_scaffold_blocks=exclude_scaffold_blocks,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k=resolved_oracle_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context_policy=context_policy,
        output_policy=resolved_output_policy.name,
        context_reconstruction=context_reconstruction,
        passage_window_tokens=passage_window_tokens,
        passage_header_tokens=passage_header_tokens,
        selection_min_blocks=selection_min_blocks,
        selection_score_ratio=selection_score_ratio,
        selection_max_total_blocks=selection_max_total_blocks,
        selection_max_children_per_parent=selection_max_children_per_parent,
        selector_config=selector_config,
        load_config_kwargs=load_kwargs,
    )
    return LongBenchOutputBenchmarkResult(
        name=name,
        model_names=tuple(model_names),
        datasets=tuple(dataset_names),
        config=config,
        rows=row_tuple,
        dataset_summaries=dataset_summaries,
        overall_summary=_summarize_output_rows("all", row_tuple),
    )


def _run_direct_context_output_benchmark(
    *,
    name: str,
    model_names: Sequence[str],
    dataset_names: Sequence[str],
    split: str,
    dataset_repo: str,
    limit_per_dataset: int | None,
    length_bucket: LengthBucket,
    prompt_cache_dir: str | Path,
    max_new_tokens: int,
    temperature: float,
    context_policy: ContextPolicy,
    output_policy: OutputPolicy,
    context_reconstruction: ContextReconstructionMode,
    passage_window_tokens: int,
    passage_header_tokens: int,
    load_kwargs: dict[str, Any],
    dataset_loader: DatasetLoader | None,
) -> LongBenchOutputBenchmarkResult:
    records = load_longbench_records(
        dataset_names=dataset_names,
        split=split,
        dataset_repo=dataset_repo,
        limit_per_dataset=limit_per_dataset,
        length_bucket=length_bucket,
        dataset_loader=dataset_loader,
    )
    if not records:
        raise ValueError("LongBench selection produced no records")
    _, samples = materialize_longbench_prompt_cases(
        records,
        prompt_dir=prompt_cache_dir,
    )

    rows: list[LongBenchOutputRunRow] = []
    for model_name in model_names:
        runtime = LocalHfRuntime(
            RuntimeLoadConfig(model_name=model_name, **load_kwargs),
            capture_config=HiddenStateCaptureConfig(),
        )
        runtime.load_model()
        for sample in samples:
            prompt_text = Path(sample.prompt_file).read_text(encoding="utf-8")
            prompt_tokens = _token_count(runtime, prompt_text)
            reconstructed_context = direct_context_from_prompt(
                runtime,
                prompt_text=prompt_text,
                answers=sample.answer_labels,
                context_policy=context_policy,
            )
            question = extract_longbench_question(prompt_text)
            generation_prompt = format_selected_context_prompt(
                question=question,
                selected_context=reconstructed_context.text,
            )
            prediction, generation_latency_sec = generate_answer(
                runtime,
                generation_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            answer_scores = score_qa_answer(prediction, sample.answer_labels)
            selected_fraction = (
                0.0
                if prompt_tokens <= 0
                else reconstructed_context.token_count / prompt_tokens
            )
            rows.append(
                LongBenchOutputRunRow(
                    dataset=sample.dataset_name,
                    sample_id=sample.sample_id,
                    model=model_name,
                    block_mode=context_policy,
                    prompt_tokens=prompt_tokens,
                    longbench_length=sample.length,
                    selected_token_count=reconstructed_context.token_count,
                    selected_token_fraction=selected_fraction,
                    reconstructed_context_token_count=reconstructed_context.token_count,
                    reconstructed_context_token_fraction=selected_fraction,
                    context_reconstruction=context_policy,
                    selected_block_count=0,
                    selection_filter_dropped_count=0,
                    mixed_fallback_used=False,
                    selector_latency_sec=0.0,
                    selector_total_latency_sec=0.0,
                    generation_latency_sec=generation_latency_sec,
                    total_latency_sec=generation_latency_sec,
                    gold_answers=sample.answer_labels,
                    prediction=prediction,
                    answer_em=answer_scores["em"],
                    answer_f1=answer_scores["f1"],
                    answer_precision=answer_scores["precision"],
                    answer_recall=answer_scores["recall"],
                    selector_recall=None,
                    selector_precision=None,
                    evidence_window_recall=None,
                    evidence_window_precision=None,
                    expected_parent_recall=None,
                    selected_ids=(),
                    selected_spans=reconstructed_context.source_spans,
                    selected_block_fraction=0.0,
                    exact_recall=None,
                    answer_quality_score=answer_scores["f1"],
                    dense_sparse_quality_delta=None,
                )
            )

    row_tuple = tuple(rows)
    if not row_tuple:
        raise ValueError("output benchmark produced no rows")
    config = _output_config(
        split=split,
        dataset_repo=dataset_repo,
        length_bucket=length_bucket.to_dict(),
        limit_per_dataset=limit_per_dataset,
        representation_source=None,
        qk_aggregation_strategy=None,
        block_modes=(context_policy,),
        coarse_top_k=None,
        mixed_refine_parent_k=None,
        mixed_global_anchor_k=None,
        mixed_fallback_margin=None,
        mixed_max_children_per_parent=None,
        mixed_child_window_radius=None,
        rerank_mode=None,
        rerank_weight=None,
        refine_top_n_tokens=None,
        refine_score_mode=None,
        stage_c_policy=None,
        exclude_scaffold_blocks=None,
        neighbor_expansion=None,
        halo_radius=None,
        max_selected_blocks=None,
        evidence_window_radius=None,
        oracle_mode="none",
        oracle_top_k=(),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        context_policy=context_policy,
        output_policy=output_policy,
        context_reconstruction=context_reconstruction,
        passage_window_tokens=passage_window_tokens,
        passage_header_tokens=passage_header_tokens,
        selection_min_blocks=0,
        selection_score_ratio=None,
        selection_max_total_blocks=None,
        selection_max_children_per_parent=None,
        selector_config=None,
        load_config_kwargs=load_kwargs,
    )
    return LongBenchOutputBenchmarkResult(
        name=name,
        model_names=tuple(model_names),
        datasets=tuple(dataset_names),
        config=config,
        rows=row_tuple,
        dataset_summaries=build_output_summaries(row_tuple),
        overall_summary=_summarize_output_rows("all", row_tuple),
    )


def resolve_output_policy_settings(
    *,
    output_policy: OutputPolicy,
    dataset_names: Sequence[str],
    length_bucket: LengthBucket,
    max_selected_blocks: int | None,
    context_reconstruction: ContextReconstructionMode,
    passage_window_tokens: int | None,
) -> ResolvedOutputPolicy:
    """Resolve explicit output benchmark policy presets.

    The guarded presets are empirical V1 benchmark defaults. They intentionally
    encode measured quality/efficiency regimes so the report harness stays
    reproducible while selector work continues.
    """

    normalized_policy = output_policy.strip().lower()
    if normalized_policy == "manual":
        return ResolvedOutputPolicy(
            name="manual",
            max_selected_blocks=max_selected_blocks,
            context_reconstruction=context_reconstruction,
            passage_window_tokens=(
                120 if passage_window_tokens is None else passage_window_tokens
            ),
        )
    if normalized_policy == "length_aware_static":
        budget = _length_aware_static_budget(
            dataset_names,
            length_bucket,
            policy_name="length_aware_static",
        )
        return ResolvedOutputPolicy(
            name="length_aware_static",
            max_selected_blocks=budget,
            context_reconstruction="passage_window",
            passage_window_tokens=(
                64 if passage_window_tokens is None else passage_window_tokens
            ),
        )
    if normalized_policy in ("quality_guarded_static", "efficiency_guarded_static"):
        return _guarded_static_output_policy(
            output_policy=normalized_policy,
            dataset_names=dataset_names,
            length_bucket=length_bucket,
            passage_window_tokens=passage_window_tokens,
        )
    raise ValueError(
        "output_policy must be one of manual, length_aware_static, "
        "quality_guarded_static, efficiency_guarded_static"
    )


def _guarded_static_output_policy(
    *,
    output_policy: str,
    dataset_names: Sequence[str],
    length_bucket: LengthBucket,
    passage_window_tokens: int | None,
) -> ResolvedOutputPolicy:
    policy_name = (
        "quality_guarded_static"
        if output_policy == "quality_guarded_static"
        else "efficiency_guarded_static"
    )
    budget = _length_aware_static_budget(
        dataset_names,
        length_bucket,
        policy_name=policy_name,
    )
    mixed_fallback_margin = 0.05
    mixed_child_window_radius = 0
    if _is_hotpot_4k_8k(dataset_names, length_bucket):
        if policy_name == "quality_guarded_static":
            budget = 12
        else:
            budget = 24
            mixed_fallback_margin = 0.0
            mixed_child_window_radius = 2

    return ResolvedOutputPolicy(
        name=policy_name,
        max_selected_blocks=budget,
        context_reconstruction="passage_window",
        passage_window_tokens=(
            64 if passage_window_tokens is None else passage_window_tokens
        ),
        block_modes=("mixed_global_refine_40_16_stride_8",),
        representation_source="query_only_last_layer",
        qk_aggregation_strategy="block_max",
        mixed_refine_parent_k=4,
        mixed_fallback_margin=mixed_fallback_margin,
        mixed_child_window_radius=mixed_child_window_radius,
        rerank_mode="dense_qk_token_refine",
        refine_top_n_tokens=4,
        refine_score_mode="softmax_mass",
        stage_c_policy="semantic_refined_mix",
        exclude_scaffold_blocks=True,
        halo_radius=2,
        evidence_window_radius=2,
        selector_shortlist_m=32,
        selector_semantic_k=8,
        selector_confidence_margin=0.05,
    )


def _length_aware_static_budget(
    dataset_names: Sequence[str],
    length_bucket: LengthBucket,
    *,
    policy_name: str = "length_aware_static",
) -> int:
    datasets = tuple(
        sorted({dataset.strip().lower() for dataset in dataset_names if dataset.strip()})
    )
    if not datasets:
        raise ValueError("dataset_names must not be empty")

    if length_bucket.max_length is not None and length_bucket.max_length <= 4000:
        budgets = {dataset: 20 for dataset in datasets}
    elif length_bucket.min_length == 4000 and length_bucket.max_length == 8000:
        empirical_4k_8k_budgets = {
            "hotpotqa": 12,
            "musique": 8,
        }
        missing = tuple(
            dataset
            for dataset in datasets
            if dataset not in empirical_4k_8k_budgets
        )
        if missing:
            raise ValueError(
                f"{policy_name} has no 4k-8k budget for datasets: "
                + ",".join(missing)
            )
        budgets = {
            dataset: empirical_4k_8k_budgets[dataset]
            for dataset in datasets
        }
    else:
        raise ValueError(
            f"{policy_name} currently supports length buckets 0-4k and 4k-8k"
        )

    unique_budgets = set(budgets.values())
    if len(unique_budgets) != 1:
        details = ", ".join(
            f"{dataset}=m{budget}" for dataset, budget in sorted(budgets.items())
        )
        raise ValueError(
            f"{policy_name} resolves different budgets for this dataset mix "
            f"({details}); run those datasets separately"
        )
    return next(iter(unique_budgets))


def _is_hotpot_4k_8k(
    dataset_names: Sequence[str],
    length_bucket: LengthBucket,
) -> bool:
    datasets = {
        dataset.strip().lower() for dataset in dataset_names if dataset.strip()
    }
    return (
        datasets == {"hotpotqa"}
        and length_bucket.min_length == 4000
        and length_bucket.max_length == 8000
    )


def apply_output_policy_selector_overrides(
    selector_config: RealBlockSelectorConfig | None,
    resolved_output_policy: ResolvedOutputPolicy,
) -> RealBlockSelectorConfig | None:
    """Apply selector-config overrides required by output policy presets."""

    updates: dict[str, Any] = {}
    if resolved_output_policy.selector_shortlist_m is not None:
        updates["shortlist_m"] = resolved_output_policy.selector_shortlist_m
    if resolved_output_policy.selector_semantic_k is not None:
        updates["semantic_k"] = resolved_output_policy.selector_semantic_k
    if resolved_output_policy.selector_confidence_margin is not None:
        updates["confidence_margin"] = (
            resolved_output_policy.selector_confidence_margin
        )
    if resolved_output_policy.qk_aggregation_strategy is not None:
        updates["qk_aggregation_strategy"] = (
            resolved_output_policy.qk_aggregation_strategy
        )
    if resolved_output_policy.representation_source is not None:
        updates["representation_source"] = resolved_output_policy.representation_source
    if not updates:
        return selector_config
    base_config = selector_config or RealBlockSelectorConfig()
    return replace(base_config, **updates)


def filter_output_selection(
    *,
    selected_block_ids: Sequence[int],
    selected_spans: Sequence[str],
    selected_blocks: Sequence[dict[str, Any]],
    selection_min_blocks: int = 0,
    selection_score_ratio: float | None = None,
    selection_max_total_blocks: int | None = None,
    selection_max_children_per_parent: int | None = None,
) -> OutputSelection:
    """Apply optional output-only budget filters while preserving rank order."""

    block_records = {
        int(record["block_id"]): dict(record)
        for record in selected_blocks
        if "block_id" in record
    }
    pairs = tuple(zip(selected_block_ids, selected_spans, strict=False))
    triples = tuple(
        (
            int(block_id),
            span,
            block_records.get(int(block_id), {}),
        )
        for block_id, span in pairs
    )
    if (
        selection_score_ratio is None
        and selection_max_total_blocks is None
        and selection_max_children_per_parent is None
    ) or not triples:
        return OutputSelection(
            block_ids=tuple(block_id for block_id, _, _ in triples),
            spans=tuple(span for _, span, _ in triples),
            blocks=tuple(record for _, _, record in triples),
            dropped_count=0,
        )

    score_filtered = _score_ratio_filter(
        triples,
        selection_min_blocks=selection_min_blocks,
        selection_score_ratio=selection_score_ratio,
    )
    kept = _parent_cap_filter(
        score_filtered,
        selection_max_total_blocks=selection_max_total_blocks,
        selection_max_children_per_parent=selection_max_children_per_parent,
    )
    return OutputSelection(
        block_ids=tuple(block_id for block_id, _, _ in kept),
        spans=tuple(span for _, span, _ in kept),
        blocks=tuple(record for _, _, record in kept),
        dropped_count=len(triples) - len(kept),
    )


def _score_ratio_filter(
    triples: Sequence[tuple[int, str, dict[str, Any]]],
    *,
    selection_min_blocks: int,
    selection_score_ratio: float | None,
) -> tuple[tuple[int, str, dict[str, Any]], ...]:
    if selection_score_ratio is None:
        return tuple(triples)

    min_keep = min(max(0, selection_min_blocks), len(triples))
    top_score = _selection_score(triples[0][2])
    if top_score is None or top_score <= 0.0:
        return tuple(triples)

    threshold = top_score * selection_score_ratio
    keep_count = len(triples)
    for index, (_, _, record) in enumerate(triples):
        if index < min_keep:
            continue
        score = _selection_score(record)
        if score is None or score < threshold:
            keep_count = index
            break
    return tuple(triples[: max(min_keep, keep_count)])


def _parent_cap_filter(
    triples: Sequence[tuple[int, str, dict[str, Any]]],
    *,
    selection_max_total_blocks: int | None,
    selection_max_children_per_parent: int | None,
) -> tuple[tuple[int, str, dict[str, Any]], ...]:
    max_total = (
        len(triples)
        if selection_max_total_blocks is None
        else min(selection_max_total_blocks, len(triples))
    )
    if selection_max_children_per_parent is None and max_total == len(triples):
        return tuple(triples)

    kept: list[tuple[int, str, dict[str, Any]]] = []
    child_counts_by_parent: dict[str, int] = {}
    for block_id, span, record in triples:
        if len(kept) >= max_total:
            break
        if (
            selection_max_children_per_parent is not None
            and _selection_is_child(record)
        ):
            parent_key = _selection_parent_key(record, block_id)
            child_count = child_counts_by_parent.get(parent_key, 0)
            if child_count >= selection_max_children_per_parent:
                continue
            child_counts_by_parent[parent_key] = child_count + 1
        kept.append((block_id, span, record))
    return tuple(kept)


def _selection_score(record: dict[str, Any]) -> float | None:
    for key in ("final_score", "refined_score", "stage_b_score", "stage_a_score"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def _selection_is_child(record: dict[str, Any]) -> bool:
    return str(record.get("candidate_role", "")) == "child"


def _selection_parent_key(record: dict[str, Any], block_id: int) -> str:
    parent_candidate_id = record.get("parent_candidate_id")
    if parent_candidate_id not in (None, ""):
        return str(parent_candidate_id)
    parent_block_id = record.get("parent_block_id")
    if parent_block_id is not None:
        return f"parent_block:{int(parent_block_id)}"
    candidate_id = record.get("candidate_id")
    if candidate_id not in (None, ""):
        return str(candidate_id)
    return f"block:{int(block_id)}"


def reconstruct_selected_context(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    selected_spans: Sequence[str],
    mode: ContextReconstructionMode = "selected_spans",
    passage_window_tokens: int = 120,
    passage_header_tokens: int = 24,
) -> ReconstructedContext:
    """Build generation context from selected spans."""

    if mode == "selected_spans":
        return selected_context_from_spans(
            runtime,
            prompt_text=prompt_text,
            selected_spans=selected_spans,
        )
    if mode == "passage_window":
        return passage_window_context_from_spans(
            runtime,
            prompt_text=prompt_text,
            selected_spans=selected_spans,
            passage_window_tokens=passage_window_tokens,
            passage_header_tokens=passage_header_tokens,
        )
    raise ValueError(f"unsupported context reconstruction mode: {mode!r}")


def full_context_from_prompt(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
) -> ReconstructedContext:
    """Extract and count the original LongBench context section."""

    bounds = _context_bounds(prompt_text)
    context = extract_longbench_context(prompt_text)
    source_spans = ()
    if bounds is not None:
        source_spans = (
            _prompt_token_span_for_char_range(
                runtime,
                prompt_text,
                start=bounds[0],
                end=bounds[1],
            ),
        )
    return ReconstructedContext(
        text=context,
        token_count=_token_count(runtime, context),
        source_spans=source_spans,
    )


def direct_context_from_prompt(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    answers: Sequence[str],
    context_policy: ContextPolicy,
) -> ReconstructedContext:
    """Build a no-selector generation context for direct baseline policies."""

    if context_policy == "full_context":
        return full_context_from_prompt(runtime, prompt_text=prompt_text)
    if context_policy == "answer_oracle":
        return answer_oracle_context_from_prompt(
            runtime,
            prompt_text=prompt_text,
            answers=answers,
        )
    raise ValueError(f"unsupported direct context policy: {context_policy!r}")


def answer_oracle_context_from_prompt(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    answers: Sequence[str],
) -> ReconstructedContext:
    """Keep only answer-containing context chunks when literal gold text exists.

    This is an upper-bound diagnostic for compression format quality, not a fair
    deployable policy. Yes/no and non-literal answers fall back to full context.
    """

    context = extract_longbench_context(prompt_text)
    chunks = _answer_oracle_context_chunks(context, answers)
    if chunks:
        oracle_context = "\n\n".join(chunk.text for chunk in chunks)
        source_spans = _source_spans_for_context_chunks(
            runtime,
            prompt_text=prompt_text,
            chunks=chunks,
        )
    else:
        full_context = full_context_from_prompt(runtime, prompt_text=prompt_text)
        oracle_context = full_context.text
        source_spans = full_context.source_spans
    return ReconstructedContext(
        text=oracle_context,
        token_count=_token_count(runtime, oracle_context),
        source_spans=source_spans,
    )


def _answer_oracle_context_chunks(
    context: str,
    answers: Sequence[str],
) -> tuple[ContextChunk, ...]:
    normalized_answers = tuple(
        normalized
        for answer in answers
        if (normalized := normalize_answer(answer))
        and normalized not in _ANSWER_ORACLE_UNHELPFUL_ANSWERS
    )
    if not normalized_answers:
        return ()

    passage_chunks = _longbench_passage_chunks(context)
    if passage_chunks:
        hits = tuple(
            chunk
            for chunk in passage_chunks
            if _chunk_contains_normalized_answer(chunk.text, normalized_answers)
        )
        if hits:
            return _dedupe_context_chunks(hits)

    sentence_hits = tuple(
        chunk
        for chunk in _sentence_like_chunks(context)
        if _chunk_contains_normalized_answer(chunk.text, normalized_answers)
    )
    return _dedupe_context_chunks(sentence_hits)


def _longbench_passage_chunks(context: str) -> tuple[ContextChunk, ...]:
    matches = tuple(_PASSAGE_MARKER_RE.finditer(context))
    if not matches:
        return ()
    chunks: list[ContextChunk] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        chunk_start, chunk_end, chunk_text = _strip_chunk_bounds(context, start, end)
        if chunk_text:
            chunks.append(ContextChunk(chunk_text, chunk_start, chunk_end))
    return tuple(chunks)


def _sentence_like_chunks(context: str) -> tuple[ContextChunk, ...]:
    chunks: list[ContextChunk] = []
    previous_end = 0
    for match in re.finditer(r"(?<=[.!?])\s+|\n+", context):
        chunk_start, chunk_end, chunk_text = _strip_chunk_bounds(
            context,
            previous_end,
            match.start(),
        )
        if chunk_text:
            chunks.append(ContextChunk(chunk_text, chunk_start, chunk_end))
        previous_end = match.end()
    chunk_start, chunk_end, chunk_text = _strip_chunk_bounds(
        context,
        previous_end,
        len(context),
    )
    if chunk_text:
        chunks.append(ContextChunk(chunk_text, chunk_start, chunk_end))
    return tuple(chunks)


def _strip_chunk_bounds(
    text: str,
    start: int,
    end: int,
) -> tuple[int, int, str]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end, text[start:end]


def _chunk_contains_normalized_answer(
    chunk: str,
    normalized_answers: Sequence[str],
) -> bool:
    normalized_chunk = normalize_answer(chunk)
    return any(answer in normalized_chunk for answer in normalized_answers)


def _dedupe_context_chunks(chunks: Sequence[ContextChunk]) -> tuple[ContextChunk, ...]:
    seen: set[str] = set()
    deduped: list[ContextChunk] = []
    for chunk in chunks:
        normalized = " ".join(chunk.text.split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(chunk)
    return tuple(deduped)


def _source_spans_for_context_chunks(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    chunks: Sequence[ContextChunk],
) -> tuple[str, ...]:
    bounds = _context_bounds(prompt_text)
    if bounds is None:
        return ()
    context_start, _ = bounds
    return tuple(
        _prompt_token_span_for_char_range(
            runtime,
            prompt_text,
            start=context_start + chunk.text_start,
            end=context_start + chunk.text_end,
        )
        for chunk in chunks
    )


def _prompt_token_span_for_char_range(
    runtime: LocalHfRuntime,
    prompt_text: str,
    *,
    start: int,
    end: int,
) -> str:
    token_start = _token_count(runtime, prompt_text[:start])
    token_end = token_start + _token_count(runtime, prompt_text[start:end])
    return f"{token_start}:{token_end}"


def selected_context_from_spans(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    selected_spans: Sequence[str],
) -> ReconstructedContext:
    """Decode selected prompt token spans into a shortened context string."""

    tokenized = runtime.tokenize(prompt_text)
    chunks: list[str] = []
    intervals: list[tuple[int, int]] = []
    for span in selected_spans:
        parsed = _parse_span(span)
        if parsed is None:
            continue
        start, end = parsed
        bounded_start = max(0, min(start, len(tokenized.token_ids)))
        bounded_end = max(bounded_start, min(end, len(tokenized.token_ids)))
        token_ids = tuple(tokenized.token_ids[bounded_start:bounded_end])
        if not token_ids:
            continue
        intervals.append((bounded_start, bounded_end))
        text = runtime.decode_token_ids(token_ids).strip()
        if text:
            chunks.append(text)
    return ReconstructedContext(
        text="\n\n".join(chunks),
        token_count=_interval_token_count(_merge_intervals(intervals)),
    )


def passage_window_context_from_spans(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    selected_spans: Sequence[str],
    passage_window_tokens: int = 120,
    passage_header_tokens: int = 24,
) -> ReconstructedContext:
    """Group selected spans by passage and add passage headers/local windows."""

    tokenized = runtime.tokenize(prompt_text)
    passages = _passage_spans(runtime, prompt_text)
    if not passages:
        return selected_context_from_spans(
            runtime,
            prompt_text=prompt_text,
            selected_spans=selected_spans,
        )
    selected_intervals = tuple(
        (start, end)
        for span in selected_spans
        if (parsed := _parse_span(span)) is not None
        for start, end in (parsed,)
    )
    sections: list[str] = []
    all_intervals: list[tuple[int, int]] = []
    for passage in passages:
        overlapping = tuple(
            (max(start, passage.token_start), min(end, passage.token_end))
            for start, end in selected_intervals
            if max(start, passage.token_start) < min(end, passage.token_end)
        )
        if not overlapping:
            continue
        intervals: list[tuple[int, int]] = []
        if passage_header_tokens > 0:
            intervals.append(
                (
                    passage.token_start,
                    min(passage.token_end, passage.token_start + passage_header_tokens),
                )
            )
        for start, end in overlapping:
            intervals.append(
                _expand_interval_to_target(
                    start,
                    end,
                    target_tokens=passage_window_tokens,
                    min_start=passage.token_start,
                    max_end=passage.token_end,
                )
            )
        merged = _merge_intervals(intervals)
        all_intervals.extend(merged)
        chunks: list[str] = []
        for start, end in merged:
            token_ids = tuple(tokenized.token_ids[start:end])
            if not token_ids:
                continue
            text = runtime.decode_token_ids(token_ids).strip()
            if text:
                chunks.append(text)
        if chunks:
            sections.append(" ... ".join(chunks))
    if not sections:
        return selected_context_from_spans(
            runtime,
            prompt_text=prompt_text,
            selected_spans=selected_spans,
        )
    return ReconstructedContext(
        text="\n\n".join(sections),
        token_count=_interval_token_count(_merge_intervals(all_intervals)),
    )


def extract_longbench_question(prompt_text: str) -> str:
    """Extract the LongBench input/question from a materialized prompt."""

    marker = "\nINPUT:\n"
    if marker not in prompt_text:
        return prompt_text.strip()
    return prompt_text.rsplit(marker, maxsplit=1)[-1].strip()


def extract_longbench_context(prompt_text: str) -> str:
    """Extract the original LongBench context from a materialized prompt."""

    bounds = _context_bounds(prompt_text)
    if bounds is None:
        return prompt_text.strip()
    start, end = bounds
    return prompt_text[start:end].strip()


def format_selected_context_prompt(*, question: str, selected_context: str) -> str:
    """Format the first-pass short-answer generation prompt."""

    return "\n".join(
        (
            "Answer the question using only the provided context. Keep the answer short.",
            "",
            "Context:",
            selected_context.strip(),
            "",
            "Question:",
            question.strip(),
            "",
            "Answer:",
        )
    )


def generate_answer(
    runtime: LocalHfRuntime,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    """Generate a short answer from a local HF runtime."""

    runtime.load_model()
    tokenizer = runtime._require_tokenizer()
    model = runtime._require_model()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = runtime._input_device()
    encoded = {
        key: value.to(input_device)
        for key, value in encoded.items()
    }
    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    started_at = perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)
    generation_latency_sec = perf_counter() - started_at
    prompt_token_count = int(encoded["input_ids"].shape[-1])
    generated_ids = output_ids[0, prompt_token_count:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _clean_prediction(prediction), generation_latency_sec


def build_output_summaries(
    rows: Sequence[LongBenchOutputRunRow],
) -> tuple[LongBenchOutputSummary, ...]:
    """Build per-dataset output summaries."""

    grouped: dict[str, list[LongBenchOutputRunRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset, []).append(row)
    return tuple(
        _summarize_output_rows(dataset, tuple(group))
        for dataset, group in sorted(grouped.items())
    )


def format_longbench_output_report(result: LongBenchOutputBenchmarkResult) -> str:
    """Return a compact human-readable output benchmark report."""

    lines = [
        "LONGBENCH OUTPUT BENCHMARK",
        f"name={result.name}",
        f"models={','.join(result.model_names)} datasets={','.join(result.datasets)}",
        "",
        "OVERALL",
        _format_summary(result.overall_summary),
        "",
        "DATASET SUMMARIES",
    ]
    for summary in result.dataset_summaries:
        lines.append(_format_summary(summary))
    lines.extend(["", "RUNS"])
    for row in result.rows:
        prediction = _compact_text(row.prediction, limit=80)
        gold = _compact_text(" | ".join(row.gold_answers), limit=80)
        lines.append(
            f"{row.dataset}:{row.sample_id} model={row.model} mode={row.block_mode} "
            f"answer_f1={row.answer_f1:.3f} answer_em={row.answer_em:.3f} "
            f"quality={row.answer_quality_score:.3f} "
            f"recon={row.context_reconstruction} "
            f"blocks={row.selected_block_count} "
            f"dropped={row.selection_filter_dropped_count} "
            f"selected_token_frac={row.selected_token_fraction:.3f} "
            f"selected_block_frac={row.selected_block_fraction:.3f} "
            f"recon_frac={row.reconstructed_context_token_fraction:.3f} "
            f"selected_tokens={row.selected_token_count} "
            f"recon_tokens={row.reconstructed_context_token_count} "
            f"fallback={row.mixed_fallback_used} "
            f"evidence_recall={_fmt_optional(row.selector_recall)} "
            f"window_recall={_fmt_optional(row.evidence_window_recall)} "
            f"exact_recall={_fmt_optional(row.exact_recall)} "
            f"quality_delta={_fmt_optional(row.dense_sparse_quality_delta)} "
            f"selector={row.selector_latency_sec:.6f}s "
            f"generation={row.generation_latency_sec:.6f}s "
            f"prediction={prediction!r} gold={gold!r}"
        )
    return "\n".join(lines)


def write_longbench_output_benchmark_outputs(
    result: LongBenchOutputBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text output benchmark reports."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    text_output.write_text(format_longbench_output_report(result), encoding="utf-8")


def _summarize_output_rows(
    dataset: str,
    rows: Sequence[LongBenchOutputRunRow],
) -> LongBenchOutputSummary:
    if not rows:
        raise ValueError("rows must not be empty")
    fallback_count = sum(1 for row in rows if row.mixed_fallback_used)
    return LongBenchOutputSummary(
        dataset=dataset,
        row_count=len(rows),
        mean_answer_em=_mean(row.answer_em for row in rows),
        mean_answer_f1=_mean(row.answer_f1 for row in rows),
        mean_answer_precision=_mean(row.answer_precision for row in rows),
        mean_answer_recall=_mean(row.answer_recall for row in rows),
        mean_selected_block_count=_mean(row.selected_block_count for row in rows),
        mean_selected_block_fraction=_mean(
            row.selected_block_fraction for row in rows
        ),
        mean_selection_filter_dropped_count=_mean(
            row.selection_filter_dropped_count for row in rows
        ),
        mean_selected_token_fraction=_mean(
            row.selected_token_fraction for row in rows
        ),
        mean_selected_tokens=_mean(row.selected_token_count for row in rows),
        mean_reconstructed_context_token_fraction=_mean(
            row.reconstructed_context_token_fraction for row in rows
        ),
        mean_reconstructed_context_tokens=_mean(
            row.reconstructed_context_token_count for row in rows
        ),
        mean_selector_latency_sec=_mean(row.selector_latency_sec for row in rows),
        mean_generation_latency_sec=_mean(
            row.generation_latency_sec for row in rows
        ),
        mean_evidence_recall=_mean_optional(row.selector_recall for row in rows),
        mean_evidence_window_recall=_mean_optional(
            row.evidence_window_recall for row in rows
        ),
        mean_exact_recall=_mean_optional(row.exact_recall for row in rows),
        mean_answer_quality_score=_mean(row.answer_quality_score for row in rows),
        mean_dense_sparse_quality_delta=_mean_optional(
            row.dense_sparse_quality_delta for row in rows
        ),
        mixed_fallback_count=fallback_count,
        mixed_fallback_rate=fallback_count / len(rows),
    )


def _output_config(**kwargs: Any) -> dict[str, Any]:
    config = dict(kwargs)
    config["block_modes"] = list(config["block_modes"])
    config["oracle_top_k"] = list(config["oracle_top_k"])
    if config["selector_config"] is not None:
        config["selector_config"] = asdict(config["selector_config"])
    return config


def _selected_token_count(selected_spans: Sequence[str]) -> int:
    total = 0
    for span in selected_spans:
        parsed = _parse_span(span)
        if parsed is None:
            continue
        start, end = parsed
        total += max(0, end - start)
    return total


def _filtered_recall(
    original_value: float | None,
    *,
    expected_ids: Sequence[int],
    selected_ids: Sequence[int],
) -> float | None:
    if original_value is None:
        return None
    expected = set(expected_ids)
    if not expected:
        return None
    return len(expected.intersection(selected_ids)) / len(expected)


def _filtered_precision(
    original_value: float | None,
    *,
    expected_ids: Sequence[int],
    selected_ids: Sequence[int],
) -> float | None:
    if original_value is None:
        return None
    selected = set(selected_ids)
    if not selected:
        return None
    return len(set(expected_ids).intersection(selected)) / len(selected)


def _filtered_window_recall(
    original_value: float | None,
    *,
    expected_ids: Sequence[int],
    selected_ids: Sequence[int],
    radius: int,
) -> float | None:
    if original_value is None:
        return None
    expected = set(expected_ids)
    if not expected:
        return None
    selected_window = _expanded_id_set(selected_ids, radius=radius)
    return len(expected.intersection(selected_window)) / len(expected)


def _filtered_window_precision(
    original_value: float | None,
    *,
    expected_ids: Sequence[int],
    selected_ids: Sequence[int],
    radius: int,
) -> float | None:
    if original_value is None:
        return None
    selected_window = _expanded_id_set(selected_ids, radius=radius)
    if not selected_window:
        return None
    return len(set(expected_ids).intersection(selected_window)) / len(selected_window)


def _expanded_id_set(ids: Sequence[int], *, radius: int) -> set[int]:
    expanded: set[int] = set()
    bounded_radius = max(0, radius)
    for block_id in ids:
        expanded.update(
            range(
                int(block_id) - bounded_radius,
                int(block_id) + bounded_radius + 1,
            )
        )
    return expanded


def _token_count(runtime: LocalHfRuntime, text: str) -> int:
    if not text:
        return 0
    tokenized = runtime.tokenize(text)
    return int(getattr(tokenized, "token_count", len(tokenized.token_ids)))


def _context_bounds(prompt_text: str) -> tuple[int, int] | None:
    context_marker_index = prompt_text.find(_CONTEXT_MARKER)
    if context_marker_index < 0:
        return None
    context_start = context_marker_index + len(_CONTEXT_MARKER)
    input_marker_index = prompt_text.find(_INPUT_MARKER, context_start)
    context_end = len(prompt_text) if input_marker_index < 0 else input_marker_index
    if context_end <= context_start:
        return None
    return context_start, context_end


def _passage_spans(
    runtime: LocalHfRuntime,
    prompt_text: str,
) -> tuple[PassageSpan, ...]:
    bounds = _context_bounds(prompt_text)
    if bounds is None:
        return ()
    context_start, context_end = bounds
    context_text = prompt_text[context_start:context_end]
    markers = tuple(_PASSAGE_MARKER_RE.finditer(context_text))
    if not markers:
        return ()

    passages: list[PassageSpan] = []
    for index, marker in enumerate(markers):
        next_start = (
            markers[index + 1].start()
            if index + 1 < len(markers)
            else len(context_text)
        )
        text_start = context_start + marker.start()
        text_end = context_start + next_start
        if text_end <= text_start:
            continue
        token_start = _token_count(runtime, prompt_text[:text_start])
        token_end = _token_count(runtime, prompt_text[:text_end])
        if token_end <= token_start:
            continue
        passages.append(
            PassageSpan(
                text_start=text_start,
                text_end=min(text_end, context_end),
                token_start=token_start,
                token_end=token_end,
            )
        )
    return tuple(passages)


def _expand_interval_to_target(
    start: int,
    end: int,
    *,
    target_tokens: int,
    min_start: int,
    max_end: int,
) -> tuple[int, int]:
    bounded_start = max(min_start, min(start, max_end))
    bounded_end = max(bounded_start, min(end, max_end))
    target = max(0, target_tokens, bounded_end - bounded_start)
    extra = max(0, target - (bounded_end - bounded_start))
    expanded_start = max(min_start, bounded_start - extra // 2)
    expanded_end = min(max_end, bounded_end + extra - extra // 2)

    missing = target - (expanded_end - expanded_start)
    if missing > 0 and expanded_start == min_start:
        expanded_end = min(max_end, expanded_end + missing)

    missing = target - (expanded_end - expanded_start)
    if missing > 0 and expanded_end == max_end:
        expanded_start = max(min_start, expanded_start - missing)

    return expanded_start, expanded_end


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized = sorted(
        (min(start, end), max(start, end))
        for start, end in intervals
        if end > start
    )
    if not normalized:
        return ()

    merged: list[tuple[int, int]] = [normalized[0]]
    for start, end in normalized[1:]:
        previous_start, previous_end = merged[-1]
        if start <= previous_end:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _interval_token_count(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, end - start) for start, end in intervals)


def _parse_span(span: str) -> tuple[int, int] | None:
    parts = span.split(":", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _clean_prediction(prediction: str) -> str:
    first_line = prediction.strip().splitlines()[0].strip() if prediction.strip() else ""
    for stop in ("\n\n", "Context:", "Question:", "Answer:"):
        if stop in first_line:
            first_line = first_line.split(stop, maxsplit=1)[0].strip()
    return first_line


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _format_summary(summary: LongBenchOutputSummary) -> str:
    return (
        f"{summary.dataset} | rows={summary.row_count} "
        f"mean_answer_f1={summary.mean_answer_f1:.3f} "
        f"mean_answer_em={summary.mean_answer_em:.3f} "
        f"mean_quality={summary.mean_answer_quality_score:.3f} "
        f"mean_blocks={summary.mean_selected_block_count:.1f} "
        f"mean_block_frac={summary.mean_selected_block_fraction:.3f} "
        f"mean_dropped={summary.mean_selection_filter_dropped_count:.1f} "
        f"mean_selected_frac={summary.mean_selected_token_fraction:.3f} "
        f"mean_selected_tokens={summary.mean_selected_tokens:.1f} "
        f"mean_recon_frac={summary.mean_reconstructed_context_token_fraction:.3f} "
        f"mean_recon_tokens={summary.mean_reconstructed_context_tokens:.1f} "
        f"mean_evidence_recall={_fmt_optional(summary.mean_evidence_recall)} "
        f"mean_window_recall={_fmt_optional(summary.mean_evidence_window_recall)} "
        f"mean_exact_recall={_fmt_optional(summary.mean_exact_recall)} "
        f"mean_quality_delta={_fmt_optional(summary.mean_dense_sparse_quality_delta)} "
        f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
        f"mean_generation={summary.mean_generation_latency_sec:.6f}s "
        f"mixed_fallback={summary.mixed_fallback_count}/{summary.row_count}"
    )


def _mean(values: Any) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _mean_optional(values: Any) -> float | None:
    materialized = tuple(float(value) for value in values if value is not None)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


def _fmt_optional(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _release_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
