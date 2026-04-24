"""Dense-only real-block selector bridge for local V1 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any

import torch

from kvblock.kv.block_manager import (
    BlockIngestConfig,
    BlockIngestResult,
    build_block_metadata_from_representations,
)
from kvblock.kv.block_modes import BlockCandidate, BlockModeName, block_mode_from_name
from kvblock.kv.block_types import BlockId
from kvblock.runtime.base import ModelPrefillOutput, RuntimeBackend
from kvblock.runtime.head_diagnostics import (
    HeadDiagnosticAggregate,
    HeadDiagnosticBlockContext,
    PerHeadBlockDiagnostic,
    build_per_head_block_diagnostics,
    summarize_head_diagnostics,
)
from kvblock.runtime.query_key_inspection import (
    QueryKeyInspectionBundle,
    build_query_key_inspection,
)
from kvblock.kv.qk_aggregation import (
    QKAggregationStrategy,
    qk_aggregation_strategy_from_name,
)
from kvblock.selector.pipeline import (
    ConfidenceTrace,
    SelectorDecisionTrace,
    SelectorPipeline,
    SelectorPipelineConfig,
)
from kvblock.selector.policies import ConfidencePolicy, StageAPolicy, StageCPolicy


@dataclass(frozen=True, slots=True)
class RealBlockSelectorConfig:
    """Config for the local dense-only real-block selector bridge."""

    block_size: int = 32
    summary_dim: int = 32
    shortlist_m: int = 16
    semantic_k: int = 10
    confidence_margin: float = 0.0
    keep_recent_blocks: int = 4
    keep_anchor_blocks: int = 2
    head_scoring_mode: str = "mean_heads"
    head_top_k: int = 2
    head_weights: tuple[float, ...] = ()
    qk_aggregation_strategy: QKAggregationStrategy = "mean_pool"
    top_token_count: int = 4
    block_mode: BlockModeName = "fixed"
    overlap_stride: int | None = None
    block_candidates: tuple[BlockCandidate, ...] = ()
    preview_chars: int = 120
    include_block_text: bool = False
    emit_head_diagnostics: bool = False
    top_heads: int = 5
    emit_query_key_inspection: bool = False
    relevant_text_fragments: tuple[str, ...] = ()
    top_unselected_blocks: int = 5
    representation_source: str | None = None
    query_prompt: str | None = None
    rail_setting: str | None = None
    prompt_id: str | None = None
    prompt_name: str | None = None

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")
        if self.shortlist_m <= 0:
            raise ValueError("shortlist_m must be > 0")
        if self.semantic_k <= 0:
            raise ValueError("semantic_k must be > 0")
        if self.confidence_margin < 0:
            raise ValueError("confidence_margin must be >= 0")
        if self.keep_recent_blocks < 0:
            raise ValueError("keep_recent_blocks must be >= 0")
        if self.keep_anchor_blocks < 0:
            raise ValueError("keep_anchor_blocks must be >= 0")
        if self.head_scoring_mode not in {
            "mean_heads",
            "max_head_score",
            "topk_head_mean",
            "weighted_head_mean",
        }:
            raise ValueError(f"unsupported head_scoring_mode: {self.head_scoring_mode!r}")
        if self.head_top_k <= 0:
            raise ValueError("head_top_k must be > 0")
        if any(weight < 0 for weight in self.head_weights):
            raise ValueError("head_weights must be >= 0")
        if self.head_weights and sum(self.head_weights) <= 0:
            raise ValueError("head_weights must contain at least one positive value")
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
        if self.preview_chars <= 0:
            raise ValueError("preview_chars must be > 0")
        if self.top_heads <= 0:
            raise ValueError("top_heads must be > 0")
        if self.top_unselected_blocks < 0:
            raise ValueError("top_unselected_blocks must be >= 0")
        if any(not fragment.strip() for fragment in self.relevant_text_fragments):
            raise ValueError("relevant_text_fragments must be non-empty when set")
        if self.representation_source is not None and not self.representation_source.strip():
            raise ValueError("representation_source must be non-empty when set")
        if self.query_prompt is not None and not self.query_prompt.strip():
            raise ValueError("query_prompt must be non-empty when set")
        if self.rail_setting is not None and not self.rail_setting.strip():
            raise ValueError("rail_setting must be non-empty when set")
        if self.prompt_id is not None and not self.prompt_id.strip():
            raise ValueError("prompt_id must be non-empty when set")
        if self.prompt_name is not None and not self.prompt_name.strip():
            raise ValueError("prompt_name must be non-empty when set")


@dataclass(frozen=True, slots=True)
class RealBlockRunSummary:
    """Compact metadata summary for one real-block selector run."""

    runtime_name: str
    representation_name: str
    token_count: int
    block_count: int
    block_size: int
    summary_dim: int
    prompt_chars: int
    block_mode: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlockInspectionRecord:
    """Human-readable block span aligned with selector trace scores."""

    block_id: int
    token_start: int
    token_end: int
    token_count: int
    selected: bool
    selected_reason: str
    stage_a_score: float | None
    stage_b_score: float | None
    final_score: float | None
    preview_text: str
    block_text: str | None = None
    candidate_id: str | None = None
    block_size: int | None = None
    stride: int | None = None
    block_mode: str | None = None
    parent_block_id: int | None = None
    parent_candidate_id: str | None = None
    parent_token_start: int | None = None
    parent_token_end: int | None = None
    candidate_role: str = "block"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly inspection record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RealBlockLatencySummary:
    """Timing breakdown for one dense-only real-block selector run."""

    model_load_sec: float
    prefill_sec: float
    metadata_sec: float
    selector_sec: float
    inspection_sec: float
    total_sec: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly latency record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class RealBlockSelectorResult:
    """Bridge output: real metadata ingest plus selector decision trace."""

    selected_block_ids: tuple[int, ...]
    selected_to_semantic_k_ratio: float
    fallback_mode: str
    confidence: ConfidenceTrace
    run_summary: RealBlockRunSummary
    latency: RealBlockLatencySummary
    trace: SelectorDecisionTrace
    block_inspections: tuple[BlockInspectionRecord, ...]
    head_diagnostics: tuple[PerHeadBlockDiagnostic, ...] = field(default_factory=tuple)
    head_diagnostic_summary: HeadDiagnosticAggregate | None = None
    query_key_inspection: QueryKeyInspectionBundle | None = None
    per_head_token_representations: torch.Tensor | None = None
    per_head_query_representation: torch.Tensor | None = None

    @property
    def selected_block_inspections(self) -> tuple[BlockInspectionRecord, ...]:
        """Return inspection records for selected blocks only."""

        return tuple(block for block in self.block_inspections if block.selected)

    @property
    def unselected_block_inspections(self) -> tuple[BlockInspectionRecord, ...]:
        """Return inspection records for unselected blocks only."""

        return tuple(block for block in self.block_inspections if not block.selected)

    @property
    def selected_head_diagnostics(self) -> tuple[PerHeadBlockDiagnostic, ...]:
        """Return per-head diagnostics for selected blocks only."""

        return tuple(block for block in self.head_diagnostics if block.selected)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result record."""

        return {
            "selected_block_ids": list(self.selected_block_ids),
            "selected_to_semantic_k_ratio": self.selected_to_semantic_k_ratio,
            "fallback_mode": self.fallback_mode,
            "confidence": self.confidence.to_dict(),
            "run_summary": self.run_summary.to_dict(),
            "latency": self.latency.to_dict(),
            "trace": self.trace.to_dict(),
            "block_inspections": [
                block.to_dict() for block in self.block_inspections
            ],
            "selected_block_inspections": [
                block.to_dict() for block in self.selected_block_inspections
            ],
            "unselected_block_inspections": [
                block.to_dict() for block in self.unselected_block_inspections
            ],
            "head_diagnostics": [
                block.to_dict() for block in self.head_diagnostics
            ],
            "selected_head_diagnostics": [
                block.to_dict() for block in self.selected_head_diagnostics
            ],
            "head_diagnostic_summary": (
                None
                if self.head_diagnostic_summary is None
                else self.head_diagnostic_summary.to_dict()
            ),
            "query_key_inspection": (
                None
                if self.query_key_inspection is None
                else self.query_key_inspection.to_dict()
            ),
            "per_head_token_representation_shape": (
                None
                if self.per_head_token_representations is None
                else list(self.per_head_token_representations.shape)
            ),
            "per_head_query_representation_shape": (
                None
                if self.per_head_query_representation is None
                else list(self.per_head_query_representation.shape)
            ),
        }


def run_real_block_selector(
    runtime: RuntimeBackend,
    prompt: str,
    config: RealBlockSelectorConfig | None = None,
) -> RealBlockSelectorResult:
    """Run prompt prefill, metadata ingest, and the existing selector pipeline."""

    resolved = config or RealBlockSelectorConfig()
    total_started_at = perf_counter()
    started_at = perf_counter()
    runtime.load_model()
    model_load_sec = perf_counter() - started_at

    started_at = perf_counter()
    prefill = runtime.prefill(prompt)
    query_prefill = (
        None
        if resolved.query_prompt is None
        else runtime.prefill(resolved.query_prompt)
    )
    per_head_query_representation = (
        prefill.per_head_query_representation
        if query_prefill is None
        else query_prefill.per_head_query_representation
    )
    prefill_sec = perf_counter() - started_at

    started_at = perf_counter()
    ingest = _ingest_prefill(prefill, resolved, query_prefill=query_prefill)
    metadata_sec = perf_counter() - started_at

    pipeline_config = _pipeline_config(resolved, ingest)
    pipeline = SelectorPipeline(pipeline_config)
    anchor_block_ids = tuple(
        block.block_id
        for block in sorted(
            ingest.metadata_blocks,
            key=lambda block: (block.token_start, block.token_len, int(block.block_id)),
        )[: resolved.keep_anchor_blocks]
    )
    started_at = perf_counter()
    selector_result = pipeline.run(
        ingest.query_summary,
        ingest.metadata_blocks,
        current_step=ingest.token_count,
        step_id="real-block-prefill",
        context_tokens=ingest.token_count,
        anchor_block_ids=anchor_block_ids,
    )
    selector_sec = perf_counter() - started_at
    selected_ids = tuple(
        int(block_id)
        for block_id in selector_result.trace.final_selection.final_selected_block_ids
    )
    started_at = perf_counter()
    block_inspections = _build_block_inspections(
        runtime=runtime,
        prefill=prefill,
        ingest=ingest,
        trace=selector_result.trace,
        preview_chars=resolved.preview_chars,
        include_block_text=resolved.include_block_text,
    )
    head_diagnostics: tuple[PerHeadBlockDiagnostic, ...] = ()
    head_diagnostic_summary: HeadDiagnosticAggregate | None = None
    if resolved.emit_head_diagnostics:
        head_diagnostics = build_per_head_block_diagnostics(
            metadata_blocks=ingest.metadata_blocks,
            query_summary=ingest.query_summary,
            block_contexts=_head_diagnostic_contexts(block_inspections),
            policy=pipeline_config.stage_a,
            representation_source=(
                resolved.representation_source or ingest.representation_name
            ),
            representation_name=ingest.representation_name,
            top_heads=resolved.top_heads,
            rail_setting=resolved.rail_setting,
            prompt_name=resolved.prompt_name,
        )
        if head_diagnostics:
            head_diagnostic_summary = summarize_head_diagnostics(
                head_diagnostics,
                prompt_name=resolved.prompt_name,
                top_n=resolved.top_heads,
            )
    query_key_inspection: QueryKeyInspectionBundle | None = None
    if resolved.emit_query_key_inspection:
        query_key_inspection = build_query_key_inspection(
            prompt_id=resolved.prompt_id,
            prompt_name=resolved.prompt_name,
            prompt_text=prefill.prompt,
            representation_source=(
                resolved.representation_source or ingest.representation_name
            ),
            representation_name=ingest.representation_name,
            rail_setting=resolved.rail_setting,
            qk_aggregation_strategy=resolved.qk_aggregation_strategy,
            selected_block_ids=selected_ids,
            metadata_blocks=ingest.metadata_blocks,
            query_summary=ingest.query_summary,
            trace=selector_result.trace,
            block_inspections=block_inspections,
            relevant_text_fragments=resolved.relevant_text_fragments,
            top_unselected_blocks=resolved.top_unselected_blocks,
        )
    inspection_sec = perf_counter() - started_at
    total_sec = perf_counter() - total_started_at

    return RealBlockSelectorResult(
        selected_block_ids=selected_ids,
        selected_to_semantic_k_ratio=len(selected_ids) / resolved.semantic_k,
        fallback_mode=selector_result.mode,
        confidence=selector_result.trace.confidence,
        run_summary=RealBlockRunSummary(
            runtime_name=prefill.runtime_name,
            representation_name=ingest.representation_name,
            token_count=ingest.token_count,
            block_count=len(ingest.metadata_blocks),
            block_size=ingest.block_size,
            summary_dim=ingest.summary_dim,
            prompt_chars=len(prefill.prompt),
            block_mode=ingest.block_mode,
        ),
        latency=RealBlockLatencySummary(
            model_load_sec=model_load_sec,
            prefill_sec=prefill_sec,
            metadata_sec=metadata_sec,
            selector_sec=selector_sec,
            inspection_sec=inspection_sec,
            total_sec=total_sec,
        ),
        trace=selector_result.trace,
        block_inspections=block_inspections,
        head_diagnostics=head_diagnostics,
        head_diagnostic_summary=head_diagnostic_summary,
        query_key_inspection=query_key_inspection,
        per_head_token_representations=prefill.per_head_token_representations,
        per_head_query_representation=per_head_query_representation,
    )


def _ingest_prefill(
    prefill: ModelPrefillOutput,
    config: RealBlockSelectorConfig,
    *,
    query_prefill: ModelPrefillOutput | None = None,
) -> BlockIngestResult:
    query_representation = (
        prefill.query_representation
        if query_prefill is None
        else query_prefill.query_representation
    )
    per_head_query_representation = (
        prefill.per_head_query_representation
        if query_prefill is None
        else query_prefill.per_head_query_representation
    )
    return build_block_metadata_from_representations(
        prefill.token_representations,
        prefill.token_ids,
        BlockIngestConfig(
            block_size=config.block_size,
            summary_dim=config.summary_dim,
            representation_name=_representation_name_with_aggregation(
                _representation_name_with_query_override(
                    prefill.representation_name,
                    query_prefill,
                ),
                config.qk_aggregation_strategy,
            ),
            qk_aggregation_strategy=config.qk_aggregation_strategy,
            top_token_count=config.top_token_count,
            block_mode=config.block_mode,
            overlap_stride=config.overlap_stride,
            block_candidates=config.block_candidates,
        ),
        query_representation=query_representation,
        per_head_token_representations=prefill.per_head_token_representations,
        per_head_query_representation=per_head_query_representation,
    )


def _representation_name_with_query_override(
    prefill_representation_name: str,
    query_prefill: ModelPrefillOutput | None,
) -> str:
    if query_prefill is None:
        return prefill_representation_name
    return f"{prefill_representation_name}_query_override_{query_prefill.representation_name}"


def _representation_name_with_aggregation(
    representation_name: str,
    strategy: QKAggregationStrategy,
) -> str:
    if strategy == "mean_pool":
        return representation_name
    return f"{representation_name}_qkagg_{strategy}"


def _pipeline_config(
    config: RealBlockSelectorConfig,
    ingest: BlockIngestResult,
) -> SelectorPipelineConfig:
    long_context_threshold = ingest.token_count + 1
    return SelectorPipelineConfig(
        stage_a=StageAPolicy(
            shortlist_size=config.shortlist_m,
            long_context_shortlist_size=config.shortlist_m,
            long_context_threshold=long_context_threshold,
            head_scoring_mode=config.head_scoring_mode,  # type: ignore[arg-type]
            head_top_k=config.head_top_k,
            head_weights=config.head_weights,
        ),
        stage_c=StageCPolicy(
            keep_recent_blocks=config.keep_recent_blocks,
            keep_anchor_blocks=config.keep_anchor_blocks,
            semantic_top_k=config.semantic_k,
            semantic_top_k_long_context=config.semantic_k,
            long_context_threshold=long_context_threshold,
        ),
        confidence=ConfidencePolicy(margin_threshold=config.confidence_margin),
    )


def _build_block_inspections(
    *,
    runtime: RuntimeBackend,
    prefill: ModelPrefillOutput,
    ingest: BlockIngestResult,
    trace: SelectorDecisionTrace,
    preview_chars: int,
    include_block_text: bool,
) -> tuple[BlockInspectionRecord, ...]:
    stage_a_scores = {score.block_id: score for score in trace.stage_a_scores}
    stage_b_scores = {score.block_id: score for score in trace.stage_b_scores}
    selected_ids = set(trace.final_selection.final_selected_block_ids)
    candidate_by_id = ingest.candidate_by_block_id

    records: list[BlockInspectionRecord] = []
    for metadata in ingest.metadata_blocks:
        block_id = int(metadata.block_id)
        candidate = candidate_by_id.get(block_id)
        token_start = metadata.token_start
        token_end = metadata.token_start + metadata.token_len
        block_text = runtime.decode_token_ids(
            tuple(prefill.token_ids[token_start:token_end])
        )
        stage_a = stage_a_scores.get(block_id)
        stage_b = stage_b_scores.get(block_id)
        records.append(
            BlockInspectionRecord(
                block_id=block_id,
                token_start=token_start,
                token_end=token_end,
                token_count=metadata.token_len,
                selected=block_id in selected_ids,
                selected_reason=_selected_reason(block_id, trace),
                stage_a_score=None if stage_a is None else stage_a.stage_a_score,
                stage_b_score=None if stage_b is None else stage_b.stage_b_score,
                final_score=(
                    stage_b.final_score
                    if stage_b is not None
                    else None if stage_a is None else stage_a.final_score
                ),
                preview_text=_preview_text(block_text, preview_chars=preview_chars),
                block_text=block_text if include_block_text else None,
                candidate_id=None if candidate is None else candidate.candidate_id,
                block_size=None if candidate is None else candidate.block_size,
                stride=None if candidate is None else candidate.stride,
                block_mode=None if candidate is None else candidate.block_mode,
                parent_block_id=None if candidate is None else candidate.parent_block_id,
                parent_candidate_id=(
                    None if candidate is None else candidate.parent_candidate_id
                ),
                parent_token_start=(
                    None if candidate is None else candidate.parent_token_start
                ),
                parent_token_end=(
                    None
                    if candidate is None or candidate.parent_token_start is None
                    or candidate.parent_token_len is None
                    else candidate.parent_token_start + candidate.parent_token_len
                ),
                candidate_role="block" if candidate is None else candidate.candidate_role,
            )
        )
    return tuple(records)


def _head_diagnostic_contexts(
    block_inspections: tuple[BlockInspectionRecord, ...],
) -> tuple[HeadDiagnosticBlockContext, ...]:
    return tuple(
        HeadDiagnosticBlockContext(
            block_id=block.block_id,
            selected=block.selected,
            selected_reason=block.selected_reason,
            token_start=block.token_start,
            token_end=block.token_end,
            token_count=block.token_count,
            preview_text=block.preview_text,
        )
        for block in block_inspections
    )


def _selected_reason(block_id: int, trace: SelectorDecisionTrace) -> str:
    reasons: list[str] = []
    selection = trace.final_selection
    if block_id in selection.recent_block_ids:
        reasons.append("recent")
    if block_id in selection.anchor_block_ids:
        reasons.append("anchor")
    if block_id in selection.semantic_block_ids:
        reasons.append("semantic")
    return "+".join(reasons) if reasons else "unselected"


def _preview_text(block_text: str, *, preview_chars: int) -> str:
    normalized = " ".join(block_text.split())
    if len(normalized) <= preview_chars:
        return normalized
    return normalized[: max(preview_chars - 3, 0)].rstrip() + "..."
