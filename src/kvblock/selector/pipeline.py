"""Thin end-to-end selector pipeline for the V1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isinf
from typing import Iterable, Sequence

from kvblock.config.models import SelectorConfig
from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import FinalSelection, QuerySummary, ScoredBlock
from kvblock.selector.confidence import (
    ConfidenceAssessment,
    ConfidenceEvaluator,
    normalized_score_margin,
    normalized_selected_mass,
    score_margin,
)
from kvblock.selector.fallback import FallbackDecision, GradedFallbackController
from kvblock.selector.policies import (
    ConfidencePolicy,
    FallbackPolicy,
    StageAPolicy,
    StageBPolicy,
    StageCPolicy,
)
from kvblock.selector.stage_a import StageAScorer
from kvblock.selector.stage_b import StageBRefiner
from kvblock.selector.stage_c import StageCSelector
from kvblock.selector.trace import (
    BlockScoreTrace,
    ConfidenceTrace,
    FallbackTrace,
    SelectionSplitTrace,
    SelectorDecisionTrace,
)
from kvblock.summaries.sign_sketch import generate_sign_sketch


@dataclass(frozen=True, slots=True)
class SelectorPipelineConfig:
    """Small container that binds the stage and fallback policies together."""

    stage_a: StageAPolicy = field(default_factory=StageAPolicy)
    stage_b: StageBPolicy = field(default_factory=StageBPolicy)
    stage_c: StageCPolicy = field(default_factory=StageCPolicy)
    confidence: ConfidencePolicy = field(default_factory=ConfidencePolicy)
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)

    @classmethod
    def from_selector_config(
        cls, config: SelectorConfig
    ) -> "SelectorPipelineConfig":
        return cls(
            stage_a=StageAPolicy.from_selector_config(config),
            stage_b=StageBPolicy.from_selector_config(config),
            stage_c=StageCPolicy.from_selector_config(config),
            confidence=ConfidencePolicy.from_selector_config(config),
            fallback=FallbackPolicy.from_selector_config(config),
        )


@dataclass(frozen=True, slots=True)
class SelectorPipelineResult:
    """Top-level selector pipeline result."""

    selected_blocks: tuple[ScoredBlock, ...]
    mode: str
    trace: SelectorDecisionTrace


class SelectorPipeline:
    """Run the V1 selector stages with a full inspectable trace."""

    def __init__(
        self,
        config: SelectorPipelineConfig | SelectorConfig | None = None,
    ) -> None:
        if config is None:
            resolved = SelectorPipelineConfig()
        elif isinstance(config, SelectorConfig):
            resolved = SelectorPipelineConfig.from_selector_config(config)
        else:
            resolved = config

        self.config = resolved
        self.stage_a = StageAScorer(resolved.stage_a)
        self.stage_b = StageBRefiner(resolved.stage_b)
        self.stage_c = StageCSelector(resolved.stage_c)
        self.confidence = ConfidenceEvaluator(resolved.confidence)
        self.fallback = GradedFallbackController(resolved.fallback)

    def run(
        self,
        query_summary: QuerySummary,
        metadata_blocks: Sequence[BlockMetadata],
        *,
        current_step: int,
        step_id: str | int | None = None,
        context_tokens: int | None = None,
        anchor_block_ids: Iterable[BlockId | int] = (),
        query_sign_sketch: int | None = None,
        semantic_top_k: int | None = None,
        keep_recent_blocks: int | None = None,
    ) -> SelectorPipelineResult:
        """Run the Stage A/B/C pipeline and produce an inspectable trace."""

        # Stage A is intentionally executed once per query. The scorer uses a
        # vectorized summary matrix path internally; slicing the sorted result
        # avoids the old trace-driven second pass over all blocks.
        stage_a_all_scores = self.stage_a.score_all(
            metadata_blocks,
            query_summary,
            current_step=current_step,
        )
        stage_a_shortlist = stage_a_all_scores[
            : self.config.stage_a.shortlist_for_context(context_tokens)
        ]

        resolved_query_sign_sketch = (
            generate_sign_sketch(query_summary)
            if query_sign_sketch is None
            else query_sign_sketch
        )
        stage_b_refined = self.stage_b.refine(stage_a_shortlist, resolved_query_sign_sketch)

        current_semantic_top_k = (
            self.config.stage_c.semantic_top_k_for_context(context_tokens)
            if semantic_top_k is None
            else semantic_top_k
        )
        current_keep_recent = (
            self.config.stage_c.keep_recent_blocks
            if keep_recent_blocks is None
            else keep_recent_blocks
        )
        resolved_anchor_block_ids = tuple(anchor_block_ids)

        pre_fallback_selection = self.stage_c.select(
            stage_b_refined,
            anchor_block_ids=resolved_anchor_block_ids,
            semantic_top_k=current_semantic_top_k,
            keep_recent_blocks=current_keep_recent,
            context_tokens=context_tokens,
        )
        semantic_ranked = _semantic_ranked_candidates(
            stage_b_refined,
            pre_fallback_selection.recent_blocks,
            pre_fallback_selection.anchor_blocks,
        )
        confidence_trace = _build_confidence_trace(
            semantic_ranked,
            semantic_included_count=len(pre_fallback_selection.semantic_blocks),
            evaluator=self.confidence,
        )
        fallback_decision = self.fallback.decide(
            ConfidenceAssessment(
                is_confident=confidence_trace.is_confident,
                score_margin=confidence_trace.raw_margin,
                normalized_margin=confidence_trace.normalized_margin,
                normalized_mass=confidence_trace.normalized_mass,
            ),
            current_top_k=current_semantic_top_k,
            current_keep_recent_blocks=current_keep_recent,
            base_top_k=self.config.stage_c.semantic_top_k_for_context(context_tokens),
            base_keep_recent_blocks=self.config.stage_c.keep_recent_blocks,
        )
        fallback_trace = _build_fallback_trace(
            fallback_decision,
            confidence_trace,
            confidence_policy=self.config.confidence,
        )

        final_selection = _apply_fallback_if_needed(
            self.stage_c,
            stage_b_refined,
            pre_fallback_selection,
            anchor_block_ids=resolved_anchor_block_ids,
            semantic_top_k=fallback_trace.next_semantic_top_k,
            keep_recent_blocks=fallback_trace.next_keep_recent_blocks,
            context_tokens=context_tokens,
            mode=fallback_trace.mode,
        )

        trace = SelectorDecisionTrace(
            step_id=step_id,
            query_sign_sketch=resolved_query_sign_sketch,
            candidate_block_ids=tuple(int(block.block_id) for block in metadata_blocks),
            stage_a_scores=tuple(_score_trace(block) for block in stage_a_all_scores),
            stage_a_shortlist_block_ids=tuple(
                int(block.block_id) for block in stage_a_shortlist
            ),
            stage_b_scores=tuple(_score_trace(block) for block in stage_b_refined),
            pre_fallback_selection=_selection_trace(
                pre_fallback_selection,
                candidates=stage_b_refined,
                anchor_block_ids=resolved_anchor_block_ids,
                semantic_top_k=current_semantic_top_k,
            ),
            final_selection=_selection_trace(
                final_selection,
                candidates=stage_b_refined,
                anchor_block_ids=resolved_anchor_block_ids,
                semantic_top_k=fallback_trace.next_semantic_top_k,
            ),
            confidence=confidence_trace,
            fallback=fallback_trace,
        )

        return SelectorPipelineResult(
            selected_blocks=final_selection.selected_blocks,
            mode=fallback_trace.mode,
            trace=trace,
        )


def _semantic_ranked_candidates(
    refined_candidates: Sequence[ScoredBlock],
    recent_blocks: Sequence[ScoredBlock],
    anchor_blocks: Sequence[ScoredBlock],
) -> list[ScoredBlock]:
    excluded_ids = {
        *(block.block_id for block in recent_blocks),
        *(block.block_id for block in anchor_blocks),
    }
    return [
        candidate for candidate in refined_candidates if candidate.block_id not in excluded_ids
    ]


def _build_confidence_trace(
    semantic_ranked: Sequence[ScoredBlock],
    *,
    semantic_included_count: int,
    evaluator: ConfidenceEvaluator,
) -> ConfidenceTrace:
    assessment = evaluator.assess(semantic_ranked, included_count=semantic_included_count)
    raw_margin = score_margin(semantic_ranked, included_count=semantic_included_count)
    normalized_margin = normalized_score_margin(
        semantic_ranked, included_count=semantic_included_count
    )
    selected_mass = _selected_positive_mass(
        semantic_ranked, included_count=semantic_included_count
    )
    normalized_mass = (
        normalized_selected_mass(semantic_ranked, included_count=semantic_included_count)
        if evaluator.policy.min_normalized_mass is not None
        else None
    )

    return ConfidenceTrace(
        semantic_candidate_block_ids=tuple(int(block.block_id) for block in semantic_ranked),
        semantic_included_count=semantic_included_count,
        raw_margin=raw_margin,
        normalized_margin=normalized_margin,
        selected_mass=selected_mass,
        normalized_mass=normalized_mass,
        is_confident=assessment.is_confident,
    )


def _selected_positive_mass(
    ranked_candidates: Sequence[ScoredBlock], *, included_count: int
) -> float | None:
    if included_count <= 0:
        return 0.0
    return sum(max(candidate.final_score, 0.0) for candidate in ranked_candidates[:included_count])


def _build_fallback_trace(
    decision: FallbackDecision,
    confidence: ConfidenceTrace,
    *,
    confidence_policy: ConfidencePolicy,
) -> FallbackTrace:
    mode_map = {
        "keep_sparse": "sparse",
        "widen_k": "widen_k",
        "add_recent_blocks": "add_recent",
        "dense_fallback": "dense",
    }
    reason_codes = _reason_codes(confidence, confidence_policy)
    reason_code = _reason_code(reason_codes)
    return FallbackTrace(
        action=decision.action,
        mode=mode_map[decision.action],
        next_semantic_top_k=decision.next_top_k,
        next_keep_recent_blocks=decision.next_keep_recent_blocks,
        use_dense_fallback=decision.use_dense_fallback,
        reason_code=reason_code,
        reason_codes=reason_codes,
    )


def _reason_codes(
    confidence: ConfidenceTrace, confidence_policy: ConfidencePolicy
) -> tuple[str, ...]:
    low_margin = confidence.raw_margin < confidence_policy.margin_threshold
    low_normalized_margin = (
        confidence_policy.normalized_margin_threshold is not None
        and (
            (confidence.normalized_margin is None and not isinf(confidence.raw_margin))
            or confidence.normalized_margin
            is not None
            and confidence.normalized_margin
            < confidence_policy.normalized_margin_threshold
        )
    )
    low_mass = (
        confidence_policy.min_normalized_mass is not None
        and (
            confidence.normalized_mass is None
            or confidence.normalized_mass < confidence_policy.min_normalized_mass
        )
    )
    reasons: list[str] = []
    if low_margin:
        reasons.append("low_margin")
    if low_normalized_margin:
        reasons.append("low_normalized_margin")
    if low_mass:
        reasons.append("low_normalized_mass")
    return tuple(reasons) or ("confident",)


def _reason_code(reason_codes: tuple[str, ...]) -> str:
    if len(reason_codes) == 1:
        return reason_codes[0]
    if reason_codes == (
        "low_margin",
        "low_normalized_margin",
        "low_normalized_mass",
    ):
        return "low_margin_low_normalized_margin_and_low_normalized_mass"
    return "_and_".join(reason_codes)


def _apply_fallback_if_needed(
    selector: StageCSelector,
    refined_candidates: Sequence[ScoredBlock],
    current_selection: FinalSelection,
    *,
    anchor_block_ids: Iterable[BlockId | int],
    semantic_top_k: int,
    keep_recent_blocks: int,
    context_tokens: int | None,
    mode: str,
) -> FinalSelection:
    if mode == "widen_k" or mode == "add_recent":
        return selector.select(
            refined_candidates,
            anchor_block_ids=anchor_block_ids,
            semantic_top_k=semantic_top_k,
            keep_recent_blocks=keep_recent_blocks,
            context_tokens=context_tokens,
        )
    return current_selection


def _selection_trace(
    selection: FinalSelection,
    *,
    candidates: Sequence[ScoredBlock],
    anchor_block_ids: Iterable[BlockId | int],
    semantic_top_k: int,
) -> SelectionSplitTrace:
    recent_ids = tuple(int(block.block_id) for block in selection.recent_blocks)
    anchor_ids = tuple(int(block.block_id) for block in selection.anchor_blocks)
    semantic_ids = tuple(int(block.block_id) for block in selection.semantic_blocks)
    rail_ids = recent_ids + tuple(
        anchor_id for anchor_id in anchor_ids if anchor_id not in recent_ids
    )
    final_ids = tuple(int(block.block_id) for block in selection.selected_blocks)
    requested_anchor_ids = _normalize_trace_anchor_ids(anchor_block_ids)
    candidate_ids = {int(candidate.block_id) for candidate in candidates}
    recent_set = set(recent_ids)
    anchor_set = set(anchor_ids)
    rail_set = set(rail_ids)
    semantic_raw_top_ids = tuple(
        int(candidate.block_id)
        for candidate in sorted(
            candidates,
            key=lambda candidate: candidate.final_score,
            reverse=True,
        )[: max(semantic_top_k, 0)]
    )
    missing_anchor_ids = tuple(
        block_id for block_id in requested_anchor_ids if block_id not in candidate_ids
    )
    deduped_anchor_ids = tuple(
        block_id
        for block_id in requested_anchor_ids
        if block_id in candidate_ids and block_id in recent_set and block_id not in anchor_set
    )
    deduped_semantic_ids = tuple(
        block_id for block_id in semantic_raw_top_ids if block_id in rail_set
    )
    deduped_ids = _ordered_union(deduped_anchor_ids, deduped_semantic_ids)
    return SelectionSplitTrace(
        recent_block_ids=recent_ids,
        anchor_block_ids=anchor_ids,
        semantic_block_ids=semantic_ids,
        final_selected_block_ids=final_ids,
        rail_block_ids=rail_ids,
        rail_block_count=len(rail_ids),
        semantic_block_count=len(semantic_ids),
        requested_anchor_block_ids=requested_anchor_ids,
        missing_anchor_block_ids=missing_anchor_ids,
        deduped_anchor_block_ids=deduped_anchor_ids,
        deduped_semantic_block_ids=deduped_semantic_ids,
        deduped_block_ids=deduped_ids,
    )


def _normalize_trace_anchor_ids(
    anchor_block_ids: Iterable[BlockId | int],
) -> tuple[int, ...]:
    normalized: list[int] = []
    seen: set[int] = set()
    for anchor_id in anchor_block_ids:
        block_id = int(anchor_id)
        if block_id in seen:
            continue
        seen.add(block_id)
        normalized.append(block_id)
    return tuple(normalized)


def _ordered_union(*groups: Sequence[int]) -> tuple[int, ...]:
    ordered: list[int] = []
    seen: set[int] = set()
    for group in groups:
        for block_id in group:
            if block_id in seen:
                continue
            seen.add(block_id)
            ordered.append(block_id)
    return tuple(ordered)


def _score_trace(block: ScoredBlock) -> BlockScoreTrace:
    return BlockScoreTrace(
        block_id=int(block.block_id),
        token_start=block.metadata.token_start,
        token_len=block.metadata.token_len,
        approx_similarity_score=block.approx_similarity_score,
        recency_score=block.recency_score,
        attn_score=block.attn_score,
        priority_score=block.priority_score,
        stage_a_score=block.stage_a_score,
        hamming_similarity=block.hamming_similarity,
        stage_b_score=block.stage_b_score,
        final_score=block.final_score,
    )
