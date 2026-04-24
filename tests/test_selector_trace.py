from __future__ import annotations

import json

import torch

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.pipeline import SelectorPipeline, SelectorPipelineConfig
from kvblock.selector.policies import (
    ConfidencePolicy,
    FallbackPolicy,
    StageAPolicy,
    StageAWeights,
    StageBPolicy,
    StageCPolicy,
)
from kvblock.selector.trace import (
    BlockScoreTrace,
    ConfidenceTrace,
    FallbackTrace,
    SelectionSplitTrace,
    SelectorDecisionTrace,
)


def _metadata(
    block_id: int,
    *,
    summary: tuple[int, ...],
    last_access_step: int = 0,
) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 32,
        token_len=32,
        summary_fp8=summary,
        summary_scale=0.25,
        sign_sketch=block_id,
        summary_norm=1.0,
        last_access_step=last_access_step,
    )


def _pipeline(
    *,
    semantic_top_k: int = 1,
    keep_recent_blocks: int = 0,
    keep_anchor_blocks: int = 0,
    margin_threshold: float = 0.0,
    allow_dense_fallback: bool = True,
) -> SelectorPipeline:
    return SelectorPipeline(
        SelectorPipelineConfig(
            stage_a=StageAPolicy(
                weights=StageAWeights(
                    summary_similarity=1.0,
                    recency=0.0,
                    attn_ema=0.0,
                    priority=0.0,
                ),
                shortlist_size=8,
                long_context_shortlist_size=8,
            ),
            stage_b=StageBPolicy(
                hamming_weight=0.0,
                base_score_weight=1.0,
                sketch_bits=64,
            ),
            stage_c=StageCPolicy(
                keep_recent_blocks=keep_recent_blocks,
                keep_anchor_blocks=keep_anchor_blocks,
                semantic_top_k=semantic_top_k,
                semantic_top_k_long_context=semantic_top_k,
            ),
            confidence=ConfidencePolicy(margin_threshold=margin_threshold),
            fallback=FallbackPolicy(
                widen_top_k_by=1,
                add_recent_blocks_by=1,
                allow_dense_fallback=allow_dense_fallback,
            ),
        )
    )


def _manual_trace() -> SelectorDecisionTrace:
    scores = (
        BlockScoreTrace(
            block_id=2,
            token_start=64,
            token_len=32,
            approx_similarity_score=0.8,
            recency_score=0.1,
            attn_score=0.0,
            priority_score=0.0,
            stage_a_score=0.8,
            hamming_similarity=0.5,
            stage_b_score=0.5,
            final_score=0.9,
        ),
        BlockScoreTrace(
            block_id=1,
            token_start=32,
            token_len=32,
            approx_similarity_score=0.7,
            recency_score=0.0,
            attn_score=0.0,
            priority_score=0.0,
            stage_a_score=0.7,
            hamming_similarity=0.25,
            stage_b_score=0.25,
            final_score=0.75,
        ),
    )
    split = SelectionSplitTrace(
        recent_block_ids=(2,),
        anchor_block_ids=(),
        semantic_block_ids=(1,),
        final_selected_block_ids=(2, 1),
        rail_block_ids=(2,),
        rail_block_count=1,
        semantic_block_count=1,
        requested_anchor_block_ids=(2,),
        deduped_anchor_block_ids=(2,),
        deduped_semantic_block_ids=(2,),
        deduped_block_ids=(2,),
    )
    return SelectorDecisionTrace(
        step_id="unit",
        query_sign_sketch=123,
        candidate_block_ids=(2, 1),
        stage_a_scores=scores,
        stage_a_shortlist_block_ids=(2, 1),
        stage_b_scores=scores,
        pre_fallback_selection=split,
        final_selection=split,
        confidence=ConfidenceTrace(
            semantic_candidate_block_ids=(1,),
            semantic_included_count=1,
            raw_margin=float("inf"),
            normalized_margin=None,
            selected_mass=0.75,
            normalized_mass=None,
            is_confident=True,
        ),
        fallback=FallbackTrace(
            action="keep_sparse",
            mode="sparse",
            next_semantic_top_k=1,
            next_keep_recent_blocks=1,
            use_dense_fallback=False,
            reason_code="confident",
            reason_codes=("confident",),
        ),
    )


def test_selector_decision_trace_serializes_to_dict_and_jsonl() -> None:
    trace = _manual_trace()

    payload = trace.to_dict()
    decoded = json.loads(trace.to_jsonl_line())
    round_tripped = SelectorDecisionTrace.from_dict(decoded)

    assert payload["candidate_block_ids"] == [2, 1]
    assert payload["candidate_count"] == 2
    assert payload["confidence"]["raw_margin"] == "Infinity"
    assert decoded == payload
    assert round_tripped == trace


def test_fallback_trace_keeps_action_and_logical_reason_codes() -> None:
    trace = FallbackTrace(
        action="dense_fallback",
        mode="dense",
        next_semantic_top_k=16,
        next_keep_recent_blocks=6,
        use_dense_fallback=True,
        reason_code="low_margin_and_low_normalized_mass",
        reason_codes=("low_margin", "low_normalized_mass"),
    )

    payload = trace.to_dict()

    assert payload["action"] == "dense_fallback"
    assert payload["mode"] == "dense"
    assert payload["use_dense_fallback"] is True
    assert payload["reason_code"] == "low_margin_and_low_normalized_mass"
    assert payload["reason_codes"] == ["low_margin", "low_normalized_mass"]


def test_pipeline_trace_exposes_overlap_and_dedup_visibility() -> None:
    pipeline = _pipeline(
        semantic_top_k=1,
        keep_recent_blocks=1,
        keep_anchor_blocks=1,
        margin_threshold=0.0,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0), last_access_step=19),
        _metadata(1, summary=(7, 1, 0, 0), last_access_step=5),
        _metadata(2, summary=(-8, 0, 0, 0), last_access_step=1),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=20,
        anchor_block_ids=[0],
    )
    selection = result.trace.final_selection

    assert selection.recent_block_ids == (0,)
    assert selection.requested_anchor_block_ids == (0,)
    assert selection.anchor_block_ids == ()
    assert selection.semantic_block_ids == (1,)
    assert selection.final_selected_block_ids == (0, 1)
    assert selection.deduped_anchor_block_ids == (0,)
    assert selection.deduped_semantic_block_ids == (0,)
    assert selection.deduped_block_ids == (0,)


def test_selector_trace_jsonl_output_is_stable_and_deterministic() -> None:
    pipeline = _pipeline(semantic_top_k=1, margin_threshold=0.0)
    blocks = [
        _metadata(0, summary=(8, 1, 0, 0)),
        _metadata(1, summary=(8, 0, 1, 0)),
        _metadata(2, summary=(-8, 0, 0, 0)),
    ]

    first = pipeline.run(
        torch.tensor([1.0, 0.1, 0.1, 0.0]),
        blocks,
        current_step=3,
        step_id="stable",
    )
    second = pipeline.run(
        torch.tensor([1.0, 0.1, 0.1, 0.0]),
        blocks,
        current_step=3,
        step_id="stable",
    )

    assert first.trace.to_dict() == second.trace.to_dict()
    assert first.trace.to_jsonl_line() == second.trace.to_jsonl_line()
