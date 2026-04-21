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


def _metadata(
    block_id: int,
    *,
    summary: tuple[int, ...],
    sign_sketch: int | None = None,
    last_access_step: int = -1,
    attn_ema: float = 0.0,
    priority: float = 0.0,
) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 32,
        token_len=32,
        summary_fp8=summary,
        summary_scale=0.25,
        sign_sketch=block_id if sign_sketch is None else sign_sketch,
        summary_norm=1.0,
        attn_ema=attn_ema,
        last_access_step=last_access_step,
        priority=priority,
    )


def _pipeline(
    *,
    stage_a_weights: StageAWeights | None = None,
    semantic_top_k: int = 1,
    keep_recent_blocks: int = 0,
    keep_anchor_blocks: int = 0,
    margin_threshold: float = 0.05,
    min_normalized_mass: float | None = None,
    widen_top_k_by: int = 1,
    add_recent_blocks_by: int = 1,
) -> SelectorPipeline:
    return SelectorPipeline(
        SelectorPipelineConfig(
            stage_a=StageAPolicy(
                weights=stage_a_weights or StageAWeights(),
                shortlist_size=8,
                long_context_shortlist_size=8,
            ),
            stage_b=StageBPolicy(hamming_weight=0.0, base_score_weight=1.0, sketch_bits=64),
            stage_c=StageCPolicy(
                keep_recent_blocks=keep_recent_blocks,
                keep_anchor_blocks=keep_anchor_blocks,
                semantic_top_k=semantic_top_k,
                semantic_top_k_long_context=semantic_top_k,
            ),
            confidence=ConfidencePolicy(
                margin_threshold=margin_threshold,
                min_normalized_mass=min_normalized_mass,
            ),
            fallback=FallbackPolicy(
                widen_top_k_by=widen_top_k_by,
                add_recent_blocks_by=add_recent_blocks_by,
                allow_dense_fallback=True,
            ),
        )
    )


def test_selector_pipeline_clear_winner_case() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        margin_threshold=0.2,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0)),
        _metadata(1, summary=(-8, 0, 0, 0)),
        _metadata(2, summary=(0, 8, 0, 0)),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=10,
        step_id="clear",
    )

    assert result.mode == "sparse"
    assert [int(block.block_id) for block in result.selected_blocks] == [0]
    assert result.trace.confidence.is_confident is True
    assert result.trace.fallback.reason_code == "confident"


def test_selector_pipeline_ambiguous_shortlist_case_is_deterministic() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        margin_threshold=0.0,
    )
    blocks = [
        _metadata(0, summary=(8, 1, 0, 0)),
        _metadata(1, summary=(8, 0, 1, 0)),
        _metadata(2, summary=(-8, 0, 0, 0)),
    ]

    first = pipeline.run(torch.tensor([1.0, 0.1, 0.1, 0.0]), blocks, current_step=1)
    second = pipeline.run(torch.tensor([1.0, 0.1, 0.1, 0.0]), blocks, current_step=1)

    assert first.trace.stage_a_shortlist_block_ids == second.trace.stage_a_shortlist_block_ids
    assert first.trace.stage_b_scores == second.trace.stage_b_scores


def test_selector_pipeline_tracks_rail_dominated_case_separately() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        keep_recent_blocks=2,
        keep_anchor_blocks=1,
        margin_threshold=0.0,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0), last_access_step=1),
        _metadata(1, summary=(0, 8, 0, 0), last_access_step=100),
        _metadata(2, summary=(0, 0, 8, 0), last_access_step=99),
        _metadata(3, summary=(-8, 0, 0, 0), last_access_step=2),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=101,
        anchor_block_ids=[3],
    )

    assert result.trace.pre_fallback_selection.recent_block_ids == (1, 2)
    assert result.trace.pre_fallback_selection.anchor_block_ids == (3,)
    assert result.trace.pre_fallback_selection.semantic_block_ids == (0,)
    assert result.trace.confidence.semantic_included_count == 1
    assert result.trace.pre_fallback_selection.rail_block_count == 3


def test_selector_pipeline_weak_confidence_triggers_widening() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        margin_threshold=0.1,
        widen_top_k_by=1,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0)),
        _metadata(1, summary=(7, 1, 0, 0)),
        _metadata(2, summary=(-8, 0, 0, 0)),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=5,
    )

    assert result.mode == "widen_k"
    assert result.trace.fallback.reason_code == "low_margin"
    assert result.trace.pre_fallback_selection.semantic_block_ids == (0,)
    assert result.trace.final_selection.semantic_block_ids == (0, 1)


def test_selector_pipeline_escalation_path_can_end_in_dense_fallback() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        keep_recent_blocks=0,
        margin_threshold=0.1,
        widen_top_k_by=1,
        add_recent_blocks_by=1,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0), last_access_step=1),
        _metadata(1, summary=(7, 1, 0, 0), last_access_step=2),
        _metadata(2, summary=(7, 0, 1, 0), last_access_step=3),
        _metadata(3, summary=(0, 8, 0, 0), last_access_step=100),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=101,
        semantic_top_k=2,
        keep_recent_blocks=1,
    )

    assert result.mode == "dense"
    assert result.trace.fallback.action == "dense_fallback"
    assert result.trace.fallback.use_dense_fallback is True
    assert result.trace.final_selection.final_selected_block_ids == (
        result.trace.pre_fallback_selection.final_selected_block_ids
    )


def test_selector_pipeline_trace_contains_required_intermediate_data() -> None:
    pipeline = _pipeline(
        stage_a_weights=StageAWeights(
            summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
        ),
        semantic_top_k=1,
        margin_threshold=0.0,
        min_normalized_mass=0.5,
    )
    blocks = [
        _metadata(0, summary=(8, 0, 0, 0)),
        _metadata(1, summary=(4, 4, 0, 0)),
    ]

    result = pipeline.run(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        blocks,
        current_step=1,
        step_id="trace-check",
    )
    trace = result.trace

    assert trace.step_id == "trace-check"
    assert len(trace.stage_a_scores) == 2
    assert len(trace.stage_a_shortlist_block_ids) == 2
    assert len(trace.stage_b_scores) == 2
    assert trace.pre_fallback_selection.final_selected_block_ids
    assert trace.final_selection.final_selected_block_ids
    assert trace.confidence.raw_margin >= 0.0
    assert trace.confidence.selected_mass is not None
    assert trace.confidence.normalized_mass is not None
    assert trace.fallback.mode in {"sparse", "widen_k", "add_recent", "dense"}
