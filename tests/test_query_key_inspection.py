from __future__ import annotations

import json

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.runtime.query_key_inspection import build_query_key_inspection
from kvblock.runtime.real_block_eval import BlockInspectionRecord
from kvblock.selector.pipeline import (
    BlockScoreTrace,
    ConfidenceTrace,
    FallbackTrace,
    SelectionSplitTrace,
    SelectorDecisionTrace,
)
from kvblock.summaries.base import SummaryEncoding


def _metadata(block_id: int) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 2,
        token_len=2,
        summary_fp8=(block_id + 1, 0, 0, 0),
        summary_scale=0.25,
        sign_sketch=block_id,
        summary_norm=float(block_id + 1),
        per_head_summary_fp8=((block_id + 1, 0, 0, 0), (0, block_id + 1, 0, 0)),
        per_head_summary_scale=(0.25, 0.5),
        per_head_summary_norm=(1.0, 2.0),
    )


def _inspection(
    block_id: int,
    *,
    selected: bool,
    reason: str,
    text: str,
    stage_a: float,
) -> BlockInspectionRecord:
    return BlockInspectionRecord(
        block_id=block_id,
        token_start=block_id * 2,
        token_end=block_id * 2 + 2,
        token_count=2,
        selected=selected,
        selected_reason=reason,
        stage_a_score=stage_a,
        stage_b_score=stage_a + 0.1 if selected else None,
        final_score=stage_a + 0.1 if selected else stage_a,
        preview_text=text,
        block_text=text,
        candidate_id=f"s2_stride2_t{block_id * 2}_{block_id * 2 + 2}",
        block_size=2,
        stride=2,
        block_mode="fixed_2",
    )


def _trace() -> SelectorDecisionTrace:
    scores = tuple(
        BlockScoreTrace(
            block_id=block_id,
            token_start=block_id * 2,
            token_len=2,
            approx_similarity_score=0.9 - block_id * 0.1,
            recency_score=0.0,
            attn_score=0.0,
            priority_score=0.0,
            stage_a_score=0.9 - block_id * 0.1,
            hamming_similarity=0.0,
            stage_b_score=0.0,
            final_score=0.9 - block_id * 0.1,
        )
        for block_id in range(4)
    )
    return SelectorDecisionTrace(
        step_id="test",
        query_sign_sketch=7,
        stage_a_scores=scores,
        stage_a_shortlist_block_ids=(0, 1, 2, 3),
        stage_b_scores=scores[:2],
        pre_fallback_selection=SelectionSplitTrace(
            recent_block_ids=(2,),
            anchor_block_ids=(),
            semantic_block_ids=(0,),
            final_selected_block_ids=(0, 2),
            rail_block_ids=(2,),
            rail_block_count=1,
            semantic_block_count=1,
        ),
        final_selection=SelectionSplitTrace(
            recent_block_ids=(2,),
            anchor_block_ids=(),
            semantic_block_ids=(0,),
            final_selected_block_ids=(0, 2),
            rail_block_ids=(2,),
            rail_block_count=1,
            semantic_block_count=1,
        ),
        confidence=ConfidenceTrace(
            semantic_candidate_block_ids=(0, 1, 3),
            semantic_included_count=1,
            raw_margin=0.1,
            normalized_margin=None,
            selected_mass=0.9,
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
        ),
    )


def test_query_key_inspection_groups_selected_and_missed_blocks() -> None:
    bundle = build_query_key_inspection(
        prompt_id="prompt-1",
        prompt_name="needle",
        prompt_text="Question: Where is TARGET evidence?",
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        rail_setting="reduced",
        selected_block_ids=(0, 2),
        metadata_blocks=tuple(_metadata(index) for index in range(4)),
        query_summary=SummaryEncoding(values=(1, 0, 0, 0), scale=0.5, summary_norm=1.0),
        trace=_trace(),
        block_inspections=(
            _inspection(0, selected=True, reason="semantic", text="TARGET selected", stage_a=0.9),
            _inspection(1, selected=False, reason="unselected", text="TARGET missed", stage_a=0.8),
            _inspection(2, selected=True, reason="recent", text="recent distractor", stage_a=0.7),
            _inspection(3, selected=False, reason="unselected", text="near miss", stage_a=0.6),
        ),
        relevant_text_fragments=("TARGET",),
        top_unselected_blocks=1,
    )

    assert bundle.selected_block_ids == (0, 2)
    assert bundle.comparison_groups.selected_relevant_block_ids == (0,)
    assert bundle.comparison_groups.selected_irrelevant_block_ids == (2,)
    assert bundle.comparison_groups.missed_relevant_block_ids == (1,)
    assert bundle.comparison_groups.high_scoring_near_miss_block_ids == (3,)
    assert bundle.query_summary_metadata.summary_dim == 4
    assert bundle.block_records[0].candidate_id == "s2_stride2_t0_2"
    assert bundle.block_records[0].block_size == 2
    assert bundle.block_records[0].block_summary_metadata.head_count == 2
    assert "semantic high-score" in bundle.block_records[0].explanation_hints
    assert "missed despite relevance" in bundle.block_records[1].explanation_hints
    assert "recent rail" in bundle.block_records[2].explanation_hints


def test_query_key_inspection_json_serialization_shape() -> None:
    bundle = build_query_key_inspection(
        prompt_id="prompt-2",
        prompt_name=None,
        prompt_text="A prompt without labels",
        representation_source="avg_mid4_hidden",
        representation_name="avg_mid4_hidden",
        rail_setting=None,
        selected_block_ids=(0,),
        metadata_blocks=(_metadata(0), _metadata(1)),
        query_summary=SummaryEncoding(values=(1, 0, 0, 0), scale=0.5, summary_norm=1.0),
        trace=_trace(),
        block_inspections=(
            _inspection(0, selected=True, reason="semantic", text="selected", stage_a=0.9),
            _inspection(1, selected=False, reason="unselected", text="unselected", stage_a=0.8),
        ),
        top_unselected_blocks=1,
    )

    payload = bundle.to_dict()
    encoded = json.dumps(payload)

    assert "query_summary_metadata" in payload
    assert payload["block_records"][0]["candidate_id"] == "s2_stride2_t0_2"
    assert payload["block_records"][0]["block_size"] == 2
    assert payload["block_records"][0]["labeled_relevant"] is None
    assert payload["high_scoring_near_miss_blocks"][0]["block_id"] == 1
    assert "prompt-2" in encoded
