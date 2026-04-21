from __future__ import annotations

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.runtime.head_diagnostics import (
    HeadDiagnosticBlockContext,
    build_per_head_block_diagnostics,
    summarize_head_diagnostics,
    summarize_head_diagnostics_by_prompt,
)
from kvblock.selector.policies import StageAPolicy
from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding


def _query_summary() -> MultiHeadQuerySummary:
    return MultiHeadQuerySummary(
        pooled=SummaryEncoding(values=(1, 1), scale=1.0, summary_norm=2**0.5),
        per_head=(
            SummaryEncoding(values=(1, 0), scale=1.0, summary_norm=1.0),
            SummaryEncoding(values=(0, 1), scale=1.0, summary_norm=1.0),
        ),
    )


def _metadata(block_id: int, heads: tuple[tuple[int, ...], ...]) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 2,
        token_len=2,
        summary_fp8=(1, 1),
        summary_scale=1.0,
        per_head_summary_fp8=heads,
        per_head_summary_scale=(1.0,) * len(heads),
        per_head_summary_norm=(1.0,) * len(heads),
    )


def _context(
    block_id: int,
    *,
    selected: bool,
    reason: str,
    prompt_name: str | None = None,
) -> HeadDiagnosticBlockContext:
    return HeadDiagnosticBlockContext(
        block_id=block_id,
        selected=selected,
        selected_reason=reason,
        token_start=block_id * 2,
        token_end=block_id * 2 + 2,
        token_count=2,
        preview_text=f"block {block_id}",
    )


def test_per_head_diagnostic_extraction_shape_and_ranking() -> None:
    diagnostics = build_per_head_block_diagnostics(
        metadata_blocks=(
            _metadata(0, ((8, 0), (0, -8))),
            _metadata(1, ((-8, 0), (0, 8))),
        ),
        query_summary=_query_summary(),
        block_contexts=(
            _context(0, selected=True, reason="semantic"),
            _context(1, selected=False, reason="unselected"),
        ),
        policy=StageAPolicy(head_scoring_mode="max_head_score"),
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        top_heads=2,
        rail_setting="no_rails",
        prompt_name="prompt",
    )

    assert len(diagnostics) == 2
    assert diagnostics[0].head_scores == (1.0, 0.0)
    assert diagnostics[0].top_contributing_heads[0].head_index == 0
    assert diagnostics[1].top_contributing_heads[0].head_index == 1
    assert diagnostics[0].selected is True
    assert diagnostics[0].representation_source == "query_mean_last_layer"


def test_per_head_diagnostic_serialization_and_aggregate() -> None:
    diagnostics = build_per_head_block_diagnostics(
        metadata_blocks=(
            _metadata(0, ((8, 0), (0, -8))),
            _metadata(1, ((-8, 0), (0, 8))),
        ),
        query_summary=_query_summary(),
        block_contexts=(
            _context(0, selected=True, reason="semantic"),
            _context(1, selected=True, reason="semantic"),
        ),
        policy=StageAPolicy(head_scoring_mode="max_head_score"),
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        top_heads=1,
        prompt_name="prompt",
    )

    summary = summarize_head_diagnostics(
        diagnostics,
        expected_block_ids=(1,),
        prompt_name="prompt",
    )
    payload = diagnostics[0].to_dict()

    assert payload["top_contributing_heads"][0]["head_index"] == 0
    assert summary.selected_count == 2
    assert summary.correct_selected_count == 1
    assert summary.selected_top_head_counts[0].count == 1
    assert summary.to_dict()["correct_selected_count"] == 1


def test_head_diagnostics_can_group_by_prompt() -> None:
    first = build_per_head_block_diagnostics(
        metadata_blocks=(_metadata(0, ((8, 0), (0, -8))),),
        query_summary=_query_summary(),
        block_contexts=(_context(0, selected=True, reason="semantic"),),
        policy=StageAPolicy(head_scoring_mode="max_head_score"),
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        prompt_name="a",
    )
    second = build_per_head_block_diagnostics(
        metadata_blocks=(_metadata(1, ((-8, 0), (0, 8))),),
        query_summary=_query_summary(),
        block_contexts=(_context(1, selected=True, reason="semantic"),),
        policy=StageAPolicy(head_scoring_mode="max_head_score"),
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        prompt_name="b",
    )

    summaries = summarize_head_diagnostics_by_prompt(first + second)

    assert [summary.prompt_name for summary in summaries] == ["a", "b"]
    assert summaries[0].selected_top_head_counts[0].head_index == 0
    assert summaries[1].selected_top_head_counts[0].head_index == 1


def test_head_diagnostics_return_empty_for_non_multi_head_query() -> None:
    diagnostics = build_per_head_block_diagnostics(
        metadata_blocks=(_metadata(0, ((8, 0), (0, -8))),),
        query_summary=SummaryEncoding(values=(1, 1), scale=1.0, summary_norm=2**0.5),
        block_contexts=(_context(0, selected=True, reason="semantic"),),
        policy=StageAPolicy(),
        representation_source="avg_mid4_hidden",
        representation_name="avg_mid4_hidden",
    )

    assert diagnostics == ()
