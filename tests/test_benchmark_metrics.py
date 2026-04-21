from kvblock.benchmark.metrics import (
    aggregate_fallback_frequency,
    average_selection_split,
    logical_selection_metrics,
    selector_quality_metrics,
    summarize_confidence,
)
from kvblock.benchmark.selector_microbench import SelectorMicrobenchRow
from kvblock.selector.oracle import compare_block_sets, dense_reference_block_set, sparse_selected_block_set


def _row(
    *,
    mode: str,
    semantic: int,
    rail: int,
    total: int,
    raw_margin: float,
    normalized_margin: float | None,
    selected_mass: float | None,
    normalized_mass: float | None,
    is_confident: bool,
) -> SelectorMicrobenchRow:
    return SelectorMicrobenchRow(
        case_id="case",
        query_index=0,
        step_id="case:0",
        seed=0,
        population_profile="default",
        num_blocks=8,
        block_size=32,
        summary_dim=32,
        shortlist_size=4,
        semantic_top_k=2,
        keep_recent_blocks=1,
        keep_anchor_blocks=1,
        confidence_margin=0.05,
        normalized_margin_threshold=None,
        min_normalized_mass=None,
        widen_top_k_by=1,
        add_recent_blocks_by=1,
        selector_latency_sec=0.001,
        stage_a_candidate_count=8,
        stage_a_shortlist_size=4,
        stage_b_refinement_count=4,
        final_selected_block_count=total,
        semantic_selected_block_count=semantic,
        rail_preserved_block_count=rail,
        fallback_mode=mode,
        fallback_action=mode,
        fallback_reason_code="reason",
        raw_margin=raw_margin,
        normalized_margin=normalized_margin,
        selected_mass=selected_mass,
        normalized_mass=normalized_mass,
        is_confident=is_confident,
        trace_size_bytes=100,
    )


def test_logical_bytes_per_token_calculation_from_page_counts() -> None:
    metrics = logical_selection_metrics(
        selected_page_count=3,
        dense_page_count=12,
        page_bytes=2048,
        output_tokens=2,
    )

    assert metrics.selected_pages_per_token == 1.5
    assert metrics.logical_kv_read_bytes_per_token == 3072.0


def test_alpha_estimation() -> None:
    metrics = logical_selection_metrics(
        selected_page_count=3,
        dense_page_count=12,
        page_bytes=1024,
        output_tokens=1,
    )

    assert metrics.estimated_alpha == 0.25


def test_selector_quality_from_overlap_comparison() -> None:
    comparison = compare_block_sets(
        dense_reference_block_set([0, 1, 2, 3]),
        sparse_selected_block_set([1, 3, 4]),
    )

    metrics = selector_quality_metrics(comparison)

    assert metrics.selector_recall_rate == 0.5
    assert metrics.selector_precision_rate == 2 / 3
    assert metrics.missed_important_count == 2
    assert metrics.extra_selected_count == 1


def test_fallback_frequency_aggregation() -> None:
    rows = [
        _row(mode="sparse", semantic=2, rail=1, total=3, raw_margin=0.2, normalized_margin=0.2, selected_mass=1.0, normalized_mass=0.8, is_confident=True),
        _row(mode="widen_k", semantic=3, rail=1, total=4, raw_margin=0.01, normalized_margin=0.02, selected_mass=0.8, normalized_mass=0.6, is_confident=False),
        _row(mode="dense", semantic=2, rail=2, total=4, raw_margin=0.0, normalized_margin=None, selected_mass=0.7, normalized_mass=None, is_confident=False),
    ]

    summary = aggregate_fallback_frequency(rows)

    assert summary.total_runs == 3
    assert summary.counts["sparse"] == 1
    assert summary.counts["widen_k"] == 1
    assert summary.counts["dense"] == 1


def test_confidence_aggregation_helpers() -> None:
    rows = [
        _row(mode="sparse", semantic=2, rail=1, total=3, raw_margin=0.2, normalized_margin=0.2, selected_mass=1.0, normalized_mass=0.8, is_confident=True),
        _row(mode="widen_k", semantic=3, rail=1, total=4, raw_margin=0.0, normalized_margin=None, selected_mass=0.6, normalized_mass=None, is_confident=False),
    ]

    split = average_selection_split(rows)
    confidence = summarize_confidence(rows)

    assert split.avg_semantic_selected_blocks == 2.5
    assert split.avg_rail_preserved_blocks == 1.0
    assert confidence.avg_raw_margin == 0.1
    assert confidence.avg_selected_mass == 0.8
    assert confidence.confident_fraction == 0.5
