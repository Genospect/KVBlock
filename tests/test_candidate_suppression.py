from __future__ import annotations

from kvblock.benchmark.candidate_suppression import (
    RankedCandidateSpan,
    span_interval_iou,
    span_overlap_fraction,
    suppress_ranked_candidates,
    suppression_modes_from_names,
)


def _candidate(
    block_id: int,
    start: int,
    end: int,
    score: float,
    rank: int,
) -> RankedCandidateSpan:
    return RankedCandidateSpan(
        block_id=block_id,
        candidate_id=f"c{block_id}",
        token_start=start,
        token_end=end,
        score=score,
        rank=rank,
        block_size=end - start,
        block_mode="test",
    )


def test_overlap_and_iou_helpers() -> None:
    left = _candidate(0, 0, 16, 1.0, 1)
    right = _candidate(1, 8, 24, 0.9, 2)

    assert span_overlap_fraction(left, right) == 0.5
    assert round(span_interval_iou(left, right), 3) == 0.333


def test_overlap_threshold_suppresses_lower_rank_duplicate() -> None:
    ranked = (
        _candidate(0, 0, 16, 1.0, 1),
        _candidate(1, 0, 24, 0.9, 2),
        _candidate(2, 32, 48, 0.8, 3),
    )

    result = suppress_ranked_candidates(
        ranked,
        mode="overlap_threshold",
        threshold=0.75,
    )

    assert result.survivor_block_ids == (0, 2)
    assert result.output_count == 2
    suppressed = result.decision_by_block_id[1]
    assert suppressed.survived is False
    assert suppressed.suppressed_by_block_id == 0
    assert suppressed.overlap_fraction == 1.0


def test_interval_iou_is_less_aggressive_than_overlap_fraction() -> None:
    ranked = (
        _candidate(0, 0, 16, 1.0, 1),
        _candidate(1, 0, 32, 0.9, 2),
    )

    result = suppress_ranked_candidates(
        ranked,
        mode="interval_iou",
        threshold=0.75,
    )

    assert result.survivor_block_ids == (0, 1)
    assert all(decision.survived for decision in result.decisions)


def test_cluster_suppression_keeps_highest_ranked_candidate() -> None:
    ranked = (
        _candidate(0, 0, 16, 1.0, 1),
        _candidate(1, 0, 24, 0.9, 2),
        _candidate(2, 48, 64, 0.8, 3),
    )

    result = suppress_ranked_candidates(
        ranked,
        mode="keep_highest_score_per_overlap_cluster",
        threshold=0.75,
    )

    assert result.survivor_block_ids == (0, 2)
    assert result.decision_by_block_id[1].suppressed_by_block_id == 0


def test_suppression_parser_validates_names() -> None:
    assert suppression_modes_from_names(("none", "interval_iou")) == (
        "none",
        "interval_iou",
    )
