from __future__ import annotations

import pytest

from kvblock.benchmark.answer_metrics import (
    exact_match_score,
    f1_score,
    metric_max_over_ground_truths,
    normalize_answer,
    score_qa_answer,
)


def test_normalize_answer_uses_hotpotqa_style_processing() -> None:
    assert normalize_answer(" The, Quick   Answer! ") == "quick answer"
    assert normalize_answer("An Example of a Result.") == "example of result"


def test_exact_match_score_compares_normalized_answers() -> None:
    assert exact_match_score("The Eiffel Tower.", "eiffel tower") == 1.0
    assert exact_match_score("Paris", "London") == 0.0


def test_f1_score_counts_token_overlap_with_duplicates() -> None:
    f1, precision, recall = f1_score("red red blue", "red blue blue")

    assert f1 == pytest.approx(2 / 3)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)


def test_f1_score_handles_yes_no_noanswer_mismatches() -> None:
    assert f1_score("yes", "no") == (0.0, 0.0, 0.0)
    assert f1_score("yes", "yes") == (1.0, 1.0, 1.0)


def test_metric_max_over_ground_truths_selects_best_answer() -> None:
    assert (
        metric_max_over_ground_truths(
            "William Shakespeare",
            ("Christopher Marlowe", "Shakespeare"),
            exact_match_score,
        )
        == 0.0
    )
    assert metric_max_over_ground_truths(
        "William Shakespeare",
        ("Christopher Marlowe", "William Shakespeare"),
        exact_match_score,
    ) == 1.0


def test_score_qa_answer_returns_best_em_and_f1() -> None:
    score = score_qa_answer("William Shakespeare", ("Shakespeare", "William Shakespeare"))

    assert score["em"] == 1.0
    assert score["f1"] == 1.0
    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
