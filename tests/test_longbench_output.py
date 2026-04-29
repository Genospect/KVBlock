from __future__ import annotations

from types import SimpleNamespace

import pytest

from kvblock.benchmark.longbench_output import (
    LongBenchOutputRunRow,
    answer_oracle_context_from_prompt,
    build_output_summaries,
    extract_longbench_context,
    extract_longbench_question,
    filter_output_selection,
    format_selected_context_prompt,
    full_context_from_prompt,
    passage_window_context_from_spans,
    resolve_output_policy_settings,
)
from kvblock.benchmark.longbench_adapter import parse_length_bucket


def _row(
    *,
    dataset: str,
    answer_f1: float,
    answer_em: float,
    selected_token_fraction: float,
    mixed_fallback_used: bool,
) -> LongBenchOutputRunRow:
    return LongBenchOutputRunRow(
        dataset=dataset,
        sample_id=f"{dataset}-1",
        model="fake/model",
        block_mode="fixed_40",
        prompt_tokens=100,
        longbench_length=100,
        selected_token_count=int(selected_token_fraction * 100),
        selected_token_fraction=selected_token_fraction,
        reconstructed_context_token_count=int(selected_token_fraction * 100),
        reconstructed_context_token_fraction=selected_token_fraction,
        context_reconstruction="selected_spans",
        selected_block_count=2,
        selection_filter_dropped_count=0,
        mixed_fallback_used=mixed_fallback_used,
        selector_latency_sec=0.01,
        selector_total_latency_sec=0.02,
        generation_latency_sec=0.03,
        total_latency_sec=0.05,
        gold_answers=("gold",),
        prediction="prediction",
        answer_em=answer_em,
        answer_f1=answer_f1,
        answer_precision=answer_f1,
        answer_recall=answer_f1,
        selector_recall=0.5,
        selector_precision=0.25,
        evidence_window_recall=1.0,
        evidence_window_precision=0.5,
        expected_parent_recall=1.0,
        selected_ids=(1, 2),
        selected_spans=("0:10", "10:20"),
    )


def test_output_summaries_aggregate_answer_and_fallback_metrics() -> None:
    summaries = build_output_summaries(
        (
            _row(
                dataset="hotpotqa",
                answer_f1=1.0,
                answer_em=1.0,
                selected_token_fraction=0.1,
                mixed_fallback_used=False,
            ),
            _row(
                dataset="hotpotqa",
                answer_f1=0.0,
                answer_em=0.0,
                selected_token_fraction=0.2,
                mixed_fallback_used=True,
            ),
        )
    )

    assert len(summaries) == 1
    assert summaries[0].dataset == "hotpotqa"
    assert summaries[0].row_count == 2
    assert summaries[0].mean_answer_f1 == 0.5
    assert summaries[0].mean_answer_em == 0.5
    assert summaries[0].mean_selected_block_count == pytest.approx(2.0)
    assert summaries[0].mean_selection_filter_dropped_count == pytest.approx(0.0)
    assert summaries[0].mean_selected_token_fraction == pytest.approx(0.15)
    assert summaries[0].mean_reconstructed_context_token_fraction == pytest.approx(
        0.15
    )
    assert summaries[0].mean_reconstructed_context_tokens == pytest.approx(15.0)
    assert summaries[0].mixed_fallback_count == 1
    assert summaries[0].mixed_fallback_rate == 0.5


def test_prompt_helpers_extract_question_and_format_selected_context() -> None:
    prompt = "DATASET: hotpotqa\n\nCONTEXT:\nalpha\n\nINPUT:\nWho won?"

    assert extract_longbench_context(prompt) == "alpha"
    assert extract_longbench_question(prompt) == "Who won?"
    assert (
        format_selected_context_prompt(question="Who won?", selected_context="alpha")
        == "Answer the question using only the provided context. Keep the answer short.\n"
        "\n"
        "Context:\n"
        "alpha\n"
        "\n"
        "Question:\n"
        "Who won?\n"
        "\n"
        "Answer:"
    )


def test_full_context_from_prompt_extracts_original_context() -> None:
    prompt = "DATASET: hotpotqa\n\nCONTEXT:\nalpha beta gamma\n\nINPUT:\nWho won?"

    reconstructed = full_context_from_prompt(
        WhitespaceRuntime(),  # type: ignore[arg-type]
        prompt_text=prompt,
    )

    assert reconstructed.text == "alpha beta gamma"
    assert reconstructed.token_count == 3


def test_answer_oracle_context_keeps_literal_answer_passage() -> None:
    prompt = (
        "DATASET: hotpotqa\n\n"
        "CONTEXT:\n"
        "Passage 1: Alpha says the answer is elsewhere.\n"
        "Passage 2: Beta says Needle Answer was the winner.\n"
        "\nINPUT:\n"
        "Who won?"
    )

    reconstructed = answer_oracle_context_from_prompt(
        WhitespaceRuntime(),  # type: ignore[arg-type]
        prompt_text=prompt,
        answers=("Needle Answer",),
    )

    assert "Passage 2:" in reconstructed.text
    assert "Passage 1:" not in reconstructed.text
    assert reconstructed.token_count == 9


def test_answer_oracle_context_falls_back_for_yes_no_answers() -> None:
    prompt = (
        "DATASET: hotpotqa\n\n"
        "CONTEXT:\n"
        "Passage 1: Alpha says a mine is in Canada.\n"
        "Passage 2: Beta says another mine is also in Canada.\n"
        "\nINPUT:\n"
        "Are both mines in Canada?"
    )

    reconstructed = answer_oracle_context_from_prompt(
        WhitespaceRuntime(),  # type: ignore[arg-type]
        prompt_text=prompt,
        answers=("yes",),
    )

    assert "Passage 1:" in reconstructed.text
    assert "Passage 2:" in reconstructed.text


def test_answer_oracle_context_uses_sentence_fallback_without_passage_markers() -> None:
    prompt = (
        "DATASET: hotpotqa\n\n"
        "CONTEXT:\n"
        "Alpha is irrelevant. Needle Answer was the winner. Gamma is irrelevant.\n"
        "\nINPUT:\n"
        "Who won?"
    )

    reconstructed = answer_oracle_context_from_prompt(
        WhitespaceRuntime(),  # type: ignore[arg-type]
        prompt_text=prompt,
        answers=("Needle Answer",),
    )

    assert reconstructed.text == "Needle Answer was the winner."
    assert reconstructed.token_count == 5


class WhitespaceRuntime:
    def tokenize(self, prompt: str) -> SimpleNamespace:
        token_ids = tuple(prompt.split())
        return SimpleNamespace(
            token_ids=token_ids,
            token_count=len(token_ids),
        )

    def decode_token_ids(self, token_ids: tuple[str, ...]) -> str:
        return " ".join(token_ids)


def test_passage_window_context_preserves_selected_passage_header_and_order() -> None:
    prompt = (
        "DATASET: hotpotqa\n\n"
        "CONTEXT:\n"
        "Passage 1: Alpha Title alpha0 alpha1 alpha2 alpha3\n"
        "Passage 2: Beta Title beta0 beta1 beta2 beta3 beta4 beta5\n"
        "\nINPUT:\n"
        "Who?"
    )
    token_ids = prompt.split()
    selected_start = token_ids.index("beta3")
    selected_span = f"{selected_start}:{selected_start + 1}"

    reconstructed = passage_window_context_from_spans(
        WhitespaceRuntime(),  # type: ignore[arg-type]
        prompt_text=prompt,
        selected_spans=(selected_span,),
        passage_window_tokens=5,
        passage_header_tokens=4,
    )

    assert "Passage 2: Beta Title" in reconstructed.text
    assert "beta3" in reconstructed.text
    assert "Passage 1:" not in reconstructed.text
    assert reconstructed.token_count >= 5


def test_filter_output_selection_keeps_min_then_stops_on_score_ratio() -> None:
    filtered = filter_output_selection(
        selected_block_ids=(1, 2, 3, 4),
        selected_spans=("0:10", "10:20", "20:30", "30:40"),
        selected_blocks=(
            {"block_id": 1, "final_score": 1.0},
            {"block_id": 2, "final_score": 0.7},
            {"block_id": 3, "final_score": 0.3},
            {"block_id": 4, "final_score": 0.2},
        ),
        selection_min_blocks=2,
        selection_score_ratio=0.5,
    )

    assert filtered.block_ids == (1, 2)
    assert filtered.spans == ("0:10", "10:20")
    assert filtered.dropped_count == 2


def test_filter_output_selection_is_disabled_without_score_ratio() -> None:
    filtered = filter_output_selection(
        selected_block_ids=(1, 2),
        selected_spans=("0:10", "10:20"),
        selected_blocks=(
            {"block_id": 1, "final_score": 1.0},
            {"block_id": 2, "final_score": 0.1},
        ),
        selection_min_blocks=1,
        selection_score_ratio=None,
    )

    assert filtered.block_ids == (1, 2)
    assert filtered.dropped_count == 0


def test_filter_output_selection_caps_children_per_parent_with_backfill() -> None:
    filtered = filter_output_selection(
        selected_block_ids=(1, 2, 3, 4, 5),
        selected_spans=("0:10", "10:20", "20:30", "30:40", "40:50"),
        selected_blocks=(
            {
                "block_id": 1,
                "candidate_role": "child",
                "parent_candidate_id": "parent-a",
            },
            {
                "block_id": 2,
                "candidate_role": "child",
                "parent_candidate_id": "parent-a",
            },
            {
                "block_id": 3,
                "candidate_role": "child",
                "parent_candidate_id": "parent-a",
            },
            {
                "block_id": 4,
                "candidate_role": "child",
                "parent_candidate_id": "parent-b",
            },
            {
                "block_id": 5,
                "candidate_role": "parent",
                "candidate_id": "parent-c",
            },
        ),
        selection_max_total_blocks=4,
        selection_max_children_per_parent=2,
    )

    assert filtered.block_ids == (1, 2, 4, 5)
    assert filtered.spans == ("0:10", "10:20", "30:40", "40:50")
    assert filtered.dropped_count == 1


def test_filter_output_selection_caps_total_blocks() -> None:
    filtered = filter_output_selection(
        selected_block_ids=(1, 2, 3),
        selected_spans=("0:10", "10:20", "20:30"),
        selected_blocks=(
            {"block_id": 1, "candidate_role": "parent"},
            {"block_id": 2, "candidate_role": "parent"},
            {"block_id": 3, "candidate_role": "parent"},
        ),
        selection_max_total_blocks=2,
    )

    assert filtered.block_ids == (1, 2)
    assert filtered.dropped_count == 1


def test_manual_output_policy_preserves_explicit_settings() -> None:
    resolved = resolve_output_policy_settings(
        output_policy="manual",
        dataset_names=("hotpotqa",),
        length_bucket=parse_length_bucket("4k-8k"),
        max_selected_blocks=17,
        context_reconstruction="selected_spans",
        passage_window_tokens=120,
    )

    assert resolved.name == "manual"
    assert resolved.max_selected_blocks == 17
    assert resolved.context_reconstruction == "selected_spans"
    assert resolved.passage_window_tokens == 120


@pytest.mark.parametrize(
    ("dataset", "length_bucket", "expected_budget"),
    (
        ("hotpotqa", "0-4k", 20),
        ("hotpotqa", "4k-8k", 12),
        ("musique", "4k-8k", 8),
    ),
)
def test_length_aware_static_output_policy_sets_empirical_budget_and_window(
    dataset: str,
    length_bucket: str,
    expected_budget: int,
) -> None:
    resolved = resolve_output_policy_settings(
        output_policy="length_aware_static",
        dataset_names=(dataset,),
        length_bucket=parse_length_bucket(length_bucket),
        max_selected_blocks=None,
        context_reconstruction="selected_spans",
        passage_window_tokens=120,
    )

    assert resolved.name == "length_aware_static"
    assert resolved.max_selected_blocks == expected_budget
    assert resolved.context_reconstruction == "passage_window"
    assert resolved.passage_window_tokens == 64


def test_length_aware_static_requires_single_budget_per_run() -> None:
    with pytest.raises(ValueError, match="run those datasets separately"):
        resolve_output_policy_settings(
            output_policy="length_aware_static",
            dataset_names=("hotpotqa", "musique"),
            length_bucket=parse_length_bucket("4k-8k"),
            max_selected_blocks=None,
            context_reconstruction="selected_spans",
            passage_window_tokens=120,
        )
