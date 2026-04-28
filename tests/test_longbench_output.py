from __future__ import annotations

import pytest

from kvblock.benchmark.longbench_output import (
    LongBenchOutputRunRow,
    build_output_summaries,
    extract_longbench_question,
    format_selected_context_prompt,
)


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
        selected_block_count=2,
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
    assert summaries[0].mean_selected_token_fraction == pytest.approx(0.15)
    assert summaries[0].mixed_fallback_count == 1
    assert summaries[0].mixed_fallback_rate == 0.5


def test_prompt_helpers_extract_question_and_format_selected_context() -> None:
    prompt = "DATASET: hotpotqa\n\nCONTEXT:\nalpha\n\nINPUT:\nWho won?"

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
