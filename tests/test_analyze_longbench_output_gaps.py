from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_longbench_output_gaps import analyze_output_gaps


def _row(
    *,
    dataset: str = "hotpotqa",
    sample_id: str,
    answer_f1: float,
    answer_em: float,
    tokens: int,
    prediction: str | None = None,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "model": "fake/model",
        "longbench_length": 5000,
        "answer_f1": answer_f1,
        "answer_em": answer_em,
        "answer_precision": answer_f1,
        "answer_recall": answer_f1,
        "reconstructed_context_token_count": tokens,
        "selected_ids": (1, 2),
        "selected_spans": ("0:10", "10:20"),
        "gold_answers": ("gold",),
        "prediction": prediction or f"prediction-{sample_id}",
    }


def _payload(*, config: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "config": config,
        "rows": rows,
        "overall_summary": {},
        "dataset_summaries": (),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_analyze_output_gaps_counts_quadrants_and_sorts_details(
    tmp_path: Path,
) -> None:
    oracle = _write_json(
        tmp_path / "oracle.json",
        _payload(
            config={"context_policy": "answer_oracle"},
            rows=[
                _row(sample_id="a", answer_f1=1.0, answer_em=1.0, tokens=2000),
                _row(sample_id="b", answer_f1=0.0, answer_em=0.0, tokens=2000),
                _row(sample_id="c", answer_f1=1.0, answer_em=1.0, tokens=2000),
            ],
        ),
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        _payload(
            config={"output_policy": "length_aware_static"},
            rows=[
                _row(sample_id="a", answer_f1=0.0, answer_em=0.0, tokens=400),
                _row(sample_id="b", answer_f1=1.0, answer_em=1.0, tokens=400),
                _row(sample_id="c", answer_f1=1.0, answer_em=1.0, tokens=400),
            ],
        ),
    )
    full = _write_json(
        tmp_path / "full.json",
        _payload(
            config={"context_policy": "full_context"},
            rows=[
                _row(sample_id="a", answer_f1=1.0, answer_em=1.0, tokens=9000),
                _row(sample_id="b", answer_f1=1.0, answer_em=1.0, tokens=9000),
                _row(sample_id="c", answer_f1=0.0, answer_em=0.0, tokens=9000),
            ],
        ),
    )

    result = analyze_output_gaps(
        (
            ("hotpot_oracle", oracle),
            ("hotpot_lenaware", candidate),
            ("hotpot_full", full),
        )
    )

    summary = result.summary_rows[0]
    assert summary["candidate_label"] == "hotpot_lenaware"
    assert summary["dataset"] == "all"
    assert summary["row_count"] == 3
    assert summary["oracle_correct_candidate_wrong"] == 1
    assert summary["candidate_correct_oracle_wrong"] == 1
    assert summary["oracle_candidate_both_correct"] == 1
    assert summary["oracle_candidate_both_wrong"] == 0
    assert summary["full_correct_candidate_wrong"] == 1
    assert summary["candidate_correct_full_wrong"] == 1
    assert summary["full_correct_oracle_wrong"] == 1
    assert summary["mean_oracle_minus_candidate_f1"] == pytest.approx(0.0)
    assert summary["mean_candidate_reconstructed_context_tokens"] == pytest.approx(400)
    assert summary["candidate_touched_oracle_span"] == 3
    assert summary["candidate_missed_oracle_span"] == 0
    assert summary["mean_candidate_oracle_span_overlap_fraction"] == pytest.approx(1.0)

    assert result.detail_rows[0]["sample_id"] == "a"
    assert result.detail_rows[0]["candidate_label"] == "hotpot_lenaware"
    assert (
        result.detail_rows[0]["oracle_candidate_category"]
        == "oracle_correct_candidate_wrong"
    )
    assert result.detail_rows[0]["candidate_selected_ids"] == "1,2"
    assert result.detail_rows[0]["candidate_oracle_span_overlap_kind"] == (
        "touches_oracle_span"
    )


def test_analyze_output_gaps_requires_oracle_and_candidate(tmp_path: Path) -> None:
    full = _write_json(
        tmp_path / "full.json",
        _payload(
            config={"context_policy": "full_context"},
            rows=[_row(sample_id="a", answer_f1=1.0, answer_em=1.0, tokens=1000)],
        ),
    )

    with pytest.raises(ValueError, match="oracle"):
        analyze_output_gaps((("hotpot_full", full),))


def test_analyze_output_gaps_supports_multiple_candidate_runs(
    tmp_path: Path,
) -> None:
    oracle = _write_json(
        tmp_path / "oracle.json",
        _payload(
            config={"context_policy": "answer_oracle"},
            rows=[_row(sample_id="a", answer_f1=1.0, answer_em=1.0, tokens=2000)],
        ),
    )
    fallback_candidate = _write_json(
        tmp_path / "fallback.json",
        _payload(
            config={"block_modes": ["mixed_global_refine_40_16_stride_8"]},
            rows=[_row(sample_id="a", answer_f1=0.5, answer_em=0.0, tokens=500)],
        ),
    )
    forced_candidate = _write_json(
        tmp_path / "forced.json",
        _payload(
            config={"block_modes": ["mixed_global_refine_40_16_stride_8"]},
            rows=[_row(sample_id="a", answer_f1=0.0, answer_em=0.0, tokens=400)],
        ),
    )

    result = analyze_output_gaps(
        (
            ("hotpot_oracle", oracle),
            ("hotpot48_fb_m12", fallback_candidate),
            ("hotpot48_forced_m24", forced_candidate),
        )
    )

    summary_by_candidate = {
        str(row["candidate_label"]): row
        for row in result.summary_rows
        if row["dataset"] == "all"
    }
    assert set(summary_by_candidate) == {"hotpot48_fb_m12", "hotpot48_forced_m24"}
    assert summary_by_candidate["hotpot48_fb_m12"][
        "mean_oracle_minus_candidate_f1"
    ] == pytest.approx(0.5)
    assert summary_by_candidate["hotpot48_forced_m24"][
        "mean_oracle_minus_candidate_f1"
    ] == pytest.approx(1.0)
