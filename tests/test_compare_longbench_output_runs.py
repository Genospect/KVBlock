from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compare_longbench_output_runs import compare_output_runs


def _row(
    *,
    dataset: str,
    sample_id: str,
    longbench_length: int,
    answer_f1: float,
    selected_tokens: int,
    mixed_fallback_used: bool = False,
) -> dict[str, object]:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "model": "fake/model",
        "longbench_length": longbench_length,
        "answer_em": answer_f1,
        "answer_f1": answer_f1,
        "answer_precision": answer_f1,
        "answer_recall": answer_f1,
        "selected_block_count": 1,
        "selection_filter_dropped_count": 0,
        "selected_token_fraction": selected_tokens / 1000,
        "selected_token_count": selected_tokens,
        "reconstructed_context_token_fraction": selected_tokens / 1000,
        "reconstructed_context_token_count": selected_tokens,
        "selector_latency_sec": 0.01,
        "generation_latency_sec": 0.02,
        "mixed_fallback_used": mixed_fallback_used,
    }


def _payload(*, config: dict[str, object], rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "config": config,
        "rows": rows,
        "overall_summary": {"row_count": len(rows), "mean_answer_f1": 0.0},
        "dataset_summaries": (),
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_quality_guarded_static_uses_fixed_for_long_hotpot_only(
    tmp_path: Path,
) -> None:
    fixed = _write_json(
        tmp_path / "fixed.json",
        _payload(
            config={"block_modes": ["fixed_40"]},
            rows=[
                _row(
                    dataset="hotpotqa",
                    sample_id="hotpot-short",
                    longbench_length=3000,
                    answer_f1=0.2,
                    selected_tokens=600,
                ),
                _row(
                    dataset="hotpotqa",
                    sample_id="hotpot-long",
                    longbench_length=5000,
                    answer_f1=0.9,
                    selected_tokens=620,
                ),
                _row(
                    dataset="musique",
                    sample_id="musique-long",
                    longbench_length=5000,
                    answer_f1=0.3,
                    selected_tokens=640,
                ),
            ],
        ),
    )
    length_aware = _write_json(
        tmp_path / "length_aware.json",
        _payload(
            config={"output_policy": "length_aware_static"},
            rows=[
                _row(
                    dataset="hotpotqa",
                    sample_id="hotpot-short",
                    longbench_length=3000,
                    answer_f1=0.8,
                    selected_tokens=400,
                ),
                _row(
                    dataset="hotpotqa",
                    sample_id="hotpot-long",
                    longbench_length=5000,
                    answer_f1=0.1,
                    selected_tokens=320,
                ),
                _row(
                    dataset="musique",
                    sample_id="musique-long",
                    longbench_length=5000,
                    answer_f1=0.7,
                    selected_tokens=220,
                ),
            ],
        ),
    )

    rows = compare_output_runs(
        (("fixed40", fixed), ("lenaware", length_aware)),
        scope="both",
        hybrid_policy="quality_guarded_static",
    )

    hybrid_all = next(
        row
        for row in rows
        if row["run_label"] == "quality_guarded_static" and row["dataset"] == "all"
    )
    assert hybrid_all["row_count"] == 3
    assert hybrid_all["mean_answer_f1"] == pytest.approx((0.8 + 0.9 + 0.7) / 3)
    assert hybrid_all["mean_selected_tokens"] == pytest.approx((400 + 620 + 220) / 3)
    assert hybrid_all["mean_answer_f1_per_1k_recon_tokens"] == pytest.approx(
        ((0.8 + 0.9 + 0.7) / 3) / (((400 + 620 + 220) / 3) / 1000)
    )

    hybrid_hotpot = next(
        row
        for row in rows
        if row["run_label"] == "quality_guarded_static"
        and row["dataset"] == "hotpotqa"
    )
    assert hybrid_hotpot["row_count"] == 2
    assert hybrid_hotpot["mean_answer_f1"] == pytest.approx((0.8 + 0.9) / 2)


def test_compare_output_runs_adds_utility_columns_to_existing_summaries(
    tmp_path: Path,
) -> None:
    payload = {
        "config": {"block_modes": ["fixed_40"]},
        "rows": [],
        "overall_summary": {
            "row_count": 2,
            "mean_answer_f1": 0.5,
            "mean_answer_em": 0.25,
            "mean_reconstructed_context_tokens": 500.0,
        },
        "dataset_summaries": (),
    }
    path = _write_json(tmp_path / "fixed.json", payload)

    rows = compare_output_runs((("fixed40", path),), scope="all")

    assert rows[0]["mean_answer_f1_per_1k_recon_tokens"] == pytest.approx(1.0)
    assert rows[0]["mean_answer_em_per_1k_recon_tokens"] == pytest.approx(0.5)


def test_quality_guarded_static_requires_length_aware_source(
    tmp_path: Path,
) -> None:
    fixed = _write_json(
        tmp_path / "fixed.json",
        _payload(
            config={"block_modes": ["fixed_40"]},
            rows=[
                _row(
                    dataset="hotpotqa",
                    sample_id="hotpot-long",
                    longbench_length=5000,
                    answer_f1=0.9,
                    selected_tokens=620,
                )
            ],
        ),
    )

    with pytest.raises(ValueError, match="length_aware_static input run"):
        compare_output_runs(
            (("fixed40", fixed),),
            scope="all",
            hybrid_policy="quality_guarded_static",
        )
