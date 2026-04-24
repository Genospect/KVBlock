from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kvblock.benchmark.longbench_comparison import (
    compare_longbench_runs,
    format_comparison_markdown,
    parse_run_inputs,
)


def _write_payload(path: Path, rows: list[dict]) -> None:
    payload = {
        "dataset_repo": "THUDM/LongBench",
        "split": "test",
        "length_bucket": {"name": "0-4k", "min_length": 0, "max_length": 4000},
        "evidence_window_radius": 2,
        "refine_score_mode": "softmax_mass",
        "stage_c_policy": "semantic_refined_mix",
        "exclude_scaffold_blocks": True,
        "oracle_mode": "none",
        "rows": rows,
        "dataset_summaries": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _row(
    *,
    dataset_name: str,
    recall: float,
    precision: float,
    window_recall: float,
    selected_spans: list[str],
    selector_latency_sec: float,
    representation_source: str = "query_only_last_layer",
    block_mode: str = "fixed_40",
    scoreable: bool = True,
    mixed_fallback_used: bool | None = None,
    mixed_fallback_margin: float | None = None,
    mixed_max_children_per_parent: int | None = None,
    mixed_child_window_radius: int | None = None,
    expected_parent_recall: float | None = None,
    child_rank_miss_count: int | None = None,
    parent_miss_count: int | None = None,
) -> dict:
    row = {
        "dataset_name": dataset_name,
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "representation_source": representation_source,
        "qk_aggregation_strategy": "block_max",
        "block_mode": block_mode,
        "rerank_mode": "dense_qk_token_refine",
        "refine_top_n_tokens": 4,
        "refine_score_mode": "softmax_mass",
        "stage_c_policy": "semantic_refined_mix",
        "halo_radius": 1,
        "max_selected_blocks": 16,
        "evidence_window_radius": 2,
        "scoreable_by_answer_presence": scoreable,
        "answer_presence_rate": 1.0 if scoreable else 0.0,
        "expected_block_count": 4,
        "target_recall": recall,
        "selected_precision": precision,
        "evidence_window_recall": window_recall,
        "evidence_window_precision": 0.5,
        "selected_count": len(selected_spans),
        "selected_to_semantic_k_ratio": len(selected_spans) / 8,
        "selected_spans": selected_spans,
        "tokens": 100,
        "candidate_block_count": 50,
        "selector_latency_sec": selector_latency_sec,
        "scaffold_excluded_count": 1,
    }
    if mixed_fallback_used is not None:
        row["mixed_fallback_used"] = mixed_fallback_used
    if mixed_fallback_margin is not None:
        row["mixed_fallback_margin"] = mixed_fallback_margin
    if mixed_max_children_per_parent is not None:
        row["mixed_max_children_per_parent"] = mixed_max_children_per_parent
    if mixed_child_window_radius is not None:
        row["mixed_child_window_radius"] = mixed_child_window_radius
    if expected_parent_recall is not None:
        row["expected_parent_recall"] = expected_parent_recall
    if child_rank_miss_count is not None:
        row["child_rank_miss_count"] = child_rank_miss_count
    if parent_miss_count is not None:
        row["expected_parent_miss_count"] = parent_miss_count
    return row


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "compare_longbench_runs.py"
    )
    spec = importlib.util.spec_from_file_location("compare_longbench_runs", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_longbench_runs_applies_named_control_deltas(tmp_path: Path) -> None:
    control_path = tmp_path / "control.json"
    experiment_path = tmp_path / "experiment.json"
    _write_payload(
        control_path,
        [
            _row(
                dataset_name="hotpotqa",
                recall=0.6,
                precision=0.1,
                window_recall=0.6,
                selected_spans=["0:40", "40:80"],
                selector_latency_sec=0.004,
            ),
            _row(
                dataset_name="musique",
                recall=1.0,
                precision=0.05,
                window_recall=1.0,
                selected_spans=["0:40", "40:80"],
                selector_latency_sec=0.006,
            ),
            _row(
                dataset_name="hotpotqa",
                recall=0.0,
                precision=0.0,
                window_recall=0.0,
                selected_spans=["0:40"],
                selector_latency_sec=0.004,
                scoreable=False,
            ),
        ],
    )
    _write_payload(
        experiment_path,
        [
            _row(
                dataset_name="hotpotqa",
                recall=0.8,
                precision=0.2,
                window_recall=0.7,
                selected_spans=["0:40"],
                selector_latency_sec=0.005,
                representation_source="query_mean_last_layer",
                mixed_fallback_used=True,
                mixed_fallback_margin=0.05,
                mixed_max_children_per_parent=1,
                mixed_child_window_radius=1,
                expected_parent_recall=0.9,
                child_rank_miss_count=1,
                parent_miss_count=0,
            ),
            _row(
                dataset_name="musique",
                recall=1.0,
                precision=0.1,
                window_recall=1.0,
                selected_spans=["0:40"],
                selector_latency_sec=0.007,
                representation_source="query_mean_last_layer",
                mixed_fallback_used=False,
                mixed_fallback_margin=0.05,
                mixed_max_children_per_parent=1,
                mixed_child_window_radius=1,
                expected_parent_recall=1.0,
                child_rank_miss_count=0,
                parent_miss_count=0,
            ),
            _row(
                dataset_name="hotpotqa",
                recall=0.0,
                precision=0.0,
                window_recall=0.0,
                selected_spans=["0:40"],
                selector_latency_sec=0.005,
                representation_source="query_mean_last_layer",
                scoreable=False,
                mixed_fallback_used=True,
                mixed_fallback_margin=0.05,
                mixed_max_children_per_parent=1,
                mixed_child_window_radius=1,
                expected_parent_recall=0.0,
                child_rank_miss_count=0,
                parent_miss_count=4,
            ),
        ],
    )

    rows = compare_longbench_runs(
        parse_run_inputs(
            (
                f"fixed40_modern_control={control_path}",
                f"rep_query_mean={experiment_path}",
            )
        ),
        control_label="fixed40_modern_control",
    )

    by_label_scope = {(row.run_label, row.scope): row for row in rows}
    control_all = by_label_scope[("fixed40_modern_control", "all")]
    experiment_all = by_label_scope[("rep_query_mean", "all")]
    experiment_hotpot = by_label_scope[("rep_query_mean", "hotpotqa")]

    assert control_all.mean_recall == pytest.approx((0.6 + 1.0 + 0.0) / 3.0)
    assert control_all.mean_scoreable_recall == pytest.approx(0.8)
    assert control_all.mean_localization_gap == pytest.approx(0.0)
    assert control_all.recall_delta_vs_control == pytest.approx(0.0)
    assert control_all.semantic_k == "8"
    assert control_all.mean_selected_token_fraction == pytest.approx(
        (0.8 + 0.8 + 0.4) / 3.0
    )
    assert experiment_all.mean_recall == pytest.approx((0.8 + 1.0 + 0.0) / 3.0)
    assert experiment_all.mean_scoreable_recall == pytest.approx(0.9)
    assert experiment_all.mean_localization_gap == pytest.approx(
        ((0.7 + 1.0 + 0.0) / 3.0) - ((0.8 + 1.0 + 0.0) / 3.0)
    )
    assert experiment_all.recall_delta_vs_control == pytest.approx(
        ((0.8 + 1.0 + 0.0) / 3.0) - ((0.6 + 1.0 + 0.0) / 3.0)
    )
    assert experiment_all.selected_token_fraction_delta_vs_control == pytest.approx(
        0.4 - ((0.8 + 0.8 + 0.4) / 3.0)
    )
    assert control_all.mixed_fallback_count is None
    assert control_all.mixed_fallback_rate is None
    assert experiment_all.mixed_fallback_count == 2
    assert experiment_all.mixed_fallback_rate == pytest.approx(2 / 3)
    assert experiment_all.mixed_fallback_margin == "0.05"
    assert experiment_all.mixed_max_children_per_parent == "1"
    assert experiment_all.mixed_child_window_radius == "1"
    assert experiment_all.mean_expected_parent_recall == pytest.approx(
        (0.9 + 1.0 + 0.0) / 3.0
    )
    assert experiment_all.mean_child_rank_miss_count == pytest.approx(1 / 3)
    assert experiment_all.mean_parent_miss_count == pytest.approx(4 / 3)
    assert experiment_hotpot.mixed_fallback_count == 2
    assert experiment_hotpot.mixed_fallback_rate == pytest.approx(1.0)
    assert experiment_all.selector_latency_delta_vs_control == pytest.approx(0.001)
    assert experiment_hotpot.window_recall_delta_vs_control == pytest.approx(0.05)
    assert experiment_all.representation_source == "query_mean_last_layer"


def test_format_comparison_markdown_uses_requested_columns(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    _write_payload(
        path,
        [
            _row(
                dataset_name="hotpotqa",
                recall=0.6,
                precision=0.1,
                window_recall=0.6,
                selected_spans=["0:40"],
                selector_latency_sec=0.004,
            )
        ],
    )
    rows = compare_longbench_runs(parse_run_inputs((f"control={path}",)), scope="all")

    table = format_comparison_markdown(
        rows,
        columns=("run_label", "scope", "mean_recall"),
        precision=2,
    )

    assert "| run_label | scope | mean_recall |" in table
    assert "| control | all | 0.60 |" in table


def test_compare_longbench_runs_cli_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    control_path = tmp_path / "control.json"
    experiment_path = tmp_path / "experiment.json"
    _write_payload(
        control_path,
        [
            _row(
                dataset_name="hotpotqa",
                recall=0.6,
                precision=0.1,
                window_recall=0.6,
                selected_spans=["0:40"],
                selector_latency_sec=0.004,
            )
        ],
    )
    _write_payload(
        experiment_path,
        [
            _row(
                dataset_name="hotpotqa",
                recall=0.8,
                precision=0.2,
                window_recall=0.7,
                selected_spans=["0:40"],
                selector_latency_sec=0.004,
            )
        ],
    )
    module = _load_script()

    rc = module.main(
        [
            f"fixed40_modern_control={control_path}",
            f"candidate={experiment_path}",
            "--control-label",
            "fixed40_modern_control",
            "--scope",
            "all",
            "--columns",
            "run_label,scope,mean_recall,recall_delta_vs_control",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "recall_delta_vs_control" in output
    assert "| candidate | all | 0.800 | 0.200 |" in output
