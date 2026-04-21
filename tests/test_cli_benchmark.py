from __future__ import annotations

from argparse import Namespace
import json
from tempfile import TemporaryDirectory

from kvblock.cli.benchmark import (
    build_specs_from_args,
    format_console_summary,
    parse_block_counts,
    parse_non_negative_float_values,
    parse_positive_int_values,
    run_selector_microbench_cli,
)


def test_parse_block_counts_supports_single_and_multiple_values() -> None:
    assert parse_block_counts("128") == (128,)
    assert parse_block_counts("128, 256,512") == (128, 256, 512)
    assert parse_positive_int_values("4,8", name="semantic-k") == (4, 8)
    assert parse_non_negative_float_values("0,0.01", name="confidence-margin") == (
        0.0,
        0.01,
    )


def test_build_specs_from_args_creates_small_sweep() -> None:
    args = Namespace(
        block_counts="64,128",
        shortlist_m="12",
        semantic_k="4,8",
        confidence_margin="0,0.05",
        normalized_margin_threshold=None,
        min_normalized_mass=None,
        seed=3,
        profile="low_confidence",
        num_queries=2,
        oracle=True,
        oracle_top_k=3,
        case_id_prefix="bench",
        run_jsonl_out=None,
        aggregate_jsonl_out=None,
    )

    specs = build_specs_from_args(args)

    assert len(specs) == 8
    assert {spec.num_blocks for spec in specs} == {64, 128}
    assert {spec.shortlist_size for spec in specs} == {12}
    assert {spec.semantic_top_k for spec in specs} == {4, 8}
    assert {spec.confidence_margin for spec in specs} == {0.0, 0.05}
    assert [spec.seed for spec in specs] == list(range(3, 11))
    assert all(spec.population_profile == "low_confidence" for spec in specs)
    assert all(spec.oracle_enabled for spec in specs)


def test_run_selector_microbench_cli_writes_optional_jsonl_outputs() -> None:
    with TemporaryDirectory() as tmpdir:
        run_path = f"{tmpdir}/runs.jsonl"
        aggregate_path = f"{tmpdir}/aggregates.jsonl"

        result = run_selector_microbench_cli(
            [
                "--block-counts",
                "32",
                "--num-queries",
                "2",
                "--seed",
                "5",
                "--confidence-margin",
                "0",
                "--run-jsonl-out",
                run_path,
                "--aggregate-jsonl-out",
                aggregate_path,
            ]
        )

        assert len(result.run_rows) == 2
        assert len(result.aggregate_rows) == 1

        with open(run_path, encoding="utf-8") as handle:
            run_payloads = [json.loads(line) for line in handle if line.strip()]
        with open(aggregate_path, encoding="utf-8") as handle:
            aggregate_payloads = [json.loads(line) for line in handle if line.strip()]

    assert len(run_payloads) == 2
    assert len(aggregate_payloads) == 1
    assert "selector_latency_sec" in run_payloads[0]
    assert "mean_selector_latency_sec" in aggregate_payloads[0]


def test_format_console_summary_includes_oracle_metrics_when_present() -> None:
    result = run_selector_microbench_cli(
        [
            "--block-counts",
            "32",
            "--num-queries",
            "1",
            "--seed",
            "6",
            "--confidence-margin",
            "0",
            "--oracle",
        ]
    )

    summary = format_console_summary(result.aggregate_rows)

    assert "mean_latency_sec=" in summary
    assert "selected/K=" in summary
    assert "fallbacks[" in summary
    assert "oracle_recall=" in summary
    assert "oracle_precision=" in summary
