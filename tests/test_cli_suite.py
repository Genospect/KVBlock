from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from kvblock.cli.suite import (
    build_parser,
    build_suite_from_args,
    format_suite_console_summary,
    run_benchmark_suite_cli,
)


def test_suite_cli_parsing_builds_overridden_suite() -> None:
    args = build_parser().parse_args(
        [
            "--suite",
            "calibration",
            "--block-counts",
            "8",
            "--shortlist-m",
            "4,6",
            "--semantic-k",
            "2",
            "--confidence-margin",
            "0,0.01",
            "--workloads",
            "default,rail_dominated",
            "--seeds",
            "0",
            "--num-queries",
            "1",
            "--no-oracle",
        ]
    )

    suite = build_suite_from_args(args)

    assert len(suite.cases) == 8
    assert {case.workload.name for case in suite.cases} == {
        "default",
        "rail_dominated",
    }
    assert {case.run_spec.oracle_enabled for case in suite.cases} == {False}


def test_suite_cli_executes_small_suite_and_writes_outputs() -> None:
    with TemporaryDirectory() as tmpdir:
        cli_result = run_benchmark_suite_cli(
            [
                "--suite",
                "default-candidate",
                "--block-counts",
                "8",
                "--workloads",
                "default",
                "--seeds",
                "0",
                "--num-queries",
                "1",
                "--out-dir",
                tmpdir,
                "--csv",
            ]
        )

        summary = format_suite_console_summary(cli_result.result)
        summary_payload = json.loads(cli_result.output_paths.summary_json.read_text())

        assert cli_result.result.summary.case_count == 1
        assert cli_result.result.summary.default_candidate_config is not None
        assert cli_result.output_paths.run_jsonl is not None
        assert cli_result.output_paths.run_jsonl.exists()
        assert cli_result.output_paths.aggregate_csv is not None
        assert cli_result.output_paths.aggregate_csv.exists()

    assert "cases=1 runs=1" in summary
    assert "default_candidate:" in summary
    assert summary_payload["case_count"] == 1
