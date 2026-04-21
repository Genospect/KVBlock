from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from kvblock.benchmark.analysis_table import (
    SelectorAnalysisAggregateRow,
    SelectorAnalysisRunRow,
    flatten_microbench_case_aggregate_rows,
    flatten_microbench_run_rows,
)
from kvblock.benchmark.jsonl_writer import write_jsonl
from kvblock.benchmark.selector_microbench import (
    SelectorMicrobenchSpec,
    run_selector_microbench_case,
    run_selector_microbench_sweep,
)


def test_analysis_table_flattens_microbench_results_into_run_rows() -> None:
    specs = [
        SelectorMicrobenchSpec(case_id="small", num_blocks=8, num_queries=2, seed=11),
        SelectorMicrobenchSpec(case_id="large", num_blocks=12, num_queries=1, seed=12),
    ]

    rows = flatten_microbench_run_rows(run_selector_microbench_sweep(specs))

    assert len(rows) == 3
    assert [row.case_id for row in rows] == ["small", "small", "large"]
    assert all(isinstance(row, SelectorAnalysisRunRow) for row in rows)
    assert all(row.shortlist_m > 0 for row in rows)


def test_analysis_table_handles_oracle_disabled_results_cleanly() -> None:
    result = run_selector_microbench_case(
        SelectorMicrobenchSpec(
            case_id="oracle-off",
            num_blocks=8,
            num_queries=2,
            seed=13,
            oracle_enabled=False,
        )
    )

    run_row = flatten_microbench_run_rows([result])[0]
    aggregate_row = flatten_microbench_case_aggregate_rows([result])[0]

    assert run_row.oracle_enabled is False
    assert run_row.oracle_recall_rate is None
    assert aggregate_row.oracle_enabled is False
    assert aggregate_row.mean_oracle_recall_rate is None
    assert aggregate_row.low_oracle_recall_dense_count is None


def test_analysis_table_handles_oracle_enabled_results_cleanly() -> None:
    result = run_selector_microbench_case(
        SelectorMicrobenchSpec(
            case_id="oracle-on",
            num_blocks=8,
            num_queries=2,
            seed=14,
            oracle_enabled=True,
            oracle_top_k=2,
        )
    )

    run_row = flatten_microbench_run_rows([result])[0]
    aggregate_row = flatten_microbench_case_aggregate_rows([result])[0]

    assert run_row.oracle_enabled is True
    assert run_row.oracle_recall_rate is not None
    assert run_row.oracle_precision_rate is not None
    assert aggregate_row.oracle_enabled is True
    assert aggregate_row.mean_oracle_recall_rate is not None
    assert aggregate_row.mean_oracle_precision_rate is not None
    assert aggregate_row.mean_oracle_overlap_count is not None


def test_analysis_table_field_names_match_serialized_keys() -> None:
    result = run_selector_microbench_case(
        SelectorMicrobenchSpec(case_id="fields", num_blocks=8, num_queries=1, seed=15)
    )

    run_row = flatten_microbench_run_rows([result])[0]
    aggregate_row = flatten_microbench_case_aggregate_rows([result])[0]

    assert tuple(run_row.to_dict().keys()) == SelectorAnalysisRunRow.field_names()
    assert tuple(aggregate_row.to_dict().keys()) == SelectorAnalysisAggregateRow.field_names()


def test_jsonl_writer_serializes_run_and_aggregate_rows() -> None:
    result = run_selector_microbench_case(
        SelectorMicrobenchSpec(
            case_id="jsonl",
            num_blocks=8,
            num_queries=2,
            seed=16,
            oracle_enabled=True,
        )
    )
    run_rows = flatten_microbench_run_rows([result])
    aggregate_rows = flatten_microbench_case_aggregate_rows([result])

    with TemporaryDirectory() as tmpdir:
        run_path = f"{tmpdir}/runs.jsonl"
        aggregate_path = f"{tmpdir}/aggregates.jsonl"

        assert write_jsonl(run_path, run_rows) == len(run_rows)
        assert write_jsonl(aggregate_path, aggregate_rows) == len(aggregate_rows)

        with open(run_path, encoding="utf-8") as handle:
            run_payloads = [json.loads(line) for line in handle if line.strip()]
        with open(aggregate_path, encoding="utf-8") as handle:
            aggregate_payloads = [json.loads(line) for line in handle if line.strip()]

    assert len(run_payloads) == len(run_rows)
    assert len(aggregate_payloads) == len(aggregate_rows)
    assert tuple(run_payloads[0].keys()) == SelectorAnalysisRunRow.field_names()
    assert tuple(aggregate_payloads[0].keys()) == SelectorAnalysisAggregateRow.field_names()
