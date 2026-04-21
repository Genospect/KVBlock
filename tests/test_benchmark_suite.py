from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from kvblock.benchmark.analysis_table import (
    STANDARD_AGGREGATE_METRIC_FIELDS,
    STANDARD_RUN_METRIC_FIELDS,
    SelectorAnalysisAggregateRow,
    SelectorAnalysisRunRow,
)
from kvblock.benchmark.cases import (
    BenchmarkMatrix,
    build_benchmark_cases,
)
from kvblock.benchmark.suite import (
    BenchmarkSuite,
    V1_DEFAULT_CANDIDATE_CONFIDENCE_MARGIN,
    V1_DEFAULT_CANDIDATE_SEMANTIC_K,
    V1_DEFAULT_CANDIDATE_SHORTLIST_M,
    build_calibration_suite,
    build_default_candidate_validation_suite,
    run_benchmark_suite,
    write_benchmark_suite_outputs,
)
from kvblock.benchmark.workloads import available_workloads, resolve_workload


def test_workload_preset_resolution_includes_placeholders() -> None:
    default = resolve_workload("default")
    repeated = resolve_workload("repeated_reference_synthetic")

    assert default.population_profile == "default"
    assert default.implementation_status == "implemented"
    assert repeated.population_profile == "rail_dominated"
    assert repeated.implementation_status.startswith("placeholder")
    assert "adversarial_semantic_synthetic" in {
        workload.name for workload in available_workloads()
    }


def test_benchmark_case_matrix_generation() -> None:
    matrix = BenchmarkMatrix(
        block_counts=(8, 12),
        shortlist_m_values=(4,),
        semantic_k_values=(2,),
        confidence_margins=(0.0, 0.05),
        oracle_enabled_values=(False, True),
        seeds=(0, 1),
        query_counts=(1,),
        workload_names=("default", "low_confidence"),
    )

    cases = build_benchmark_cases(suite_name="unit-suite", matrix=matrix)

    assert len(cases) == 32
    assert {case.workload.name for case in cases} == {"default", "low_confidence"}
    assert {case.microbench_spec.num_blocks for case in cases} == {8, 12}
    assert {case.microbench_spec.shortlist_size for case in cases} == {4}
    assert {case.microbench_spec.semantic_top_k for case in cases} == {2}
    assert {case.microbench_spec.oracle_enabled for case in cases} == {False, True}


def test_suite_presets_include_current_default_candidate() -> None:
    suite = build_default_candidate_validation_suite(block_counts=(8,), query_counts=(1,))

    assert suite.cases
    assert all(
        case.run_spec.shortlist_m == V1_DEFAULT_CANDIDATE_SHORTLIST_M
        for case in suite.cases
    )
    assert all(
        case.run_spec.semantic_k == V1_DEFAULT_CANDIDATE_SEMANTIC_K
        for case in suite.cases
    )
    assert all(
        case.run_spec.confidence_margin == V1_DEFAULT_CANDIDATE_CONFIDENCE_MARGIN
        for case in suite.cases
    )


def test_metric_schema_constants_are_stable() -> None:
    assert STANDARD_RUN_METRIC_FIELDS == SelectorAnalysisRunRow.field_names()
    assert STANDARD_AGGREGATE_METRIC_FIELDS == SelectorAnalysisAggregateRow.field_names()
    assert "selector_latency_sec" in STANDARD_RUN_METRIC_FIELDS
    assert "mean_selector_latency_sec" in STANDARD_AGGREGATE_METRIC_FIELDS
    assert "mean_oracle_recall_rate" in STANDARD_AGGREGATE_METRIC_FIELDS


def test_suite_execution_and_output_writing() -> None:
    matrix = BenchmarkMatrix(
        block_counts=(8,),
        shortlist_m_values=(4,),
        semantic_k_values=(2,),
        confidence_margins=(0.0,),
        oracle_enabled_values=(True,),
        seeds=(2,),
        query_counts=(1,),
        workload_names=("default",),
    )
    suite = BenchmarkSuite(
        name="unit-suite",
        preset="unit",
        description="Unit test suite",
        cases=build_benchmark_cases(suite_name="unit-suite", matrix=matrix),
    )

    result = run_benchmark_suite(suite)

    assert result.summary.case_count == 1
    assert result.summary.run_count == 1
    assert result.summary.best_balanced_config is not None
    assert result.summary.default_candidate_config is None

    with TemporaryDirectory() as tmpdir:
        paths = write_benchmark_suite_outputs(
            result,
            tmpdir,
            write_jsonl_outputs=True,
            write_csv_outputs=True,
        )

        assert paths.run_jsonl is not None and paths.run_jsonl.exists()
        assert paths.aggregate_jsonl is not None and paths.aggregate_jsonl.exists()
        assert paths.cases_jsonl is not None and paths.cases_jsonl.exists()
        assert paths.summary_json is not None and paths.summary_json.exists()
        assert paths.run_csv is not None and paths.run_csv.exists()
        assert paths.aggregate_csv is not None and paths.aggregate_csv.exists()

        summary = json.loads(paths.summary_json.read_text())

    assert summary["case_count"] == 1
    assert summary["run_count"] == 1


def test_calibration_suite_construction_can_be_overridden_small() -> None:
    suite = build_calibration_suite(
        block_counts=(8,),
        shortlist_m_values=(4, 6),
        semantic_k_values=(2,),
        confidence_margins=(0.0, 0.01),
        workload_names=("default",),
        query_counts=(1,),
    )

    assert len(suite.cases) == 4
    assert {case.run_spec.shortlist_m for case in suite.cases} == {4, 6}
    assert {case.run_spec.confidence_margin for case in suite.cases} == {0.0, 0.01}
