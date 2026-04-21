"""Research-grade benchmark suite orchestration for the V1 selector lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from kvblock.benchmark.analysis_table import (
    SelectorAnalysisAggregateRow,
    SelectorAnalysisRunRow,
    flatten_microbench_case_aggregate_rows,
    flatten_microbench_run_rows,
)
from kvblock.benchmark.cases import (
    BenchmarkCase,
    BenchmarkMatrix,
    build_benchmark_cases,
)
from kvblock.benchmark.csv_writer import write_csv
from kvblock.benchmark.jsonl_writer import write_jsonl
from kvblock.benchmark.selector_microbench import (
    SelectorMicrobenchCaseResult,
    run_selector_microbench_sweep,
)

SUITE_CALIBRATION = "calibration"
SUITE_DEFAULT_CANDIDATE = "default-candidate"
SUITE_STRESS = "stress"
SUITE_PRESETS = (SUITE_CALIBRATION, SUITE_DEFAULT_CANDIDATE, SUITE_STRESS)

V1_DEFAULT_CANDIDATE_SHORTLIST_M = 16
V1_DEFAULT_CANDIDATE_SEMANTIC_K = 10
V1_DEFAULT_CANDIDATE_CONFIDENCE_MARGIN = 0.0


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    """Executable benchmark suite definition."""

    name: str
    preset: str
    description: str
    cases: tuple[BenchmarkCase, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly suite definition."""

        return {
            "name": self.name,
            "preset": self.preset,
            "description": self.description,
            "case_count": len(self.cases),
            "cases": [case.to_dict() for case in self.cases],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteSummary:
    """Suite-level summary for reports and console output."""

    suite_name: str
    preset: str
    case_count: int
    run_count: int
    run_row_count: int
    aggregate_row_count: int
    best_balanced_config: dict[str, Any] | None
    best_quality_config: dict[str, Any] | None
    best_low_latency_config: dict[str, Any] | None
    default_candidate_config: dict[str, Any] | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly suite summary."""

        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteResult:
    """Executed suite output bundle."""

    suite: BenchmarkSuite
    case_results: tuple[SelectorMicrobenchCaseResult, ...]
    run_rows: tuple[SelectorAnalysisRunRow, ...]
    aggregate_rows: tuple[SelectorAnalysisAggregateRow, ...]
    summary: BenchmarkSuiteSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly suite result."""

        return {
            "suite": self.suite.to_dict(),
            "summary": self.summary.to_dict(),
            "run_rows": [row.to_dict() for row in self.run_rows],
            "aggregate_rows": [row.to_dict() for row in self.aggregate_rows],
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteOutputPaths:
    """Paths written by :func:`write_benchmark_suite_outputs`."""

    output_dir: Path
    run_jsonl: Path | None = None
    aggregate_jsonl: Path | None = None
    cases_jsonl: Path | None = None
    summary_json: Path | None = None
    run_csv: Path | None = None
    aggregate_csv: Path | None = None


def build_calibration_suite(
    *,
    block_counts: Sequence[int] = (128, 256, 512, 1024),
    shortlist_m_values: Sequence[int] = (16, 24, 32),
    semantic_k_values: Sequence[int] = (6, 8, 10),
    confidence_margins: Sequence[float] = (0.0, 0.01, 0.025, 0.05),
    oracle_enabled_values: Sequence[bool] = (True,),
    seeds: Sequence[int] = (41,),
    query_counts: Sequence[int] = (8,),
    workload_names: Sequence[str] = ("default",),
) -> BenchmarkSuite:
    """Build the selector calibration suite used for M/K/margin sweeps."""

    matrix = BenchmarkMatrix(
        block_counts=tuple(block_counts),
        shortlist_m_values=tuple(shortlist_m_values),
        semantic_k_values=tuple(semantic_k_values),
        confidence_margins=tuple(confidence_margins),
        oracle_enabled_values=tuple(oracle_enabled_values),
        seeds=tuple(seeds),
        query_counts=tuple(query_counts),
        workload_names=tuple(workload_names),
    )
    return _suite_from_matrix(
        name="v1-selector-calibration",
        preset=SUITE_CALIBRATION,
        description="Selector calibration sweep over block count, shortlist M, semantic K, and confidence margin.",
        matrix=matrix,
    )


def build_default_candidate_validation_suite(
    *,
    block_counts: Sequence[int] = (128, 256, 512),
    shortlist_m_values: Sequence[int] = (V1_DEFAULT_CANDIDATE_SHORTLIST_M,),
    semantic_k_values: Sequence[int] = (V1_DEFAULT_CANDIDATE_SEMANTIC_K,),
    confidence_margins: Sequence[float] = (V1_DEFAULT_CANDIDATE_CONFIDENCE_MARGIN,),
    oracle_enabled_values: Sequence[bool] = (True,),
    seeds: Sequence[int] = (41,),
    query_counts: Sequence[int] = (8,),
    workload_names: Sequence[str] = ("default", "low_confidence", "rail_dominated"),
) -> BenchmarkSuite:
    """Build a suite centered on the provisional V1 default candidate."""

    matrix = BenchmarkMatrix(
        block_counts=tuple(block_counts),
        shortlist_m_values=tuple(shortlist_m_values),
        semantic_k_values=tuple(semantic_k_values),
        confidence_margins=tuple(confidence_margins),
        oracle_enabled_values=tuple(oracle_enabled_values),
        seeds=tuple(seeds),
        query_counts=tuple(query_counts),
        workload_names=tuple(workload_names),
    )
    return _suite_from_matrix(
        name="v1-selector-default-candidate-validation",
        preset=SUITE_DEFAULT_CANDIDATE,
        description="Validation suite for the provisional V1 selector default candidate.",
        matrix=matrix,
    )


def build_stress_suite(
    *,
    block_counts: Sequence[int] = (512, 1024, 2048),
    shortlist_m_values: Sequence[int] = (16, 32),
    semantic_k_values: Sequence[int] = (8, 10),
    confidence_margins: Sequence[float] = (0.0, 0.01),
    oracle_enabled_values: Sequence[bool] = (True,),
    seeds: Sequence[int] = (61,),
    query_counts: Sequence[int] = (4,),
    workload_names: Sequence[str] = ("default", "low_confidence", "rail_dominated"),
) -> BenchmarkSuite:
    """Build a local-CPU-friendly stress suite for selector scalability."""

    matrix = BenchmarkMatrix(
        block_counts=tuple(block_counts),
        shortlist_m_values=tuple(shortlist_m_values),
        semantic_k_values=tuple(semantic_k_values),
        confidence_margins=tuple(confidence_margins),
        oracle_enabled_values=tuple(oracle_enabled_values),
        seeds=tuple(seeds),
        query_counts=tuple(query_counts),
        workload_names=tuple(workload_names),
    )
    return _suite_from_matrix(
        name="v1-selector-stress",
        preset=SUITE_STRESS,
        description="Selector-only stress suite for larger synthetic block populations.",
        matrix=matrix,
    )


def build_suite_preset(preset: str, **overrides: Any) -> BenchmarkSuite:
    """Build one of the named suite presets with optional matrix overrides."""

    if preset == SUITE_CALIBRATION:
        return build_calibration_suite(**_preset_kwargs(overrides))
    if preset == SUITE_DEFAULT_CANDIDATE:
        return build_default_candidate_validation_suite(**_preset_kwargs(overrides))
    if preset == SUITE_STRESS:
        return build_stress_suite(**_preset_kwargs(overrides))
    raise ValueError(f"Unknown benchmark suite preset: {preset}")


def run_benchmark_suite(suite: BenchmarkSuite) -> BenchmarkSuiteResult:
    """Execute a benchmark suite through the current selector microbench harness."""

    case_results = tuple(
        run_selector_microbench_sweep([case.microbench_spec])[0]
        for case in suite.cases
    )
    run_rows = tuple(flatten_microbench_run_rows(case_results))
    aggregate_rows = tuple(flatten_microbench_case_aggregate_rows(case_results))
    summary = summarize_benchmark_suite(
        suite=suite,
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
    )
    return BenchmarkSuiteResult(
        suite=suite,
        case_results=case_results,
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
        summary=summary,
    )


def summarize_benchmark_suite(
    *,
    suite: BenchmarkSuite,
    run_rows: Sequence[SelectorAnalysisRunRow],
    aggregate_rows: Sequence[SelectorAnalysisAggregateRow],
) -> BenchmarkSuiteSummary:
    """Build a compact suite-level summary."""

    warnings = _suite_warnings(aggregate_rows)
    return BenchmarkSuiteSummary(
        suite_name=suite.name,
        preset=suite.preset,
        case_count=len(suite.cases),
        run_count=sum(row.query_count for row in aggregate_rows),
        run_row_count=len(run_rows),
        aggregate_row_count=len(aggregate_rows),
        best_balanced_config=_best_config(aggregate_rows, mode="balanced"),
        best_quality_config=_best_config(aggregate_rows, mode="quality"),
        best_low_latency_config=_best_config(aggregate_rows, mode="latency"),
        default_candidate_config=_default_candidate_config(aggregate_rows),
        warnings=warnings,
    )


def write_benchmark_suite_outputs(
    result: BenchmarkSuiteResult,
    output_dir: str | Path,
    *,
    write_jsonl_outputs: bool = True,
    write_csv_outputs: bool = False,
) -> BenchmarkSuiteOutputPaths:
    """Write suite outputs to an analysis-friendly directory."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_jsonl = aggregate_jsonl = cases_jsonl = None
    if write_jsonl_outputs:
        run_jsonl = out_dir / "runs.jsonl"
        aggregate_jsonl = out_dir / "aggregates.jsonl"
        cases_jsonl = out_dir / "cases.jsonl"
        write_jsonl(run_jsonl, result.run_rows)
        write_jsonl(aggregate_jsonl, result.aggregate_rows)
        write_jsonl(cases_jsonl, result.suite.cases)

    summary_json = out_dir / "summary.json"
    summary_json.write_text(
        json.dumps(result.summary.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    run_csv = aggregate_csv = None
    if write_csv_outputs:
        run_csv = out_dir / "runs.csv"
        aggregate_csv = out_dir / "aggregates.csv"
        write_csv(run_csv, result.run_rows)
        write_csv(aggregate_csv, result.aggregate_rows)

    return BenchmarkSuiteOutputPaths(
        output_dir=out_dir,
        run_jsonl=run_jsonl,
        aggregate_jsonl=aggregate_jsonl,
        cases_jsonl=cases_jsonl,
        summary_json=summary_json,
        run_csv=run_csv,
        aggregate_csv=aggregate_csv,
    )


def _suite_from_matrix(
    *,
    name: str,
    preset: str,
    description: str,
    matrix: BenchmarkMatrix,
) -> BenchmarkSuite:
    return BenchmarkSuite(
        name=name,
        preset=preset,
        description=description,
        cases=build_benchmark_cases(suite_name=name, matrix=matrix, case_id_prefix=preset),
    )


def _preset_kwargs(overrides: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in overrides.items() if value is not None}


def _best_config(
    rows: Sequence[SelectorAnalysisAggregateRow], *, mode: str
) -> dict[str, Any] | None:
    if not rows:
        return None
    if mode == "latency":
        selected = min(rows, key=lambda row: (row.mean_selector_latency_sec, row.mean_selected_to_semantic_k_ratio))
    elif mode == "quality":
        selected = max(
            rows,
            key=lambda row: (
                _optional(row.mean_oracle_recall_rate),
                _optional(row.mean_oracle_precision_rate),
                -row.mean_selected_to_semantic_k_ratio,
                -row.mean_selector_latency_sec,
            ),
        )
    elif mode == "balanced":
        selected = max(rows, key=lambda row: _balanced_score(row, rows))
    else:
        raise ValueError(f"Unknown best-config mode: {mode}")
    return _config_summary(selected)


def _balanced_score(
    row: SelectorAnalysisAggregateRow,
    rows: Sequence[SelectorAnalysisAggregateRow],
) -> float:
    recall = _normalize(
        _optional(row.mean_oracle_recall_rate),
        [_optional(item.mean_oracle_recall_rate) for item in rows],
    )
    precision = _normalize(
        _optional(row.mean_oracle_precision_rate),
        [_optional(item.mean_oracle_precision_rate) for item in rows],
    )
    latency = 1.0 - _normalize(
        row.mean_selector_latency_sec,
        [item.mean_selector_latency_sec for item in rows],
    )
    selected_ratio = 1.0 - _normalize(
        row.mean_selected_to_semantic_k_ratio,
        [item.mean_selected_to_semantic_k_ratio for item in rows],
    )
    return 0.30 * recall + 0.30 * precision + 0.20 * latency + 0.20 * selected_ratio


def _normalize(value: float, values: Sequence[float]) -> float:
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return 1.0
    return (value - minimum) / (maximum - minimum)


def _optional(value: float | None) -> float:
    return 0.0 if value is None else value


def _default_candidate_config(
    rows: Sequence[SelectorAnalysisAggregateRow],
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row.shortlist_m == V1_DEFAULT_CANDIDATE_SHORTLIST_M
        and row.semantic_k == V1_DEFAULT_CANDIDATE_SEMANTIC_K
        and row.confidence_margin == V1_DEFAULT_CANDIDATE_CONFIDENCE_MARGIN
    ]
    if not candidates:
        return None
    return _best_config(candidates, mode="balanced")


def _suite_warnings(rows: Sequence[SelectorAnalysisAggregateRow]) -> tuple[str, ...]:
    if not rows:
        return ("suite produced no aggregate rows",)

    warnings: list[str] = []
    mean_widen_rate = sum(row.widen_k_rate for row in rows) / len(rows)
    if mean_widen_rate >= 0.5:
        warnings.append(
            f"widen_k fallback dominates this suite: mean widen_k_rate={mean_widen_rate:.2%}"
        )
    if any(row.dense_rate > 0 for row in rows):
        warnings.append("dense fallback occurred in at least one case")
    return tuple(warnings)


def _config_summary(row: SelectorAnalysisAggregateRow) -> dict[str, Any]:
    return {
        "case_id": row.case_id,
        "workload_profile": row.workload_profile,
        "num_blocks": row.num_blocks,
        "shortlist_m": row.shortlist_m,
        "semantic_k": row.semantic_k,
        "confidence_margin": row.confidence_margin,
        "oracle_enabled": row.oracle_enabled,
        "query_count": row.query_count,
        "mean_selector_latency_sec": row.mean_selector_latency_sec,
        "mean_selected_to_semantic_k_ratio": row.mean_selected_to_semantic_k_ratio,
        "mean_final_selected_block_count": row.mean_final_selected_block_count,
        "sparse_rate": row.sparse_rate,
        "widen_k_rate": row.widen_k_rate,
        "dense_rate": row.dense_rate,
        "mean_oracle_recall_rate": row.mean_oracle_recall_rate,
        "mean_oracle_precision_rate": row.mean_oracle_precision_rate,
        "mean_oracle_overlap_count": row.mean_oracle_overlap_count,
    }
