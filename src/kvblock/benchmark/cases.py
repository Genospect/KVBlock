"""Benchmark case and matrix construction for V1 selector experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Sequence

from kvblock.benchmark.selector_microbench import SelectorMicrobenchSpec
from kvblock.benchmark.workloads import BenchmarkWorkload, resolve_workload


@dataclass(frozen=True, slots=True)
class BenchmarkRunSpec:
    """Selector-lab knobs shared by generated benchmark cases."""

    block_count: int
    shortlist_m: int
    semantic_k: int
    confidence_margin: float
    oracle_enabled: bool
    seed: int
    query_count: int
    block_size: int = 32
    summary_dim: int = 32

    def __post_init__(self) -> None:
        if self.block_count <= 0:
            raise ValueError("block_count must be > 0")
        if self.shortlist_m <= 0:
            raise ValueError("shortlist_m must be > 0")
        if self.semantic_k <= 0:
            raise ValueError("semantic_k must be > 0")
        if self.confidence_margin < 0:
            raise ValueError("confidence_margin must be >= 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.query_count <= 0:
            raise ValueError("query_count must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly run-spec record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One executable selector benchmark case."""

    case_id: str
    suite_name: str
    workload: BenchmarkWorkload
    run_spec: BenchmarkRunSpec
    microbench_spec: SelectorMicrobenchSpec
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark case record."""

        return {
            "case_id": self.case_id,
            "suite_name": self.suite_name,
            "workload": self.workload.to_dict(),
            "run_spec": self.run_spec.to_dict(),
            "microbench_spec": self.microbench_spec.to_dict()
            if hasattr(self.microbench_spec, "to_dict")
            else asdict(self.microbench_spec),
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkMatrix:
    """Cartesian-product matrix for selector benchmark cases."""

    block_counts: tuple[int, ...]
    shortlist_m_values: tuple[int, ...]
    semantic_k_values: tuple[int, ...]
    confidence_margins: tuple[float, ...]
    oracle_enabled_values: tuple[bool, ...] = (True,)
    seeds: tuple[int, ...] = (0,)
    query_counts: tuple[int, ...] = (8,)
    workload_names: tuple[str, ...] = ("default",)
    block_size: int = 32
    summary_dim: int = 32

    def __post_init__(self) -> None:
        _ensure_positive_tuple("block_counts", self.block_counts)
        _ensure_positive_tuple("shortlist_m_values", self.shortlist_m_values)
        _ensure_positive_tuple("semantic_k_values", self.semantic_k_values)
        if not self.confidence_margins:
            raise ValueError("confidence_margins must not be empty")
        if any(value < 0 for value in self.confidence_margins):
            raise ValueError("confidence_margins must all be >= 0")
        if not self.oracle_enabled_values:
            raise ValueError("oracle_enabled_values must not be empty")
        _ensure_non_negative_tuple("seeds", self.seeds)
        _ensure_positive_tuple("query_counts", self.query_counts)
        if not self.workload_names:
            raise ValueError("workload_names must not be empty")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly matrix record."""

        return asdict(self)


def build_benchmark_cases(
    *,
    suite_name: str,
    matrix: BenchmarkMatrix,
    case_id_prefix: str | None = None,
) -> tuple[BenchmarkCase, ...]:
    """Expand a matrix into executable benchmark cases."""

    prefix = case_id_prefix or suite_name
    entry_count = matrix_entry_count(matrix)
    cases: list[BenchmarkCase] = []
    for index, values in enumerate(
        product(
            matrix.workload_names,
            matrix.block_counts,
            matrix.shortlist_m_values,
            matrix.semantic_k_values,
            matrix.confidence_margins,
            matrix.oracle_enabled_values,
            matrix.seeds,
            matrix.query_counts,
        )
    ):
        (
            workload_name,
            block_count,
            shortlist_m,
            semantic_k,
            confidence_margin,
            oracle_enabled,
            seed,
            query_count,
        ) = values
        workload = resolve_workload(workload_name)
        run_spec = BenchmarkRunSpec(
            block_count=block_count,
            shortlist_m=shortlist_m,
            semantic_k=semantic_k,
            confidence_margin=confidence_margin,
            oracle_enabled=oracle_enabled,
            seed=seed,
            query_count=query_count,
            block_size=matrix.block_size,
            summary_dim=matrix.summary_dim,
        )
        case_id = _case_id(
            prefix=prefix,
            workload_name=workload.name,
            run_spec=run_spec,
            index=index,
            is_matrix=entry_count > 1,
        )
        microbench_spec = SelectorMicrobenchSpec(
            case_id=case_id,
            num_blocks=run_spec.block_count,
            block_size=run_spec.block_size,
            summary_dim=run_spec.summary_dim,
            shortlist_size=run_spec.shortlist_m,
            semantic_top_k=run_spec.semantic_k,
            confidence_margin=run_spec.confidence_margin,
            num_queries=run_spec.query_count,
            seed=run_spec.seed,
            population_profile=workload.population_profile,
            oracle_enabled=run_spec.oracle_enabled,
        )
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                suite_name=suite_name,
                workload=workload,
                run_spec=run_spec,
                microbench_spec=microbench_spec,
                tags=_case_tags(workload=workload, run_spec=run_spec),
            )
        )
    return tuple(cases)


def matrix_entry_count(matrix: BenchmarkMatrix) -> int:
    """Return the number of entries in a benchmark matrix."""

    return (
        len(matrix.workload_names)
        * len(matrix.block_counts)
        * len(matrix.shortlist_m_values)
        * len(matrix.semantic_k_values)
        * len(matrix.confidence_margins)
        * len(matrix.oracle_enabled_values)
        * len(matrix.seeds)
        * len(matrix.query_counts)
    )


def _case_id(
    *,
    prefix: str,
    workload_name: str,
    run_spec: BenchmarkRunSpec,
    index: int,
    is_matrix: bool,
) -> str:
    margin = f"{run_spec.confidence_margin:g}".replace(".", "p")
    oracle = "oracle" if run_spec.oracle_enabled else "nooracle"
    base = (
        f"{prefix}-{workload_name}-b{run_spec.block_count}"
        f"-m{run_spec.shortlist_m}-k{run_spec.semantic_k}-c{margin}"
        f"-q{run_spec.query_count}-s{run_spec.seed}-{oracle}"
    )
    return f"{base}-i{index}" if is_matrix else base


def _case_tags(
    *, workload: BenchmarkWorkload, run_spec: BenchmarkRunSpec
) -> tuple[str, ...]:
    tags = [
        workload.name,
        workload.implementation_status,
        "oracle" if run_spec.oracle_enabled else "no_oracle",
    ]
    return tuple(tags)


def _ensure_positive_tuple(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} must all be > 0")


def _ensure_non_negative_tuple(name: str, values: Sequence[int]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must all be >= 0")
