"""CLI helpers for running benchmark suite presets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kvblock.benchmark.suite import (
    SUITE_PRESETS,
    BenchmarkSuiteOutputPaths,
    BenchmarkSuiteResult,
    build_suite_preset,
    run_benchmark_suite,
    write_benchmark_suite_outputs,
)
from kvblock.benchmark.workloads import workload_names
from kvblock.cli.benchmark import (
    parse_non_negative_float_values,
    parse_positive_int_values,
)


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteCliResult:
    """Executed suite plus written output paths."""

    result: BenchmarkSuiteResult
    output_paths: BenchmarkSuiteOutputPaths


def build_parser() -> argparse.ArgumentParser:
    """Build the benchmark suite CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run V1 selector benchmark suite presets.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_PRESETS,
        default="calibration",
        help="Benchmark suite preset to execute.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/baselines/v1_suite",
        help="Directory for benchmark suite outputs.",
    )
    parser.add_argument(
        "--block-counts",
        default=None,
        help="Optional comma-separated block-count override.",
    )
    parser.add_argument(
        "--shortlist-m",
        default=None,
        help="Optional comma-separated shortlist M override.",
    )
    parser.add_argument(
        "--semantic-k",
        default=None,
        help="Optional comma-separated semantic K override.",
    )
    parser.add_argument(
        "--confidence-margin",
        default=None,
        help="Optional comma-separated confidence-margin override.",
    )
    parser.add_argument(
        "--workloads",
        default=None,
        help=f"Optional comma-separated workload override. Known: {', '.join(workload_names())}.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed override.",
    )
    parser.add_argument(
        "--query-counts",
        default=None,
        help="Optional comma-separated query-count override.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=None,
        help="Convenience alias for a single query-count override.",
    )
    parser.set_defaults(oracle_enabled=None)
    parser.add_argument(
        "--oracle",
        dest="oracle_enabled",
        action="store_true",
        help="Force synthetic oracle comparisons on.",
    )
    parser.add_argument(
        "--no-oracle",
        dest="oracle_enabled",
        action="store_false",
        help="Force synthetic oracle comparisons off.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Do not write JSONL row outputs.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write CSV row outputs.",
    )
    return parser


def build_suite_from_args(args: argparse.Namespace):
    """Build a benchmark suite from parsed CLI args."""

    overrides = {
        "block_counts": _optional_ints(args.block_counts, name="block-counts"),
        "shortlist_m_values": _optional_ints(args.shortlist_m, name="shortlist-m"),
        "semantic_k_values": _optional_ints(args.semantic_k, name="semantic-k"),
        "confidence_margins": _optional_floats(
            args.confidence_margin, name="confidence-margin"
        ),
        "workload_names": _optional_workloads(args.workloads),
        "seeds": _optional_non_negative_ints(args.seeds, name="seeds"),
        "query_counts": _query_counts(args),
        "oracle_enabled_values": (
            None
            if args.oracle_enabled is None
            else (bool(args.oracle_enabled),)
        ),
    }
    return build_suite_preset(args.suite, **overrides)


def run_benchmark_suite_cli(
    argv: Sequence[str] | None = None,
) -> BenchmarkSuiteCliResult:
    """Run a benchmark suite from CLI-style args."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    suite = build_suite_from_args(args)
    result = run_benchmark_suite(suite)
    output_paths = write_benchmark_suite_outputs(
        result,
        Path(args.out_dir),
        write_jsonl_outputs=not args.no_jsonl,
        write_csv_outputs=args.csv,
    )
    return BenchmarkSuiteCliResult(result=result, output_paths=output_paths)


def format_suite_console_summary(result: BenchmarkSuiteResult) -> str:
    """Format a compact report-grade console summary."""

    summary = result.summary
    lines = [
        f"suite={summary.suite_name} preset={summary.preset}",
        f"cases={summary.case_count} runs={summary.run_count}",
    ]
    lines.append(_format_config("best_balanced", summary.best_balanced_config))
    lines.append(_format_config("best_quality", summary.best_quality_config))
    lines.append(_format_config("best_low_latency", summary.best_low_latency_config))
    lines.append(_format_config("default_candidate", summary.default_candidate_config))
    if summary.warnings:
        lines.extend(f"warning: {warning}" for warning in summary.warnings)
    return "\n".join(line for line in lines if line)


def _format_config(label: str, config: dict | None) -> str:
    if config is None:
        return f"{label}=not_present"
    recall = config.get("mean_oracle_recall_rate")
    precision = config.get("mean_oracle_precision_rate")
    metrics = (
        f"{label}: case={config['case_id']} "
        f"blocks={config['num_blocks']} M={config['shortlist_m']} "
        f"K={config['semantic_k']} margin={config['confidence_margin']:.4f} "
        f"lat_ms={config['mean_selector_latency_sec'] * 1000:.3f} "
        f"selected/K={config['mean_selected_to_semantic_k_ratio']:.3f} "
        f"widen={config['widen_k_rate']:.0%}"
    )
    if recall is not None and precision is not None:
        metrics += f" recall={recall:.4f} precision={precision:.4f}"
    return metrics


def _optional_ints(value: str | None, *, name: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    return parse_positive_int_values(value, name=name)


def _optional_non_negative_ints(
    value: str | None, *, name: str
) -> tuple[int, ...] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",")]
    values = tuple(int(part) for part in parts if part)
    if not values:
        raise ValueError(f"{name} must contain at least one integer")
    if any(item < 0 for item in values):
        raise ValueError(f"{name} must all be >= 0")
    return values


def _optional_floats(value: str | None, *, name: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    return parse_non_negative_float_values(value, name=name)


def _optional_workloads(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    workloads = tuple(item.strip() for item in value.split(",") if item.strip())
    if not workloads:
        raise ValueError("workloads must contain at least one value")
    return workloads


def _query_counts(args: argparse.Namespace) -> tuple[int, ...] | None:
    if args.query_counts is not None and args.num_queries is not None:
        raise ValueError("Use either --query-counts or --num-queries, not both")
    if args.query_counts is not None:
        return parse_positive_int_values(args.query_counts, name="query-counts")
    if args.num_queries is not None:
        if args.num_queries <= 0:
            raise ValueError("num-queries must be > 0")
        return (args.num_queries,)
    return None
