"""Minimal CLI wrapper for the selector microbenchmark harness."""

from __future__ import annotations

import argparse
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from kvblock.benchmark.analysis_table import (
    SelectorAnalysisAggregateRow,
    SelectorAnalysisRunRow,
    flatten_microbench_case_aggregate_rows,
    flatten_microbench_run_rows,
)
from kvblock.benchmark.jsonl_writer import write_jsonl
from kvblock.benchmark.selector_microbench import (
    PopulationProfile,
    SelectorMicrobenchSpec,
    run_selector_microbench_sweep,
)


@dataclass(frozen=True, slots=True)
class SelectorMicrobenchCliResult:
    """Flattened outputs from one CLI-triggered microbenchmark run."""

    run_rows: tuple[SelectorAnalysisRunRow, ...]
    aggregate_rows: tuple[SelectorAnalysisAggregateRow, ...]
    run_jsonl_path: Path | None = None
    aggregate_jsonl_path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    """Build the selector microbenchmark CLI parser."""

    parser = argparse.ArgumentParser(
        description="Run the synthetic V1 selector microbenchmark harness.",
    )
    parser.add_argument(
        "--block-counts",
        default="128",
        help="Comma-separated block counts to run, e.g. 128 or 128,512,2048.",
    )
    parser.add_argument(
        "--shortlist-m",
        default="24",
        help="Stage A shortlist size or comma-separated sweep values.",
    )
    parser.add_argument(
        "--semantic-k",
        default="8",
        help="Stage C semantic top-K budget or comma-separated sweep values.",
    )
    parser.add_argument(
        "--confidence-margin",
        default="0.05",
        help="Raw confidence margin or comma-separated sweep values.",
    )
    parser.add_argument(
        "--normalized-margin-threshold",
        type=float,
        default=None,
        help="Optional normalized confidence margin threshold.",
    )
    parser.add_argument(
        "--min-normalized-mass",
        type=float,
        default=None,
        help="Optional selected positive-mass threshold for confidence.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for synthetic generation.",
    )
    parser.add_argument(
        "--profile",
        choices=("default", "low_confidence", "rail_dominated"),
        default="default",
        help="Synthetic workload profile.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=8,
        help="Number of synthetic query runs per case.",
    )
    parser.add_argument(
        "--oracle",
        action="store_true",
        help="Enable synthetic dense-oracle comparison.",
    )
    parser.add_argument(
        "--oracle-top-k",
        type=int,
        default=None,
        help="Optional synthetic oracle top-K override.",
    )
    parser.add_argument(
        "--case-id-prefix",
        default="selector-microbench",
        help="Prefix used when building case ids for the sweep.",
    )
    parser.add_argument(
        "--run-jsonl-out",
        default=None,
        help="Optional path for flattened per-run JSONL output.",
    )
    parser.add_argument(
        "--aggregate-jsonl-out",
        default=None,
        help="Optional path for per-case aggregate JSONL output.",
    )
    return parser


def build_specs_from_args(args: argparse.Namespace) -> list[SelectorMicrobenchSpec]:
    """Build one or more microbenchmark specs from parsed CLI args."""

    block_counts = parse_block_counts(args.block_counts)
    shortlist_values = parse_positive_int_values(args.shortlist_m, name="shortlist-m")
    semantic_values = parse_positive_int_values(args.semantic_k, name="semantic-k")
    margin_values = parse_non_negative_float_values(
        args.confidence_margin, name="confidence-margin"
    )
    profile = args.profile
    specs: list[SelectorMicrobenchSpec] = []
    sweep_items = list(
        product(block_counts, shortlist_values, semantic_values, margin_values)
    )
    for index, (block_count, shortlist_m, semantic_k, confidence_margin) in enumerate(
        sweep_items
    ):
        specs.append(
            SelectorMicrobenchSpec(
                case_id=_build_case_id(
                    prefix=args.case_id_prefix,
                    profile=profile,
                    block_count=block_count,
                    shortlist_m=shortlist_m,
                    semantic_k=semantic_k,
                    confidence_margin=confidence_margin,
                    index=index,
                    is_sweep=len(sweep_items) > 1,
                ),
                num_blocks=block_count,
                shortlist_size=shortlist_m,
                semantic_top_k=semantic_k,
                confidence_margin=confidence_margin,
                normalized_margin_threshold=args.normalized_margin_threshold,
                min_normalized_mass=args.min_normalized_mass,
                seed=args.seed + index,
                population_profile=profile,
                num_queries=args.num_queries,
                oracle_enabled=args.oracle,
                oracle_top_k=args.oracle_top_k,
            )
        )
    return specs


def run_selector_microbench_cli(
    argv: Sequence[str] | None = None,
) -> SelectorMicrobenchCliResult:
    """Run the selector microbenchmark from CLI-style arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    specs = build_specs_from_args(args)
    case_results = run_selector_microbench_sweep(specs)
    run_rows = tuple(flatten_microbench_run_rows(case_results))
    aggregate_rows = tuple(flatten_microbench_case_aggregate_rows(case_results))

    run_jsonl_path = Path(args.run_jsonl_out) if args.run_jsonl_out else None
    aggregate_jsonl_path = (
        Path(args.aggregate_jsonl_out) if args.aggregate_jsonl_out else None
    )
    if run_jsonl_path is not None:
        write_jsonl(run_jsonl_path, run_rows)
    if aggregate_jsonl_path is not None:
        write_jsonl(aggregate_jsonl_path, aggregate_rows)

    return SelectorMicrobenchCliResult(
        run_rows=run_rows,
        aggregate_rows=aggregate_rows,
        run_jsonl_path=run_jsonl_path,
        aggregate_jsonl_path=aggregate_jsonl_path,
    )


def format_console_summary(
    aggregate_rows: Sequence[SelectorAnalysisAggregateRow],
) -> str:
    """Format a compact human-readable summary for console output."""

    if not aggregate_rows:
        return "No microbenchmark cases executed."

    lines = []
    for row in aggregate_rows:
        fallback_counts = (
            f"sparse={row.sparse_count} ({row.sparse_rate:.0%}), "
            f"widen_k={row.widen_k_count} ({row.widen_k_rate:.0%}), "
            f"add_recent={row.add_recent_count} ({row.add_recent_rate:.0%}), "
            f"dense={row.dense_count} ({row.dense_rate:.0%})"
        )
        summary = (
            f"{row.case_id}: blocks={row.num_blocks}, M={row.shortlist_m}, "
            f"K={row.semantic_k}, margin={row.confidence_margin:.4f}, "
            f"profile={row.workload_profile}, "
            f"mean_latency_sec={row.mean_selector_latency_sec:.6f}, "
            f"mean_selected={row.mean_final_selected_block_count:.2f}, "
            f"selected/K={row.mean_selected_to_semantic_k_ratio:.2f}, "
            f"mean_semantic={row.mean_semantic_selected_block_count:.2f}, "
            f"mean_rail={row.mean_rail_preserved_block_count:.2f}, "
            f"fallbacks[{fallback_counts}]"
        )
        if row.mean_oracle_recall_rate is not None:
            summary += (
                f", oracle_recall={row.mean_oracle_recall_rate:.4f}, "
                f"oracle_precision={row.mean_oracle_precision_rate:.4f}"
            )
        lines.append(summary)
    return "\n".join(lines)


def parse_block_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated block-count string into a stable tuple."""

    return parse_positive_int_values(value, name="block-counts")


def parse_positive_int_values(value: str, *, name: str) -> tuple[int, ...]:
    """Parse comma-separated positive integer values."""

    parts = [part.strip() for part in value.split(",")]
    counts = tuple(int(part) for part in parts if part)
    if not counts:
        raise ValueError(f"{name} must contain at least one integer")
    if any(count <= 0 for count in counts):
        raise ValueError(f"{name} must all be > 0")
    return counts


def parse_non_negative_float_values(value: str, *, name: str) -> tuple[float, ...]:
    """Parse comma-separated non-negative floating-point values."""

    parts = [part.strip() for part in value.split(",")]
    values = tuple(float(part) for part in parts if part)
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(item < 0 for item in values):
        raise ValueError(f"{name} must all be >= 0")
    return values


def _build_case_id(
    *,
    prefix: str,
    profile: PopulationProfile,
    block_count: int,
    shortlist_m: int,
    semantic_k: int,
    confidence_margin: float,
    index: int,
    is_sweep: bool,
) -> str:
    normalized_prefix = prefix.strip() or "selector-microbench"
    if is_sweep:
        margin_label = f"{confidence_margin:g}".replace(".", "p")
        return (
            f"{normalized_prefix}-{profile}-b{block_count}-m{shortlist_m}"
            f"-k{semantic_k}-c{margin_label}-i{index}"
        )
    return f"{normalized_prefix}-{profile}-b{block_count}"
