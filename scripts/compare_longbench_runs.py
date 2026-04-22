#!/usr/bin/env python3
"""Compare LongBench selector benchmark JSON outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a compact comparison table from LongBench selector benchmark "
            "JSON outputs. Inputs may be paths or LABEL=path pairs."
        ),
    )
    parser.add_argument(
        "runs",
        nargs="+",
        help="LongBench JSON outputs as path or LABEL=path.",
    )
    parser.add_argument(
        "--control-label",
        default=None,
        help="Optional run label used for delta columns, e.g. fixed40_modern_control.",
    )
    parser.add_argument(
        "--scope",
        default="both",
        choices=("all", "datasets", "both"),
        help="Rows to emit: aggregate all datasets, per-dataset rows, or both.",
    )
    parser.add_argument(
        "--format",
        default="markdown",
        choices=("markdown", "csv", "json"),
        help="Output format.",
    )
    parser.add_argument(
        "--precision",
        type=int,
        default=3,
        help="Decimal precision for markdown float cells.",
    )
    parser.add_argument(
        "--columns",
        default="",
        help="Optional comma-separated column list. Defaults to the standard table.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional output path. If omitted, prints to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.longbench_comparison import (
        DEFAULT_COMPARISON_COLUMNS,
        compare_longbench_runs,
        format_comparison_csv,
        format_comparison_json,
        format_comparison_markdown,
        parse_run_inputs,
        write_comparison_output,
    )

    args = build_parser().parse_args(argv)
    columns = (
        tuple(item.strip() for item in args.columns.split(",") if item.strip())
        if args.columns.strip()
        else DEFAULT_COMPARISON_COLUMNS
    )
    rows = compare_longbench_runs(
        parse_run_inputs(args.runs),
        control_label=args.control_label,
        scope=args.scope,
    )
    if args.format == "markdown":
        output = format_comparison_markdown(
            rows,
            columns=columns,
            precision=args.precision,
        )
    elif args.format == "csv":
        output = format_comparison_csv(rows, columns=columns)
    else:
        output = format_comparison_json(rows)

    write_comparison_output(output, args.out)
    if args.out is None:
        print(output, end="")
    else:
        print(f"comparison written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
