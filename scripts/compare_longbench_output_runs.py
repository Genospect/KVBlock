#!/usr/bin/env python3
"""Compare LongBench output benchmark JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

DEFAULT_COLUMNS: tuple[str, ...] = (
    "run_label",
    "dataset",
    "row_count",
    "mean_answer_f1",
    "mean_answer_em",
    "mean_selected_block_count",
    "mean_selection_filter_dropped_count",
    "mean_selected_token_fraction",
    "mean_selected_tokens",
    "mean_reconstructed_context_token_fraction",
    "mean_reconstructed_context_tokens",
    "mean_selector_latency_sec",
    "mean_generation_latency_sec",
    "mixed_fallback_rate",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="Run specs as label=path.json, or bare path.json.",
    )
    parser.add_argument(
        "--scope",
        default="both",
        choices=("all", "dataset", "both"),
        help="Rows to print.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated columns to include.",
    )
    parser.add_argument("--precision", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compare_output_runs(parse_run_inputs(args.runs), scope=args.scope)
    columns = tuple(item.strip() for item in args.columns.split(",") if item.strip())
    print(format_markdown(rows, columns=columns, precision=args.precision))
    return 0


def parse_run_inputs(specs: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" in spec:
            label, path_text = spec.split("=", maxsplit=1)
            label = label.strip()
            if not label:
                raise ValueError(f"empty run label in {spec!r}")
        else:
            path_text = spec
            label = Path(path_text).stem
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((label, path))
    return tuple(parsed)


def compare_output_runs(
    run_inputs: Sequence[tuple[str, Path]],
    *,
    scope: str = "both",
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for label, path in run_inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if scope in {"all", "both"}:
            rows.append(_summary_row(label, "all", payload["overall_summary"]))
        if scope in {"dataset", "both"}:
            for summary in payload.get("dataset_summaries", ()):
                rows.append(_summary_row(label, str(summary["dataset"]), summary))
    return tuple(rows)


def format_markdown(
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str] = DEFAULT_COLUMNS,
    precision: int = 3,
) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(_format_cell(row.get(column), precision=precision) for column in columns)
        + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def _summary_row(label: str, dataset: str, summary: dict[str, Any]) -> dict[str, Any]:
    row = dict(summary)
    row["run_label"] = label
    row["dataset"] = dataset
    return row


def _format_cell(value: Any, *, precision: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
