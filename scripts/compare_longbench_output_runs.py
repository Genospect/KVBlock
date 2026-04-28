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
    parser.add_argument(
        "--hybrid-policy",
        default=None,
        choices=("quality_guarded_static",),
        help=(
            "Append a synthetic policy summary. quality_guarded_static uses fixed40 "
            "for HotpotQA samples with LongBench length >= 4000 and length-aware "
            "mixed rows elsewhere."
        ),
    )
    parser.add_argument("--precision", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = compare_output_runs(
        parse_run_inputs(args.runs),
        scope=args.scope,
        hybrid_policy=args.hybrid_policy,
    )
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
    hybrid_policy: str | None = None,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    payloads: list[tuple[str, dict[str, Any]]] = []
    for label, path in run_inputs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payloads.append((label, payload))
        if scope in {"all", "both"}:
            rows.append(_summary_row(label, "all", payload["overall_summary"]))
        if scope in {"dataset", "both"}:
            for summary in payload.get("dataset_summaries", ()):
                rows.append(_summary_row(label, str(summary["dataset"]), summary))
    if hybrid_policy is not None:
        rows.extend(_hybrid_summary_rows(hybrid_policy, payloads, scope=scope))
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


def _hybrid_summary_rows(
    policy: str,
    payloads: Sequence[tuple[str, dict[str, Any]]],
    *,
    scope: str,
) -> tuple[dict[str, Any], ...]:
    if policy != "quality_guarded_static":
        raise ValueError("unsupported hybrid policy")

    fixed_rows = _rows_by_key(payloads, role="fixed")
    length_aware_rows = _rows_by_key(payloads, role="length_aware")
    if not fixed_rows:
        raise ValueError("quality_guarded_static requires a fixed_40 input run")
    if not length_aware_rows:
        raise ValueError(
            "quality_guarded_static requires a length_aware_static input run"
        )

    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    for key in sorted(set(fixed_rows).union(length_aware_rows)):
        source = fixed_rows.get(key) or length_aware_rows.get(key)
        if source is None:
            continue
        use_fixed = _quality_guarded_uses_fixed(source)
        row = fixed_rows.get(key) if use_fixed else length_aware_rows.get(key)
        if row is None:
            missing.append(_format_row_key(key))
            continue
        selected.append(row)

    if missing:
        preview = ", ".join(missing[:5])
        if len(missing) > 5:
            preview += f", ... ({len(missing)} total)"
        raise ValueError(f"missing required rows for quality_guarded_static: {preview}")
    if not selected:
        raise ValueError("quality_guarded_static produced no rows")

    rows: list[dict[str, Any]] = []
    label = "quality_guarded_static"
    if scope in {"all", "both"}:
        rows.append(_summary_row(label, "all", _summarize_rows(selected)))
    if scope in {"dataset", "both"}:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in selected:
            grouped.setdefault(str(row["dataset"]), []).append(row)
        for dataset, dataset_rows in sorted(grouped.items()):
            rows.append(_summary_row(label, dataset, _summarize_rows(dataset_rows)))
    return tuple(rows)


def _rows_by_key(
    payloads: Sequence[tuple[str, dict[str, Any]]],
    *,
    role: str,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for label, payload in payloads:
        if _payload_role(label, payload) != role:
            continue
        for row in payload.get("rows", ()):
            key = _row_key(row)
            if key in rows_by_key:
                raise ValueError(
                    f"duplicate {role} row for {_format_row_key(key)}; "
                    "pass one source run per sample for hybrid comparison"
                )
            rows_by_key[key] = row
    return rows_by_key


def _payload_role(label: str, payload: dict[str, Any]) -> str | None:
    config = payload.get("config", {})
    output_policy = str(config.get("output_policy", "")).lower()
    if output_policy == "length_aware_static":
        return "length_aware"

    normalized_label = label.lower()
    if "lenaware" in normalized_label or "length_aware" in normalized_label:
        return "length_aware"

    block_modes = tuple(str(item) for item in config.get("block_modes", ()))
    if block_modes == ("fixed_40",):
        return "fixed"
    if "fixed" in normalized_label:
        return "fixed"
    return None


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        str(row.get("model", "")),
    )


def _format_row_key(key: tuple[str, str, str]) -> str:
    dataset, sample_id, model = key
    return f"{dataset}:{sample_id}:{model}"


def _quality_guarded_uses_fixed(row: dict[str, Any]) -> bool:
    return (
        str(row.get("dataset", "")).lower() == "hotpotqa"
        and _row_longbench_length(row) >= 4000
    )


def _row_longbench_length(row: dict[str, Any]) -> int:
    value = row.get("longbench_length")
    if value is None:
        value = row.get("prompt_tokens")
    return int(value or 0)


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    fallback_count = sum(1 for row in rows if row.get("mixed_fallback_used"))
    return {
        "row_count": len(rows),
        "mean_answer_em": _mean(row.get("answer_em", 0.0) for row in rows),
        "mean_answer_f1": _mean(row.get("answer_f1", 0.0) for row in rows),
        "mean_answer_precision": _mean(
            row.get("answer_precision", 0.0) for row in rows
        ),
        "mean_answer_recall": _mean(row.get("answer_recall", 0.0) for row in rows),
        "mean_selected_block_count": _mean(
            row.get("selected_block_count", 0.0) for row in rows
        ),
        "mean_selection_filter_dropped_count": _mean(
            row.get("selection_filter_dropped_count", 0.0) for row in rows
        ),
        "mean_selected_token_fraction": _mean(
            row.get("selected_token_fraction", 0.0) for row in rows
        ),
        "mean_selected_tokens": _mean(
            row.get("selected_token_count", 0.0) for row in rows
        ),
        "mean_reconstructed_context_token_fraction": _mean(
            row.get("reconstructed_context_token_fraction", 0.0) for row in rows
        ),
        "mean_reconstructed_context_tokens": _mean(
            row.get("reconstructed_context_token_count", 0.0) for row in rows
        ),
        "mean_selector_latency_sec": _mean(
            row.get("selector_latency_sec", 0.0) for row in rows
        ),
        "mean_generation_latency_sec": _mean(
            row.get("generation_latency_sec", 0.0) for row in rows
        ),
        "mixed_fallback_count": fallback_count,
        "mixed_fallback_rate": fallback_count / len(rows),
    }


def _mean(values: Any) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _format_cell(value: Any, *, precision: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
