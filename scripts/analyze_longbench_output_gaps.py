#!/usr/bin/env python3
"""Analyze row-level LongBench output gaps between oracle and sparse runs."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

DEFAULT_DETAIL_COLUMNS: tuple[str, ...] = (
    "dataset",
    "sample_id",
    "oracle_candidate_category",
    "full_candidate_category",
    "oracle_answer_f1",
    "candidate_answer_f1",
    "full_answer_f1",
    "oracle_answer_em",
    "candidate_answer_em",
    "full_answer_em",
    "oracle_minus_candidate_f1",
    "full_minus_candidate_f1",
    "candidate_reconstructed_context_tokens",
    "oracle_reconstructed_context_tokens",
    "full_reconstructed_context_tokens",
    "candidate_oracle_span_overlap_kind",
    "candidate_oracle_span_overlap_fraction",
    "candidate_oracle_span_overlap_tokens",
    "candidate_selected_ids",
    "gold_answers",
    "candidate_prediction",
    "oracle_prediction",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "dataset",
    "row_count",
    "oracle_correct_candidate_wrong",
    "candidate_correct_oracle_wrong",
    "oracle_candidate_both_correct",
    "oracle_candidate_both_wrong",
    "full_correct_candidate_wrong",
    "candidate_correct_full_wrong",
    "full_correct_oracle_wrong",
    "mean_oracle_minus_candidate_f1",
    "mean_full_minus_candidate_f1",
    "mean_candidate_reconstructed_context_tokens",
    "mean_oracle_reconstructed_context_tokens",
    "mean_full_reconstructed_context_tokens",
    "candidate_touched_oracle_span",
    "candidate_missed_oracle_span",
    "mean_candidate_oracle_span_overlap_fraction",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help=(
            "Run specs as label=path.json. Roles are inferred from labels/config: "
            "oracle, lenaware/candidate, full, fixed."
        ),
    )
    parser.add_argument(
        "--correct-field",
        default="answer_em",
        choices=("answer_em", "answer_f1"),
        help="Per-row metric used for correct/wrong quadrant counts.",
    )
    parser.add_argument(
        "--correct-threshold",
        type=float,
        default=1.0,
        help="A row is correct when correct-field is at least this value.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of largest oracle-candidate F1 gaps to print.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary counts, not row-level details.",
    )
    parser.add_argument(
        "--only-category",
        default=None,
        help=(
            "Optional detail filter against oracle_candidate_category or "
            "full_candidate_category, for example oracle_correct_candidate_wrong."
        ),
    )
    parser.add_argument(
        "--detail-columns",
        default=",".join(DEFAULT_DETAIL_COLUMNS),
        help="Comma-separated detail columns.",
    )
    parser.add_argument("--precision", type=int, default=3)
    parser.add_argument(
        "--max-cell-chars",
        type=int,
        default=96,
        help="Maximum Markdown cell length before truncation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = analyze_output_gaps(
        parse_run_inputs(args.runs),
        correct_field=args.correct_field,
        correct_threshold=args.correct_threshold,
    )
    print(
        format_markdown(
            result.summary_rows,
            columns=SUMMARY_COLUMNS,
            precision=args.precision,
            max_cell_chars=args.max_cell_chars,
        )
    )
    if not args.summary_only and args.top_k > 0:
        detail_columns = tuple(
            item.strip() for item in args.detail_columns.split(",") if item.strip()
        )
        detail_rows = result.detail_rows
        if args.only_category is not None:
            detail_rows = tuple(
                row
                for row in detail_rows
                if row.get("oracle_candidate_category") == args.only_category
                or row.get("full_candidate_category") == args.only_category
            )
        print()
        print(
            format_markdown(
                detail_rows[: args.top_k],
                columns=detail_columns,
                precision=args.precision,
                max_cell_chars=args.max_cell_chars,
            )
        )
    return 0


def parse_run_inputs(specs: Sequence[str]) -> tuple[tuple[str, Path], ...]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"run spec must be label=path.json: {spec!r}")
        label, path_text = spec.split("=", maxsplit=1)
        label = label.strip()
        if not label:
            raise ValueError(f"empty run label in {spec!r}")
        path = Path(path_text)
        if not path.exists():
            raise FileNotFoundError(path)
        parsed.append((label, path))
    return tuple(parsed)


class GapAnalysisResult(tuple):
    """Tuple-compatible wrapper for summary and detail rows."""

    __slots__ = ()

    @property
    def summary_rows(self) -> tuple[dict[str, Any], ...]:
        return self[0]

    @property
    def detail_rows(self) -> tuple[dict[str, Any], ...]:
        return self[1]


def analyze_output_gaps(
    run_inputs: Sequence[tuple[str, Path]],
    *,
    correct_field: str = "answer_em",
    correct_threshold: float = 1.0,
) -> GapAnalysisResult:
    payloads = tuple(
        (label, json.loads(path.read_text(encoding="utf-8")))
        for label, path in run_inputs
    )
    rows_by_role = _rows_by_role(payloads)
    oracle_rows = rows_by_role.get("oracle", {})
    candidate_rows = rows_by_role.get("candidate", {})
    if not oracle_rows:
        raise ValueError("gap analysis requires an oracle/answer_oracle run")
    if not candidate_rows:
        raise ValueError("gap analysis requires a lenaware/candidate run")

    full_rows = rows_by_role.get("full", {})
    fixed_rows = rows_by_role.get("fixed", {})
    common_keys = sorted(set(oracle_rows).intersection(candidate_rows))
    if not common_keys:
        raise ValueError("oracle and candidate runs share no sample rows")

    detail_rows = tuple(
        _gap_detail_row(
            key,
            oracle=oracle_rows[key],
            candidate=candidate_rows[key],
            full=full_rows.get(key),
            fixed=fixed_rows.get(key),
            correct_field=correct_field,
            correct_threshold=correct_threshold,
        )
        for key in common_keys
    )
    sorted_detail_rows = tuple(
        sorted(
            detail_rows,
            key=lambda row: (
                float(row["oracle_minus_candidate_f1"]),
                float(row.get("full_minus_candidate_f1") or 0.0),
            ),
            reverse=True,
        )
    )
    summary_rows = _gap_summary_rows(
        detail_rows,
        include_full=bool(full_rows),
    )
    return GapAnalysisResult((summary_rows, sorted_detail_rows))


def format_markdown(
    rows: Sequence[dict[str, Any]],
    *,
    columns: Sequence[str],
    precision: int,
    max_cell_chars: int = 96,
) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| "
        + " | ".join(
            _format_cell(
                row.get(column),
                precision=precision,
                max_cell_chars=max_cell_chars,
            )
            for column in columns
        )
        + " |"
        for row in rows
    ]
    return "\n".join((header, separator, *body))


def _rows_by_role(
    payloads: Sequence[tuple[str, dict[str, Any]]],
) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    grouped: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    for label, payload in payloads:
        role = _payload_role(label, payload)
        if role is None:
            continue
        role_rows = grouped.setdefault(role, {})
        for row in payload.get("rows", ()):
            key = _row_key(row)
            if key in role_rows:
                raise ValueError(
                    f"duplicate {role} row for {_format_row_key(key)}; "
                    "pass one source run per sample for each role"
                )
            role_rows[key] = row
    return grouped


def _payload_role(label: str, payload: dict[str, Any]) -> str | None:
    normalized_label = label.lower()
    config = payload.get("config", {})
    context_policy = str(config.get("context_policy", "")).lower()
    output_policy = str(config.get("output_policy", "")).lower()
    block_modes = tuple(str(item).lower() for item in config.get("block_modes", ()))

    if (
        context_policy == "answer_oracle"
        or block_modes == ("answer_oracle",)
        or "answeroracle" in normalized_label
        or "answer_oracle" in normalized_label
        or normalized_label.endswith("_oracle")
    ):
        return "oracle"
    if (
        context_policy == "full_context"
        or block_modes == ("full_context",)
        or "fullctx" in normalized_label
        or normalized_label.endswith("_full")
    ):
        return "full"
    if (
        output_policy == "length_aware_static"
        or "lenaware" in normalized_label
        or "length_aware" in normalized_label
        or normalized_label.endswith("_candidate")
    ):
        return "candidate"
    if block_modes == ("fixed_40",) or "fixed40" in normalized_label:
        return "fixed"
    return None


def _gap_detail_row(
    key: tuple[str, str, str],
    *,
    oracle: dict[str, Any],
    candidate: dict[str, Any],
    full: dict[str, Any] | None,
    fixed: dict[str, Any] | None,
    correct_field: str,
    correct_threshold: float,
) -> dict[str, Any]:
    oracle_correct = _is_correct(
        oracle,
        field=correct_field,
        threshold=correct_threshold,
    )
    candidate_correct = _is_correct(
        candidate,
        field=correct_field,
        threshold=correct_threshold,
    )
    full_correct = (
        _is_correct(full, field=correct_field, threshold=correct_threshold)
        if full is not None
        else None
    )
    fixed_correct = (
        _is_correct(fixed, field=correct_field, threshold=correct_threshold)
        if fixed is not None
        else None
    )

    row: dict[str, Any] = {
        "dataset": key[0],
        "sample_id": key[1],
        "model": key[2],
        "oracle_correct": oracle_correct,
        "candidate_correct": candidate_correct,
        "full_correct": full_correct,
        "fixed_correct": fixed_correct,
        "oracle_candidate_category": _pair_category(
            first_name="oracle",
            first_correct=oracle_correct,
            second_name="candidate",
            second_correct=candidate_correct,
        ),
        "full_candidate_category": (
            _pair_category(
                first_name="full",
                first_correct=bool(full_correct),
                second_name="candidate",
                second_correct=candidate_correct,
            )
            if full_correct is not None
            else "n/a"
        ),
        "oracle_answer_f1": _score(oracle, "answer_f1"),
        "candidate_answer_f1": _score(candidate, "answer_f1"),
        "full_answer_f1": _score(full, "answer_f1"),
        "fixed_answer_f1": _score(fixed, "answer_f1"),
        "oracle_answer_em": _score(oracle, "answer_em"),
        "candidate_answer_em": _score(candidate, "answer_em"),
        "full_answer_em": _score(full, "answer_em"),
        "fixed_answer_em": _score(fixed, "answer_em"),
        "oracle_minus_candidate_f1": _score(oracle, "answer_f1")
        - _score(candidate, "answer_f1"),
        "full_minus_candidate_f1": (
            _score(full, "answer_f1") - _score(candidate, "answer_f1")
            if full is not None
            else None
        ),
        "candidate_reconstructed_context_tokens": _token_count(candidate),
        "oracle_reconstructed_context_tokens": _token_count(oracle),
        "full_reconstructed_context_tokens": _token_count(full),
        "fixed_reconstructed_context_tokens": _token_count(fixed),
        "candidate_selected_ids": _compact_sequence(candidate.get("selected_ids", ())),
        "candidate_selected_spans": _compact_sequence(
            candidate.get("selected_spans", ())
        ),
        "gold_answers": _compact_sequence(candidate.get("gold_answers", ())),
        "candidate_prediction": candidate.get("prediction", ""),
        "oracle_prediction": oracle.get("prediction", ""),
        "full_prediction": full.get("prediction", "") if full is not None else "",
        "fixed_prediction": fixed.get("prediction", "") if fixed is not None else "",
    }
    row.update(_span_overlap_fields(candidate=candidate, oracle=oracle))
    row["full_correct_oracle_wrong"] = bool(full_correct) and not oracle_correct
    return row


def _gap_summary_rows(
    detail_rows: Sequence[dict[str, Any]],
    *,
    include_full: bool,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = [
        _summarize_gap_rows("all", detail_rows, include_full=include_full)
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in detail_rows:
        grouped.setdefault(str(row["dataset"]), []).append(row)
    for dataset, dataset_rows in sorted(grouped.items()):
        rows.append(_summarize_gap_rows(dataset, dataset_rows, include_full=include_full))
    return tuple(rows)


def _summarize_gap_rows(
    dataset: str,
    rows: Sequence[dict[str, Any]],
    *,
    include_full: bool,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "row_count": len(rows),
        "oracle_correct_candidate_wrong": _count_category(
            rows,
            "oracle_candidate_category",
            "oracle_correct_candidate_wrong",
        ),
        "candidate_correct_oracle_wrong": _count_category(
            rows,
            "oracle_candidate_category",
            "candidate_correct_oracle_wrong",
        ),
        "oracle_candidate_both_correct": _count_category(
            rows,
            "oracle_candidate_category",
            "oracle_candidate_both_correct",
        ),
        "oracle_candidate_both_wrong": _count_category(
            rows,
            "oracle_candidate_category",
            "oracle_candidate_both_wrong",
        ),
        "full_correct_candidate_wrong": (
            _count_category(
                rows,
                "full_candidate_category",
                "full_correct_candidate_wrong",
            )
            if include_full
            else None
        ),
        "candidate_correct_full_wrong": (
            _count_category(
                rows,
                "full_candidate_category",
                "candidate_correct_full_wrong",
            )
            if include_full
            else None
        ),
        "full_correct_oracle_wrong": (
            sum(1 for row in rows if row.get("full_correct_oracle_wrong"))
            if include_full
            else None
        ),
        "mean_oracle_minus_candidate_f1": _mean(
            row["oracle_minus_candidate_f1"] for row in rows
        ),
        "mean_full_minus_candidate_f1": (
            _mean(
                row["full_minus_candidate_f1"]
                for row in rows
                if row["full_minus_candidate_f1"] is not None
            )
            if include_full
            else None
        ),
        "mean_candidate_reconstructed_context_tokens": _mean(
            row["candidate_reconstructed_context_tokens"] for row in rows
        ),
        "mean_oracle_reconstructed_context_tokens": _mean(
            row["oracle_reconstructed_context_tokens"] for row in rows
        ),
        "mean_full_reconstructed_context_tokens": (
            _mean(
                row["full_reconstructed_context_tokens"]
                for row in rows
                if row["full_reconstructed_context_tokens"] is not None
            )
            if include_full
            else None
        ),
        "candidate_touched_oracle_span": sum(
            1
            for row in rows
            if row.get("candidate_oracle_span_overlap_kind") == "touches_oracle_span"
        ),
        "candidate_missed_oracle_span": sum(
            1
            for row in rows
            if row.get("candidate_oracle_span_overlap_kind") == "misses_oracle_span"
        ),
        "mean_candidate_oracle_span_overlap_fraction": _mean(
            row["candidate_oracle_span_overlap_fraction"]
            for row in rows
            if row["candidate_oracle_span_overlap_fraction"] is not None
        ),
    }


def _is_correct(
    row: dict[str, Any] | None,
    *,
    field: str,
    threshold: float,
) -> bool:
    if row is None:
        return False
    return float(row.get(field, 0.0) or 0.0) >= threshold


def _pair_category(
    *,
    first_name: str,
    first_correct: bool,
    second_name: str,
    second_correct: bool,
) -> str:
    if first_correct and not second_correct:
        return f"{first_name}_correct_{second_name}_wrong"
    if second_correct and not first_correct:
        return f"{second_name}_correct_{first_name}_wrong"
    if first_correct and second_correct:
        return f"{first_name}_{second_name}_both_correct"
    return f"{first_name}_{second_name}_both_wrong"


def _row_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset", "")),
        str(row.get("sample_id", "")),
        str(row.get("model", "")),
    )


def _format_row_key(key: tuple[str, str, str]) -> str:
    dataset, sample_id, model = key
    return f"{dataset}:{sample_id}:{model}"


def _span_overlap_fields(
    *,
    candidate: dict[str, Any],
    oracle: dict[str, Any],
) -> dict[str, Any]:
    candidate_intervals = _parse_token_spans(candidate.get("selected_spans", ()))
    oracle_intervals = _parse_token_spans(oracle.get("selected_spans", ()))
    if not candidate_intervals or not oracle_intervals:
        return {
            "candidate_oracle_span_overlap_kind": "n/a",
            "candidate_oracle_span_overlap_tokens": None,
            "candidate_oracle_span_overlap_fraction": None,
        }

    merged_candidate = _merge_intervals(candidate_intervals)
    merged_oracle = _merge_intervals(oracle_intervals)
    overlap_tokens = _interval_overlap_token_count(merged_candidate, merged_oracle)
    oracle_tokens = _interval_token_count(merged_oracle)
    overlap_fraction = None
    if oracle_tokens > 0:
        overlap_fraction = overlap_tokens / oracle_tokens
    return {
        "candidate_oracle_span_overlap_kind": (
            "touches_oracle_span" if overlap_tokens > 0 else "misses_oracle_span"
        ),
        "candidate_oracle_span_overlap_tokens": overlap_tokens,
        "candidate_oracle_span_overlap_fraction": overlap_fraction,
    }


def _parse_token_spans(value: Any) -> tuple[tuple[int, int], ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = tuple(item for item in value.split(",") if item)
    elif isinstance(value, Sequence):
        items = tuple(str(item) for item in value)
    else:
        return ()
    intervals: list[tuple[int, int]] = []
    for item in items:
        if ":" not in item:
            continue
        start_text, end_text = item.split(":", maxsplit=1)
        try:
            start = int(start_text)
            end = int(end_text)
        except ValueError:
            continue
        if end > start:
            intervals.append((start, end))
    return tuple(intervals)


def _merge_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not intervals:
        return ()
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        previous_start, previous_end = merged[-1]
        merged[-1] = (previous_start, max(previous_end, end))
    return tuple(merged)


def _interval_overlap_token_count(
    left: Sequence[tuple[int, int]],
    right: Sequence[tuple[int, int]],
) -> int:
    overlap = 0
    left_index = 0
    right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0, min(left_end, right_end) - max(left_start, right_start))
        if left_end < right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def _interval_token_count(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(max(0, end - start) for start, end in intervals)


def _score(row: dict[str, Any] | None, field: str) -> float:
    if row is None:
        return 0.0
    return float(row.get(field, 0.0) or 0.0)


def _token_count(row: dict[str, Any] | None) -> float | None:
    if row is None:
        return None
    return float(row.get("reconstructed_context_token_count", 0.0) or 0.0)


def _count_category(
    rows: Sequence[dict[str, Any]],
    field: str,
    value: str,
) -> int:
    return sum(1 for row in rows if row.get(field) == value)


def _mean(values: Iterable[Any]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _compact_sequence(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ",".join(str(item) for item in value)
    return str(value)


def _format_cell(
    value: Any,
    *,
    precision: int,
    max_cell_chars: int,
) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        text = f"{value:.{precision}f}"
    else:
        text = str(value)
    text = text.replace("\n", " ").replace("|", "\\|")
    if max_cell_chars > 0 and len(text) > max_cell_chars:
        return text[: max_cell_chars - 3] + "..."
    return text


if __name__ == "__main__":
    raise SystemExit(main())
