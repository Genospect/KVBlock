"""Comparison helpers for LongBench selector benchmark JSON outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_COMPARISON_COLUMNS: tuple[str, ...] = (
    "run_label",
    "scope",
    "model_names",
    "representation_source",
    "block_modes",
    "rerank_mode",
    "refine_score_mode",
    "stage_c_policy",
    "semantic_k",
    "max_selected_blocks",
    "halo_radius",
    "mixed_fallback_margin",
    "mixed_max_children_per_parent",
    "row_count",
    "scoreable_run_count",
    "mixed_fallback_count",
    "mixed_fallback_rate",
    "mean_recall",
    "recall_delta_vs_control",
    "mean_precision",
    "mean_evidence_window_recall",
    "window_recall_delta_vs_control",
    "mean_evidence_window_precision",
    "mean_selected_to_semantic_k_ratio",
    "mean_selected_count",
    "mean_selected_tokens",
    "mean_selected_token_fraction",
    "selected_token_fraction_delta_vs_control",
    "mean_candidate_block_count",
    "mean_selector_latency_sec",
    "selector_latency_delta_vs_control",
)


@dataclass(frozen=True, slots=True)
class LongBenchRunInput:
    """One labeled LongBench JSON artifact."""

    label: str
    path: Path


@dataclass(frozen=True, slots=True)
class LongBenchComparisonRow:
    """One aggregate comparison row for a run and dataset scope."""

    run_label: str
    scope: str
    source_path: str
    dataset_repo: str
    split: str
    length_bucket: str
    datasets: str
    model_names: str
    representation_source: str
    qk_aggregation_strategy: str
    block_modes: str
    rerank_mode: str
    refine_top_n_tokens: str
    refine_score_mode: str
    stage_c_policy: str
    semantic_k: str
    max_selected_blocks: str
    halo_radius: str
    mixed_refine_parent_k: str
    mixed_global_anchor_k: str
    mixed_fallback_margin: str
    mixed_max_children_per_parent: str
    evidence_window_radius: str
    exclude_scaffold_blocks: bool | None
    oracle_mode: str
    row_count: int
    scoreable_run_count: int
    mixed_fallback_count: int | None
    mixed_fallback_rate: float | None
    mean_expected_block_count: float | None
    mean_answer_presence_rate: float | None
    mean_recall: float | None
    recall_delta_vs_control: float | None = None
    mean_precision: float | None = None
    mean_evidence_window_recall: float | None = None
    window_recall_delta_vs_control: float | None = None
    mean_evidence_window_precision: float | None = None
    mean_selected_to_semantic_k_ratio: float | None = None
    mean_selected_count: float | None = None
    mean_selected_tokens: float | None = None
    mean_selected_token_fraction: float | None = None
    selected_token_fraction_delta_vs_control: float | None = None
    mean_candidate_block_count: float | None = None
    mean_selector_latency_sec: float | None = None
    selector_latency_delta_vs_control: float | None = None
    mean_scaffold_excluded_count: float | None = None
    mean_oracle_selected_mass_fraction: float | None = None
    mean_oracle_expected_mass_fraction: float | None = None
    mean_scoreable_recall: float | None = None
    mean_scoreable_precision: float | None = None
    mean_scoreable_evidence_window_recall: float | None = None
    mean_scoreable_evidence_window_precision: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/CSV-friendly row record."""

        return asdict(self)


def parse_run_inputs(values: Sequence[str]) -> tuple[LongBenchRunInput, ...]:
    """Parse positional CLI inputs as ``path`` or ``label=path`` values."""

    if not values:
        raise ValueError("at least one LongBench JSON path is required")

    parsed: list[LongBenchRunInput] = []
    seen_labels: set[str] = set()
    for value in values:
        label, path = _parse_run_input(value)
        if label in seen_labels:
            raise ValueError(f"duplicate run label {label!r}")
        seen_labels.add(label)
        parsed.append(LongBenchRunInput(label=label, path=path))
    return tuple(parsed)


def compare_longbench_runs(
    run_inputs: Sequence[LongBenchRunInput],
    *,
    control_label: str | None = None,
    scope: str = "both",
) -> tuple[LongBenchComparisonRow, ...]:
    """Load LongBench benchmark JSONs and return aggregate comparison rows."""

    if scope not in {"all", "datasets", "both"}:
        raise ValueError("scope must be one of all, datasets, both")
    rows: list[LongBenchComparisonRow] = []
    for run_input in run_inputs:
        payload = _load_longbench_payload(run_input.path)
        run_rows = _payload_comparison_rows(
            payload,
            run_label=run_input.label,
            source_path=run_input.path,
            scope=scope,
        )
        rows.extend(run_rows)
    if control_label is not None:
        rows = _apply_control_deltas(rows, control_label=control_label)
    return tuple(rows)


def format_comparison_markdown(
    rows: Sequence[LongBenchComparisonRow],
    *,
    columns: Sequence[str] = DEFAULT_COMPARISON_COLUMNS,
    precision: int = 3,
) -> str:
    """Format comparison rows as a GitHub-flavored Markdown table."""

    if not rows:
        return ""
    headers = tuple(columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = row.to_dict()
        cells = [
            _escape_markdown_cell(_format_value(values.get(column), precision=precision))
            for column in headers
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def format_comparison_csv(
    rows: Sequence[LongBenchComparisonRow],
    *,
    columns: Sequence[str] = DEFAULT_COMPARISON_COLUMNS,
) -> str:
    """Format comparison rows as CSV text."""

    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=list(columns), extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        payload = {
            key: "" if value is None else value for key, value in row.to_dict().items()
        }
        writer.writerow(payload)
    return output.getvalue()


def format_comparison_json(rows: Sequence[LongBenchComparisonRow]) -> str:
    """Format comparison rows as indented JSON text."""

    return json.dumps([row.to_dict() for row in rows], indent=2) + "\n"


def write_comparison_output(text: str, path: str | Path | None) -> None:
    """Write comparison output to ``path`` when provided."""

    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def _parse_run_input(value: str) -> tuple[str, Path]:
    if "=" in value:
        raw_label, raw_path = value.split("=", 1)
        label = raw_label.strip()
        path = Path(raw_path.strip())
        if not label:
            raise ValueError(f"run label must be non-empty in {value!r}")
        if not str(path):
            raise ValueError(f"run path must be non-empty in {value!r}")
        return label, path
    path = Path(value)
    return path.stem, path


def _load_longbench_payload(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    if "rows" not in payload or "dataset_summaries" not in payload:
        raise ValueError(f"{path} is not a LongBench benchmark JSON output")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} contains no LongBench rows")
    return payload


def _payload_comparison_rows(
    payload: Mapping[str, Any],
    *,
    run_label: str,
    source_path: Path,
    scope: str,
) -> tuple[LongBenchComparisonRow, ...]:
    raw_rows = [row for row in payload["rows"] if isinstance(row, dict)]
    groups: list[tuple[str, list[Mapping[str, Any]]]] = []
    if scope in {"all", "both"}:
        groups.append(("all", raw_rows))
    if scope in {"datasets", "both"}:
        for dataset_name in sorted(_unique_strings(row.get("dataset_name") for row in raw_rows)):
            groups.append(
                (
                    dataset_name,
                    [row for row in raw_rows if row.get("dataset_name") == dataset_name],
                )
            )
    return tuple(
        _comparison_row_from_group(
            group_rows,
            payload=payload,
            run_label=run_label,
            source_path=source_path,
            scope_name=scope_name,
        )
        for scope_name, group_rows in groups
        if group_rows
    )


def _comparison_row_from_group(
    rows: Sequence[Mapping[str, Any]],
    *,
    payload: Mapping[str, Any],
    run_label: str,
    source_path: Path,
    scope_name: str,
) -> LongBenchComparisonRow:
    scoreable_rows = [row for row in rows if _is_scoreable(row)]
    token_fractions = [_selected_token_fraction(row) for row in rows]
    selected_tokens = [_selected_token_count(row) for row in rows]
    mixed_fallback_count = _mixed_fallback_count(rows)
    return LongBenchComparisonRow(
        run_label=run_label,
        scope=scope_name,
        source_path=str(source_path),
        dataset_repo=str(payload.get("dataset_repo", "")),
        split=str(payload.get("split", "")),
        length_bucket=_length_bucket_name(payload.get("length_bucket")),
        datasets=_join_unique(row.get("dataset_name") for row in rows),
        model_names=_join_unique(row.get("model_name") for row in rows),
        representation_source=_join_unique(row.get("representation_source") for row in rows),
        qk_aggregation_strategy=_join_unique(
            row.get("qk_aggregation_strategy") for row in rows
        ),
        block_modes=_join_unique(row.get("block_mode") for row in rows),
        rerank_mode=_join_unique(row.get("rerank_mode") for row in rows),
        refine_top_n_tokens=_join_unique(row.get("refine_top_n_tokens") for row in rows),
        refine_score_mode=_join_unique(
            row.get("refine_score_mode", payload.get("refine_score_mode")) for row in rows
        ),
        stage_c_policy=_join_unique(
            row.get("stage_c_policy", payload.get("stage_c_policy")) for row in rows
        ),
        semantic_k=_join_unique(_infer_semantic_k(row) for row in rows),
        max_selected_blocks=_join_unique(row.get("max_selected_blocks") for row in rows),
        halo_radius=_join_unique(row.get("halo_radius") for row in rows),
        mixed_refine_parent_k=_join_unique(
            row.get("mixed_refine_parent_k") for row in rows
        ),
        mixed_global_anchor_k=_join_unique(
            row.get("mixed_global_anchor_k") for row in rows
        ),
        mixed_fallback_margin=_join_unique(
            row.get("mixed_fallback_margin") for row in rows
        ),
        mixed_max_children_per_parent=_join_unique(
            row.get("mixed_max_children_per_parent") for row in rows
        ),
        evidence_window_radius=_join_unique(
            row.get("evidence_window_radius", payload.get("evidence_window_radius"))
            for row in rows
        ),
        exclude_scaffold_blocks=_optional_bool(payload.get("exclude_scaffold_blocks")),
        oracle_mode=str(payload.get("oracle_mode", "")),
        row_count=len(rows),
        scoreable_run_count=len(scoreable_rows),
        mixed_fallback_count=mixed_fallback_count,
        mixed_fallback_rate=(
            None if mixed_fallback_count is None else mixed_fallback_count / len(rows)
        ),
        mean_expected_block_count=_mean(
            _optional_float(row.get("expected_block_count")) for row in scoreable_rows
        ),
        mean_answer_presence_rate=_mean(
            _optional_float(row.get("answer_presence_rate")) for row in rows
        ),
        # Match LongBench DATASET SUMMARIES: unscoreable answer-absent rows still
        # carry numeric zero metrics and are included in the reported means.
        mean_recall=_mean(_optional_float(row.get("target_recall")) for row in rows),
        mean_precision=_mean(
            _optional_float(row.get("selected_precision")) for row in rows
        ),
        mean_evidence_window_recall=_mean(
            _optional_float(row.get("evidence_window_recall")) for row in rows
        ),
        mean_evidence_window_precision=_mean(
            _optional_float(row.get("evidence_window_precision")) for row in rows
        ),
        mean_selected_to_semantic_k_ratio=_mean(
            _optional_float(row.get("selected_to_semantic_k_ratio")) for row in rows
        ),
        mean_selected_count=_mean(
            _optional_float(row.get("selected_count")) for row in rows
        ),
        mean_selected_tokens=_mean(selected_tokens),
        mean_selected_token_fraction=_mean(token_fractions),
        mean_candidate_block_count=_mean(
            _optional_float(row.get("candidate_block_count")) for row in rows
        ),
        mean_selector_latency_sec=_mean(
            _optional_float(row.get("selector_latency_sec")) for row in rows
        ),
        mean_scaffold_excluded_count=_mean(
            _optional_float(row.get("scaffold_excluded_count")) for row in rows
        ),
        mean_oracle_selected_mass_fraction=_mean(
            _optional_float(row.get("oracle_selected_mass_fraction")) for row in rows
        ),
        mean_oracle_expected_mass_fraction=_mean(
            _optional_float(row.get("oracle_expected_mass_fraction")) for row in rows
        ),
        mean_scoreable_recall=_mean(
            _optional_float(row.get("target_recall")) for row in scoreable_rows
        ),
        mean_scoreable_precision=_mean(
            _optional_float(row.get("selected_precision")) for row in scoreable_rows
        ),
        mean_scoreable_evidence_window_recall=_mean(
            _optional_float(row.get("evidence_window_recall")) for row in scoreable_rows
        ),
        mean_scoreable_evidence_window_precision=_mean(
            _optional_float(row.get("evidence_window_precision")) for row in scoreable_rows
        ),
    )


def _apply_control_deltas(
    rows: Sequence[LongBenchComparisonRow],
    *,
    control_label: str,
) -> list[LongBenchComparisonRow]:
    controls = {
        row.scope: row
        for row in rows
        if row.run_label == control_label
    }
    if not controls:
        raise ValueError(f"control label {control_label!r} was not found")
    updated: list[LongBenchComparisonRow] = []
    for row in rows:
        control = controls.get(row.scope)
        updated.append(
            LongBenchComparisonRow(
                **{
                    **row.to_dict(),
                    "recall_delta_vs_control": _delta(
                        row.mean_recall,
                        None if control is None else control.mean_recall,
                    ),
                    "window_recall_delta_vs_control": _delta(
                        row.mean_evidence_window_recall,
                        None if control is None else control.mean_evidence_window_recall,
                    ),
                    "selected_token_fraction_delta_vs_control": _delta(
                        row.mean_selected_token_fraction,
                        None if control is None else control.mean_selected_token_fraction,
                    ),
                    "selector_latency_delta_vs_control": _delta(
                        row.mean_selector_latency_sec,
                        None if control is None else control.mean_selector_latency_sec,
                    ),
                }
            )
        )
    return updated


def _delta(value: float | None, control: float | None) -> float | None:
    if value is None or control is None:
        return None
    return value - control


def _is_scoreable(row: Mapping[str, Any]) -> bool:
    explicit = row.get("scoreable_by_answer_presence")
    if explicit is not None:
        return bool(explicit)
    return _optional_float(row.get("target_recall")) is not None


def _mixed_fallback_count(rows: Sequence[Mapping[str, Any]]) -> int | None:
    if not any("mixed_fallback_used" in row for row in rows):
        return None
    return sum(bool(row.get("mixed_fallback_used")) for row in rows)


def _selected_token_count(row: Mapping[str, Any]) -> float | None:
    spans = row.get("selected_spans")
    if isinstance(spans, Sequence) and not isinstance(spans, str):
        lengths = [_span_length(span) for span in spans]
        valid_lengths = [length for length in lengths if length is not None]
        if valid_lengths:
            return float(sum(valid_lengths))

    selected_blocks = row.get("selected_blocks")
    if isinstance(selected_blocks, Sequence) and not isinstance(selected_blocks, str):
        lengths = []
        for block in selected_blocks:
            if not isinstance(block, Mapping):
                continue
            start = _optional_float(block.get("token_start"))
            end = _optional_float(block.get("token_end"))
            if start is not None and end is not None and end >= start:
                lengths.append(end - start)
        if lengths:
            return float(sum(lengths))
    return None


def _selected_token_fraction(row: Mapping[str, Any]) -> float | None:
    selected_tokens = _selected_token_count(row)
    token_count = _optional_float(row.get("tokens"))
    if selected_tokens is None or token_count is None or token_count <= 0:
        return None
    return selected_tokens / token_count


def _span_length(value: Any) -> float | None:
    if not isinstance(value, str) or ":" not in value:
        return None
    start_text, end_text = value.split(":", 1)
    try:
        start = float(start_text)
        end = float(end_text)
    except ValueError:
        return None
    if end < start:
        return None
    return end - start


def _infer_semantic_k(row: Mapping[str, Any]) -> int | None:
    selected_count = _optional_float(row.get("selected_count"))
    ratio = _optional_float(row.get("selected_to_semantic_k_ratio"))
    if selected_count is None or ratio is None or ratio <= 0:
        return None
    return int(round(selected_count / ratio))


def _length_bucket_name(value: Any) -> str:
    if isinstance(value, Mapping):
        name = value.get("name")
        return "" if name is None else str(name)
    return "" if value is None else str(value)


def _join_unique(values: Iterable[Any]) -> str:
    strings = _unique_strings(values)
    if not strings:
        return ""
    return ",".join(strings)


def _unique_strings(values: Iterable[Any]) -> tuple[str, ...]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text:
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _mean(values: Iterable[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return sum(valid_values) / len(valid_values)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _format_value(value: Any, *, precision: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _escape_markdown_cell(value: str) -> str:
    return value.replace("|", "\\|")
