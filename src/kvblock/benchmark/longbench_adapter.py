"""Minimal LongBench adapter for real-block selector benchmarks.

This module converts Hugging Face LongBench records into the existing
``PromptRetrievalCase`` path used by ``dynamic_block_benchmark``. It does not
adopt LongBench's full task evaluator; V1 uses answer-string evidence as a
selector-quality proxy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from kvblock.benchmark.dynamic_block_benchmark import (
    DynamicBlockBenchmarkResult,
    run_dynamic_block_benchmark,
)
from kvblock.benchmark.real_block_representation_sweep import PromptRetrievalCase
from kvblock.kv.block_modes import BlockModeName
from kvblock.kv.qk_aggregation import QKAggregationStrategy
from kvblock.runtime.hooks import RepresentationSource
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

SUPPORTED_LONGBENCH_DATASETS: tuple[str, ...] = (
    "narrativeqa",
    "qasper",
    "hotpotqa",
    "musique",
    "lcc",
    "repobench-p",
)

DEFAULT_LONGBENCH_DATASETS: tuple[str, ...] = (
    "narrativeqa",
    "hotpotqa",
    "lcc",
)

DatasetLoader = Callable[[str, str, str], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class LengthBucket:
    """Approximate LongBench length filter."""

    name: str
    min_length: int | None = None
    max_length: int | None = None

    def contains(self, length: int) -> bool:
        """Return whether ``length`` is inside this bucket."""

        if self.min_length is not None and length < self.min_length:
            return False
        if self.max_length is not None and length >= self.max_length:
            return False
        return True

    def to_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-friendly bucket record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchRecord:
    """Normalized LongBench sample."""

    dataset_name: str
    sample_id: str
    input_text: str
    context: str
    answers: tuple[str, ...]
    length: int | None = None
    raw_dataset_name: str | None = None

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name must be non-empty")
        if not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty")
        if not self.input_text.strip() and not self.context.strip():
            raise ValueError("LongBench record must contain input or context text")
        if not self.answers:
            raise ValueError("LongBench record must contain at least one answer label")
        if any(not answer.strip() for answer in self.answers):
            raise ValueError("answers must be non-empty")
        if self.length is not None and self.length < 0:
            raise ValueError("length must be >= 0 when provided")

    @property
    def prompt_text(self) -> str:
        """Return the stable KVBlock prompt text for this sample."""

        return format_longbench_prompt(self)

    @property
    def approximate_length(self) -> int:
        """Return LongBench length metadata or a word-count fallback."""

        if self.length is not None:
            return self.length
        return len(self.prompt_text.split())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly normalized record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchPromptMetadata:
    """Sidecar metadata for one materialized LongBench prompt file."""

    prompt_name: str
    prompt_file: str
    dataset_name: str
    sample_id: str
    length: int | None
    approximate_length: int
    answer_labels: tuple[str, ...]
    prompt_chars: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly sidecar record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchBenchmarkRunRow:
    """LongBench-enriched dynamic-block benchmark row."""

    dataset_name: str
    sample_id: str
    longbench_length: int | None
    approximate_length: int
    answer_count: int
    prompt_name: str
    prompt_file: str
    model_name: str
    representation_source: str
    representation_name: str
    qk_aggregation_strategy: str
    block_mode: str
    suppression_mode: str
    suppression_threshold: float
    tokens: int
    candidate_block_count: int
    candidate_count_after_suppression: int
    selected_count: int
    selected_to_semantic_k_ratio: float
    selector_latency_sec: float
    prefill_latency_sec: float
    metadata_latency_sec: float
    total_latency_sec: float
    target_recall: float | None
    selected_precision: float | None
    target_hit: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchDatasetSummary:
    """Aggregate LongBench metrics grouped by dataset."""

    dataset_name: str
    run_count: int
    mean_length: float
    mean_tokens: float
    mean_candidate_block_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float
    mean_recall: float | None
    mean_precision: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchBenchmarkResult:
    """Full LongBench adapter benchmark result."""

    samples: tuple[LongBenchPromptMetadata, ...]
    rows: tuple[LongBenchBenchmarkRunRow, ...]
    dataset_summaries: tuple[LongBenchDatasetSummary, ...]
    dynamic_result: DynamicBlockBenchmarkResult
    dataset_repo: str
    split: str
    length_bucket: LengthBucket

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark payload."""

        return {
            "dataset_repo": self.dataset_repo,
            "split": self.split,
            "length_bucket": self.length_bucket.to_dict(),
            "samples": [sample.to_dict() for sample in self.samples],
            "rows": [row.to_dict() for row in self.rows],
            "dataset_summaries": [
                summary.to_dict() for summary in self.dataset_summaries
            ],
            "dynamic_result": self.dynamic_result.to_dict(),
        }


def parse_dataset_names(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse and validate LongBench dataset names."""

    names = (
        tuple(item.strip() for item in value.split(",") if item.strip())
        if isinstance(value, str)
        else tuple(item.strip() for item in value if item.strip())
    )
    if not names:
        raise ValueError("at least one LongBench dataset name is required")
    invalid = [name for name in names if name not in SUPPORTED_LONGBENCH_DATASETS]
    if invalid:
        valid = ", ".join(SUPPORTED_LONGBENCH_DATASETS)
        raise ValueError(f"unsupported LongBench dataset(s) {invalid!r}; valid: {valid}")
    return names


def parse_length_bucket(value: str) -> LengthBucket:
    """Parse a compact length bucket such as ``0-4k`` or ``8k+``."""

    normalized = value.strip().lower()
    if normalized in {"", "all", "any"}:
        return LengthBucket(name="all")
    ranges = {
        "0-4k": LengthBucket("0-4k", min_length=0, max_length=4000),
        "4k-8k": LengthBucket("4k-8k", min_length=4000, max_length=8000),
        "8k+": LengthBucket("8k+", min_length=8000, max_length=None),
    }
    try:
        return ranges[normalized]
    except KeyError as exc:
        raise ValueError("length bucket must be one of all, 0-4k, 4k-8k, 8k+") from exc


def load_longbench_records(
    *,
    dataset_names: Sequence[str] = DEFAULT_LONGBENCH_DATASETS,
    split: str = "test",
    dataset_repo: str = "THUDM/LongBench",
    limit_per_dataset: int | None = None,
    length_bucket: LengthBucket | str = "all",
    dataset_loader: DatasetLoader | None = None,
) -> tuple[LongBenchRecord, ...]:
    """Load and normalize LongBench records via Hugging Face datasets."""

    names = parse_dataset_names(dataset_names)
    if not split.strip():
        raise ValueError("split must be non-empty")
    if not dataset_repo.strip():
        raise ValueError("dataset_repo must be non-empty")
    if limit_per_dataset is not None and limit_per_dataset <= 0:
        raise ValueError("limit_per_dataset must be > 0 when provided")
    bucket = (
        parse_length_bucket(length_bucket)
        if isinstance(length_bucket, str)
        else length_bucket
    )
    loader = dataset_loader or _load_hf_longbench_dataset

    records: list[LongBenchRecord] = []
    for dataset_name in names:
        kept = 0
        for index, raw in enumerate(loader(dataset_repo, dataset_name, split)):
            record = longbench_record_from_mapping(
                raw,
                dataset_name=dataset_name,
                index=index,
            )
            if not bucket.contains(record.approximate_length):
                continue
            records.append(record)
            kept += 1
            if limit_per_dataset is not None and kept >= limit_per_dataset:
                break
    return tuple(records)


def longbench_record_from_mapping(
    row: Mapping[str, Any],
    *,
    dataset_name: str,
    index: int,
) -> LongBenchRecord:
    """Normalize one raw LongBench row."""

    input_text = _first_text(row, ("input", "question", "query", "instruction"))
    context = _first_text(row, ("context", "document", "passage", "text"))
    answers = _normalize_answers(
        row.get("answers", row.get("answer", row.get("outputs", ())))
    )
    length = _parse_optional_int(
        row.get("length", row.get("token_length", row.get("context_length")))
    )
    sample_id = str(row.get("_id", row.get("id", index)))
    raw_dataset_name = row.get("dataset")
    return LongBenchRecord(
        dataset_name=dataset_name,
        sample_id=sample_id,
        input_text=input_text,
        context=context,
        answers=answers,
        length=length,
        raw_dataset_name=None if raw_dataset_name is None else str(raw_dataset_name),
    )


def format_longbench_prompt(record: LongBenchRecord) -> str:
    """Format one LongBench sample as a stable KVBlock prompt."""

    length = "unknown" if record.length is None else str(record.length)
    return "\n".join(
        (
            f"DATASET: {record.dataset_name}",
            f"SAMPLE_ID: {record.sample_id}",
            f"LENGTH: {length}",
            "",
            "CONTEXT:",
            record.context.strip(),
            "",
            "INPUT:",
            record.input_text.strip(),
            "",
            "KVBlock selector target: identify the context blocks needed to answer the input.",
            "",
        )
    )


def materialize_longbench_prompt_cases(
    records: Sequence[LongBenchRecord],
    *,
    prompt_dir: str | Path,
) -> tuple[tuple[PromptRetrievalCase, ...], tuple[LongBenchPromptMetadata, ...]]:
    """Write prompt files and return matching dynamic-benchmark prompt cases."""

    if not records:
        raise ValueError("records must not be empty")
    output_dir = Path(prompt_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases: list[PromptRetrievalCase] = []
    metadata: list[LongBenchPromptMetadata] = []
    for ordinal, record in enumerate(records):
        prompt_name = _prompt_name(record, ordinal=ordinal)
        path = output_dir / f"{prompt_name}.txt"
        prompt_text = record.prompt_text
        path.write_text(prompt_text, encoding="utf-8")
        cases.append(
            PromptRetrievalCase(
                name=prompt_name,
                path=path,
                target_fragments=record.answers,
            )
        )
        metadata.append(
            LongBenchPromptMetadata(
                prompt_name=prompt_name,
                prompt_file=str(path),
                dataset_name=record.dataset_name,
                sample_id=record.sample_id,
                length=record.length,
                approximate_length=record.approximate_length,
                answer_labels=record.answers,
                prompt_chars=len(prompt_text),
            )
        )
    return tuple(cases), tuple(metadata)


def run_longbench_selector_benchmark(
    *,
    model_names: Sequence[str],
    dataset_names: Sequence[str] = DEFAULT_LONGBENCH_DATASETS,
    split: str = "test",
    dataset_repo: str = "THUDM/LongBench",
    limit_per_dataset: int | None = 1,
    length_bucket: LengthBucket | str = "all",
    prompt_cache_dir: str | Path = "results/longbench/prompts",
    block_modes: Sequence[BlockModeName] = ("fixed_40",),
    representation_source: RepresentationSource = "query_mean_last_layer",
    qk_aggregation_strategy: QKAggregationStrategy = "block_max",
    needle_qk_aggregation_strategy: QKAggregationStrategy | None = None,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> LongBenchBenchmarkResult:
    """Run LongBench samples through the existing dynamic-block benchmark."""

    bucket = (
        parse_length_bucket(length_bucket)
        if isinstance(length_bucket, str)
        else length_bucket
    )
    records = load_longbench_records(
        dataset_names=dataset_names,
        split=split,
        dataset_repo=dataset_repo,
        limit_per_dataset=limit_per_dataset,
        length_bucket=bucket,
        dataset_loader=dataset_loader,
    )
    if not records:
        raise ValueError("LongBench selection produced no records")
    cases, sample_metadata = materialize_longbench_prompt_cases(
        records,
        prompt_dir=prompt_cache_dir,
    )
    dynamic_result = run_dynamic_block_benchmark(
        model_names=model_names,
        prompt_cases=cases,
        block_modes=block_modes,
        representation_source=representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy,
        needle_qk_aggregation_strategy=needle_qk_aggregation_strategy,
        load_config_kwargs=load_config_kwargs,
        selector_config=selector_config,
    )
    rows = _longbench_rows(dynamic_result, sample_metadata)
    return LongBenchBenchmarkResult(
        samples=sample_metadata,
        rows=rows,
        dataset_summaries=_dataset_summaries(rows),
        dynamic_result=dynamic_result,
        dataset_repo=dataset_repo,
        split=split,
        length_bucket=bucket,
    )


def write_longbench_benchmark_outputs(
    result: LongBenchBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write LongBench JSON and text reports."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    text_output.write_text(format_longbench_benchmark_report(result), encoding="utf-8")


def format_longbench_benchmark_report(result: LongBenchBenchmarkResult) -> str:
    """Format a compact LongBench benchmark report."""

    lines = [
        "LONGBENCH SELECTOR BENCHMARK",
        f"dataset_repo={result.dataset_repo} split={result.split} length_bucket={result.length_bucket.name}",
        f"samples={len(result.samples)} rows={len(result.rows)}",
        "",
        "DATASET SUMMARIES",
    ]
    for summary in result.dataset_summaries:
        lines.append(
            f"{summary.dataset_name} | runs={summary.run_count} "
            f"mean_length={summary.mean_length:.1f} "
            f"mean_tokens={summary.mean_tokens:.1f} "
            f"mean_candidates={summary.mean_candidate_block_count:.1f} "
            f"mean_selected/K={summary.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
            f"mean_recall={_fmt_optional(summary.mean_recall)} "
            f"mean_precision={_fmt_optional(summary.mean_precision)}"
        )
    lines.append("")
    lines.append("RUNS")
    for row in result.rows:
        lines.append(
            f"{row.dataset_name}:{row.sample_id} | model={row.model_name} "
            f"mode={row.block_mode} qk={row.qk_aggregation_strategy} "
            f"length={row.longbench_length} tokens={row.tokens} "
            f"candidates={row.candidate_block_count} selected/K={row.selected_to_semantic_k_ratio:.3f} "
            f"selector={row.selector_latency_sec:.6f}s "
            f"recall={_fmt_optional(row.target_recall)} "
            f"precision={_fmt_optional(row.selected_precision)}"
        )
    return "\n".join(lines)


def _load_hf_longbench_dataset(
    dataset_repo: str,
    dataset_name: str,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "LongBench loading requires the optional 'datasets' dependency. "
            "Install it in the benchmark environment."
        ) from exc
    return load_dataset(dataset_repo, dataset_name, split=split)


def _longbench_rows(
    dynamic_result: DynamicBlockBenchmarkResult,
    samples: Sequence[LongBenchPromptMetadata],
) -> tuple[LongBenchBenchmarkRunRow, ...]:
    by_prompt = {sample.prompt_name: sample for sample in samples}
    rows: list[LongBenchBenchmarkRunRow] = []
    for row in dynamic_result.rows:
        sample = by_prompt[row.prompt_name]
        rows.append(
            LongBenchBenchmarkRunRow(
                dataset_name=sample.dataset_name,
                sample_id=sample.sample_id,
                longbench_length=sample.length,
                approximate_length=sample.approximate_length,
                answer_count=len(sample.answer_labels),
                prompt_name=row.prompt_name,
                prompt_file=row.prompt_file,
                model_name=row.model_name,
                representation_source=row.representation_source,
                representation_name=row.representation_name,
                qk_aggregation_strategy=row.qk_aggregation_strategy,
                block_mode=row.block_mode,
                suppression_mode=row.suppression_mode,
                suppression_threshold=row.suppression_threshold,
                tokens=row.tokens,
                candidate_block_count=row.candidate_block_count,
                candidate_count_after_suppression=row.candidate_count_after_suppression,
                selected_count=row.selected_count,
                selected_to_semantic_k_ratio=row.selected_to_semantic_k_ratio,
                selector_latency_sec=row.selector_latency_sec,
                prefill_latency_sec=row.prefill_latency_sec,
                metadata_latency_sec=row.metadata_latency_sec,
                total_latency_sec=row.total_latency_sec,
                target_recall=row.retrieval_quality.target_recall,
                selected_precision=row.retrieval_quality.selected_precision,
                target_hit=row.retrieval_quality.target_hit,
            )
        )
    return tuple(rows)


def _dataset_summaries(
    rows: Sequence[LongBenchBenchmarkRunRow],
) -> tuple[LongBenchDatasetSummary, ...]:
    grouped: dict[str, list[LongBenchBenchmarkRunRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dataset_name].append(row)
    return tuple(
        LongBenchDatasetSummary(
            dataset_name=dataset_name,
            run_count=len(group),
            mean_length=_mean(row.approximate_length for row in group) or 0.0,
            mean_tokens=_mean(row.tokens for row in group) or 0.0,
            mean_candidate_block_count=_mean(row.candidate_block_count for row in group) or 0.0,
            mean_selected_to_semantic_k_ratio=_mean(
                row.selected_to_semantic_k_ratio for row in group
            )
            or 0.0,
            mean_selector_latency_sec=_mean(row.selector_latency_sec for row in group) or 0.0,
            mean_recall=_mean_optional(row.target_recall for row in group),
            mean_precision=_mean_optional(row.selected_precision for row in group),
        )
        for dataset_name, group in sorted(grouped.items())
    )


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = _coerce_text(value)
            if text.strip():
                return text
    return ""


def _normalize_answers(value: Any) -> tuple[str, ...]:
    flattened: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        text = _coerce_text(item).strip()
        if text:
            flattened.append(text)

    visit(value)
    deduped = tuple(dict.fromkeys(flattened))
    if not deduped:
        raise ValueError("LongBench answers must contain at least one non-empty label")
    return deduped


def _coerce_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(_coerce_text(item) for item in value)
    return str(value)


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _prompt_name(record: LongBenchRecord, *, ordinal: int) -> str:
    sample = _slugify(record.sample_id)
    return f"longbench_{record.dataset_name}_{ordinal:04d}_{sample}"


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug[:80].strip("_") or "sample"


def _mean(values: Iterable[float | int]) -> float | None:
    items = [float(value) for value in values]
    if not items:
        return None
    return sum(items) / len(items)


def _mean_optional(values: Iterable[float | None]) -> float | None:
    items = [float(value) for value in values if value is not None]
    if not items:
        return None
    return sum(items) / len(items)


def _fmt_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
