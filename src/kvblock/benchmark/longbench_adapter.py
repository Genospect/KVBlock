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
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Literal, Mapping, Sequence, cast
import zipfile

import torch

from kvblock.benchmark.dynamic_block_benchmark import (
    DynamicBlockBenchmarkResult,
    DynamicBlockRunRow,
    RerankMode,
    query_prompt_override_for_representation,
    run_dynamic_block_benchmark,
)
from kvblock.benchmark.real_block_representation_sweep import PromptRetrievalCase
from kvblock.kv.block_modes import BlockModeName
from kvblock.kv.qk_aggregation import QKAggregationStrategy
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
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
OracleMode = Literal["none", "dense_qk"]
VALID_ORACLE_MODES: tuple[OracleMode, ...] = ("none", "dense_qk")
DEFAULT_ORACLE_TOP_K: tuple[int, ...] = (4, 8, 16, 32)


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
    answer_present_count: int
    answer_missing_count: int
    answer_presence_rate: float
    answer_present: bool
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
    answer_present_count: int
    answer_missing_count: int
    answer_presence_rate: float
    scoreable_by_answer_presence: bool
    expected_block_count: int
    selected_expected_block_count: int
    missed_expected_block_count: int
    expected_block_ids: tuple[int, ...]
    expected_block_ranks: tuple[int, ...]
    best_expected_rank: int | None
    best_expected_score: float | None
    score_gap_to_top1: float | None
    recall_at_4: float | None
    recall_at_8: float | None
    recall_at_16: float | None
    recall_at_32: float | None
    recall_at_64: float | None
    neighbor_recall_at_1: float | None
    neighbor_recall_at_2: float | None
    best_neighbor_distance: int | None
    evidence_window_radius: int
    evidence_window_recall: float | None
    evidence_window_precision: float | None
    evidence_window_recall_at_0: float | None
    evidence_window_recall_at_1: float | None
    evidence_window_recall_at_2: float | None
    evidence_window_precision_at_0: float | None
    evidence_window_precision_at_1: float | None
    evidence_window_precision_at_2: float | None
    semantic_selected_block_ids: tuple[int, ...]
    selected_block_ids: tuple[int, ...]
    missed_expected_block_ids: tuple[int, ...]
    selected_spans: tuple[str, ...]
    expected_block_distance: int | None
    selected_blocks: tuple[dict[str, Any], ...]
    expected_blocks: tuple[dict[str, Any], ...]
    missed_expected_blocks: tuple[dict[str, Any], ...]
    top_ranked_blocks: tuple[dict[str, Any], ...]
    prompt_name: str
    prompt_file: str
    model_name: str
    representation_source: str
    representation_name: str
    qk_aggregation_strategy: str
    rerank_mode: str
    rerank_weight: float
    refine_top_n_tokens: int
    neighbor_expansion: int
    halo_radius: int
    max_selected_blocks: int | None
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
    expected_rank_movements: tuple[dict[str, int | None], ...] = ()
    oracle_mode: str = "none"
    oracle_top_k_values: tuple[int, ...] = ()
    oracle_top_block_ids: tuple[int, ...] = ()
    oracle_total_mass: float | None = None
    oracle_selected_mass_fraction: float | None = None
    oracle_semantic_selected_mass_fraction: float | None = None
    oracle_expected_mass_fraction: float | None = None
    oracle_topk_recall_at_4: float | None = None
    oracle_topk_recall_at_8: float | None = None
    oracle_topk_recall_at_16: float | None = None
    oracle_topk_recall_at_32: float | None = None
    oracle_expected_block_ranks: tuple[int, ...] = ()
    selected_vs_oracle_jaccard_at_4: float | None = None
    selected_vs_oracle_jaccard_at_8: float | None = None
    selected_vs_oracle_jaccard_at_16: float | None = None
    selected_vs_oracle_jaccard_at_32: float | None = None

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
    mean_answer_presence_rate: float
    scoreable_run_count: int
    mean_expected_block_count: float
    mean_selected_to_semantic_k_ratio: float
    mean_selector_latency_sec: float
    mean_recall: float | None
    mean_precision: float | None
    mean_evidence_window_recall: float | None
    mean_evidence_window_precision: float | None
    mean_oracle_selected_mass_fraction: float | None
    mean_oracle_semantic_selected_mass_fraction: float | None
    mean_oracle_expected_mass_fraction: float | None
    mean_oracle_topk_recall_at_4: float | None
    mean_oracle_topk_recall_at_8: float | None
    mean_oracle_topk_recall_at_16: float | None
    mean_oracle_topk_recall_at_32: float | None

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
    evidence_window_radius: int
    oracle_mode: str
    oracle_top_k_values: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly benchmark payload."""

        return {
            "dataset_repo": self.dataset_repo,
            "split": self.split,
            "length_bucket": self.length_bucket.to_dict(),
            "evidence_window_radius": self.evidence_window_radius,
            "oracle_mode": self.oracle_mode,
            "oracle_top_k_values": list(self.oracle_top_k_values),
            "samples": [sample.to_dict() for sample in self.samples],
            "rows": [row.to_dict() for row in self.rows],
            "dataset_summaries": [
                summary.to_dict() for summary in self.dataset_summaries
            ],
            "dynamic_result": self.dynamic_result.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class DenseQKOracleDiagnostics:
    """Dense QK-derived block mass ranking for one model/prompt/block mode."""

    top_block_ids: tuple[int, ...]
    total_mass: float
    mass_by_block_id: dict[int, float]
    rank_by_block_id: dict[int, int]


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


def parse_oracle_mode(name: str) -> OracleMode:
    """Validate and return a LongBench oracle diagnostic mode."""

    normalized = name.strip()
    if normalized not in VALID_ORACLE_MODES:
        valid = ", ".join(VALID_ORACLE_MODES)
        raise ValueError(f"unknown oracle mode {name!r}; valid: {valid}")
    return cast(OracleMode, normalized)


def parse_oracle_top_k(value: str | Sequence[int]) -> tuple[int, ...]:
    """Parse oracle reporting cutoffs such as ``4,8,16,32``."""

    values = (
        tuple(int(item.strip()) for item in value.split(",") if item.strip())
        if isinstance(value, str)
        else tuple(int(item) for item in value)
    )
    if not values:
        raise ValueError("oracle_top_k must contain at least one cutoff")
    if any(item <= 0 for item in values):
        raise ValueError("oracle_top_k values must be > 0")
    return tuple(sorted(dict.fromkeys(values)))


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
    # The selector's query summary is derived from the prompt tail. Keep the
    # actual LongBench input/question last; a generic adapter instruction here
    # would dominate query_mean_last_layer and obscure the retrieval target.
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
        present_answers, missing_answers = _answer_presence(
            prompt_text,
            record.answers,
        )
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
                answer_present_count=len(present_answers),
                answer_missing_count=len(missing_answers),
                answer_presence_rate=len(present_answers) / len(record.answers),
                answer_present=bool(present_answers),
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
    rerank_mode: RerankMode = "none",
    rerank_weight: float = 0.3,
    refine_top_n_tokens: int = 4,
    neighbor_expansion: int = 0,
    halo_radius: int = 0,
    max_selected_blocks: int | None = None,
    evidence_window_radius: int = 0,
    oracle_mode: OracleMode = "none",
    oracle_top_k: Sequence[int] = DEFAULT_ORACLE_TOP_K,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> LongBenchBenchmarkResult:
    """Run LongBench samples through the existing dynamic-block benchmark."""

    if evidence_window_radius < 0:
        raise ValueError("evidence_window_radius must be >= 0")
    resolved_oracle_mode = parse_oracle_mode(oracle_mode)
    resolved_oracle_top_k = parse_oracle_top_k(oracle_top_k)
    if resolved_oracle_mode != "none" and not all(
        str(block_mode).startswith("fixed_") for block_mode in block_modes
    ):
        raise ValueError("dense_qk oracle currently supports fixed block modes only")
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
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        load_config_kwargs=load_config_kwargs,
        selector_config=selector_config,
        include_block_inspections=True,
    )
    oracle_by_key = (
        {}
        if resolved_oracle_mode == "none"
        else _build_dense_qk_oracles(
            dynamic_result,
            sample_metadata,
            model_names=model_names,
            representation_source=representation_source,
            load_config_kwargs=load_config_kwargs,
            max_top_k=max(resolved_oracle_top_k),
        )
    )
    rows = _longbench_rows(
        dynamic_result,
        sample_metadata,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k=resolved_oracle_top_k,
        oracle_by_key=oracle_by_key,
    )
    return LongBenchBenchmarkResult(
        samples=sample_metadata,
        rows=rows,
        dataset_summaries=_dataset_summaries(rows),
        dynamic_result=dynamic_result,
        dataset_repo=dataset_repo,
        split=split,
        length_bucket=bucket,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k_values=resolved_oracle_top_k,
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
        f"evidence_window_radius={result.evidence_window_radius}",
        f"oracle_mode={result.oracle_mode} oracle_top_k={list(result.oracle_top_k_values)}",
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
            f"answer_presence={summary.mean_answer_presence_rate:.3f} "
            f"scoreable_runs={summary.scoreable_run_count}/{summary.run_count} "
            f"mean_expected_blocks={summary.mean_expected_block_count:.1f} "
            f"mean_selected/K={summary.mean_selected_to_semantic_k_ratio:.3f} "
            f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
            f"mean_recall={_fmt_optional(summary.mean_recall)} "
            f"mean_precision={_fmt_optional(summary.mean_precision)} "
            f"mean_window_recall={_fmt_optional(summary.mean_evidence_window_recall)} "
            f"mean_window_precision={_fmt_optional(summary.mean_evidence_window_precision)} "
            f"mean_oracle_mass={_fmt_optional(summary.mean_oracle_selected_mass_fraction)} "
            f"mean_oracle_semantic_mass={_fmt_optional(summary.mean_oracle_semantic_selected_mass_fraction)} "
            f"mean_oracle_expected_mass={_fmt_optional(summary.mean_oracle_expected_mass_fraction)} "
            f"mean_oracle_recall@4/8/16/32="
            f"{_fmt_optional(summary.mean_oracle_topk_recall_at_4)}/"
            f"{_fmt_optional(summary.mean_oracle_topk_recall_at_8)}/"
            f"{_fmt_optional(summary.mean_oracle_topk_recall_at_16)}/"
            f"{_fmt_optional(summary.mean_oracle_topk_recall_at_32)}"
        )
    lines.append("")
    lines.append("RUNS")
    for row in result.rows:
        lines.append(
            f"{row.dataset_name}:{row.sample_id} | model={row.model_name} "
            f"mode={row.block_mode} qk={row.qk_aggregation_strategy} "
            f"rerank={row.rerank_mode}@{row.rerank_weight:.2f} "
            f"refine_top_n={row.refine_top_n_tokens} "
            f"neighbor_expansion={row.neighbor_expansion} "
            f"halo={row.halo_radius} cap={_fmt_int_optional(row.max_selected_blocks)} "
            f"length={row.longbench_length} tokens={row.tokens} "
            f"candidates={row.candidate_block_count} selected/K={row.selected_to_semantic_k_ratio:.3f} "
            f"answer_presence={row.answer_present_count}/{row.answer_count} "
            f"expected_blocks={row.expected_block_count} "
            f"expected_distance={_fmt_optional(row.expected_block_distance)} "
            f"expected_ranks={list(row.expected_block_ranks)} "
            f"expected_rank_moves={_format_rank_movements(row.expected_rank_movements)} "
            f"best_expected_rank={_fmt_optional(row.best_expected_rank)} "
            f"score_gap={_fmt_optional(row.score_gap_to_top1)} "
            f"recall@4/8/16/32/64="
            f"{_fmt_optional(row.recall_at_4)}/"
            f"{_fmt_optional(row.recall_at_8)}/"
            f"{_fmt_optional(row.recall_at_16)}/"
            f"{_fmt_optional(row.recall_at_32)}/"
            f"{_fmt_optional(row.recall_at_64)} "
            f"neighbor_recall@1={_fmt_optional(row.neighbor_recall_at_1)} "
            f"neighbor_recall@2={_fmt_optional(row.neighbor_recall_at_2)} "
            f"best_neighbor_distance={_fmt_optional(row.best_neighbor_distance)} "
            f"evidence_window@{row.evidence_window_radius}="
            f"{_fmt_optional(row.evidence_window_recall)}/"
            f"{_fmt_optional(row.evidence_window_precision)} "
            f"evidence_recall@0/1/2="
            f"{_fmt_optional(row.evidence_window_recall_at_0)}/"
            f"{_fmt_optional(row.evidence_window_recall_at_1)}/"
            f"{_fmt_optional(row.evidence_window_recall_at_2)} "
            f"evidence_precision@0/1/2="
            f"{_fmt_optional(row.evidence_window_precision_at_0)}/"
            f"{_fmt_optional(row.evidence_window_precision_at_1)}/"
            f"{_fmt_optional(row.evidence_window_precision_at_2)} "
            f"oracle_mass={_fmt_optional(row.oracle_selected_mass_fraction)} "
            f"oracle_semantic_mass={_fmt_optional(row.oracle_semantic_selected_mass_fraction)} "
            f"oracle_expected_mass={_fmt_optional(row.oracle_expected_mass_fraction)} "
            f"oracle_recall@4/8/16/32="
            f"{_fmt_optional(row.oracle_topk_recall_at_4)}/"
            f"{_fmt_optional(row.oracle_topk_recall_at_8)}/"
            f"{_fmt_optional(row.oracle_topk_recall_at_16)}/"
            f"{_fmt_optional(row.oracle_topk_recall_at_32)} "
            f"oracle_expected_ranks={list(row.oracle_expected_block_ranks)} "
            f"selected_vs_oracle_jaccard@4/8/16/32="
            f"{_fmt_optional(row.selected_vs_oracle_jaccard_at_4)}/"
            f"{_fmt_optional(row.selected_vs_oracle_jaccard_at_8)}/"
            f"{_fmt_optional(row.selected_vs_oracle_jaccard_at_16)}/"
            f"{_fmt_optional(row.selected_vs_oracle_jaccard_at_32)} "
            f"oracle_top_ids={list(row.oracle_top_block_ids)} "
            f"selected_ids={list(row.selected_block_ids)} "
            f"semantic_selected_ids={list(row.semantic_selected_block_ids)} "
            f"expected_ids={list(row.expected_block_ids)} "
            f"selector={row.selector_latency_sec:.6f}s "
            f"recall={_fmt_optional(row.target_recall)} "
            f"precision={_fmt_optional(row.selected_precision)}"
        )
        if row.scoreable_by_answer_presence:
            lines.append(f"  selected_blocks: {_format_block_records(row.selected_blocks)}")
            lines.append(f"  expected_blocks: {_format_block_records(row.expected_blocks)}")
            lines.append(f"  top_ranked_blocks: {_format_block_records(row.top_ranked_blocks)}")
    return "\n".join(lines)


def _load_hf_longbench_dataset(
    dataset_repo: str,
    dataset_name: str,
    split: str,
) -> Iterable[Mapping[str, Any]]:
    # Newer `datasets` releases reject Hub datasets backed by Python loading
    # scripts. LongBench's script only downloads `data.zip` and reads one JSONL
    # file, so prefer that script-free path and keep load_dataset as a fallback.
    if split != "test":
        raise ValueError("LongBench adapter currently supports the test split only")
    direct_error: Exception | None = None
    try:
        return _load_longbench_jsonl_from_hub(dataset_repo, dataset_name)
    except Exception as exc:
        direct_error = exc

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise ImportError(
            "LongBench loading requires the optional 'datasets' dependency. "
            "Install it in the benchmark environment."
        ) from exc
    try:
        return load_dataset(dataset_repo, dataset_name, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise
        detail = "" if direct_error is None else f" Direct data.zip error: {direct_error}"
        raise RuntimeError(
            "This datasets version no longer supports LongBench's Hub loading "
            "script, and direct data.zip loading also failed. Install the "
            "optional huggingface_hub dependency or use a datasets version that "
            f"still supports script-based datasets.{detail}"
        ) from exc


def _load_longbench_jsonl_from_hub(
    dataset_repo: str,
    dataset_name: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load LongBench rows directly from the repository's data.zip archive."""

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency of datasets
        raise ImportError(
            "Direct LongBench loading requires huggingface_hub."
        ) from exc
    zip_path = Path(
        hf_hub_download(
            repo_id=dataset_repo,
            filename="data.zip",
            repo_type="dataset",
        )
    )
    member_name = f"data/{dataset_name}.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        if member_name not in archive.namelist():
            raise FileNotFoundError(
                f"{member_name!r} not found in LongBench data.zip"
            )
        rows: list[Mapping[str, Any]] = []
        with archive.open(member_name) as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped.decode("utf-8")))
    return tuple(rows)


def _longbench_rows(
    dynamic_result: DynamicBlockBenchmarkResult,
    samples: Sequence[LongBenchPromptMetadata],
    *,
    evidence_window_radius: int,
    oracle_mode: OracleMode,
    oracle_top_k: Sequence[int],
    oracle_by_key: Mapping[tuple[str, str, str], DenseQKOracleDiagnostics],
) -> tuple[LongBenchBenchmarkRunRow, ...]:
    by_prompt = {sample.prompt_name: sample for sample in samples}
    rows: list[LongBenchBenchmarkRunRow] = []
    for row in dynamic_result.rows:
        sample = by_prompt[row.prompt_name]
        oracle = oracle_by_key.get((row.model_name, row.prompt_name, row.block_mode))
        oracle_metrics = _oracle_metrics(
            oracle,
            selected_ids=row.selected_ids,
            semantic_selected_ids=row.semantic_selected_ids,
            expected_ids=row.retrieval_quality.expected_block_ids,
            top_k_values=oracle_top_k,
        )
        inspection_by_id = _inspection_by_id(row.block_inspection_records)
        rank_by_id = _rank_by_block_id(row.suppression_decisions)
        score_by_id = _score_by_block_id(row.suppression_decisions)
        expected_ranks = _expected_ranks(
            row.retrieval_quality.expected_block_ids,
            rank_by_id=rank_by_id,
        )
        best_expected_rank = min(expected_ranks) if expected_ranks else None
        best_expected_score = _best_expected_score(
            row.retrieval_quality.expected_block_ids,
            score_by_id=score_by_id,
        )
        top1_score = _top1_score(row.suppression_decisions)
        score_gap_to_top1 = (
            None
            if top1_score is None or best_expected_score is None
            else top1_score - best_expected_score
        )
        rows.append(
            LongBenchBenchmarkRunRow(
                dataset_name=sample.dataset_name,
                sample_id=sample.sample_id,
                longbench_length=sample.length,
                approximate_length=sample.approximate_length,
                answer_count=len(sample.answer_labels),
                answer_present_count=sample.answer_present_count,
                answer_missing_count=sample.answer_missing_count,
                answer_presence_rate=sample.answer_presence_rate,
                scoreable_by_answer_presence=sample.answer_present,
                expected_block_count=len(row.retrieval_quality.expected_block_ids),
                selected_expected_block_count=len(
                    row.retrieval_quality.selected_expected_block_ids
                ),
                missed_expected_block_count=len(
                    row.retrieval_quality.missed_expected_block_ids
                ),
                expected_block_ids=row.retrieval_quality.expected_block_ids,
                expected_block_ranks=expected_ranks,
                best_expected_rank=best_expected_rank,
                best_expected_score=best_expected_score,
                score_gap_to_top1=score_gap_to_top1,
                recall_at_4=_block_recall_at_k(
                    row.retrieval_quality.expected_block_ids,
                    ranked_ids=tuple(rank_by_id),
                    k=4,
                ),
                recall_at_8=_block_recall_at_k(
                    row.retrieval_quality.expected_block_ids,
                    ranked_ids=tuple(rank_by_id),
                    k=8,
                ),
                recall_at_16=_block_recall_at_k(
                    row.retrieval_quality.expected_block_ids,
                    ranked_ids=tuple(rank_by_id),
                    k=16,
                ),
                recall_at_32=_block_recall_at_k(
                    row.retrieval_quality.expected_block_ids,
                    ranked_ids=tuple(rank_by_id),
                    k=32,
                ),
                recall_at_64=_block_recall_at_k(
                    row.retrieval_quality.expected_block_ids,
                    ranked_ids=tuple(rank_by_id),
                    k=64,
                ),
                neighbor_recall_at_1=_neighbor_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.semantic_selected_ids,
                    radius=1,
                ),
                neighbor_recall_at_2=_neighbor_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.semantic_selected_ids,
                    radius=2,
                ),
                best_neighbor_distance=_expected_block_distance(
                    selected_ids=row.semantic_selected_ids,
                    expected_ids=row.retrieval_quality.expected_block_ids,
                ),
                evidence_window_radius=evidence_window_radius,
                evidence_window_recall=_window_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=evidence_window_radius,
                ),
                evidence_window_precision=_window_precision(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=evidence_window_radius,
                ),
                evidence_window_recall_at_0=_window_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=0,
                ),
                evidence_window_recall_at_1=_window_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=1,
                ),
                evidence_window_recall_at_2=_window_recall(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=2,
                ),
                evidence_window_precision_at_0=_window_precision(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=0,
                ),
                evidence_window_precision_at_1=_window_precision(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=1,
                ),
                evidence_window_precision_at_2=_window_precision(
                    row.retrieval_quality.expected_block_ids,
                    selected_ids=row.selected_ids,
                    radius=2,
                ),
                semantic_selected_block_ids=row.semantic_selected_ids,
                selected_block_ids=row.selected_ids,
                missed_expected_block_ids=row.retrieval_quality.missed_expected_block_ids,
                selected_spans=row.selected_spans,
                expected_block_distance=_expected_block_distance(
                    selected_ids=row.selected_ids,
                    expected_ids=row.retrieval_quality.expected_block_ids,
                ),
                selected_blocks=_records_for_ids(inspection_by_id, row.selected_ids),
                expected_blocks=_records_for_ids(
                    inspection_by_id,
                    row.retrieval_quality.expected_block_ids,
                ),
                missed_expected_blocks=_records_for_ids(
                    inspection_by_id,
                    row.retrieval_quality.missed_expected_block_ids,
                ),
                top_ranked_blocks=_top_ranked_blocks(
                    row.suppression_decisions,
                    inspection_by_id=inspection_by_id,
                    limit=10,
                ),
                prompt_name=row.prompt_name,
                prompt_file=row.prompt_file,
                model_name=row.model_name,
                representation_source=row.representation_source,
                representation_name=row.representation_name,
                qk_aggregation_strategy=row.qk_aggregation_strategy,
                rerank_mode=row.rerank_mode,
                rerank_weight=row.rerank_weight,
                refine_top_n_tokens=row.refine_top_n_tokens,
                neighbor_expansion=row.neighbor_expansion,
                halo_radius=row.halo_radius,
                max_selected_blocks=row.max_selected_blocks,
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
                expected_rank_movements=_expected_rank_movements(
                    row.retrieval_quality.expected_block_ids,
                    suppression_decisions=row.suppression_decisions,
                ),
                oracle_mode=oracle_mode,
                oracle_top_k_values=tuple(oracle_top_k),
                oracle_top_block_ids=oracle_metrics["oracle_top_block_ids"],
                oracle_total_mass=oracle_metrics["oracle_total_mass"],
                oracle_selected_mass_fraction=oracle_metrics[
                    "oracle_selected_mass_fraction"
                ],
                oracle_semantic_selected_mass_fraction=oracle_metrics[
                    "oracle_semantic_selected_mass_fraction"
                ],
                oracle_expected_mass_fraction=oracle_metrics[
                    "oracle_expected_mass_fraction"
                ],
                oracle_topk_recall_at_4=oracle_metrics["oracle_topk_recall_at_4"],
                oracle_topk_recall_at_8=oracle_metrics["oracle_topk_recall_at_8"],
                oracle_topk_recall_at_16=oracle_metrics["oracle_topk_recall_at_16"],
                oracle_topk_recall_at_32=oracle_metrics["oracle_topk_recall_at_32"],
                oracle_expected_block_ranks=oracle_metrics[
                    "oracle_expected_block_ranks"
                ],
                selected_vs_oracle_jaccard_at_4=oracle_metrics[
                    "selected_vs_oracle_jaccard_at_4"
                ],
                selected_vs_oracle_jaccard_at_8=oracle_metrics[
                    "selected_vs_oracle_jaccard_at_8"
                ],
                selected_vs_oracle_jaccard_at_16=oracle_metrics[
                    "selected_vs_oracle_jaccard_at_16"
                ],
                selected_vs_oracle_jaccard_at_32=oracle_metrics[
                    "selected_vs_oracle_jaccard_at_32"
                ],
            )
        )
    return tuple(rows)


def _build_dense_qk_oracles(
    dynamic_result: DynamicBlockBenchmarkResult,
    samples: Sequence[LongBenchPromptMetadata],
    *,
    model_names: Sequence[str],
    representation_source: RepresentationSource,
    load_config_kwargs: dict[str, Any] | None,
    max_top_k: int,
) -> dict[tuple[str, str, str], DenseQKOracleDiagnostics]:
    """Build dense QK block-mass rankings without materializing LxL attention."""

    by_prompt = {sample.prompt_name: sample for sample in samples}
    exemplar_by_key: dict[tuple[str, str, str], DynamicBlockRunRow] = {}
    for row in dynamic_result.rows:
        key = (row.model_name, row.prompt_name, row.block_mode)
        exemplar_by_key.setdefault(key, row)

    load_kwargs = dict(load_config_kwargs or {})
    diagnostics: dict[tuple[str, str, str], DenseQKOracleDiagnostics] = {}
    for model_name in model_names:
        runtime = LocalHfRuntime(
            RuntimeLoadConfig(model_name=model_name, **load_kwargs),
            capture_config=HiddenStateCaptureConfig(
                representation_source=representation_source,
            ),
        )
        runtime.load_model()
        for key, row in exemplar_by_key.items():
            if key[0] != model_name:
                continue
            sample = by_prompt[row.prompt_name]
            prompt_text = Path(sample.prompt_file).read_text(encoding="utf-8")
            diagnostics[key] = _dense_qk_oracle_for_row(
                runtime,
                prompt_text=prompt_text,
                row=row,
                representation_source=representation_source,
                max_top_k=max_top_k,
            )
    return diagnostics


def _dense_qk_oracle_for_row(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    row: DynamicBlockRunRow,
    representation_source: RepresentationSource,
    max_top_k: int,
) -> DenseQKOracleDiagnostics:
    prompt_prefill = runtime.prefill(prompt_text)
    query_prompt = query_prompt_override_for_representation(
        prompt_text,
        representation_source=representation_source,
    )
    query_prefill = (
        None if query_prompt is None else runtime.prefill(query_prompt)
    )
    token_heads = prompt_prefill.per_head_token_representations
    query_heads = (
        prompt_prefill.per_head_query_representation
        if query_prefill is None
        else query_prefill.per_head_query_representation
    )
    if token_heads is None or query_heads is None:
        raise ValueError("dense_qk oracle requires a query/key representation source")

    token_mass = _dense_qk_token_mass(token_heads, query_heads)
    spans = _oracle_block_spans(row.block_inspection_records)
    mass_by_block_id: dict[int, float] = {}
    for block_id, start, end in spans:
        clipped_start = max(0, min(start, token_mass.numel()))
        clipped_end = max(clipped_start, min(end, token_mass.numel()))
        mass_by_block_id[block_id] = float(token_mass[clipped_start:clipped_end].sum())
    total_mass = sum(mass_by_block_id.values())
    ranked = tuple(
        block_id
        for block_id, _mass in sorted(
            mass_by_block_id.items(),
            key=lambda item: (item[1], -item[0]),
            reverse=True,
        )
    )
    rank_by_block_id = {
        block_id: rank
        for rank, block_id in enumerate(ranked, start=1)
    }
    return DenseQKOracleDiagnostics(
        top_block_ids=ranked[:max_top_k],
        total_mass=total_mass,
        mass_by_block_id=mass_by_block_id,
        rank_by_block_id=rank_by_block_id,
    )


def _dense_qk_token_mass(
    per_head_token_representations: torch.Tensor,
    per_head_query_representation: torch.Tensor,
) -> torch.Tensor:
    """Return mean per-head softmax QK mass for each prompt token."""

    token_heads = per_head_token_representations.detach().to(dtype=torch.float32, device="cpu")
    query_heads = per_head_query_representation.detach().to(dtype=torch.float32, device="cpu")
    if token_heads.ndim != 3:
        raise ValueError("per_head_token_representations must have shape [heads, tokens, dim]")
    if query_heads.ndim != 2:
        raise ValueError("per_head_query_representation must have shape [heads, dim]")
    if token_heads.shape[0] != query_heads.shape[0]:
        raise ValueError("query/key head counts must match")
    if token_heads.shape[2] != query_heads.shape[1]:
        raise ValueError("query/key head dimensions must match")
    scale = math.sqrt(float(token_heads.shape[2]))
    logits = (token_heads * query_heads[:, None, :]).sum(dim=-1) / scale
    return torch.softmax(logits, dim=-1).mean(dim=0)


def _oracle_block_spans(
    block_inspection_records: Sequence[Mapping[str, Any]],
) -> tuple[tuple[int, int, int], ...]:
    spans: list[tuple[int, int, int]] = []
    for record in block_inspection_records:
        if "block_id" not in record:
            continue
        start = record.get("token_start")
        end = record.get("token_end")
        if start is None or end is None:
            continue
        spans.append((int(record["block_id"]), int(start), int(end)))
    spans.sort(key=lambda item: (item[1], item[2], item[0]))
    return tuple(spans)


def _oracle_metrics(
    oracle: DenseQKOracleDiagnostics | None,
    *,
    selected_ids: Sequence[int],
    semantic_selected_ids: Sequence[int],
    expected_ids: Sequence[int],
    top_k_values: Sequence[int],
) -> dict[str, Any]:
    if oracle is None:
        return {
            "oracle_top_block_ids": (),
            "oracle_total_mass": None,
            "oracle_selected_mass_fraction": None,
            "oracle_semantic_selected_mass_fraction": None,
            "oracle_expected_mass_fraction": None,
            "oracle_topk_recall_at_4": None,
            "oracle_topk_recall_at_8": None,
            "oracle_topk_recall_at_16": None,
            "oracle_topk_recall_at_32": None,
            "oracle_expected_block_ranks": (),
            "selected_vs_oracle_jaccard_at_4": None,
            "selected_vs_oracle_jaccard_at_8": None,
            "selected_vs_oracle_jaccard_at_16": None,
            "selected_vs_oracle_jaccard_at_32": None,
        }
    selected = tuple(int(block_id) for block_id in selected_ids)
    semantic_selected = tuple(int(block_id) for block_id in semantic_selected_ids)
    expected = tuple(int(block_id) for block_id in expected_ids)
    total_mass = oracle.total_mass

    metrics: dict[str, Any] = {
        "oracle_top_block_ids": oracle.top_block_ids,
        "oracle_total_mass": total_mass,
        "oracle_selected_mass_fraction": _oracle_mass_fraction(
            selected,
            oracle=oracle,
        ),
        "oracle_semantic_selected_mass_fraction": _oracle_mass_fraction(
            semantic_selected,
            oracle=oracle,
        ),
        "oracle_expected_mass_fraction": _oracle_mass_fraction(
            expected,
            oracle=oracle,
        ),
        "oracle_expected_block_ranks": tuple(
            oracle.rank_by_block_id[block_id]
            for block_id in expected
            if block_id in oracle.rank_by_block_id
        ),
    }
    for k in (4, 8, 16, 32):
        metrics[f"oracle_topk_recall_at_{k}"] = (
            _oracle_topk_recall(selected, oracle=oracle, k=k)
            if k in top_k_values
            else None
        )
        metrics[f"selected_vs_oracle_jaccard_at_{k}"] = (
            _oracle_jaccard(selected, oracle=oracle, k=k)
            if k in top_k_values
            else None
        )
    return metrics


def _oracle_mass_fraction(
    block_ids: Sequence[int],
    *,
    oracle: DenseQKOracleDiagnostics,
) -> float | None:
    if oracle.total_mass <= 0:
        return None
    mass = sum(oracle.mass_by_block_id.get(int(block_id), 0.0) for block_id in block_ids)
    return mass / oracle.total_mass


def _oracle_topk_recall(
    selected_ids: Sequence[int],
    *,
    oracle: DenseQKOracleDiagnostics,
    k: int,
) -> float | None:
    top = set(oracle.top_block_ids[:k])
    if not top:
        return None
    selected = {int(block_id) for block_id in selected_ids}
    return len(selected & top) / len(top)


def _oracle_jaccard(
    selected_ids: Sequence[int],
    *,
    oracle: DenseQKOracleDiagnostics,
    k: int,
) -> float | None:
    top = set(oracle.top_block_ids[:k])
    selected = {int(block_id) for block_id in selected_ids}
    union = selected | top
    if not union:
        return None
    return len(selected & top) / len(union)


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
            mean_answer_presence_rate=_mean(row.answer_presence_rate for row in group) or 0.0,
            scoreable_run_count=sum(1 for row in group if row.scoreable_by_answer_presence),
            mean_expected_block_count=_mean(row.expected_block_count for row in group) or 0.0,
            mean_selected_to_semantic_k_ratio=_mean(
                row.selected_to_semantic_k_ratio for row in group
            )
            or 0.0,
            mean_selector_latency_sec=_mean(row.selector_latency_sec for row in group) or 0.0,
            mean_recall=_mean_optional(row.target_recall for row in group),
            mean_precision=_mean_optional(row.selected_precision for row in group),
            mean_evidence_window_recall=_mean_optional(
                row.evidence_window_recall for row in group
            ),
            mean_evidence_window_precision=_mean_optional(
                row.evidence_window_precision for row in group
            ),
            mean_oracle_selected_mass_fraction=_mean_optional(
                row.oracle_selected_mass_fraction for row in group
            ),
            mean_oracle_semantic_selected_mass_fraction=_mean_optional(
                row.oracle_semantic_selected_mass_fraction for row in group
            ),
            mean_oracle_expected_mass_fraction=_mean_optional(
                row.oracle_expected_mass_fraction for row in group
            ),
            mean_oracle_topk_recall_at_4=_mean_optional(
                row.oracle_topk_recall_at_4 for row in group
            ),
            mean_oracle_topk_recall_at_8=_mean_optional(
                row.oracle_topk_recall_at_8 for row in group
            ),
            mean_oracle_topk_recall_at_16=_mean_optional(
                row.oracle_topk_recall_at_16 for row in group
            ),
            mean_oracle_topk_recall_at_32=_mean_optional(
                row.oracle_topk_recall_at_32 for row in group
            ),
        )
        for dataset_name, group in sorted(grouped.items())
    )


def _rank_by_block_id(
    suppression_decisions: Sequence[Mapping[str, Any]],
) -> dict[int, int]:
    """Return ranked candidate positions keyed by block id."""

    return {
        int(decision["block_id"]): rank
        for rank, decision in enumerate(suppression_decisions, start=1)
        if "block_id" in decision
    }


def _score_by_block_id(
    suppression_decisions: Sequence[Mapping[str, Any]],
) -> dict[int, float]:
    """Return ranked candidate scores keyed by block id when available."""

    scores: dict[int, float] = {}
    for decision in suppression_decisions:
        if "block_id" not in decision or decision.get("final_score") is None:
            continue
        scores[int(decision["block_id"])] = float(decision["final_score"])
    return scores


def _expected_ranks(
    expected_ids: Sequence[int],
    *,
    rank_by_id: Mapping[int, int],
) -> tuple[int, ...]:
    """Return available ranks for expected answer-bearing blocks."""

    return tuple(
        rank_by_id[int(block_id)]
        for block_id in expected_ids
        if int(block_id) in rank_by_id
    )


def _expected_rank_movements(
    expected_ids: Sequence[int],
    *,
    suppression_decisions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int | None], ...]:
    """Return old/new rerank positions for expected answer-bearing blocks."""

    decision_by_id = {
        int(decision["block_id"]): decision
        for decision in suppression_decisions
        if "block_id" in decision
    }
    movements: list[dict[str, int | None]] = []
    for block_id in expected_ids:
        decision = decision_by_id.get(int(block_id))
        if decision is None:
            continue
        new_rank = _optional_int(decision.get("rerank_new_rank"))
        original_rank = _optional_int(decision.get("rerank_original_rank"))
        if new_rank is None:
            new_rank = _optional_int(decision.get("rank"))
        if original_rank is None and new_rank is None:
            continue
        delta = (
            None
            if original_rank is None or new_rank is None
            else original_rank - new_rank
        )
        movements.append(
            {
                "block_id": int(block_id),
                "original_rank": original_rank,
                "new_rank": new_rank,
                "delta": delta,
            }
        )
    return tuple(movements)


def _best_expected_score(
    expected_ids: Sequence[int],
    *,
    score_by_id: Mapping[int, float],
) -> float | None:
    """Return the best score among expected answer-bearing blocks."""

    scores = [
        score_by_id[int(block_id)]
        for block_id in expected_ids
        if int(block_id) in score_by_id
    ]
    if not scores:
        return None
    return max(scores)


def _top1_score(suppression_decisions: Sequence[Mapping[str, Any]]) -> float | None:
    """Return the top-ranked final score when available."""

    if not suppression_decisions:
        return None
    score = suppression_decisions[0].get("final_score")
    return None if score is None else float(score)


def _block_recall_at_k(
    expected_ids: Sequence[int],
    *,
    ranked_ids: Sequence[int],
    k: int,
) -> float | None:
    """Return expected-block recall against the top-k ranked candidates."""

    expected = {int(block_id) for block_id in expected_ids}
    if not expected:
        return None
    top = {int(block_id) for block_id in ranked_ids[:k]}
    return len(expected & top) / len(expected)


def _neighbor_recall(
    expected_ids: Sequence[int],
    *,
    selected_ids: Sequence[int],
    radius: int,
) -> float | None:
    """Return expected-block recall allowing selected-id +/- radius matches."""

    expected = {int(block_id) for block_id in expected_ids}
    selected = tuple(int(block_id) for block_id in selected_ids)
    if not expected:
        return None
    if not selected:
        return 0.0
    hits = {
        expected_id
        for expected_id in expected
        if any(abs(expected_id - selected_id) <= radius for selected_id in selected)
    }
    return len(hits) / len(expected)


def _window_recall(
    expected_ids: Sequence[int],
    *,
    selected_ids: Sequence[int],
    radius: int,
) -> float | None:
    """Return evidence recall when selected ids may land near expected ids."""

    return _neighbor_recall(expected_ids, selected_ids=selected_ids, radius=radius)


def _window_precision(
    expected_ids: Sequence[int],
    *,
    selected_ids: Sequence[int],
    radius: int,
) -> float | None:
    """Return selected-block precision allowing +/- radius evidence windows."""

    expected = tuple(int(block_id) for block_id in expected_ids)
    selected = tuple(int(block_id) for block_id in selected_ids)
    if not expected:
        return None
    if not selected:
        return 0.0
    hits = [
        selected_id
        for selected_id in selected
        if any(abs(selected_id - expected_id) <= radius for expected_id in expected)
    ]
    return len(hits) / len(selected)


def _expected_block_distance(
    *,
    selected_ids: Sequence[int],
    expected_ids: Sequence[int],
) -> int | None:
    """Return the nearest block-id distance from any selected to expected block."""

    if not selected_ids or not expected_ids:
        return None
    return min(
        abs(int(selected) - int(expected))
        for selected in selected_ids
        for expected in expected_ids
    )


def _inspection_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    """Return compact inspection records keyed by block id."""

    return {
        int(record["block_id"]): record
        for record in records
        if "block_id" in record
    }


def _records_for_ids(
    inspection_by_id: Mapping[int, Mapping[str, Any]],
    block_ids: Sequence[int],
) -> tuple[dict[str, Any], ...]:
    """Return inspection records for the requested block ids in order."""

    return tuple(
        _compact_inspection_record(inspection_by_id[block_id])
        for block_id in block_ids
        if block_id in inspection_by_id
    )


def _top_ranked_blocks(
    suppression_decisions: Sequence[Mapping[str, Any]],
    *,
    inspection_by_id: Mapping[int, Mapping[str, Any]],
    limit: int = 10,
) -> tuple[dict[str, Any], ...]:
    """Return the top ranked candidate records from suppression decisions."""

    top: list[dict[str, Any]] = []
    for rank, decision in enumerate(suppression_decisions[:limit], start=1):
        block_id = int(decision["block_id"])
        record = dict(inspection_by_id.get(block_id, decision))
        compact = _compact_inspection_record(record)
        compact["rank"] = rank
        if "final_score" not in compact and "final_score" in decision:
            compact["final_score"] = decision["final_score"]
        for key in (
            "refined_score",
            "rerank_original_rank",
            "rerank_new_rank",
            "rerank_rank_delta",
        ):
            if key in decision:
                compact[key] = decision[key]
        top.append(compact)
    return tuple(top)


def _compact_inspection_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable, JSON-friendly block preview record."""

    return {
        "block_id": int(record.get("block_id", -1)),
        "candidate_id": str(record.get("candidate_id", record.get("block_id", ""))),
        "token_start": record.get("token_start"),
        "token_end": record.get("token_end"),
        "block_size": record.get("block_size"),
        "stage_a_score": record.get("stage_a_score"),
        "stage_b_score": record.get("stage_b_score"),
        "final_score": record.get("final_score"),
        "refined_score": record.get("refined_score"),
        "rerank_original_rank": record.get("rerank_original_rank"),
        "rerank_new_rank": record.get("rerank_new_rank"),
        "rerank_rank_delta": record.get("rerank_rank_delta"),
        "selected": bool(record.get("selected", False)),
        "selected_reason": str(record.get("selected_reason", "")),
        "preview_text": str(record.get("preview_text", "")),
    }


def _format_block_records(records: Sequence[Mapping[str, Any]]) -> str:
    """Format compact block records for text reports."""

    if not records:
        return "[]"
    formatted: list[str] = []
    for record in records:
        preview = " ".join(str(record.get("preview_text", "")).split())
        if len(preview) > 96:
            preview = preview[:93] + "..."
        score = record.get("final_score")
        score_text = "n/a" if score is None else f"{float(score):.4f}"
        refined = record.get("refined_score")
        refined_text = (
            ""
            if refined is None
            else f" refined={float(refined):.4f}"
        )
        rank = record.get("rank")
        rank_text = "" if rank is None else f"rank={rank} "
        old_rank = record.get("rerank_original_rank")
        new_rank = record.get("rerank_new_rank")
        move_text = (
            ""
            if old_rank is None or new_rank is None
            else f" old_rank={old_rank} new_rank={new_rank}"
        )
        formatted.append(
            f"{rank_text}id={record.get('block_id')} "
            f"span={record.get('token_start')}:{record.get('token_end')} "
            f"score={score_text}{refined_text}{move_text} preview={preview!r}"
        )
    return "[" + "; ".join(formatted) + "]"


def _format_rank_movements(movements: Sequence[Mapping[str, Any]]) -> str:
    if not movements:
        return "[]"
    parts = []
    for movement in movements:
        parts.append(
            f"id={movement.get('block_id')}:"
            f"{movement.get('original_rank')}->{movement.get('new_rank')}"
            f"({movement.get('delta')})"
        )
    return "[" + ", ".join(parts) + "]"


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


def _answer_presence(
    prompt_text: str,
    answers: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split answer labels by whether they appear in the materialized prompt.

    This is a LongBench diagnostic only. Some tasks use abstractive or
    completion labels, so answer-string retrieval is not always scoreable.
    """

    present: list[str] = []
    missing: list[str] = []
    for answer in answers:
        if _contains_fragment(prompt_text, answer):
            present.append(answer)
        else:
            missing.append(answer)
    return tuple(present), tuple(missing)


def _contains_fragment(text: str, fragment: str) -> bool:
    """Return whether ``fragment`` appears in ``text`` with light normalization."""

    text_folded = text.casefold()
    fragment_folded = fragment.strip().casefold()
    if not fragment_folded:
        return False
    if fragment_folded in text_folded:
        return True
    normalized_text = " ".join(text_folded.split())
    normalized_fragment = " ".join(fragment_folded.split())
    return bool(normalized_fragment and normalized_fragment in normalized_text)


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


def _fmt_int_optional(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
