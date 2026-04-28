"""Output-based LongBench QA benchmark over KVBlock-selected context."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import torch

from kvblock.benchmark.answer_metrics import score_qa_answer
from kvblock.benchmark.longbench_adapter import (
    DEFAULT_LONGBENCH_DATASETS,
    DEFAULT_ORACLE_TOP_K,
    DatasetLoader,
    LengthBucket,
    OracleMode,
    parse_length_bucket,
    parse_oracle_mode,
    parse_oracle_top_k,
    run_longbench_selector_benchmark,
)
from kvblock.benchmark.dynamic_block_benchmark import (
    RefineScoreMode,
    RerankMode,
    StageCPolicyMode,
)
from kvblock.kv.block_modes import BlockModeName
from kvblock.kv.qk_aggregation import QKAggregationStrategy
from kvblock.runtime.base import RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig


@dataclass(frozen=True, slots=True)
class LongBenchOutputRunRow:
    """One generated-answer evaluation row."""

    dataset: str
    sample_id: str
    model: str
    block_mode: str
    prompt_tokens: int
    longbench_length: int | None
    selected_token_count: int
    selected_token_fraction: float
    selected_block_count: int
    mixed_fallback_used: bool
    selector_latency_sec: float
    selector_total_latency_sec: float
    generation_latency_sec: float
    total_latency_sec: float
    gold_answers: tuple[str, ...]
    prediction: str
    answer_em: float
    answer_f1: float
    answer_precision: float
    answer_recall: float
    selector_recall: float | None
    selector_precision: float | None
    evidence_window_recall: float | None
    evidence_window_precision: float | None
    expected_parent_recall: float | None
    selected_ids: tuple[int, ...]
    selected_spans: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchOutputSummary:
    """Aggregate answer-quality and bandwidth proxy metrics."""

    dataset: str
    row_count: int
    mean_answer_em: float
    mean_answer_f1: float
    mean_answer_precision: float
    mean_answer_recall: float
    mean_selected_token_fraction: float
    mean_selected_tokens: float
    mean_selector_latency_sec: float
    mean_generation_latency_sec: float
    mixed_fallback_count: int
    mixed_fallback_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class LongBenchOutputBenchmarkResult:
    """Full output-based benchmark payload."""

    name: str
    model_names: tuple[str, ...]
    datasets: tuple[str, ...]
    config: dict[str, Any]
    rows: tuple[LongBenchOutputRunRow, ...]
    dataset_summaries: tuple[LongBenchOutputSummary, ...]
    overall_summary: LongBenchOutputSummary

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly payload."""

        return {
            "name": self.name,
            "model_names": list(self.model_names),
            "datasets": list(self.datasets),
            "config": self.config,
            "rows": [row.to_dict() for row in self.rows],
            "dataset_summaries": [
                summary.to_dict() for summary in self.dataset_summaries
            ],
            "overall_summary": self.overall_summary.to_dict(),
        }


def run_longbench_output_benchmark(
    *,
    name: str,
    model_names: Sequence[str],
    dataset_names: Sequence[str] = DEFAULT_LONGBENCH_DATASETS,
    split: str = "test",
    dataset_repo: str = "THUDM/LongBench",
    limit_per_dataset: int | None = 1,
    length_bucket: LengthBucket | str = "all",
    prompt_cache_dir: str | Path = "results/longbench_output/prompts",
    block_modes: Sequence[BlockModeName] = ("fixed_40",),
    representation_source: RepresentationSource = "query_mean_last_layer",
    qk_aggregation_strategy: QKAggregationStrategy = "block_max",
    needle_qk_aggregation_strategy: QKAggregationStrategy | None = None,
    coarse_top_k: int = 2,
    mixed_refine_parent_k: int = 8,
    mixed_global_anchor_k: int = 8,
    mixed_fallback_margin: float = 0.05,
    mixed_max_children_per_parent: int | None = None,
    mixed_child_window_radius: int = 0,
    rerank_mode: RerankMode = "none",
    rerank_weight: float = 0.3,
    refine_top_n_tokens: int = 4,
    refine_score_mode: RefineScoreMode = "raw_topn_mean",
    stage_c_policy: StageCPolicyMode = "refined_only",
    exclude_scaffold_blocks: bool = False,
    neighbor_expansion: int = 0,
    halo_radius: int = 0,
    max_selected_blocks: int | None = None,
    evidence_window_radius: int = 0,
    oracle_mode: OracleMode = "none",
    oracle_top_k: Sequence[int] = DEFAULT_ORACLE_TOP_K,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
    load_config_kwargs: dict[str, Any] | None = None,
    selector_config: RealBlockSelectorConfig | None = None,
    dataset_loader: DatasetLoader | None = None,
) -> LongBenchOutputBenchmarkResult:
    """Run generation on contexts shortened to KVBlock-selected spans."""

    if not name.strip():
        raise ValueError("name must be non-empty")
    if not model_names:
        raise ValueError("model_names must not be empty")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be > 0")
    if temperature < 0.0:
        raise ValueError("temperature must be >= 0")

    bucket = (
        parse_length_bucket(length_bucket)
        if isinstance(length_bucket, str)
        else length_bucket
    )
    resolved_oracle_mode = parse_oracle_mode(oracle_mode)
    resolved_oracle_top_k = parse_oracle_top_k(oracle_top_k)
    load_kwargs = dict(load_config_kwargs or {})
    selector_result = run_longbench_selector_benchmark(
        model_names=model_names,
        dataset_names=dataset_names,
        split=split,
        dataset_repo=dataset_repo,
        limit_per_dataset=limit_per_dataset,
        length_bucket=bucket,
        prompt_cache_dir=prompt_cache_dir,
        block_modes=block_modes,
        representation_source=representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy,
        needle_qk_aggregation_strategy=needle_qk_aggregation_strategy,
        coarse_top_k=coarse_top_k,
        mixed_refine_parent_k=mixed_refine_parent_k,
        mixed_global_anchor_k=mixed_global_anchor_k,
        mixed_fallback_margin=mixed_fallback_margin,
        mixed_max_children_per_parent=mixed_max_children_per_parent,
        mixed_child_window_radius=mixed_child_window_radius,
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        stage_c_policy=stage_c_policy,
        exclude_scaffold_blocks=exclude_scaffold_blocks,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k=resolved_oracle_top_k,
        load_config_kwargs=load_kwargs,
        selector_config=selector_config,
        dataset_loader=dataset_loader,
    )
    _release_torch_cache()

    answers_by_prompt = {
        sample.prompt_name: sample.answer_labels
        for sample in selector_result.samples
    }
    rows: list[LongBenchOutputRunRow] = []
    for model_name in model_names:
        runtime = LocalHfRuntime(
            RuntimeLoadConfig(model_name=model_name, **load_kwargs),
            capture_config=HiddenStateCaptureConfig(
                representation_source=representation_source,
            ),
        )
        runtime.load_model()
        for selector_row in selector_result.rows:
            if selector_row.model_name != model_name:
                continue
            prompt_text = Path(selector_row.prompt_file).read_text(encoding="utf-8")
            selected_context = selected_context_from_spans(
                runtime,
                prompt_text=prompt_text,
                selected_spans=selector_row.selected_spans,
            )
            question = extract_longbench_question(prompt_text)
            generation_prompt = format_selected_context_prompt(
                question=question,
                selected_context=selected_context,
            )
            prediction, generation_latency_sec = generate_answer(
                runtime,
                generation_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            gold_answers = answers_by_prompt.get(selector_row.prompt_name, ())
            answer_scores = score_qa_answer(prediction, gold_answers)
            selected_token_count = _selected_token_count(selector_row.selected_spans)
            selected_token_fraction = (
                0.0
                if selector_row.tokens <= 0
                else selected_token_count / selector_row.tokens
            )
            rows.append(
                LongBenchOutputRunRow(
                    dataset=selector_row.dataset_name,
                    sample_id=selector_row.sample_id,
                    model=selector_row.model_name,
                    block_mode=selector_row.block_mode,
                    prompt_tokens=selector_row.tokens,
                    longbench_length=selector_row.longbench_length,
                    selected_token_count=selected_token_count,
                    selected_token_fraction=selected_token_fraction,
                    selected_block_count=selector_row.selected_count,
                    mixed_fallback_used=selector_row.mixed_fallback_used,
                    selector_latency_sec=selector_row.selector_latency_sec,
                    selector_total_latency_sec=selector_row.total_latency_sec,
                    generation_latency_sec=generation_latency_sec,
                    total_latency_sec=(
                        selector_row.total_latency_sec + generation_latency_sec
                    ),
                    gold_answers=gold_answers,
                    prediction=prediction,
                    answer_em=answer_scores["em"],
                    answer_f1=answer_scores["f1"],
                    answer_precision=answer_scores["precision"],
                    answer_recall=answer_scores["recall"],
                    selector_recall=selector_row.target_recall,
                    selector_precision=selector_row.selected_precision,
                    evidence_window_recall=selector_row.evidence_window_recall,
                    evidence_window_precision=selector_row.evidence_window_precision,
                    expected_parent_recall=selector_row.expected_parent_recall,
                    selected_ids=selector_row.selected_block_ids,
                    selected_spans=selector_row.selected_spans,
                )
            )
    row_tuple = tuple(rows)
    if not row_tuple:
        raise ValueError("output benchmark produced no rows")
    dataset_summaries = build_output_summaries(row_tuple)
    config = _output_config(
        split=split,
        dataset_repo=dataset_repo,
        length_bucket=bucket.to_dict(),
        limit_per_dataset=limit_per_dataset,
        representation_source=representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy,
        block_modes=block_modes,
        coarse_top_k=coarse_top_k,
        mixed_refine_parent_k=mixed_refine_parent_k,
        mixed_global_anchor_k=mixed_global_anchor_k,
        mixed_fallback_margin=mixed_fallback_margin,
        mixed_max_children_per_parent=mixed_max_children_per_parent,
        mixed_child_window_radius=mixed_child_window_radius,
        rerank_mode=rerank_mode,
        rerank_weight=rerank_weight,
        refine_top_n_tokens=refine_top_n_tokens,
        refine_score_mode=refine_score_mode,
        stage_c_policy=stage_c_policy,
        exclude_scaffold_blocks=exclude_scaffold_blocks,
        neighbor_expansion=neighbor_expansion,
        halo_radius=halo_radius,
        max_selected_blocks=max_selected_blocks,
        evidence_window_radius=evidence_window_radius,
        oracle_mode=resolved_oracle_mode,
        oracle_top_k=resolved_oracle_top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        selector_config=selector_config,
        load_config_kwargs=load_kwargs,
    )
    return LongBenchOutputBenchmarkResult(
        name=name,
        model_names=tuple(model_names),
        datasets=tuple(dataset_names),
        config=config,
        rows=row_tuple,
        dataset_summaries=dataset_summaries,
        overall_summary=_summarize_output_rows("all", row_tuple),
    )


def selected_context_from_spans(
    runtime: LocalHfRuntime,
    *,
    prompt_text: str,
    selected_spans: Sequence[str],
) -> str:
    """Decode selected prompt token spans into a shortened context string."""

    tokenized = runtime.tokenize(prompt_text)
    chunks: list[str] = []
    for span in selected_spans:
        parsed = _parse_span(span)
        if parsed is None:
            continue
        start, end = parsed
        bounded_start = max(0, min(start, len(tokenized.token_ids)))
        bounded_end = max(bounded_start, min(end, len(tokenized.token_ids)))
        token_ids = tuple(tokenized.token_ids[bounded_start:bounded_end])
        if not token_ids:
            continue
        text = runtime.decode_token_ids(token_ids).strip()
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)


def extract_longbench_question(prompt_text: str) -> str:
    """Extract the LongBench input/question from a materialized prompt."""

    marker = "\nINPUT:\n"
    if marker not in prompt_text:
        return prompt_text.strip()
    return prompt_text.rsplit(marker, maxsplit=1)[-1].strip()


def format_selected_context_prompt(*, question: str, selected_context: str) -> str:
    """Format the first-pass short-answer generation prompt."""

    return "\n".join(
        (
            "Answer the question using only the provided context. Keep the answer short.",
            "",
            "Context:",
            selected_context.strip(),
            "",
            "Question:",
            question.strip(),
            "",
            "Answer:",
        )
    )


def generate_answer(
    runtime: LocalHfRuntime,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float]:
    """Generate a short answer from a local HF runtime."""

    runtime.load_model()
    tokenizer = runtime._require_tokenizer()
    model = runtime._require_model()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_device = runtime._input_device()
    encoded = {
        key: value.to(input_device)
        for key, value in encoded.items()
    }
    do_sample = temperature > 0.0
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
    started_at = perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**encoded, **generation_kwargs)
    generation_latency_sec = perf_counter() - started_at
    prompt_token_count = int(encoded["input_ids"].shape[-1])
    generated_ids = output_ids[0, prompt_token_count:]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return _clean_prediction(prediction), generation_latency_sec


def build_output_summaries(
    rows: Sequence[LongBenchOutputRunRow],
) -> tuple[LongBenchOutputSummary, ...]:
    """Build per-dataset output summaries."""

    grouped: dict[str, list[LongBenchOutputRunRow]] = {}
    for row in rows:
        grouped.setdefault(row.dataset, []).append(row)
    return tuple(
        _summarize_output_rows(dataset, tuple(group))
        for dataset, group in sorted(grouped.items())
    )


def format_longbench_output_report(result: LongBenchOutputBenchmarkResult) -> str:
    """Return a compact human-readable output benchmark report."""

    lines = [
        "LONGBENCH OUTPUT BENCHMARK",
        f"name={result.name}",
        f"models={','.join(result.model_names)} datasets={','.join(result.datasets)}",
        "",
        "OVERALL",
        _format_summary(result.overall_summary),
        "",
        "DATASET SUMMARIES",
    ]
    for summary in result.dataset_summaries:
        lines.append(_format_summary(summary))
    lines.extend(["", "RUNS"])
    for row in result.rows:
        prediction = _compact_text(row.prediction, limit=80)
        gold = _compact_text(" | ".join(row.gold_answers), limit=80)
        lines.append(
            f"{row.dataset}:{row.sample_id} model={row.model} mode={row.block_mode} "
            f"answer_f1={row.answer_f1:.3f} answer_em={row.answer_em:.3f} "
            f"selected_frac={row.selected_token_fraction:.3f} "
            f"selected_tokens={row.selected_token_count} "
            f"fallback={row.mixed_fallback_used} "
            f"selector={row.selector_latency_sec:.6f}s "
            f"generation={row.generation_latency_sec:.6f}s "
            f"prediction={prediction!r} gold={gold!r}"
        )
    return "\n".join(lines)


def write_longbench_output_benchmark_outputs(
    result: LongBenchOutputBenchmarkResult,
    *,
    json_path: str | Path,
    text_path: str | Path,
) -> None:
    """Write JSON and text output benchmark reports."""

    json_output = Path(json_path)
    text_output = Path(text_path)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    text_output.write_text(format_longbench_output_report(result), encoding="utf-8")


def _summarize_output_rows(
    dataset: str,
    rows: Sequence[LongBenchOutputRunRow],
) -> LongBenchOutputSummary:
    if not rows:
        raise ValueError("rows must not be empty")
    fallback_count = sum(1 for row in rows if row.mixed_fallback_used)
    return LongBenchOutputSummary(
        dataset=dataset,
        row_count=len(rows),
        mean_answer_em=_mean(row.answer_em for row in rows),
        mean_answer_f1=_mean(row.answer_f1 for row in rows),
        mean_answer_precision=_mean(row.answer_precision for row in rows),
        mean_answer_recall=_mean(row.answer_recall for row in rows),
        mean_selected_token_fraction=_mean(
            row.selected_token_fraction for row in rows
        ),
        mean_selected_tokens=_mean(row.selected_token_count for row in rows),
        mean_selector_latency_sec=_mean(row.selector_latency_sec for row in rows),
        mean_generation_latency_sec=_mean(
            row.generation_latency_sec for row in rows
        ),
        mixed_fallback_count=fallback_count,
        mixed_fallback_rate=fallback_count / len(rows),
    )


def _output_config(**kwargs: Any) -> dict[str, Any]:
    config = dict(kwargs)
    config["block_modes"] = list(config["block_modes"])
    config["oracle_top_k"] = list(config["oracle_top_k"])
    if config["selector_config"] is not None:
        config["selector_config"] = asdict(config["selector_config"])
    return config


def _selected_token_count(selected_spans: Sequence[str]) -> int:
    total = 0
    for span in selected_spans:
        parsed = _parse_span(span)
        if parsed is None:
            continue
        start, end = parsed
        total += max(0, end - start)
    return total


def _parse_span(span: str) -> tuple[int, int] | None:
    parts = span.split(":", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _clean_prediction(prediction: str) -> str:
    first_line = prediction.strip().splitlines()[0].strip() if prediction.strip() else ""
    for stop in ("\n\n", "Context:", "Question:", "Answer:"):
        if stop in first_line:
            first_line = first_line.split(stop, maxsplit=1)[0].strip()
    return first_line


def _compact_text(text: str, *, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _format_summary(summary: LongBenchOutputSummary) -> str:
    return (
        f"{summary.dataset} | rows={summary.row_count} "
        f"mean_answer_f1={summary.mean_answer_f1:.3f} "
        f"mean_answer_em={summary.mean_answer_em:.3f} "
        f"mean_selected_frac={summary.mean_selected_token_fraction:.3f} "
        f"mean_selected_tokens={summary.mean_selected_tokens:.1f} "
        f"mean_selector={summary.mean_selector_latency_sec:.6f}s "
        f"mean_generation={summary.mean_generation_latency_sec:.6f}s "
        f"mixed_fallback={summary.mixed_fallback_count}/{summary.row_count}"
    )


def _mean(values: Any) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def _release_torch_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
