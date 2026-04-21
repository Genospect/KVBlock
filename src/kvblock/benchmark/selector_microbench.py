"""Synthetic selector microbenchmark harness for the V1 scaffold."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from time import perf_counter
from typing import Any, Literal, Sequence

import torch

from kvblock.benchmark.metrics import selector_quality_metrics
from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.oracle import (
    SyntheticDenseOracle,
    SyntheticDenseOracleConfig,
    compare_block_sets,
    sparse_selected_block_set,
)
from kvblock.selector.pipeline import SelectorPipeline, SelectorPipelineConfig
from kvblock.selector.policies import (
    ConfidencePolicy,
    FallbackPolicy,
    StageAPolicy,
    StageAWeights,
    StageBPolicy,
    StageCPolicy,
)
from kvblock.summaries.fp8_summary import FP8SummaryBuilder
from kvblock.summaries.sign_sketch import generate_sign_sketch

PopulationProfile = Literal["default", "low_confidence", "rail_dominated"]


@dataclass(frozen=True, slots=True)
class SyntheticSelectorPopulation:
    """Synthetic metadata/query bundle for selector microbench runs."""

    metadata_blocks: tuple[BlockMetadata, ...]
    query_summaries: tuple[torch.Tensor, ...]
    anchor_block_ids: tuple[BlockId, ...]


@dataclass(frozen=True, slots=True)
class SelectorMicrobenchSpec:
    """Single synthetic selector microbenchmark case."""

    case_id: str
    num_blocks: int
    block_size: int = 32
    summary_dim: int = 32
    shortlist_size: int = 24
    semantic_top_k: int = 8
    keep_recent_blocks: int = 4
    keep_anchor_blocks: int = 2
    confidence_margin: float = 0.05
    normalized_margin_threshold: float | None = None
    min_normalized_mass: float | None = None
    widen_top_k_by: int = 4
    add_recent_blocks_by: int = 2
    allow_dense_fallback: bool = True
    num_queries: int = 8
    seed: int = 0
    population_profile: PopulationProfile = "default"
    oracle_enabled: bool = False
    oracle_top_k: int | None = None
    low_oracle_recall_threshold: float = 0.5
    stage_a_weights: StageAWeights = field(default_factory=StageAWeights)
    stage_b_hamming_weight: float = 0.2
    stage_b_base_score_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.num_blocks <= 0:
            raise ValueError("num_blocks must be > 0")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.summary_dim <= 0:
            raise ValueError("summary_dim must be > 0")
        if self.shortlist_size <= 0:
            raise ValueError("shortlist_size must be > 0")
        if self.semantic_top_k <= 0:
            raise ValueError("semantic_top_k must be > 0")
        if self.keep_recent_blocks < 0:
            raise ValueError("keep_recent_blocks must be >= 0")
        if self.keep_anchor_blocks < 0:
            raise ValueError("keep_anchor_blocks must be >= 0")
        if self.confidence_margin < 0:
            raise ValueError("confidence_margin must be >= 0")
        if (
            self.normalized_margin_threshold is not None
            and self.normalized_margin_threshold < 0
        ):
            raise ValueError("normalized_margin_threshold must be >= 0")
        if self.num_queries <= 0:
            raise ValueError("num_queries must be > 0")
        if self.oracle_top_k is not None and self.oracle_top_k <= 0:
            raise ValueError("oracle_top_k must be > 0 when provided")
        if not (0.0 <= self.low_oracle_recall_threshold <= 1.0):
            raise ValueError("low_oracle_recall_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SelectorMicrobenchRow:
    """Single row of synthetic selector microbenchmark metrics."""

    case_id: str
    query_index: int
    step_id: str
    seed: int
    population_profile: PopulationProfile
    num_blocks: int
    block_size: int
    summary_dim: int
    shortlist_size: int
    semantic_top_k: int
    keep_recent_blocks: int
    keep_anchor_blocks: int
    confidence_margin: float
    normalized_margin_threshold: float | None
    min_normalized_mass: float | None
    widen_top_k_by: int
    add_recent_blocks_by: int
    selector_latency_sec: float
    stage_a_candidate_count: int
    stage_a_shortlist_size: int
    stage_b_refinement_count: int
    final_selected_block_count: int
    semantic_selected_block_count: int
    rail_preserved_block_count: int
    fallback_mode: str
    fallback_action: str
    fallback_reason_code: str
    raw_margin: float
    normalized_margin: float | None
    selected_mass: float | None
    normalized_mass: float | None
    is_confident: bool
    trace_size_bytes: int
    oracle_recall_rate: float | None = None
    oracle_precision_rate: float | None = None
    oracle_overlap_count: int | None = None
    oracle_missed_important_block_ids: tuple[int, ...] = ()
    oracle_missed_important_count: int | None = None
    oracle_extra_selected_block_ids: tuple[int, ...] = ()
    oracle_extra_selected_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly row record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class OracleAggregateSummary:
    """Aggregate oracle-comparison metrics for one microbenchmark case."""

    mean_recall_rate: float
    mean_precision_rate: float
    mean_overlap_count: float
    low_recall_fallback_frequency_by_mode: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectorMicrobenchCaseResult:
    """Full result for one synthetic microbenchmark case."""

    spec: SelectorMicrobenchSpec
    rows: tuple[SelectorMicrobenchRow, ...]
    fallback_frequency_by_mode: dict[str, int]
    oracle_summary: OracleAggregateSummary | None = None

    @property
    def query_count(self) -> int:
        """Return the number of executed query runs."""

        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        return {
            "spec": asdict(self.spec),
            "rows": [row.to_dict() for row in self.rows],
            "fallback_frequency_by_mode": dict(self.fallback_frequency_by_mode),
            "oracle_summary": None if self.oracle_summary is None else self.oracle_summary.to_dict(),
        }


def build_selector_pipeline_config(
    spec: SelectorMicrobenchSpec,
) -> SelectorPipelineConfig:
    """Construct a pipeline config from one benchmark case spec."""

    return SelectorPipelineConfig(
        stage_a=StageAPolicy(
            weights=spec.stage_a_weights,
            shortlist_size=spec.shortlist_size,
            long_context_shortlist_size=spec.shortlist_size,
            long_context_threshold=spec.num_blocks * spec.block_size + 1,
        ),
        stage_b=StageBPolicy(
            hamming_weight=spec.stage_b_hamming_weight,
            base_score_weight=spec.stage_b_base_score_weight,
            sketch_bits=64,
        ),
        stage_c=StageCPolicy(
            keep_recent_blocks=spec.keep_recent_blocks,
            keep_anchor_blocks=spec.keep_anchor_blocks,
            semantic_top_k=spec.semantic_top_k,
            semantic_top_k_long_context=spec.semantic_top_k,
            long_context_threshold=spec.num_blocks * spec.block_size + 1,
        ),
        confidence=ConfidencePolicy(
            margin_threshold=spec.confidence_margin,
            normalized_margin_threshold=spec.normalized_margin_threshold,
            min_normalized_mass=spec.min_normalized_mass,
        ),
        fallback=FallbackPolicy(
            widen_top_k_by=spec.widen_top_k_by,
            add_recent_blocks_by=spec.add_recent_blocks_by,
            allow_dense_fallback=spec.allow_dense_fallback,
        ),
    )


def generate_synthetic_selector_population(
    spec: SelectorMicrobenchSpec,
) -> SyntheticSelectorPopulation:
    """Generate a deterministic synthetic metadata/query bundle."""

    generator = torch.Generator().manual_seed(spec.seed)
    builder = FP8SummaryBuilder(summary_dim=spec.summary_dim)

    if spec.population_profile == "low_confidence":
        common_center = torch.randn(spec.summary_dim, generator=generator)
    else:
        common_center = torch.zeros(spec.summary_dim, dtype=torch.float32)

    metadata_blocks: list[BlockMetadata] = []
    for block_index in range(spec.num_blocks):
        if spec.population_profile == "default":
            center = torch.randn(spec.summary_dim, generator=generator)
        elif spec.population_profile == "low_confidence":
            center = common_center + 0.03 * torch.randn(
                spec.summary_dim, generator=generator
            )
        else:
            center = torch.randn(spec.summary_dim, generator=generator)

        block_states = center.unsqueeze(0) + 0.05 * torch.randn(
            spec.block_size, spec.summary_dim, generator=generator
        )
        encoding = builder.build(block_states)

        if spec.population_profile == "rail_dominated":
            recent_base = spec.num_blocks * 10
            last_access = (
                recent_base + block_index
                if block_index >= max(spec.num_blocks - max(spec.keep_recent_blocks, 2), 0)
                else block_index
            )
        else:
            last_access = int(
                torch.randint(0, spec.num_blocks * 4 + 1, (1,), generator=generator).item()
            )

        metadata_blocks.append(
            BlockMetadata(
                block_id=BlockId(block_index),
                pool_id=0,
                token_start=block_index * spec.block_size,
                token_len=spec.block_size,
                precision_tier="fp16",
                summary_fp8=encoding.values,
                summary_scale=encoding.scale,
                sign_sketch=generate_sign_sketch(encoding),
                summary_norm=encoding.summary_norm,
                attn_ema=float(torch.rand((), generator=generator).item()),
                attn_var=float(torch.rand((), generator=generator).item()) * 0.1,
                last_access_step=last_access,
                hit_count=int(
                    torch.randint(0, 6, (1,), generator=generator).item()
                ),
                priority=float(torch.rand((), generator=generator).item()),
                rope_bucket=block_index // max(spec.block_size, 1),
                fallback_miss_count=int(
                    torch.randint(0, 3, (1,), generator=generator).item()
                ),
            )
        )

    query_summaries = tuple(
        _generate_query_summary(
            metadata_blocks, query_index=query_index, spec=spec, generator=generator
        )
        for query_index in range(spec.num_queries)
    )
    anchor_count = min(spec.keep_anchor_blocks, spec.num_blocks, 2)
    anchor_ids = tuple(BlockId(index) for index in range(anchor_count))
    return SyntheticSelectorPopulation(
        metadata_blocks=tuple(metadata_blocks),
        query_summaries=query_summaries,
        anchor_block_ids=anchor_ids,
    )


def run_selector_microbench_case(
    spec: SelectorMicrobenchSpec,
) -> SelectorMicrobenchCaseResult:
    """Execute one synthetic selector microbenchmark case."""

    population = generate_synthetic_selector_population(spec)
    pipeline = SelectorPipeline(build_selector_pipeline_config(spec))
    oracle = _build_oracle(spec)

    rows: list[SelectorMicrobenchRow] = []
    fallback_counter: Counter[str] = Counter()
    context_tokens = spec.num_blocks * spec.block_size

    for query_index, query_summary in enumerate(population.query_summaries):
        started_at = perf_counter()
        result = pipeline.run(
            query_summary,
            population.metadata_blocks,
            current_step=context_tokens + query_index,
            step_id=f"{spec.case_id}:{query_index}",
            context_tokens=context_tokens,
            anchor_block_ids=population.anchor_block_ids,
        )
        latency = perf_counter() - started_at
        fallback_counter[result.mode] += 1

        trace_payload = {
            "stage_a_scores": [asdict(item) for item in result.trace.stage_a_scores],
            "stage_a_shortlist_block_ids": list(result.trace.stage_a_shortlist_block_ids),
            "stage_b_scores": [asdict(item) for item in result.trace.stage_b_scores],
            "pre_fallback_selection": asdict(result.trace.pre_fallback_selection),
            "final_selection": asdict(result.trace.final_selection),
            "confidence": asdict(result.trace.confidence),
            "fallback": asdict(result.trace.fallback),
        }
        trace_size_bytes = len(json.dumps(trace_payload))
        oracle_metrics = _oracle_metrics_for_run(
            spec,
            oracle=oracle,
            query_summary=query_summary,
            metadata_blocks=population.metadata_blocks,
            selected_block_ids=result.trace.final_selection.final_selected_block_ids,
            step_id=result.trace.step_id,
        )

        rows.append(
            SelectorMicrobenchRow(
                case_id=spec.case_id,
                query_index=query_index,
                step_id=str(result.trace.step_id),
                seed=spec.seed,
                population_profile=spec.population_profile,
                num_blocks=spec.num_blocks,
                block_size=spec.block_size,
                summary_dim=spec.summary_dim,
                shortlist_size=spec.shortlist_size,
                semantic_top_k=spec.semantic_top_k,
                keep_recent_blocks=spec.keep_recent_blocks,
                keep_anchor_blocks=spec.keep_anchor_blocks,
                confidence_margin=spec.confidence_margin,
                normalized_margin_threshold=spec.normalized_margin_threshold,
                min_normalized_mass=spec.min_normalized_mass,
                widen_top_k_by=spec.widen_top_k_by,
                add_recent_blocks_by=spec.add_recent_blocks_by,
                selector_latency_sec=latency,
                stage_a_candidate_count=len(result.trace.stage_a_scores),
                stage_a_shortlist_size=len(result.trace.stage_a_shortlist_block_ids),
                stage_b_refinement_count=len(result.trace.stage_b_scores),
                final_selected_block_count=len(
                    result.trace.final_selection.final_selected_block_ids
                ),
                semantic_selected_block_count=result.trace.final_selection.semantic_block_count,
                rail_preserved_block_count=result.trace.final_selection.rail_block_count,
                fallback_mode=result.mode,
                fallback_action=result.trace.fallback.action,
                fallback_reason_code=result.trace.fallback.reason_code,
                raw_margin=result.trace.confidence.raw_margin,
                normalized_margin=result.trace.confidence.normalized_margin,
                selected_mass=result.trace.confidence.selected_mass,
                normalized_mass=result.trace.confidence.normalized_mass,
                is_confident=result.trace.confidence.is_confident,
                trace_size_bytes=trace_size_bytes,
                oracle_recall_rate=oracle_metrics["oracle_recall_rate"],
                oracle_precision_rate=oracle_metrics["oracle_precision_rate"],
                oracle_overlap_count=oracle_metrics["oracle_overlap_count"],
                oracle_missed_important_block_ids=oracle_metrics["oracle_missed_important_block_ids"],
                oracle_missed_important_count=oracle_metrics["oracle_missed_important_count"],
                oracle_extra_selected_block_ids=oracle_metrics["oracle_extra_selected_block_ids"],
                oracle_extra_selected_count=oracle_metrics["oracle_extra_selected_count"],
            )
        )

    return SelectorMicrobenchCaseResult(
        spec=spec,
        rows=tuple(rows),
        fallback_frequency_by_mode=dict(fallback_counter),
        oracle_summary=_aggregate_oracle_summary(tuple(rows), spec=spec),
    )


def run_selector_microbench_sweep(
    specs: Sequence[SelectorMicrobenchSpec],
) -> list[SelectorMicrobenchCaseResult]:
    """Execute a sweep of synthetic selector microbenchmark cases."""

    return [run_selector_microbench_case(spec) for spec in specs]


def _generate_query_summary(
    metadata_blocks: Sequence[BlockMetadata],
    *,
    query_index: int,
    spec: SelectorMicrobenchSpec,
    generator: torch.Generator,
) -> torch.Tensor:
    summary_dim = spec.summary_dim
    if spec.population_profile == "default":
        target = metadata_blocks[query_index % len(metadata_blocks)].dequantize_summary()
        noise = 0.05 * torch.randn(summary_dim, generator=generator)
        return target + noise

    if spec.population_profile == "low_confidence":
        limit = min(4, len(metadata_blocks))
        mean_summary = torch.stack(
            [metadata_blocks[index].dequantize_summary() for index in range(limit)]
        ).mean(dim=0)
        noise = 0.01 * torch.randn(summary_dim, generator=generator)
        return mean_summary + noise

    # rail_dominated: use a random query that is weakly related to recency so the
    # rails can dominate selection under the chosen policy settings.
    return torch.randn(summary_dim, generator=generator)


def _build_oracle(spec: SelectorMicrobenchSpec) -> SyntheticDenseOracle | None:
    if not spec.oracle_enabled:
        return None
    return SyntheticDenseOracle(
        SyntheticDenseOracleConfig(top_k=spec.oracle_top_k or spec.semantic_top_k)
    )


def _oracle_metrics_for_run(
    spec: SelectorMicrobenchSpec,
    *,
    oracle: SyntheticDenseOracle | None,
    query_summary: torch.Tensor,
    metadata_blocks: Sequence[BlockMetadata],
    selected_block_ids: Sequence[int],
    step_id: str | int | None,
) -> dict[str, Any]:
    if oracle is None:
        return {
            "oracle_recall_rate": None,
            "oracle_precision_rate": None,
            "oracle_overlap_count": None,
            "oracle_missed_important_block_ids": (),
            "oracle_missed_important_count": None,
            "oracle_extra_selected_block_ids": (),
            "oracle_extra_selected_count": None,
        }

    dense_reference = oracle.reference_blocks(
        query_summary,
        metadata_blocks,
        step_id=step_id,
    )
    sparse_selection = sparse_selected_block_set(selected_block_ids, step_id=step_id)
    comparison = compare_block_sets(dense_reference, sparse_selection)
    quality = selector_quality_metrics(comparison)

    return {
        "oracle_recall_rate": quality.selector_recall_rate,
        "oracle_precision_rate": quality.selector_precision_rate,
        "oracle_overlap_count": quality.overlap_count,
        "oracle_missed_important_block_ids": tuple(
            int(block_id) for block_id in comparison.missed_important_block_ids
        ),
        "oracle_missed_important_count": quality.missed_important_count,
        "oracle_extra_selected_block_ids": tuple(
            int(block_id) for block_id in comparison.extra_selected_block_ids
        ),
        "oracle_extra_selected_count": quality.extra_selected_count,
    }


def _aggregate_oracle_summary(
    rows: Sequence[SelectorMicrobenchRow],
    *,
    spec: SelectorMicrobenchSpec,
) -> OracleAggregateSummary | None:
    if not spec.oracle_enabled:
        return None

    oracle_rows = [
        row
        for row in rows
        if row.oracle_recall_rate is not None
        and row.oracle_precision_rate is not None
        and row.oracle_overlap_count is not None
    ]
    if not oracle_rows:
        return OracleAggregateSummary(
            mean_recall_rate=0.0,
            mean_precision_rate=0.0,
            mean_overlap_count=0.0,
            low_recall_fallback_frequency_by_mode={},
        )

    low_recall_counter: Counter[str] = Counter(
        row.fallback_mode
        for row in oracle_rows
        if row.oracle_recall_rate is not None
        and row.oracle_recall_rate < spec.low_oracle_recall_threshold
    )

    return OracleAggregateSummary(
        mean_recall_rate=sum(row.oracle_recall_rate for row in oracle_rows if row.oracle_recall_rate is not None)
        / len(oracle_rows),
        mean_precision_rate=sum(
            row.oracle_precision_rate for row in oracle_rows if row.oracle_precision_rate is not None
        )
        / len(oracle_rows),
        mean_overlap_count=sum(
            row.oracle_overlap_count for row in oracle_rows if row.oracle_overlap_count is not None
        )
        / len(oracle_rows),
        low_recall_fallback_frequency_by_mode=dict(low_recall_counter),
    )
