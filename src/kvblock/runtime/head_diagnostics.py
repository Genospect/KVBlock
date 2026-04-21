"""Per-head Q/K contribution diagnostics for dense-only real-block runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.policies import StageAPolicy
from kvblock.summaries.base import MultiHeadQuerySummary


@dataclass(frozen=True, slots=True)
class HeadContribution:
    """Score assigned to one attention head for a block."""

    head_index: int
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadFrequency:
    """Frequency with which one head appears as the top contributor."""

    head_index: int
    count: int
    fraction: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class HeadDiagnosticBlockContext:
    """Text/span context attached to one block diagnostic record."""

    block_id: int
    selected: bool
    selected_reason: str
    token_start: int
    token_end: int
    token_count: int
    preview_text: str


@dataclass(frozen=True, slots=True)
class PerHeadBlockDiagnostic:
    """Per-head contribution record for one metadata block."""

    block_id: int
    selected: bool
    selected_reason: str
    head_scores: tuple[float, ...]
    top_contributing_heads: tuple[HeadContribution, ...]
    aggregated_score: float
    token_start: int
    token_end: int
    token_count: int
    preview_text: str
    representation_source: str
    representation_name: str
    rail_setting: str | None = None
    prompt_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly diagnostic record."""

        payload = asdict(self)
        payload["top_contributing_heads"] = [
            head.to_dict() for head in self.top_contributing_heads
        ]
        return payload


@dataclass(frozen=True, slots=True)
class HeadDiagnosticAggregate:
    """Aggregate head contribution summary for one run or prompt group."""

    prompt_name: str | None
    block_count: int
    selected_count: int
    correct_selected_count: int | None
    head_count: int
    top_head_counts: tuple[HeadFrequency, ...]
    selected_top_head_counts: tuple[HeadFrequency, ...]
    correct_selected_top_head_counts: tuple[HeadFrequency, ...]
    mean_head_scores: tuple[float, ...]
    selected_mean_head_scores: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly aggregate record."""

        payload = asdict(self)
        payload["top_head_counts"] = [
            item.to_dict() for item in self.top_head_counts
        ]
        payload["selected_top_head_counts"] = [
            item.to_dict() for item in self.selected_top_head_counts
        ]
        payload["correct_selected_top_head_counts"] = [
            item.to_dict() for item in self.correct_selected_top_head_counts
        ]
        return payload


def build_per_head_block_diagnostics(
    *,
    metadata_blocks: Sequence[BlockMetadata],
    query_summary: object,
    block_contexts: Sequence[HeadDiagnosticBlockContext],
    policy: StageAPolicy,
    representation_source: str,
    representation_name: str,
    top_heads: int = 5,
    rail_setting: str | None = None,
    prompt_name: str | None = None,
) -> tuple[PerHeadBlockDiagnostic, ...]:
    """Build per-head Q/K contribution records when per-head summaries exist."""

    if top_heads <= 0:
        raise ValueError("top_heads must be > 0")
    if not isinstance(query_summary, MultiHeadQuerySummary):
        return ()

    contexts = {context.block_id: context for context in block_contexts}
    per_head_query = query_summary.dequantize_heads()
    pooled_query = query_summary.dequantize()
    records: list[PerHeadBlockDiagnostic] = []
    for metadata in metadata_blocks:
        if not metadata.per_head_summary_fp8:
            continue
        context = contexts.get(int(metadata.block_id))
        if context is None:
            continue
        per_head_scores = _per_head_similarity_scores(
            per_head_query,
            metadata.dequantize_per_head_summary(),
        )
        aggregated_score = _aggregate_similarity_score(
            per_head_scores,
            policy=policy,
            pooled_query=pooled_query,
            metadata=metadata,
        )
        records.append(
            PerHeadBlockDiagnostic(
                block_id=context.block_id,
                selected=context.selected,
                selected_reason=context.selected_reason,
                head_scores=per_head_scores,
                top_contributing_heads=_top_contributing_heads(
                    per_head_scores,
                    top_heads=top_heads,
                ),
                aggregated_score=aggregated_score,
                token_start=context.token_start,
                token_end=context.token_end,
                token_count=context.token_count,
                preview_text=context.preview_text,
                representation_source=representation_source,
                representation_name=representation_name,
                rail_setting=rail_setting,
                prompt_name=prompt_name,
            )
        )
    return tuple(records)


def summarize_head_diagnostics(
    diagnostics: Sequence[PerHeadBlockDiagnostic],
    *,
    expected_block_ids: Sequence[int] | None = None,
    prompt_name: str | None = None,
    top_n: int = 5,
) -> HeadDiagnosticAggregate:
    """Summarize top-ranked heads across all or selected diagnostic records."""

    if top_n <= 0:
        raise ValueError("top_n must be > 0")
    if not diagnostics:
        return HeadDiagnosticAggregate(
            prompt_name=prompt_name,
            block_count=0,
            selected_count=0,
            correct_selected_count=None if expected_block_ids is None else 0,
            head_count=0,
            top_head_counts=(),
            selected_top_head_counts=(),
            correct_selected_top_head_counts=(),
            mean_head_scores=(),
            selected_mean_head_scores=(),
        )

    expected = None if expected_block_ids is None else set(int(item) for item in expected_block_ids)
    head_count = max(len(record.head_scores) for record in diagnostics)
    selected = tuple(record for record in diagnostics if record.selected)
    correct_selected = (
        ()
        if expected is None
        else tuple(record for record in selected if record.block_id in expected)
    )
    return HeadDiagnosticAggregate(
        prompt_name=prompt_name,
        block_count=len(diagnostics),
        selected_count=len(selected),
        correct_selected_count=None if expected is None else len(correct_selected),
        head_count=head_count,
        top_head_counts=_top_head_frequencies(diagnostics, top_n=top_n),
        selected_top_head_counts=_top_head_frequencies(selected, top_n=top_n),
        correct_selected_top_head_counts=_top_head_frequencies(correct_selected, top_n=top_n),
        mean_head_scores=_mean_head_scores(diagnostics, head_count=head_count),
        selected_mean_head_scores=_mean_head_scores(selected, head_count=head_count),
    )


def summarize_head_diagnostics_by_prompt(
    diagnostics: Sequence[PerHeadBlockDiagnostic],
    *,
    expected_block_ids_by_prompt: Mapping[str, Sequence[int]] | None = None,
    top_n: int = 5,
) -> tuple[HeadDiagnosticAggregate, ...]:
    """Group diagnostics by prompt name and summarize each prompt independently."""

    grouped: dict[str | None, list[PerHeadBlockDiagnostic]] = {}
    for record in diagnostics:
        grouped.setdefault(record.prompt_name, []).append(record)

    summaries: list[HeadDiagnosticAggregate] = []
    for prompt_name, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        expected = (
            None
            if prompt_name is None or expected_block_ids_by_prompt is None
            else expected_block_ids_by_prompt.get(prompt_name)
        )
        summaries.append(
            summarize_head_diagnostics(
                records,
                expected_block_ids=expected,
                prompt_name=prompt_name,
                top_n=top_n,
            )
        )
    return tuple(summaries)


def _per_head_similarity_scores(
    per_head_query: torch.Tensor,
    per_head_summary: torch.Tensor,
) -> tuple[float, ...]:
    if per_head_query.shape != per_head_summary.shape:
        raise ValueError(
            "per-head query and block summary shapes must match, got "
            f"{tuple(per_head_query.shape)} and {tuple(per_head_summary.shape)}"
        )
    query_norms = torch.linalg.vector_norm(per_head_query, dim=1)
    block_norms = torch.linalg.vector_norm(per_head_summary, dim=1)
    denominators = query_norms * block_norms
    dots = torch.sum(per_head_query * per_head_summary, dim=1)
    cosine = torch.where(
        denominators > 0,
        dots / denominators.clamp_min(1e-12),
        torch.zeros_like(denominators),
    )
    scores = (cosine + 1.0) * 0.5
    return tuple(float(value) for value in scores.tolist())


def _aggregate_similarity_score(
    head_scores: Sequence[float],
    *,
    policy: StageAPolicy,
    pooled_query: torch.Tensor,
    metadata: BlockMetadata,
) -> float:
    if not head_scores:
        return 0.0
    if policy.head_scoring_mode == "mean_heads":
        return _approx_similarity_score(pooled_query, metadata.dequantize_summary())
    if policy.head_scoring_mode == "max_head_score":
        return max(head_scores)
    if policy.head_scoring_mode == "topk_head_mean":
        k = min(policy.head_top_k, len(head_scores))
        return sum(sorted(head_scores, reverse=True)[:k]) / k
    if policy.head_scoring_mode == "weighted_head_mean":
        if len(policy.head_weights) != len(head_scores):
            raise ValueError(
                "head_weights length must match available head count, got "
                f"{len(policy.head_weights)} and {len(head_scores)}"
            )
        total = sum(policy.head_weights)
        return sum(score * weight for score, weight in zip(head_scores, policy.head_weights)) / total
    raise ValueError(f"unsupported head_scoring_mode: {policy.head_scoring_mode!r}")


def _approx_similarity_score(query: torch.Tensor, block_summary: torch.Tensor) -> float:
    query = query.reshape(-1)
    block = block_summary.reshape(-1).to(dtype=query.dtype, device=query.device)
    denominator = torch.linalg.vector_norm(query) * torch.linalg.vector_norm(block)
    if float(denominator.item()) <= 0:
        return 0.0
    cosine = torch.dot(query, block) / denominator.clamp_min(1e-12)
    return float(((cosine + 1.0) * 0.5).item())


def _top_contributing_heads(
    head_scores: Sequence[float],
    *,
    top_heads: int,
) -> tuple[HeadContribution, ...]:
    ranked = sorted(
        enumerate(head_scores),
        key=lambda item: (-item[1], item[0]),
    )[:top_heads]
    return tuple(
        HeadContribution(head_index=head_index, score=float(score), rank=rank)
        for rank, (head_index, score) in enumerate(ranked, start=1)
    )


def _top_head_frequencies(
    diagnostics: Sequence[PerHeadBlockDiagnostic],
    *,
    top_n: int,
) -> tuple[HeadFrequency, ...]:
    counts: dict[int, int] = {}
    total = 0
    for record in diagnostics:
        if not record.top_contributing_heads:
            continue
        total += 1
        head_index = record.top_contributing_heads[0].head_index
        counts[head_index] = counts.get(head_index, 0) + 1
    if total == 0:
        return ()
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_n]
    return tuple(
        HeadFrequency(
            head_index=head_index,
            count=count,
            fraction=count / total,
        )
        for head_index, count in ranked
    )


def _mean_head_scores(
    diagnostics: Sequence[PerHeadBlockDiagnostic],
    *,
    head_count: int,
) -> tuple[float, ...]:
    if not diagnostics or head_count == 0:
        return ()
    sums = [0.0] * head_count
    counts = [0] * head_count
    for record in diagnostics:
        for index, score in enumerate(record.head_scores):
            if index >= head_count:
                break
            sums[index] += score
            counts[index] += 1
    return tuple(
        0.0 if count == 0 else sums[index] / count
        for index, count in enumerate(counts)
    )
