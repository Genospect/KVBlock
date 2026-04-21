"""Stage A coarse scoring for the V1 selector skeleton."""

from __future__ import annotations

from typing import Sequence

import torch

from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import (
    QuerySummary,
    ScoredBlock,
    coerce_per_head_query_summary,
    coerce_query_summary,
    metadata_summary_tensor,
)
from kvblock.selector.policies import StageAPolicy


class StageAScorer:
    """Compute weighted coarse scores over block metadata."""

    def __init__(self, policy: StageAPolicy | None = None) -> None:
        self.policy = policy or StageAPolicy()

    def score(
        self,
        metadata_blocks: Sequence[BlockMetadata],
        query_summary: QuerySummary,
        *,
        current_step: int,
        shortlist_size: int | None = None,
        context_tokens: int | None = None,
    ) -> list[ScoredBlock]:
        """Return the Stage A shortlist sorted by descending coarse score."""

        scored = self.score_all(
            metadata_blocks,
            query_summary,
            current_step=current_step,
        )
        target_size = shortlist_size or self.policy.shortlist_for_context(context_tokens)
        return scored[:target_size]

    def score_all(
        self,
        metadata_blocks: Sequence[BlockMetadata],
        query_summary: QuerySummary,
        *,
        current_step: int,
    ) -> list[ScoredBlock]:
        """Return all Stage A scores sorted by descending coarse score.

        The numeric feature path is vectorized: summaries are materialized into
        one matrix and scored against the query with one batched dot/cosine-style
        operation. The remaining per-block work builds trace-friendly
        ``ScoredBlock`` records and applies deterministic Python sorting.
        """

        if not metadata_blocks:
            return []

        query_tensor = coerce_query_summary(query_summary)
        summary_matrix = _metadata_summary_matrix(
            metadata_blocks, dtype=query_tensor.dtype, device=query_tensor.device
        )
        if summary_matrix.shape[1] != query_tensor.numel():
            raise ValueError(
                "query_summary and block summaries must have matching dimensions, got "
                f"{query_tensor.numel()} and {summary_matrix.shape[1]}"
            )

        approx_similarity_scores = _stage_a_similarity_scores(
            metadata_blocks,
            query_summary,
            query_tensor,
            summary_matrix,
            policy=self.policy,
        )
        last_access_steps = torch.tensor(
            [block.last_access_step for block in metadata_blocks],
            dtype=torch.float32,
            device=query_tensor.device,
        )
        if bool(torch.any(last_access_steps > float(current_step)).item()):
            raise ValueError("current_step must be >= all non-negative last_access_step values")
        recency_scores = torch.where(
            last_access_steps < 0,
            torch.zeros_like(last_access_steps),
            1.0 / (1.0 + float(current_step) - last_access_steps),
        )
        attn_scores = _min_max_normalize_tensor(
            torch.tensor(
                [block.attn_ema for block in metadata_blocks],
                dtype=torch.float32,
                device=query_tensor.device,
            )
        )
        priority_scores = _min_max_normalize_tensor(
            torch.tensor(
                [block.priority for block in metadata_blocks],
                dtype=torch.float32,
                device=query_tensor.device,
            )
        )
        weights = self.policy.weights
        stage_a_scores = (
            weights.summary_similarity * approx_similarity_scores
            + weights.recency * recency_scores
            + weights.attn_ema * attn_scores
            + weights.priority * priority_scores
        )

        scored: list[ScoredBlock] = []
        similarity_values = approx_similarity_scores.tolist()
        recency_values = recency_scores.tolist()
        attn_values = attn_scores.tolist()
        priority_values = priority_scores.tolist()
        stage_a_values = stage_a_scores.tolist()
        for index, block in enumerate(metadata_blocks):
            scored.append(
                ScoredBlock(
                    metadata=block,
                    approx_similarity_score=similarity_values[index],
                    recency_score=recency_values[index],
                    attn_score=attn_values[index],
                    priority_score=priority_values[index],
                    stage_a_score=stage_a_values[index],
                    final_score=stage_a_values[index],
                )
            )

        scored.sort(key=_rank_key, reverse=True)
        return scored


def approx_cosine_similarity(query_summary: QuerySummary, block: BlockMetadata) -> float:
    """Compute a scale-aware approximate cosine score from low-precision summaries.

    This uses the dequantized `int8 + scale` metadata representation and therefore
    should be treated as a heuristic ranking signal, not an exact cosine similarity
    over original full-precision summaries.
    """

    query_tensor = coerce_query_summary(query_summary)
    block_tensor = metadata_summary_tensor(block).to(
        dtype=query_tensor.dtype, device=query_tensor.device
    )
    if block_tensor.numel() != query_tensor.numel():
        raise ValueError(
            "query_summary and block summary must have matching dimensions, got "
            f"{query_tensor.numel()} and {block_tensor.numel()}"
        )

    return float(
        _batched_approx_cosine_similarity(query_tensor, block_tensor.unsqueeze(0))[0].item()
    )


def _recency_score(last_access_step: int, current_step: int) -> float:
    if last_access_step < 0:
        return 0.0
    if current_step < last_access_step:
        raise ValueError(
            "current_step must be >= last_access_step, got "
            f"{current_step} < {last_access_step}"
        )
    distance = current_step - last_access_step
    return 1.0 / (1.0 + float(distance))


def _min_max_normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    tensor = torch.tensor(tuple(values), dtype=torch.float32)
    return _min_max_normalize_tensor(tensor).tolist()


def _metadata_summary_matrix(
    metadata_blocks: Sequence[BlockMetadata],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    expected_dim = len(metadata_blocks[0].summary_fp8)
    if any(len(block.summary_fp8) != expected_dim for block in metadata_blocks):
        raise ValueError("all block summaries must have the same dimension")

    quantized = torch.tensor(
        [block.summary_fp8 for block in metadata_blocks],
        dtype=torch.float32,
        device=device,
    )
    scales = torch.tensor(
        [block.summary_scale for block in metadata_blocks],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(1)
    return quantized.mul(scales).to(dtype=dtype)


def _stage_a_similarity_scores(
    metadata_blocks: Sequence[BlockMetadata],
    query_summary: QuerySummary,
    query_tensor: torch.Tensor,
    summary_matrix: torch.Tensor,
    *,
    policy: StageAPolicy,
) -> torch.Tensor:
    # ``mean_heads`` intentionally preserves the current pooled Q/K baseline:
    # query-key sources mean-pool heads before metadata construction and score
    # that pooled summary here. The other modes opt into per-head metadata.
    if policy.head_scoring_mode == "mean_heads":
        return _batched_approx_cosine_similarity(query_tensor, summary_matrix)

    per_head_query = coerce_per_head_query_summary(query_summary)
    if per_head_query is None:
        return _batched_approx_cosine_similarity(query_tensor, summary_matrix)
    if any(not block.per_head_summary_fp8 for block in metadata_blocks):
        return _batched_approx_cosine_similarity(query_tensor, summary_matrix)

    per_head_query = per_head_query.to(dtype=query_tensor.dtype, device=query_tensor.device)
    per_head_matrix = _metadata_per_head_summary_matrix(
        metadata_blocks,
        dtype=query_tensor.dtype,
        device=query_tensor.device,
    )
    if per_head_matrix.shape[1:] != per_head_query.shape:
        raise ValueError(
            "per-head query summaries and block summaries must have matching "
            f"shape, got {tuple(per_head_query.shape)} and {tuple(per_head_matrix.shape[1:])}"
        )
    per_head_scores = _batched_per_head_approx_cosine_similarity(
        per_head_query,
        per_head_matrix,
    )
    if policy.head_scoring_mode == "max_head_score":
        return torch.max(per_head_scores, dim=1).values
    if policy.head_scoring_mode == "topk_head_mean":
        k = min(policy.head_top_k, per_head_scores.shape[1])
        return torch.topk(per_head_scores, k=k, dim=1).values.mean(dim=1)
    if policy.head_scoring_mode == "weighted_head_mean":
        if len(policy.head_weights) != per_head_scores.shape[1]:
            raise ValueError(
                "head_weights length must match available head count, got "
                f"{len(policy.head_weights)} and {per_head_scores.shape[1]}"
            )
        weights = torch.tensor(
            policy.head_weights,
            dtype=per_head_scores.dtype,
            device=per_head_scores.device,
        )
        weights = weights / weights.sum().clamp_min(1e-12)
        return per_head_scores.matmul(weights)
    raise ValueError(f"unsupported head_scoring_mode: {policy.head_scoring_mode!r}")


def _metadata_per_head_summary_matrix(
    metadata_blocks: Sequence[BlockMetadata],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    expected_heads = len(metadata_blocks[0].per_head_summary_fp8)
    expected_dim = len(metadata_blocks[0].summary_fp8)
    if expected_heads == 0:
        raise ValueError("per-head summaries are not available")
    for block in metadata_blocks:
        if len(block.per_head_summary_fp8) != expected_heads:
            raise ValueError("all blocks must have the same number of per-head summaries")
        if any(len(values) != expected_dim for values in block.per_head_summary_fp8):
            raise ValueError("all per-head summaries must have the same dimension")

    quantized = torch.tensor(
        [block.per_head_summary_fp8 for block in metadata_blocks],
        dtype=torch.float32,
        device=device,
    )
    scales = torch.tensor(
        [block.per_head_summary_scale for block in metadata_blocks],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(2)
    return quantized.mul(scales).to(dtype=dtype)


def _batched_approx_cosine_similarity(
    query_tensor: torch.Tensor, summary_matrix: torch.Tensor
) -> torch.Tensor:
    query = query_tensor.reshape(-1)
    query_norm = torch.linalg.vector_norm(query)
    block_norms = torch.linalg.vector_norm(summary_matrix, dim=1)
    denominators = block_norms * query_norm
    cosine = torch.where(
        denominators > 0,
        summary_matrix.matmul(query) / denominators.clamp_min(1e-12),
        torch.zeros_like(denominators),
    )
    return (cosine + 1.0) * 0.5


def _batched_per_head_approx_cosine_similarity(
    per_head_query: torch.Tensor,
    per_head_summary_matrix: torch.Tensor,
) -> torch.Tensor:
    query_norms = torch.linalg.vector_norm(per_head_query, dim=1).unsqueeze(0)
    block_norms = torch.linalg.vector_norm(per_head_summary_matrix, dim=2)
    denominators = block_norms * query_norms
    dots = torch.sum(per_head_summary_matrix * per_head_query.unsqueeze(0), dim=2)
    cosine = torch.where(
        denominators > 0,
        dots / denominators.clamp_min(1e-12),
        torch.zeros_like(denominators),
    )
    return (cosine + 1.0) * 0.5


def _min_max_normalize_tensor(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values
    minimum = torch.min(values)
    maximum = torch.max(values)
    if bool((minimum == maximum).item()):
        return torch.zeros_like(values)
    return (values - minimum) / (maximum - minimum)


def _rank_key(candidate: ScoredBlock) -> tuple[float, int, int]:
    return (
        candidate.final_score,
        candidate.metadata.last_access_step,
        candidate.metadata.token_start,
    )
