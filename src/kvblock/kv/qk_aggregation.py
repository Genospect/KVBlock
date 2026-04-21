"""Constrained query/key aggregation strategies for V1 experiments."""

from __future__ import annotations

from typing import Literal, Sequence, cast

import torch

QKAggregationStrategy = Literal[
    "mean_pool",
    "max_pool",
    "norm_weighted_mean",
    "top_token_mean",
    "block_max",
]

VALID_QK_AGGREGATION_STRATEGIES: tuple[QKAggregationStrategy, ...] = (
    "mean_pool",
    "max_pool",
    "norm_weighted_mean",
    "top_token_mean",
    "block_max",
)


def qk_aggregation_strategy_from_name(name: str) -> QKAggregationStrategy:
    """Validate and return one query/key aggregation strategy name."""

    normalized = name.strip()
    if normalized not in VALID_QK_AGGREGATION_STRATEGIES:
        valid = ", ".join(VALID_QK_AGGREGATION_STRATEGIES)
        raise ValueError(f"unknown query/key aggregation {name!r}; valid: {valid}")
    return cast(QKAggregationStrategy, normalized)


def qk_aggregation_strategies_from_names(
    names: Sequence[str],
) -> tuple[QKAggregationStrategy, ...]:
    """Resolve one or more aggregation names with stable validation."""

    if not names:
        raise ValueError("aggregation strategy names must not be empty")
    strategies = tuple(
        qk_aggregation_strategy_from_name(name)
        for name in names
        if name.strip()
    )
    if not strategies:
        raise ValueError("aggregation strategy names must not be empty")
    return strategies


def aggregate_query_key_heads(
    *,
    per_head_token_representations: torch.Tensor,
    per_head_query_representation: torch.Tensor,
    strategy: QKAggregationStrategy,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-head K streams and latest Q heads into pooled vectors.

    The output shape intentionally matches the existing selector bridge:
    ``[tokens, features]`` for keys and ``[features]`` for the current query.
    """

    keys = _normalize_per_head_tokens(per_head_token_representations)
    query = _normalize_per_head_query(per_head_query_representation)
    if keys.shape[0] != query.shape[0]:
        raise ValueError("per-head key and query head counts must match")
    if keys.shape[2] != query.shape[1]:
        raise ValueError("per-head key and query feature dims must match")

    if strategy in {"mean_pool", "top_token_mean", "block_max"}:
        return keys.mean(dim=0), query.mean(dim=0)
    if strategy == "max_pool":
        return _signed_absmax(keys, dim=0), _signed_absmax(query, dim=0)
    if strategy == "norm_weighted_mean":
        weights = torch.linalg.vector_norm(query, dim=1)
        weights = weights / weights.sum().clamp_min(1e-12)
        return (
            torch.sum(keys * weights.reshape(-1, 1, 1), dim=0),
            torch.sum(query * weights.reshape(-1, 1), dim=0),
        )
    raise ValueError(f"unsupported query/key aggregation strategy: {strategy!r}")


def aggregate_block_states_for_summary(
    block_states: torch.Tensor,
    query_vector: torch.Tensor,
    *,
    strategy: QKAggregationStrategy,
    top_token_count: int = 4,
) -> torch.Tensor:
    """Return block states to feed into the existing low-precision summary builder."""

    states = _normalize_block_states(block_states)
    query = _normalize_query(query_vector)
    if states.shape[1] != query.numel():
        raise ValueError("block states and query vector feature dims must match")
    if top_token_count <= 0:
        raise ValueError("top_token_count must be > 0")

    if strategy == "top_token_mean":
        k = min(top_token_count, states.shape[0])
        scores = _cosine_scores(states, query)
        top_indices = torch.topk(scores, k=k).indices.sort().values
        return states.index_select(0, top_indices)
    if strategy == "block_max":
        return _signed_absmax(states, dim=0).unsqueeze(0)
    return states


def _signed_absmax(values: torch.Tensor, *, dim: int) -> torch.Tensor:
    abs_values = values.abs()
    indices = torch.argmax(abs_values, dim=dim, keepdim=True)
    return torch.gather(values, dim, indices).squeeze(dim)


def _cosine_scores(states: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    state_norms = torch.linalg.vector_norm(states, dim=1)
    query_norm = torch.linalg.vector_norm(query)
    denom = state_norms * query_norm
    dots = states.matmul(query)
    return torch.where(
        denom > 0,
        dots / denom.clamp_min(1e-12),
        torch.zeros_like(dots),
    )


def _normalize_per_head_tokens(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("per_head_token_representations must be a torch.Tensor")
    if values.ndim != 3:
        raise ValueError(
            "per_head_token_representations must have shape [heads, tokens, features]"
        )
    if 0 in values.shape:
        raise ValueError("per_head_token_representations dimensions must be non-empty")
    return values.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _normalize_per_head_query(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("per_head_query_representation must be a torch.Tensor")
    if values.ndim != 2:
        raise ValueError("per_head_query_representation must have shape [heads, features]")
    if 0 in values.shape:
        raise ValueError("per_head_query_representation dimensions must be non-empty")
    return values.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _normalize_block_states(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("block_states must be a torch.Tensor")
    if values.ndim != 2:
        raise ValueError("block_states must have shape [tokens, features]")
    if 0 in values.shape:
        raise ValueError("block_states dimensions must be non-empty")
    return values.detach().to(dtype=torch.float32, device="cpu").contiguous()


def _normalize_query(values: torch.Tensor) -> torch.Tensor:
    if not isinstance(values, torch.Tensor):
        raise TypeError("query_vector must be a torch.Tensor")
    if values.ndim != 1:
        raise ValueError("query_vector must have shape [features]")
    if values.numel() == 0:
        raise ValueError("query_vector must not be empty")
    return values.detach().to(dtype=torch.float32, device="cpu").contiguous()
