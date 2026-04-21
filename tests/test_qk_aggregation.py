from __future__ import annotations

import torch

from kvblock.kv.qk_aggregation import (
    aggregate_block_states_for_summary,
    aggregate_query_key_heads,
    qk_aggregation_strategies_from_names,
    qk_aggregation_strategy_from_name,
)


def test_qk_aggregation_parser_validates_names() -> None:
    assert qk_aggregation_strategy_from_name("mean_pool") == "mean_pool"
    assert qk_aggregation_strategies_from_names(("mean_pool", "block_max")) == (
        "mean_pool",
        "block_max",
    )


def test_query_key_head_aggregation_outputs_are_deterministic() -> None:
    keys = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0]],
            [[-5.0, 1.0], [2.0, 8.0]],
        ]
    )
    query = torch.tensor([[3.0, 4.0], [0.0, 10.0]])

    mean_keys, mean_query = aggregate_query_key_heads(
        per_head_token_representations=keys,
        per_head_query_representation=query,
        strategy="mean_pool",
    )
    max_keys, max_query = aggregate_query_key_heads(
        per_head_token_representations=keys,
        per_head_query_representation=query,
        strategy="max_pool",
    )
    weighted_keys, weighted_query = aggregate_query_key_heads(
        per_head_token_representations=keys,
        per_head_query_representation=query,
        strategy="norm_weighted_mean",
    )

    assert torch.equal(mean_keys, torch.tensor([[-2.0, -0.5], [2.5, 2.0]]))
    assert torch.equal(mean_query, torch.tensor([1.5, 7.0]))
    assert torch.equal(max_keys, torch.tensor([[-5.0, -2.0], [3.0, 8.0]]))
    assert torch.equal(max_query, torch.tensor([3.0, 10.0]))
    assert weighted_keys.shape == (2, 2)
    assert weighted_query.shape == (2,)
    assert torch.allclose(
        weighted_query,
        torch.tensor([1.0, 8.0]),
    )


def test_block_summary_aggregation_selects_top_tokens_and_block_max() -> None:
    states = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    query = torch.tensor([1.0, 0.0])

    top_states = aggregate_block_states_for_summary(
        states,
        query,
        strategy="top_token_mean",
        top_token_count=2,
    )
    max_state = aggregate_block_states_for_summary(
        states,
        query,
        strategy="block_max",
    )

    assert torch.equal(top_states, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    assert torch.equal(max_state, torch.tensor([[2.0, 1.0]]))
