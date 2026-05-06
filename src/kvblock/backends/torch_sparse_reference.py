"""Sparse PyTorch reference decode attention.

This backend proves selection semantics only. It gathers selected token ranges
with PyTorch indexing and does not demonstrate physical sparse speedup.
"""

from __future__ import annotations

from typing import Any

import torch

from kvblock.backends.torch_dense import dense_decode_attention
from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


class TorchSparseReferenceBackend:
    """Correctness-first sparse attention backend."""

    name = "torch_sparse_reference"

    def run_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: SelectedKVPlan,
        layout: BlockLayout,
        **kwargs: Any,
    ) -> torch.Tensor:
        return sparse_reference_decode_attention(query, key_cache, value_cache, plan, layout)


def sparse_reference_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    plan: SelectedKVPlan,
    layout: BlockLayout,
) -> torch.Tensor:
    """Run dense attention over gathered selected token positions only."""

    token_ids = _selected_token_ids(plan, layout, key_cache.device)
    if key_cache.ndim in {2, 3}:
        sparse_key = key_cache.index_select(0, token_ids)
        sparse_value = value_cache.index_select(0, token_ids)
    elif key_cache.ndim == 4:
        sparse_key = key_cache.index_select(1, token_ids)
        sparse_value = value_cache.index_select(1, token_ids)
    else:
        raise ValueError("unsupported key_cache rank")
    return dense_decode_attention(query, sparse_key, sparse_value)


def _selected_token_ids(
    plan: SelectedKVPlan,
    layout: BlockLayout,
    device: torch.device,
) -> torch.Tensor:
    ranges = plan.selected_token_ranges
    if not ranges and plan.logical_block_ids:
        ranges = layout.token_ranges_for_blocks(plan.logical_block_ids)
    ids: set[int] = set()
    for start, end in ranges:
        if start < 0 or end > layout.total_tokens:
            raise ValueError("selected token ranges must fit inside layout")
        ids.update(range(start, end))
    if not ids:
        raise ValueError("sparse reference attention requires at least one selected token")
    return torch.tensor(sorted(ids), dtype=torch.long, device=device)
