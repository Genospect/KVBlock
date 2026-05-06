"""Dense PyTorch reference decode attention."""

from __future__ import annotations

from math import sqrt
from typing import Any

import torch

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


class TorchDenseAttentionBackend:
    """Correctness-first dense attention backend."""

    name = "torch_dense"

    def run_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: SelectedKVPlan,
        layout: BlockLayout,
        **kwargs: Any,
    ) -> torch.Tensor:
        return dense_decode_attention(query, key_cache, value_cache)


def dense_decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Run scaled dot-product decode attention over all cached tokens."""

    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")
    if query.numel() == 0 or key_cache.numel() == 0:
        raise ValueError("query and caches must not be empty")

    scale = 1.0 / sqrt(query.shape[-1])
    if query.ndim == 1 and key_cache.ndim == 2:
        scores = torch.einsum("d,td->t", query, key_cache) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("t,td->d", probs, value_cache)
    if query.ndim == 2 and key_cache.ndim == 3:
        scores = torch.einsum("hd,thd->ht", query, key_cache) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("ht,thd->hd", probs, value_cache)
    if query.ndim == 3 and key_cache.ndim == 4:
        scores = torch.einsum("bhd,bthd->bht", query, key_cache) * scale
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("bht,bthd->bhd", probs, value_cache)
    raise ValueError(
        "supported shapes are query [D]/K [T,D], query [H,D]/K [T,H,D], "
        "or query [B,H,D]/K [B,T,H,D]"
    )
