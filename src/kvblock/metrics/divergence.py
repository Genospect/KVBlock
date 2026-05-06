"""Logit divergence metrics for dense-vs-sparse correctness checks."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence_logits(
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
) -> float:
    """Return KL(dense || sparse) averaged over leading rows."""

    dense, sparse = _flatten_logits(dense_logits, sparse_logits)
    dense_probs = F.softmax(dense, dim=-1)
    sparse_log_probs = F.log_softmax(sparse, dim=-1)
    value = float(F.kl_div(sparse_log_probs, dense_probs, reduction="batchmean").item())
    if abs(value) < 1e-7:
        return 0.0
    return max(0.0, value)


def topk_agreement(
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    *,
    k: int = 1,
) -> float:
    """Return the row fraction where dense and sparse top-k sets match exactly."""

    if k <= 0:
        raise ValueError("k must be > 0")
    dense, sparse = _flatten_logits(dense_logits, sparse_logits)
    k = min(k, dense.shape[-1])
    dense_top = torch.topk(dense, k=k, dim=-1).indices.sort(dim=-1).values
    sparse_top = torch.topk(sparse, k=k, dim=-1).indices.sort(dim=-1).values
    return float((dense_top == sparse_top).all(dim=-1).float().mean().item())


def topk_overlap(
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
    *,
    k: int = 5,
) -> float:
    """Return the mean fraction of dense top-k ids recovered by sparse top-k."""

    if k <= 0:
        raise ValueError("k must be > 0")
    dense, sparse = _flatten_logits(dense_logits, sparse_logits)
    k = min(k, dense.shape[-1])
    dense_top = torch.topk(dense, k=k, dim=-1).indices
    sparse_top = torch.topk(sparse, k=k, dim=-1).indices
    overlaps = []
    for dense_row, sparse_row in zip(dense_top, sparse_top, strict=True):
        overlaps.append(
            len(set(dense_row.tolist()) & set(sparse_row.tolist())) / float(k)
        )
    return float(sum(overlaps) / len(overlaps))


def max_abs_logit_diff(
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
) -> float:
    """Return the maximum absolute logit difference."""

    dense, sparse = _flatten_logits(dense_logits, sparse_logits)
    return float(torch.max(torch.abs(dense - sparse)).item())


def _flatten_logits(
    dense_logits: torch.Tensor,
    sparse_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dense_logits.shape != sparse_logits.shape:
        raise ValueError("dense_logits and sparse_logits must have the same shape")
    if dense_logits.numel() == 0:
        raise ValueError("logits must not be empty")
    dense = dense_logits.detach().to(dtype=torch.float32)
    sparse = sparse_logits.detach().to(dtype=torch.float32, device=dense.device)
    return dense.reshape(-1, dense.shape[-1]), sparse.reshape(-1, sparse.shape[-1])
