"""64-bit sign sketch utilities for the V1 scaffold."""

from __future__ import annotations

from typing import Sequence

import torch

from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding

UINT64_MASK = (1 << 64) - 1


def generate_sign_sketch(
    summary: MultiHeadQuerySummary | SummaryEncoding | Sequence[float] | torch.Tensor,
    *,
    bits: int = 64,
) -> int:
    """Generate a deterministic sign sketch using a small SimHash-style projection."""

    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits!r}")

    vector = _coerce_summary(summary)
    sketch = 0

    for bit_index in range(bits):
        score = 0.0
        for dim_index, value in enumerate(vector.tolist()):
            hashed = _splitmix64(((bit_index + 1) << 32) ^ (dim_index + 1))
            coeff = 1.0 if (hashed & 1) else -1.0
            score += value * coeff
        if score >= 0:
            sketch |= 1 << bit_index

    return sketch


def hamming_similarity(left: int, right: int, *, bits: int = 64) -> float:
    """Return normalized Hamming similarity in ``[0, 1]`` for two sketches."""

    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits!r}")
    distance = (left ^ right).bit_count()
    return 1.0 - (distance / bits)


def _coerce_summary(
    summary: MultiHeadQuerySummary | SummaryEncoding | Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    if isinstance(summary, MultiHeadQuerySummary):
        vector = summary.dequantize()
    elif isinstance(summary, SummaryEncoding):
        vector = summary.dequantize()
    elif isinstance(summary, torch.Tensor):
        vector = summary.detach().to(dtype=torch.float32).reshape(-1)
    else:
        vector = torch.tensor(tuple(summary), dtype=torch.float32)

    if vector.ndim != 1:
        vector = vector.reshape(-1)
    if vector.numel() == 0:
        raise ValueError("summary must not be empty")
    return vector.cpu()


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & UINT64_MASK
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & UINT64_MASK
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & UINT64_MASK
    return (value ^ (value >> 31)) & UINT64_MASK
