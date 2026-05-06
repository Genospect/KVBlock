"""Logical KV read byte calculators."""

from __future__ import annotations


def logical_kv_read_bytes(
    *,
    selected_tokens: int,
    num_kv_heads: int,
    head_dim: int,
    bytes_per_element: int,
) -> int:
    """Estimate logical bytes touched for K and V reads in one decode step."""

    if selected_tokens < 0 or num_kv_heads <= 0 or head_dim <= 0:
        raise ValueError("selected_tokens must be >= 0 and dimensions must be > 0")
    if bytes_per_element <= 0:
        raise ValueError("bytes_per_element must be > 0")
    return int(selected_tokens * num_kv_heads * head_dim * bytes_per_element * 2)
