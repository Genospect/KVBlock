"""Fallback widening helpers."""

from __future__ import annotations


def widen_budget(current: int, *, by: int, total: int) -> int:
    """Increase a sparse selection budget without exceeding total blocks."""

    if current < 0 or by <= 0 or total < 0:
        raise ValueError("current and total must be >= 0, by must be > 0")
    return min(total, current + by)
