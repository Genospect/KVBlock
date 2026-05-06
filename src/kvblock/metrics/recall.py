"""Recall helpers for selected block/page reports."""

from __future__ import annotations


def set_recall(selected: set[int], oracle: set[int]) -> float:
    """Return oracle set recall for selected ids."""

    if not oracle:
        return 1.0
    return len(selected & oracle) / len(oracle)
