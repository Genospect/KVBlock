"""Small confidence helpers for package-level fallback decisions."""

from __future__ import annotations


def is_confident_margin(margin: float, threshold: float = 0.05) -> bool:
    """Return true when a selector margin clears the fallback threshold."""

    if threshold < 0:
        raise ValueError("threshold must be >= 0")
    return margin >= threshold
