"""Dense-vs-sparse correctness harness helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DenseVsSparseSummary:
    """Compact summary for one correctness comparison."""

    selected_token_fraction: float
    kl_divergence: float
    top1_agreement: float
    top5_overlap: float
    max_abs_logit_diff: float
