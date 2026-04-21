"""Graded fallback decisions for the V1 selector skeleton."""

from __future__ import annotations

from dataclasses import dataclass

from kvblock.selector.confidence import ConfidenceAssessment
from kvblock.selector.policies import FallbackPolicy


@dataclass(frozen=True, slots=True)
class FallbackDecision:
    """Single fallback decision step for the sparse selector."""

    action: str
    next_top_k: int
    next_keep_recent_blocks: int
    use_dense_fallback: bool = False


class GradedFallbackController:
    """Apply the required fallback order: widen K, add recent, then dense."""

    def __init__(self, policy: FallbackPolicy | None = None) -> None:
        self.policy = policy or FallbackPolicy()

    def decide(
        self,
        assessment: ConfidenceAssessment,
        *,
        current_top_k: int,
        current_keep_recent_blocks: int,
        base_top_k: int,
        base_keep_recent_blocks: int,
    ) -> FallbackDecision:
        """Choose the next fallback action from the current sparse state."""

        if current_top_k <= 0:
            raise ValueError("current_top_k must be > 0")
        if current_keep_recent_blocks < 0:
            raise ValueError("current_keep_recent_blocks must be >= 0")
        if base_top_k <= 0:
            raise ValueError("base_top_k must be > 0")
        if base_keep_recent_blocks < 0:
            raise ValueError("base_keep_recent_blocks must be >= 0")

        if assessment.is_confident:
            return FallbackDecision(
                action="keep_sparse",
                next_top_k=current_top_k,
                next_keep_recent_blocks=current_keep_recent_blocks,
            )

        widened_top_k = base_top_k + self.policy.widen_top_k_by
        if current_top_k < widened_top_k:
            return FallbackDecision(
                action="widen_k",
                next_top_k=widened_top_k,
                next_keep_recent_blocks=current_keep_recent_blocks,
            )

        expanded_recent = base_keep_recent_blocks + self.policy.add_recent_blocks_by
        if current_keep_recent_blocks < expanded_recent:
            return FallbackDecision(
                action="add_recent_blocks",
                next_top_k=current_top_k,
                next_keep_recent_blocks=expanded_recent,
            )

        if self.policy.allow_dense_fallback:
            return FallbackDecision(
                action="dense_fallback",
                next_top_k=current_top_k,
                next_keep_recent_blocks=current_keep_recent_blocks,
                use_dense_fallback=True,
            )

        return FallbackDecision(
            action="keep_sparse",
            next_top_k=current_top_k,
            next_keep_recent_blocks=current_keep_recent_blocks,
        )
