"""Confidence helpers for the V1 selector skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Sequence

from kvblock.selector.base import ScoredBlock
from kvblock.selector.policies import ConfidencePolicy


@dataclass(frozen=True, slots=True)
class ConfidenceAssessment:
    """Inspectable result of a sparse-selection confidence check."""

    is_confident: bool
    score_margin: float
    normalized_margin: float | None = None
    normalized_mass: float | None = None


class ConfidenceEvaluator:
    """Evaluate confidence from ranked selector scores."""

    def __init__(self, policy: ConfidencePolicy | None = None) -> None:
        self.policy = policy or ConfidencePolicy()

    def assess(
        self,
        ranked_candidates: Sequence[ScoredBlock],
        *,
        included_count: int,
    ) -> ConfidenceAssessment:
        """Assess confidence using score margin and optional normalized mass."""

        margin = score_margin(ranked_candidates, included_count=included_count)
        normalized_margin = normalized_score_margin(
            ranked_candidates, included_count=included_count
        )
        mass = (
            normalized_selected_mass(ranked_candidates, included_count=included_count)
            if self.policy.min_normalized_mass is not None
            else None
        )

        is_confident = margin >= self.policy.margin_threshold
        if self.policy.normalized_margin_threshold is not None:
            normalized_margin_ok = (
                margin == inf
                if normalized_margin is None
                else normalized_margin >= self.policy.normalized_margin_threshold
            )
            is_confident = is_confident and normalized_margin_ok
        if self.policy.min_normalized_mass is not None:
            is_confident = is_confident and (mass is not None) and (
                mass >= self.policy.min_normalized_mass
            )

        return ConfidenceAssessment(
            is_confident=is_confident,
            score_margin=margin,
            normalized_margin=normalized_margin,
            normalized_mass=mass,
        )


def score_margin(ranked_candidates: Sequence[ScoredBlock], *, included_count: int) -> float:
    """Compute the score gap between the last included and first excluded block."""

    if included_count < 0:
        raise ValueError("included_count must be >= 0")
    if included_count == 0:
        return 0.0
    if included_count >= len(ranked_candidates):
        return inf
    last_included = ranked_candidates[included_count - 1].final_score
    first_excluded = ranked_candidates[included_count].final_score
    return last_included - first_excluded


def normalized_score_margin(
    ranked_candidates: Sequence[ScoredBlock], *, included_count: int
) -> float | None:
    """Compute a scale-normalized boundary margin for calibration sweeps."""

    if included_count < 0:
        raise ValueError("included_count must be >= 0")
    if included_count <= 0 or included_count >= len(ranked_candidates):
        return None
    last_included = ranked_candidates[included_count - 1].final_score
    first_excluded = ranked_candidates[included_count].final_score
    denominator = max(abs(last_included), abs(first_excluded), 1e-6)
    return (last_included - first_excluded) / denominator


def normalized_selected_mass(
    ranked_candidates: Sequence[ScoredBlock], *, included_count: int
) -> float:
    """Compute selected positive score mass divided by total positive score mass."""

    if included_count < 0:
        raise ValueError("included_count must be >= 0")
    positive_scores = [max(candidate.final_score, 0.0) for candidate in ranked_candidates]
    total_mass = sum(positive_scores)
    if total_mass == 0.0:
        return 0.0
    selected_mass = sum(positive_scores[:included_count])
    return selected_mass / total_mass
