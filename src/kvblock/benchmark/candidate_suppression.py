"""Benchmark-only interval suppression for mixed block candidate sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Sequence, cast

SuppressionMode = Literal[
    "none",
    "overlap_threshold",
    "interval_iou",
    "keep_highest_score_per_overlap_cluster",
]

VALID_SUPPRESSION_MODES: tuple[SuppressionMode, ...] = (
    "none",
    "overlap_threshold",
    "interval_iou",
    "keep_highest_score_per_overlap_cluster",
)


@dataclass(frozen=True, slots=True)
class RankedCandidateSpan:
    """One ranked candidate span from selector trace output."""

    block_id: int
    candidate_id: str
    token_start: int
    token_end: int
    score: float
    rank: int
    block_size: int | None = None
    block_mode: str | None = None

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise ValueError("block_id must be >= 0")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.token_start < 0:
            raise ValueError("token_start must be >= 0")
        if self.token_end <= self.token_start:
            raise ValueError("token_end must be > token_start")
        if self.rank <= 0:
            raise ValueError("rank must be > 0")
        if self.block_size is not None and self.block_size <= 0:
            raise ValueError("block_size must be > 0 when set")

    @property
    def token_len(self) -> int:
        """Return the candidate span length."""

        return self.token_end - self.token_start

    def to_dict(self) -> dict[str, int | float | str | None]:
        """Return a JSON-friendly ranked candidate."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SuppressionDecision:
    """Suppression decision for one ranked candidate."""

    block_id: int
    candidate_id: str
    survived: bool
    suppressed_by_block_id: int | None = None
    suppressed_by_candidate_id: str | None = None
    reason: str | None = None
    overlap_tokens: int = 0
    overlap_fraction: float = 0.0
    interval_iou: float = 0.0

    def to_dict(self) -> dict[str, int | float | str | bool | None]:
        """Return a JSON-friendly decision record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class SuppressionResult:
    """Result of benchmark-only interval suppression."""

    mode: SuppressionMode
    threshold: float
    input_count: int
    output_count: int
    survivor_block_ids: tuple[int, ...]
    decisions: tuple[SuppressionDecision, ...]

    @property
    def decision_by_block_id(self) -> dict[int, SuppressionDecision]:
        """Return decisions keyed by block id."""

        return {decision.block_id: decision for decision in self.decisions}

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly suppression result."""

        return {
            "mode": self.mode,
            "threshold": self.threshold,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "survivor_block_ids": list(self.survivor_block_ids),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }


def suppression_mode_from_name(name: str) -> SuppressionMode:
    """Validate and return one suppression mode name."""

    normalized = name.strip()
    if normalized not in VALID_SUPPRESSION_MODES:
        valid = ", ".join(VALID_SUPPRESSION_MODES)
        raise ValueError(f"unknown suppression mode {name!r}; valid: {valid}")
    return cast(SuppressionMode, normalized)


def suppression_modes_from_names(names: Sequence[str]) -> tuple[SuppressionMode, ...]:
    """Resolve a non-empty sequence of suppression mode names."""

    if not names:
        raise ValueError("suppression mode names must not be empty")
    modes = tuple(suppression_mode_from_name(name) for name in names if name.strip())
    if not modes:
        raise ValueError("suppression mode names must not be empty")
    return modes


def suppress_ranked_candidates(
    ranked_candidates: Sequence[RankedCandidateSpan],
    *,
    mode: SuppressionMode = "none",
    threshold: float = 0.75,
) -> SuppressionResult:
    """Suppress overlapping candidate spans with a simple greedy NMS pass."""

    resolved = suppression_mode_from_name(mode)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be in [0, 1]")
    ranked = tuple(sorted(ranked_candidates, key=lambda item: (item.rank, -item.score, item.block_id)))
    if resolved == "none":
        return SuppressionResult(
            mode=resolved,
            threshold=threshold,
            input_count=len(ranked),
            output_count=len(ranked),
            survivor_block_ids=tuple(candidate.block_id for candidate in ranked),
            decisions=tuple(
                SuppressionDecision(
                    block_id=candidate.block_id,
                    candidate_id=candidate.candidate_id,
                    survived=True,
                )
                for candidate in ranked
            ),
        )

    survivors: list[RankedCandidateSpan] = []
    decisions: list[SuppressionDecision] = []
    decision_by_block_id: dict[int, SuppressionDecision] = {}
    for candidate in ranked:
        suppressor, stats = _find_suppressor(
            candidate,
            survivors,
            mode=resolved,
            threshold=threshold,
        )
        if suppressor is None:
            survivors.append(candidate)
            decision = SuppressionDecision(
                block_id=candidate.block_id,
                candidate_id=candidate.candidate_id,
                survived=True,
            )
        else:
            decision = SuppressionDecision(
                block_id=candidate.block_id,
                candidate_id=candidate.candidate_id,
                survived=False,
                suppressed_by_block_id=suppressor.block_id,
                suppressed_by_candidate_id=suppressor.candidate_id,
                reason=resolved,
                overlap_tokens=stats[0],
                overlap_fraction=stats[1],
                interval_iou=stats[2],
            )
        decision_by_block_id[candidate.block_id] = decision

    decisions = [
        decision_by_block_id[candidate.block_id]
        for candidate in ranked
    ]
    return SuppressionResult(
        mode=resolved,
        threshold=threshold,
        input_count=len(ranked),
        output_count=len(survivors),
        survivor_block_ids=tuple(candidate.block_id for candidate in survivors),
        decisions=tuple(decisions),
    )


def span_overlap_tokens(left: RankedCandidateSpan, right: RankedCandidateSpan) -> int:
    """Return integer token overlap between two half-open spans."""

    return max(0, min(left.token_end, right.token_end) - max(left.token_start, right.token_start))


def span_overlap_fraction(left: RankedCandidateSpan, right: RankedCandidateSpan) -> float:
    """Return overlap divided by the shorter span length."""

    overlap = span_overlap_tokens(left, right)
    if overlap <= 0:
        return 0.0
    return overlap / float(min(left.token_len, right.token_len))


def span_interval_iou(left: RankedCandidateSpan, right: RankedCandidateSpan) -> float:
    """Return interval IoU over token spans."""

    overlap = span_overlap_tokens(left, right)
    if overlap <= 0:
        return 0.0
    union = max(left.token_end, right.token_end) - min(left.token_start, right.token_start)
    return overlap / float(union)


def _find_suppressor(
    candidate: RankedCandidateSpan,
    survivors: Sequence[RankedCandidateSpan],
    *,
    mode: SuppressionMode,
    threshold: float,
) -> tuple[RankedCandidateSpan | None, tuple[int, float, float]]:
    best: tuple[RankedCandidateSpan | None, tuple[int, float, float]] = (
        None,
        (0, 0.0, 0.0),
    )
    for survivor in survivors:
        overlap = span_overlap_tokens(candidate, survivor)
        overlap_fraction = span_overlap_fraction(candidate, survivor)
        iou = span_interval_iou(candidate, survivor)
        if _should_suppress(mode, overlap_fraction=overlap_fraction, iou=iou, threshold=threshold):
            stats = (overlap, overlap_fraction, iou)
            if best[0] is None or stats[1:] > best[1][1:]:
                best = (survivor, stats)
    return best


def _should_suppress(
    mode: SuppressionMode,
    *,
    overlap_fraction: float,
    iou: float,
    threshold: float,
) -> bool:
    if mode == "overlap_threshold":
        return overlap_fraction >= threshold
    if mode == "interval_iou":
        return iou >= threshold
    if mode == "keep_highest_score_per_overlap_cluster":
        return overlap_fraction >= threshold or iou >= threshold
    if mode == "none":
        return False
    raise ValueError(f"unsupported suppression mode: {mode!r}")
