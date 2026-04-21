"""Small policy dataclasses for the V1 selector skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kvblock.config.models import SelectorConfig

HeadScoringMode = Literal[
    "mean_heads",
    "max_head_score",
    "topk_head_mean",
    "weighted_head_mean",
]


@dataclass(frozen=True, slots=True)
class StageAWeights:
    """Feature weights for Stage A coarse scoring."""

    summary_similarity: float = 0.6
    recency: float = 0.2
    attn_ema: float = 0.15
    priority: float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.summary_similarity + self.recency + self.attn_ema + self.priority
        )
        if total <= 0:
            raise ValueError("At least one Stage A weight must be positive")
        if min(
            self.summary_similarity, self.recency, self.attn_ema, self.priority
        ) < 0:
            raise ValueError("Stage A weights must be >= 0")


@dataclass(frozen=True, slots=True)
class StageAPolicy:
    """Policy for shortlist sizing and coarse scoring."""

    weights: StageAWeights = field(default_factory=StageAWeights)
    shortlist_size: int = 24
    long_context_shortlist_size: int = 48
    long_context_threshold: int = 32768
    head_scoring_mode: HeadScoringMode = "mean_heads"
    head_top_k: int = 2
    head_weights: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.shortlist_size <= 0:
            raise ValueError("shortlist_size must be > 0")
        if self.long_context_shortlist_size <= 0:
            raise ValueError("long_context_shortlist_size must be > 0")
        if self.long_context_threshold <= 0:
            raise ValueError("long_context_threshold must be > 0")
        if self.head_scoring_mode not in {
            "mean_heads",
            "max_head_score",
            "topk_head_mean",
            "weighted_head_mean",
        }:
            raise ValueError(f"unsupported head_scoring_mode: {self.head_scoring_mode!r}")
        if self.head_top_k <= 0:
            raise ValueError("head_top_k must be > 0")
        if any(weight < 0 for weight in self.head_weights):
            raise ValueError("head_weights must be >= 0")
        if self.head_weights and sum(self.head_weights) <= 0:
            raise ValueError("head_weights must contain at least one positive value")

    def shortlist_for_context(self, context_tokens: int | None) -> int:
        """Choose the Stage A shortlist size for the active context length."""

        if context_tokens is not None and context_tokens > self.long_context_threshold:
            return self.long_context_shortlist_size
        return self.shortlist_size

    @classmethod
    def from_selector_config(
        cls, config: SelectorConfig, *, weights: StageAWeights | None = None
    ) -> "StageAPolicy":
        return cls(
            weights=weights or StageAWeights(),
            shortlist_size=config.stage_a_shortlist,
            long_context_shortlist_size=config.stage_a_long_context_shortlist,
            long_context_threshold=config.long_context_threshold,
        )


@dataclass(frozen=True, slots=True)
class StageBPolicy:
    """Policy for Stage B Hamming refinement."""

    hamming_weight: float = 0.2
    base_score_weight: float = 1.0
    sketch_bits: int = 64

    def __post_init__(self) -> None:
        if self.hamming_weight < 0:
            raise ValueError("hamming_weight must be >= 0")
        if self.base_score_weight < 0:
            raise ValueError("base_score_weight must be >= 0")
        if self.base_score_weight + self.hamming_weight <= 0:
            raise ValueError("Stage B must have at least one positive weight")
        if self.sketch_bits <= 0:
            raise ValueError("sketch_bits must be > 0")

    @classmethod
    def from_selector_config(cls, config: SelectorConfig) -> "StageBPolicy":
        return cls(sketch_bits=config.sign_sketch_bits)


@dataclass(frozen=True, slots=True)
class StageCPolicy:
    """Policy for rails and semantic block count in Stage C."""

    keep_recent_blocks: int = 4
    keep_anchor_blocks: int = 2
    semantic_top_k: int = 8
    semantic_top_k_long_context: int = 16
    long_context_threshold: int = 32768

    def __post_init__(self) -> None:
        if self.keep_recent_blocks < 0:
            raise ValueError("keep_recent_blocks must be >= 0")
        if self.keep_anchor_blocks < 0:
            raise ValueError("keep_anchor_blocks must be >= 0")
        if self.semantic_top_k <= 0:
            raise ValueError("semantic_top_k must be > 0")
        if self.semantic_top_k_long_context <= 0:
            raise ValueError("semantic_top_k_long_context must be > 0")
        if self.long_context_threshold <= 0:
            raise ValueError("long_context_threshold must be > 0")

    def semantic_top_k_for_context(self, context_tokens: int | None) -> int:
        """Choose the semantic K budget for the active context length."""

        if context_tokens is not None and context_tokens > self.long_context_threshold:
            return self.semantic_top_k_long_context
        return self.semantic_top_k

    @classmethod
    def from_selector_config(cls, config: SelectorConfig) -> "StageCPolicy":
        return cls(
            keep_recent_blocks=config.keep_recent_blocks,
            keep_anchor_blocks=config.keep_anchor_blocks,
            semantic_top_k=config.final_top_k,
            semantic_top_k_long_context=config.final_top_k_long_context,
            long_context_threshold=config.long_context_threshold,
        )


@dataclass(frozen=True, slots=True)
class ConfidencePolicy:
    """Thresholds for sparse-selection confidence checks."""

    margin_threshold: float = 0.05
    normalized_margin_threshold: float | None = None
    min_normalized_mass: float | None = None

    def __post_init__(self) -> None:
        if self.margin_threshold < 0:
            raise ValueError("margin_threshold must be >= 0")
        if (
            self.normalized_margin_threshold is not None
            and self.normalized_margin_threshold < 0
        ):
            raise ValueError("normalized_margin_threshold must be >= 0")
        if self.min_normalized_mass is not None and not (
            0.0 <= self.min_normalized_mass <= 1.0
        ):
            raise ValueError("min_normalized_mass must be in [0, 1]")

    @classmethod
    def from_selector_config(cls, config: SelectorConfig) -> "ConfidencePolicy":
        return cls(
            margin_threshold=config.confidence_margin,
            normalized_margin_threshold=config.confidence_normalized_margin,
        )


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Policy controlling graded fallback escalation."""

    widen_top_k_by: int = 4
    add_recent_blocks_by: int = 2
    allow_dense_fallback: bool = True

    def __post_init__(self) -> None:
        if self.widen_top_k_by <= 0:
            raise ValueError("widen_top_k_by must be > 0")
        if self.add_recent_blocks_by <= 0:
            raise ValueError("add_recent_blocks_by must be > 0")

    @classmethod
    def from_selector_config(cls, config: SelectorConfig) -> "FallbackPolicy":
        return cls(allow_dense_fallback=config.allow_dense_fallback)
