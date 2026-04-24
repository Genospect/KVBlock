"""Runtime-independent selector decision trace records."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from math import inf, isfinite, isnan
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BlockScoreTrace:
    """Serializable per-block score record for pipeline traces."""

    block_id: int
    token_start: int
    token_len: int
    approx_similarity_score: float
    recency_score: float
    attn_score: float
    priority_score: float
    stage_a_score: float
    hamming_similarity: float
    stage_b_score: float
    final_score: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly score record."""

        return {
            "block_id": self.block_id,
            "token_start": self.token_start,
            "token_len": self.token_len,
            "approx_similarity_score": _json_float(self.approx_similarity_score),
            "recency_score": _json_float(self.recency_score),
            "attn_score": _json_float(self.attn_score),
            "priority_score": _json_float(self.priority_score),
            "stage_a_score": _json_float(self.stage_a_score),
            "hamming_similarity": _json_float(self.hamming_similarity),
            "stage_b_score": _json_float(self.stage_b_score),
            "final_score": _json_float(self.final_score),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlockScoreTrace":
        """Rebuild a score record from :meth:`to_dict` output."""

        return cls(
            block_id=int(data["block_id"]),
            token_start=int(data["token_start"]),
            token_len=int(data["token_len"]),
            approx_similarity_score=_float_value(data["approx_similarity_score"]),
            recency_score=_float_value(data["recency_score"]),
            attn_score=_float_value(data["attn_score"]),
            priority_score=_float_value(data["priority_score"]),
            stage_a_score=_float_value(data["stage_a_score"]),
            hamming_similarity=_float_value(data["hamming_similarity"]),
            stage_b_score=_float_value(data["stage_b_score"]),
            final_score=_float_value(data["final_score"]),
        )


@dataclass(frozen=True, slots=True)
class SelectionSplitTrace:
    """Trace of rail-preserved vs semantic selections.

    The first fields mirror Stage C's de-duplicated output. The overlap fields make
    rail collisions visible without changing the selected block order.
    """

    recent_block_ids: tuple[int, ...]
    anchor_block_ids: tuple[int, ...]
    semantic_block_ids: tuple[int, ...]
    final_selected_block_ids: tuple[int, ...]
    rail_block_ids: tuple[int, ...]
    rail_block_count: int
    semantic_block_count: int
    requested_anchor_block_ids: tuple[int, ...] = field(default_factory=tuple)
    missing_anchor_block_ids: tuple[int, ...] = field(default_factory=tuple)
    deduped_anchor_block_ids: tuple[int, ...] = field(default_factory=tuple)
    deduped_semantic_block_ids: tuple[int, ...] = field(default_factory=tuple)
    deduped_block_ids: tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly Stage C split record."""

        return {
            "recent_block_ids": list(self.recent_block_ids),
            "anchor_block_ids": list(self.anchor_block_ids),
            "semantic_block_ids": list(self.semantic_block_ids),
            "final_selected_block_ids": list(self.final_selected_block_ids),
            "rail_block_ids": list(self.rail_block_ids),
            "rail_block_count": self.rail_block_count,
            "semantic_block_count": self.semantic_block_count,
            "requested_anchor_block_ids": list(self.requested_anchor_block_ids),
            "missing_anchor_block_ids": list(self.missing_anchor_block_ids),
            "deduped_anchor_block_ids": list(self.deduped_anchor_block_ids),
            "deduped_semantic_block_ids": list(self.deduped_semantic_block_ids),
            "deduped_block_ids": list(self.deduped_block_ids),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectionSplitTrace":
        """Rebuild a Stage C split record from :meth:`to_dict` output."""

        return cls(
            recent_block_ids=_int_tuple(data, "recent_block_ids"),
            anchor_block_ids=_int_tuple(data, "anchor_block_ids"),
            semantic_block_ids=_int_tuple(data, "semantic_block_ids"),
            final_selected_block_ids=_int_tuple(data, "final_selected_block_ids"),
            rail_block_ids=_int_tuple(data, "rail_block_ids"),
            rail_block_count=int(data["rail_block_count"]),
            semantic_block_count=int(data["semantic_block_count"]),
            requested_anchor_block_ids=_int_tuple(data, "requested_anchor_block_ids"),
            missing_anchor_block_ids=_int_tuple(data, "missing_anchor_block_ids"),
            deduped_anchor_block_ids=_int_tuple(data, "deduped_anchor_block_ids"),
            deduped_semantic_block_ids=_int_tuple(data, "deduped_semantic_block_ids"),
            deduped_block_ids=_int_tuple(data, "deduped_block_ids"),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceTrace:
    """Traceable confidence metrics for one selector invocation."""

    semantic_candidate_block_ids: tuple[int, ...]
    semantic_included_count: int
    raw_margin: float
    normalized_margin: float | None
    selected_mass: float | None
    normalized_mass: float | None
    is_confident: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly confidence record."""

        return {
            "semantic_candidate_block_ids": list(self.semantic_candidate_block_ids),
            "semantic_included_count": self.semantic_included_count,
            "raw_margin": _json_float(self.raw_margin),
            "normalized_margin": _json_optional_float(self.normalized_margin),
            "selected_mass": _json_optional_float(self.selected_mass),
            "normalized_mass": _json_optional_float(self.normalized_mass),
            "is_confident": self.is_confident,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConfidenceTrace":
        """Rebuild a confidence record from :meth:`to_dict` output."""

        return cls(
            semantic_candidate_block_ids=_int_tuple(
                data, "semantic_candidate_block_ids"
            ),
            semantic_included_count=int(data["semantic_included_count"]),
            raw_margin=_float_value(data["raw_margin"]),
            normalized_margin=_optional_float_value(data.get("normalized_margin")),
            selected_mass=_optional_float_value(data.get("selected_mass")),
            normalized_mass=_optional_float_value(data.get("normalized_mass")),
            is_confident=bool(data["is_confident"]),
        )


@dataclass(frozen=True, slots=True)
class FallbackTrace:
    """Trace-friendly fallback decision record."""

    action: str
    mode: str
    next_semantic_top_k: int
    next_keep_recent_blocks: int
    use_dense_fallback: bool
    reason_code: str
    reason_codes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly fallback record."""

        reason_codes = self.reason_codes or (self.reason_code,)
        return {
            "action": self.action,
            "mode": self.mode,
            "next_semantic_top_k": self.next_semantic_top_k,
            "next_keep_recent_blocks": self.next_keep_recent_blocks,
            "use_dense_fallback": self.use_dense_fallback,
            "reason_code": self.reason_code,
            "reason_codes": list(reason_codes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FallbackTrace":
        """Rebuild a fallback record from :meth:`to_dict` output."""

        reason_code = str(data["reason_code"])
        reason_codes = tuple(str(value) for value in data.get("reason_codes", ()))
        return cls(
            action=str(data["action"]),
            mode=str(data["mode"]),
            next_semantic_top_k=int(data["next_semantic_top_k"]),
            next_keep_recent_blocks=int(data["next_keep_recent_blocks"]),
            use_dense_fallback=bool(data["use_dense_fallback"]),
            reason_code=reason_code,
            reason_codes=reason_codes or (reason_code,),
        )


@dataclass(frozen=True, slots=True)
class SelectorDecisionTrace:
    """Complete end-to-end decision trace for one query/step."""

    step_id: str | int | None
    query_sign_sketch: int
    stage_a_scores: tuple[BlockScoreTrace, ...]
    stage_a_shortlist_block_ids: tuple[int, ...]
    stage_b_scores: tuple[BlockScoreTrace, ...]
    pre_fallback_selection: SelectionSplitTrace
    final_selection: SelectionSplitTrace
    confidence: ConfidenceTrace
    fallback: FallbackTrace
    candidate_block_ids: tuple[int, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly selector trace payload."""

        candidate_ids = self.candidate_block_ids or tuple(
            score.block_id for score in self.stage_a_scores
        )
        return {
            "step_id": self.step_id,
            "query_sign_sketch": self.query_sign_sketch,
            "candidate_block_ids": list(candidate_ids),
            "candidate_count": len(candidate_ids),
            "stage_a_scores": [score.to_dict() for score in self.stage_a_scores],
            "stage_a_shortlist_block_ids": list(self.stage_a_shortlist_block_ids),
            "stage_b_scores": [score.to_dict() for score in self.stage_b_scores],
            "pre_fallback_selection": self.pre_fallback_selection.to_dict(),
            "final_selection": self.final_selection.to_dict(),
            "confidence": self.confidence.to_dict(),
            "fallback": self.fallback.to_dict(),
        }

    def to_jsonl_line(self) -> str:
        """Return one deterministic JSON object suitable for JSONL output."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectorDecisionTrace":
        """Rebuild a selector trace from :meth:`to_dict` output."""

        return cls(
            step_id=data.get("step_id"),
            query_sign_sketch=int(data["query_sign_sketch"]),
            candidate_block_ids=_int_tuple(data, "candidate_block_ids"),
            stage_a_scores=tuple(
                BlockScoreTrace.from_dict(item)
                for item in data.get("stage_a_scores", ())
            ),
            stage_a_shortlist_block_ids=_int_tuple(
                data, "stage_a_shortlist_block_ids"
            ),
            stage_b_scores=tuple(
                BlockScoreTrace.from_dict(item)
                for item in data.get("stage_b_scores", ())
            ),
            pre_fallback_selection=SelectionSplitTrace.from_dict(
                data["pre_fallback_selection"]
            ),
            final_selection=SelectionSplitTrace.from_dict(data["final_selection"]),
            confidence=ConfidenceTrace.from_dict(data["confidence"]),
            fallback=FallbackTrace.from_dict(data["fallback"]),
        )


def _int_tuple(data: Mapping[str, Any], key: str) -> tuple[int, ...]:
    return tuple(int(value) for value in data.get(key, ()))


def _json_optional_float(value: float | None) -> float | str | None:
    if value is None:
        return None
    return _json_float(value)


def _json_float(value: float) -> float | str:
    value = float(value)
    if isfinite(value):
        return value
    if isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def _optional_float_value(value: Any) -> float | None:
    if value is None:
        return None
    return _float_value(value)


def _float_value(value: Any) -> float:
    if value == "Infinity":
        return inf
    if value == "-Infinity":
        return -inf
    if value == "NaN":
        return float("nan")
    return float(value)
