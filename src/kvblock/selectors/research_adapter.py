"""Compatibility adapters from research selector outputs to SelectedKVPlan."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan
from kvblock.policies import KVBlockPolicy

DEFAULT_CONFIDENCE = 1.0


@dataclass(frozen=True, slots=True)
class ExistingSelectorSelectedPlanAdapter:
    """Wrap existing research selector outputs in the package selector API.

    The adapter intentionally does not rerun or rewrite selector internals. It
    accepts an already-produced research result/row as ``query_state`` and
    returns the corresponding package-level ``SelectedKVPlan``.
    """

    name: str = "existing_research_selector"

    def select(
        self,
        query_state: Any,
        kv_metadata: Any,
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        """Convert an existing selector result/row into ``SelectedKVPlan``."""

        return selected_kv_plan_from_legacy_output(
            query_state,
            layout=layout,
            selector_name=self.name,
            policy_name=policy.name,
        )


class MixedGlobalRefineSelectedPlanAdapter(ExistingSelectorSelectedPlanAdapter):
    """Adapter name for the current LongBench mixed/global/refine path."""

    def __init__(self) -> None:
        super().__init__(name="mixed_global_refine_40_16_stride_8")


def selected_kv_plan_from_legacy_output(
    legacy_output: Any,
    *,
    layout: BlockLayout | None = None,
    selector_name: str | None = None,
    policy_name: str = "legacy",
) -> SelectedKVPlan:
    """Convert a known legacy selector output shape into ``SelectedKVPlan``."""

    if isinstance(legacy_output, Mapping):
        return _plan_from_mapping(
            legacy_output,
            layout=layout,
            selector_name=selector_name,
            policy_name=policy_name,
        )
    return _plan_from_object(
        legacy_output,
        layout=layout,
        selector_name=selector_name,
        policy_name=policy_name,
    )


def selected_kv_plan_from_longbench_selector_row(
    row: Any,
    *,
    selected_block_ids: Sequence[int] | None = None,
    selected_spans: Sequence[str] | None = None,
    selected_blocks: Sequence[Mapping[str, Any]] | None = None,
    policy_name: str = "longbench_output",
) -> SelectedKVPlan:
    """Build a plan from one LongBench selector row plus optional output filter."""

    row_selected_blocks = (
        tuple(getattr(row, "selected_blocks", ()) or ())
        if selected_blocks is None
        else tuple(dict(block) for block in selected_blocks)
    )
    payload: dict[str, Any] = {
        "logical_block_ids": (
            tuple(getattr(row, "selected_block_ids", ()) or ())
            if selected_block_ids is None
            else tuple(selected_block_ids)
        ),
        "selected_spans": (
            tuple(getattr(row, "selected_spans", ()) or ())
            if selected_spans is None
            else tuple(selected_spans)
        ),
        "semantic_selected_ids": tuple(
            getattr(row, "semantic_selected_block_ids", ()) or ()
        ),
        "selected_blocks": row_selected_blocks,
        "total_blocks": int(getattr(row, "candidate_block_count", 0) or 0),
        "total_tokens": int(getattr(row, "tokens", 0) or 0),
        "confidence": _confidence_from_row(row),
        "fallback_triggered": bool(getattr(row, "mixed_fallback_used", False)),
        "fallback_reason": (
            "mixed_global_refine_fallback"
            if bool(getattr(row, "mixed_fallback_used", False))
            else None
        ),
        "selector_name": _selector_name_from_row(row),
        "policy_name": policy_name,
        "metadata": {
            "legacy_type": type(row).__name__,
            "dataset_name": getattr(row, "dataset_name", None),
            "sample_id": getattr(row, "sample_id", None),
            "selector_latency_sec": getattr(row, "selector_latency_sec", None),
            "evidence_recall": getattr(row, "target_recall", None),
            "evidence_window_recall": getattr(row, "evidence_window_recall", None),
            "exact_recall": getattr(row, "target_recall", None),
            "selected_precision": getattr(row, "selected_precision", None),
            "block_mode": getattr(row, "block_mode", None),
            "qk_aggregation_strategy": getattr(row, "qk_aggregation_strategy", None),
            "rerank_mode": getattr(row, "rerank_mode", None),
            "refine_score_mode": getattr(row, "refine_score_mode", None),
            "stage_c_policy": getattr(row, "stage_c_policy", None),
            "scaffold_excluded_count": getattr(row, "scaffold_excluded_count", None),
        },
    }
    return _plan_from_mapping(payload, layout=None, selector_name=None, policy_name=policy_name)


def selected_kv_plan_report_from_longbench_output_row(row: Any) -> dict[str, Any]:
    """Return report fields sourced from a row's selected KV plan."""

    plan_payload = getattr(row, "selected_kv_plan", None) or {}
    plan = SelectedKVPlan.from_dict(plan_payload) if plan_payload else None
    return {
        "selected_token_fraction": (
            getattr(row, "selected_token_fraction", None)
            if plan is None
            else plan.selected_token_fraction
        ),
        "selected_block_fraction": (
            getattr(row, "selected_block_fraction", None)
            if plan is None
            else plan.selected_block_fraction
        ),
        "selector_latency_sec": getattr(row, "selector_latency_sec", None),
        "fallback_triggered": (
            getattr(row, "mixed_fallback_used", None)
            if plan is None
            else plan.fallback_triggered
        ),
        "evidence_recall": getattr(row, "selector_recall", None),
        "evidence_window_recall": getattr(row, "evidence_window_recall", None),
        "exact_recall": getattr(row, "exact_recall", None),
        "answer_quality_score": getattr(row, "answer_quality_score", None),
        "dense_sparse_quality_delta": getattr(row, "dense_sparse_quality_delta", None),
    }


def _plan_from_object(
    legacy_output: Any,
    *,
    layout: BlockLayout | None,
    selector_name: str | None,
    policy_name: str,
) -> SelectedKVPlan:
    selected_blocks = tuple(getattr(legacy_output, "selected_blocks", ()) or ())
    selected_block_inspections = tuple(
        getattr(legacy_output, "selected_block_inspections", ()) or ()
    )
    block_inspections = tuple(getattr(legacy_output, "block_inspections", ()) or ())
    selected_ids = _first_non_empty_int_sequence(
        getattr(legacy_output, "selected_block_ids", None),
        getattr(legacy_output, "selected_ids", None),
        getattr(legacy_output, "logical_block_ids", None),
    )
    selected_spans = _first_non_empty_span_sequence(
        getattr(legacy_output, "selected_spans", None),
        _spans_from_blocks(selected_block_inspections),
        _spans_from_selected_blocks(block_inspections),
    )
    trace = getattr(legacy_output, "trace", None)
    final_selection = getattr(trace, "final_selection", None)
    metadata = {
        "legacy_type": type(legacy_output).__name__,
        "fallback_mode": getattr(legacy_output, "fallback_mode", None),
        "selected_to_semantic_k_ratio": getattr(
            legacy_output,
            "selected_to_semantic_k_ratio",
            None,
        ),
        "run_summary": _maybe_to_dict(getattr(legacy_output, "run_summary", None)),
        "latency": _maybe_to_dict(getattr(legacy_output, "latency", None)),
        "confidence": _maybe_to_dict(getattr(legacy_output, "confidence", None)),
        "trace": _maybe_to_dict(trace),
    }
    return _build_plan(
        logical_block_ids=selected_ids,
        selected_token_ranges=_parse_spans(selected_spans),
        selected_blocks=selected_blocks,
        layout=layout,
        recent_block_ids=_int_tuple(getattr(final_selection, "recent_block_ids", ())),
        anchor_block_ids=_int_tuple(getattr(final_selection, "anchor_block_ids", ())),
        halo_block_ids=(),
        linked_block_ids=_linked_block_ids_from_records(selected_block_inspections),
        confidence=_confidence_from_result(legacy_output),
        fallback_triggered=_fallback_triggered(legacy_output),
        fallback_reason=_fallback_reason(legacy_output),
        selector_name=selector_name or _selector_name_from_result(legacy_output),
        policy_name=policy_name,
        total_blocks=_total_blocks_from_result(legacy_output),
        total_tokens=_total_tokens_from_result(legacy_output),
        metadata=metadata,
    )


def _plan_from_mapping(
    payload: Mapping[str, Any],
    *,
    layout: BlockLayout | None,
    selector_name: str | None,
    policy_name: str,
) -> SelectedKVPlan:
    selected_blocks = tuple(payload.get("selected_blocks", ()) or ())
    semantic_ids = _int_tuple(payload.get("semantic_selected_ids", ()))
    logical_ids = _first_non_empty_int_sequence(
        payload.get("logical_block_ids"),
        payload.get("selected_block_ids"),
        payload.get("selected_ids"),
        payload.get("block_ids"),
    )
    ranges = _parse_spans(
        _first_non_empty_span_sequence(
            payload.get("selected_token_ranges"),
            payload.get("selected_spans"),
            _spans_from_blocks(selected_blocks),
        )
    )
    fallback_triggered = bool(payload.get("fallback_triggered", False))
    fallback_reason = payload.get("fallback_reason")
    metadata = dict(payload.get("metadata", {}))
    if "confidence" not in payload:
        metadata["confidence_default"] = "missing_confidence_assumed_1.0"
    halo_ids = _int_tuple(payload.get("halo_block_ids", ()))
    if not halo_ids and int(payload.get("halo_radius", 0) or 0) > 0 and semantic_ids:
        semantic = set(semantic_ids)
        halo_ids = tuple(block_id for block_id in logical_ids if block_id not in semantic)
    return _build_plan(
        logical_block_ids=logical_ids,
        selected_token_ranges=ranges,
        selected_blocks=selected_blocks,
        layout=layout,
        recent_block_ids=_int_tuple(payload.get("recent_block_ids", ())),
        anchor_block_ids=_int_tuple(payload.get("anchor_block_ids", ())),
        halo_block_ids=halo_ids,
        linked_block_ids=_int_tuple(payload.get("linked_block_ids", ()))
        or _linked_block_ids_from_records(selected_blocks),
        confidence=_coerce_confidence(payload.get("confidence", DEFAULT_CONFIDENCE)),
        fallback_triggered=fallback_triggered,
        fallback_reason=None if fallback_reason is None else str(fallback_reason),
        selector_name=selector_name or str(payload.get("selector_name", "legacy")),
        policy_name=str(payload.get("policy_name", policy_name)),
        total_blocks=int(payload.get("total_blocks", 0) or 0),
        total_tokens=int(payload.get("total_tokens", 0) or 0),
        metadata=metadata,
    )


def _build_plan(
    *,
    logical_block_ids: Sequence[int],
    selected_token_ranges: Sequence[tuple[int, int]],
    selected_blocks: Sequence[Any],
    layout: BlockLayout | None,
    recent_block_ids: Sequence[int],
    anchor_block_ids: Sequence[int],
    halo_block_ids: Sequence[int],
    linked_block_ids: Sequence[int],
    confidence: float,
    fallback_triggered: bool,
    fallback_reason: str | None,
    selector_name: str,
    policy_name: str,
    total_blocks: int,
    total_tokens: int,
    metadata: Mapping[str, Any],
) -> SelectedKVPlan:
    ranges = list(selected_token_ranges)
    ids = list(dict.fromkeys(int(block_id) for block_id in logical_block_ids))
    if not ids and ranges and layout is not None:
        ids = list(
            dict.fromkeys(
                block_id
                for start, end in ranges
                for block_id in layout.token_range_to_block_ids(start, end)
            )
        )
    if not ranges and ids and layout is not None:
        ranges = layout.token_ranges_for_blocks(ids)

    inferred_total_tokens = total_tokens
    if layout is not None:
        inferred_total_tokens = max(inferred_total_tokens, layout.total_tokens)
    if not inferred_total_tokens:
        inferred_total_tokens = _max_range_end(ranges)
    selected_tokens = _merged_token_count(ranges)
    inferred_total_tokens = max(inferred_total_tokens, selected_tokens)

    inferred_total_blocks = total_blocks
    if layout is not None:
        inferred_total_blocks = max(inferred_total_blocks, layout.block_count)
    if not inferred_total_blocks:
        inferred_total_blocks = _max_block_id(ids, selected_blocks) + 1
    inferred_total_blocks = max(inferred_total_blocks, len(ids))

    resolved_fallback_reason = fallback_reason
    if fallback_triggered and not resolved_fallback_reason:
        resolved_fallback_reason = "legacy_fallback"

    return SelectedKVPlan(
        logical_block_ids=ids,
        selected_token_ranges=ranges,
        recent_block_ids=list(recent_block_ids),
        anchor_block_ids=list(anchor_block_ids),
        halo_block_ids=list(halo_block_ids),
        linked_block_ids=list(dict.fromkeys(int(block_id) for block_id in linked_block_ids)),
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        fallback_reason=resolved_fallback_reason,
        selector_name=selector_name,
        policy_name=policy_name,
        total_blocks=inferred_total_blocks,
        selected_blocks=len(ids),
        total_tokens=inferred_total_tokens,
        selected_tokens=selected_tokens,
        metadata=dict(metadata),
    )


def _first_non_empty_int_sequence(*values: Any) -> tuple[int, ...]:
    for value in values:
        if value is None:
            continue
        items = tuple(int(item) for item in value)
        if items:
            return tuple(dict.fromkeys(items))
    return ()


def _first_non_empty_span_sequence(*values: Any) -> tuple[Any, ...]:
    for value in values:
        if value is None:
            continue
        items = tuple(value)
        if items:
            return items
    return ()


def _parse_spans(spans: Sequence[Any]) -> tuple[tuple[int, int], ...]:
    parsed: list[tuple[int, int]] = []
    for span in spans:
        if isinstance(span, str):
            start_text, end_text = span.split(":", maxsplit=1)
            parsed.append((int(start_text), int(end_text)))
            continue
        if len(span) != 2:
            raise ValueError("selected spans must be 'start:end' or two-item ranges")
        start, end = span
        parsed.append((int(start), int(end)))
    return tuple(parsed)


def _spans_from_blocks(blocks: Sequence[Any]) -> tuple[str, ...]:
    spans: list[str] = []
    for block in blocks:
        start = _get_value(block, "token_start")
        end = _get_value(block, "token_end")
        if start is None or end is None:
            continue
        spans.append(f"{int(start)}:{int(end)}")
    return tuple(spans)


def _spans_from_selected_blocks(blocks: Sequence[Any]) -> tuple[str, ...]:
    return _spans_from_blocks(
        tuple(block for block in blocks if bool(_get_value(block, "selected", False)))
    )


def _linked_block_ids_from_records(records: Sequence[Any]) -> tuple[int, ...]:
    linked: list[int] = []
    for record in records:
        parent = _get_value(record, "parent_block_id")
        if parent is not None:
            linked.append(int(parent))
    return tuple(dict.fromkeys(linked))


def _confidence_from_result(result: Any) -> float:
    confidence = getattr(result, "confidence", None)
    if confidence is None:
        return DEFAULT_CONFIDENCE
    if isinstance(confidence, (int, float)):
        return _coerce_confidence(confidence)
    return _coerce_confidence(
        getattr(confidence, "normalized_margin", None)
        if getattr(confidence, "normalized_margin", None) is not None
        else getattr(confidence, "raw_margin", DEFAULT_CONFIDENCE)
    )


def _confidence_from_row(row: Any) -> float:
    for attr in ("confidence", "normalized_margin", "raw_margin"):
        value = getattr(row, attr, None)
        if value is not None:
            return _coerce_confidence(value)
    return DEFAULT_CONFIDENCE


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    if not isfinite(confidence):
        return DEFAULT_CONFIDENCE
    return max(0.0, min(1.0, confidence))


def _fallback_triggered(result: Any) -> bool:
    explicit = getattr(result, "fallback_triggered", None)
    if explicit is not None:
        return bool(explicit)
    if bool(getattr(result, "mixed_fallback_used", False)):
        return True
    mode = getattr(result, "fallback_mode", None)
    if mode is None:
        return False
    return str(mode) not in {"", "none", "sparse"}


def _fallback_reason(result: Any) -> str | None:
    explicit = getattr(result, "fallback_reason", None)
    if explicit not in (None, ""):
        return str(explicit)
    if bool(getattr(result, "mixed_fallback_used", False)):
        return "mixed_global_refine_fallback"
    mode = getattr(result, "fallback_mode", None)
    if mode is None or str(mode) in {"", "none", "sparse"}:
        return None
    return str(mode)


def _selector_name_from_result(result: Any) -> str:
    summary = getattr(result, "run_summary", None)
    block_mode = getattr(summary, "block_mode", None)
    return "legacy_selector" if block_mode in (None, "") else str(block_mode)


def _selector_name_from_row(row: Any) -> str:
    block_mode = getattr(row, "block_mode", None)
    if block_mode:
        return str(block_mode)
    return "longbench_selector"


def _total_blocks_from_result(result: Any) -> int:
    summary = getattr(result, "run_summary", None)
    if summary is not None and getattr(summary, "block_count", None) is not None:
        return int(summary.block_count)
    return len(tuple(getattr(result, "block_inspections", ()) or ()))


def _total_tokens_from_result(result: Any) -> int:
    summary = getattr(result, "run_summary", None)
    if summary is not None and getattr(summary, "token_count", None) is not None:
        return int(summary.token_count)
    return 0


def _int_tuple(values: Any) -> tuple[int, ...]:
    if values is None:
        return ()
    return tuple(dict.fromkeys(int(value) for value in values))


def _max_range_end(ranges: Sequence[tuple[int, int]]) -> int:
    return max((end for _, end in ranges), default=0)


def _max_block_id(ids: Sequence[int], records: Sequence[Any]) -> int:
    values = [int(block_id) for block_id in ids]
    for record in records:
        block_id = _get_value(record, "block_id")
        if block_id is not None:
            values.append(int(block_id))
    return max(values, default=-1)


def _merged_token_count(ranges: Sequence[tuple[int, int]]) -> int:
    intervals = sorted((max(0, start), max(start, end)) for start, end in ranges)
    if not intervals:
        return 0
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue
        merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(end - start for start, end in merged)


def _get_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _maybe_to_dict(value: Any) -> Any:
    if value is None:
        return None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    return None
