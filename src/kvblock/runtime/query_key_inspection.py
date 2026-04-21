"""Query/key qualitative inspection records for real-block selector runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import re
from typing import Any, Sequence

from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.pipeline import SelectorDecisionTrace
from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding
from kvblock.summaries.sign_sketch import generate_sign_sketch


@dataclass(frozen=True, slots=True)
class SummaryInspectionMetadata:
    """Compact summary metadata for query and block vectors."""

    summary_dim: int
    summary_scale: float | None
    summary_norm: float | None
    sign_sketch: int | None
    quantized_values_preview: tuple[int, ...]
    has_per_head: bool = False
    head_count: int = 0
    mean_per_head_summary_scale: float | None = None
    mean_per_head_summary_norm: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary metadata record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryKeyBlockInspectionRecord:
    """One block's qualitative query/key matching record."""

    block_id: int
    token_start: int
    token_end: int
    token_count: int
    preview_text: str
    selected: bool
    selected_reason: str
    stage_a_score: float | None
    stage_b_score: float | None
    final_score: float | None
    rank_position: int | None
    stage_a_rank: int | None
    stage_b_rank: int | None
    final_rank: int | None
    labeled_relevant: bool | None
    matched_relevance_fragments: tuple[str, ...]
    explanation_hints: tuple[str, ...]
    block_summary_metadata: SummaryInspectionMetadata
    candidate_id: str | None = None
    block_size: int | None = None
    stride: int | None = None
    block_mode: str | None = None
    parent_block_id: int | None = None
    parent_candidate_id: str | None = None
    parent_token_start: int | None = None
    parent_token_end: int | None = None
    candidate_role: str = "block"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly per-block inspection record."""

        payload = asdict(self)
        payload["block_summary_metadata"] = self.block_summary_metadata.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class QueryKeyComparisonGroups:
    """Block groups useful for manual selector failure analysis."""

    selected_relevant_block_ids: tuple[int, ...]
    selected_irrelevant_block_ids: tuple[int, ...]
    missed_relevant_block_ids: tuple[int, ...]
    high_scoring_near_miss_block_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly comparison group record."""

        return asdict(self)


@dataclass(frozen=True, slots=True)
class QueryKeyInspectionBundle:
    """Full qualitative query/key inspection payload for one selector run."""

    prompt_id: str | None
    prompt_name: str | None
    representation_source: str
    representation_name: str
    qk_aggregation_strategy: str
    rail_setting: str | None
    selected_block_ids: tuple[int, ...]
    missed_relevant_block_ids: tuple[int, ...]
    relevance_fragments: tuple[str, ...]
    query_summary_metadata: SummaryInspectionMetadata
    comparison_groups: QueryKeyComparisonGroups
    block_records: tuple[QueryKeyBlockInspectionRecord, ...]

    @property
    def selected_relevant_blocks(self) -> tuple[QueryKeyBlockInspectionRecord, ...]:
        """Return selected blocks that contain a known relevance fragment."""

        ids = set(self.comparison_groups.selected_relevant_block_ids)
        return tuple(record for record in self.block_records if record.block_id in ids)

    @property
    def selected_irrelevant_blocks(self) -> tuple[QueryKeyBlockInspectionRecord, ...]:
        """Return selected blocks that do not contain known relevance fragments."""

        ids = set(self.comparison_groups.selected_irrelevant_block_ids)
        return tuple(record for record in self.block_records if record.block_id in ids)

    @property
    def missed_relevant_blocks(self) -> tuple[QueryKeyBlockInspectionRecord, ...]:
        """Return relevant blocks that were not selected."""

        ids = set(self.comparison_groups.missed_relevant_block_ids)
        return tuple(record for record in self.block_records if record.block_id in ids)

    @property
    def high_scoring_near_miss_blocks(self) -> tuple[QueryKeyBlockInspectionRecord, ...]:
        """Return high-ranked unselected blocks retained for manual review."""

        ids = set(self.comparison_groups.high_scoring_near_miss_block_ids)
        return tuple(record for record in self.block_records if record.block_id in ids)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly qualitative inspection payload."""

        return {
            "prompt_id": self.prompt_id,
            "prompt_name": self.prompt_name,
            "representation_source": self.representation_source,
            "representation_name": self.representation_name,
            "qk_aggregation_strategy": self.qk_aggregation_strategy,
            "rail_setting": self.rail_setting,
            "selected_block_ids": list(self.selected_block_ids),
            "missed_relevant_block_ids": list(self.missed_relevant_block_ids),
            "relevance_fragments": list(self.relevance_fragments),
            "query_summary_metadata": self.query_summary_metadata.to_dict(),
            "comparison_groups": self.comparison_groups.to_dict(),
            "block_records": [record.to_dict() for record in self.block_records],
            "selected_relevant_blocks": [
                record.to_dict() for record in self.selected_relevant_blocks
            ],
            "selected_irrelevant_blocks": [
                record.to_dict() for record in self.selected_irrelevant_blocks
            ],
            "missed_relevant_blocks": [
                record.to_dict() for record in self.missed_relevant_blocks
            ],
            "high_scoring_near_miss_blocks": [
                record.to_dict() for record in self.high_scoring_near_miss_blocks
            ],
        }


def build_query_key_inspection(
    *,
    prompt_id: str | None,
    prompt_name: str | None,
    prompt_text: str,
    representation_source: str,
    representation_name: str,
    rail_setting: str | None,
    qk_aggregation_strategy: str = "mean_pool",
    selected_block_ids: Sequence[int],
    metadata_blocks: Sequence[BlockMetadata],
    query_summary: SummaryEncoding | MultiHeadQuerySummary | object,
    trace: SelectorDecisionTrace,
    block_inspections: Sequence[Any],
    relevant_text_fragments: Sequence[str] = (),
    top_unselected_blocks: int = 5,
) -> QueryKeyInspectionBundle:
    """Build a structured qualitative inspection bundle without changing selection."""

    if top_unselected_blocks < 0:
        raise ValueError("top_unselected_blocks must be >= 0")
    selected_ids = tuple(int(block_id) for block_id in selected_block_ids)
    relevant_fragments = tuple(
        fragment for fragment in (item.strip() for item in relevant_text_fragments)
        if fragment
    )
    query_terms = _question_terms(prompt_text)
    metadata_by_id = {int(block.block_id): block for block in metadata_blocks}
    stage_a_rank = {
        score.block_id: rank
        for rank, score in enumerate(trace.stage_a_scores, start=1)
    }
    stage_b_rank = {
        score.block_id: rank
        for rank, score in enumerate(trace.stage_b_scores, start=1)
    }
    final_rank = {
        block_id: rank
        for rank, block_id in enumerate(
            trace.final_selection.final_selected_block_ids,
            start=1,
        )
    }

    records: list[QueryKeyBlockInspectionRecord] = []
    for block in block_inspections:
        block_id = int(block.block_id)
        metadata = metadata_by_id[block_id]
        block_text = block.block_text or block.preview_text
        matched_fragments = _matched_fragments(block_text, relevant_fragments)
        labeled_relevant = (
            None if not relevant_fragments else bool(matched_fragments)
        )
        records.append(
            QueryKeyBlockInspectionRecord(
                block_id=block_id,
                token_start=block.token_start,
                token_end=block.token_end,
                token_count=block.token_count,
                preview_text=block.preview_text,
                selected=bool(block.selected),
                selected_reason=block.selected_reason,
                stage_a_score=block.stage_a_score,
                stage_b_score=block.stage_b_score,
                final_score=block.final_score,
                rank_position=stage_a_rank.get(block_id),
                stage_a_rank=stage_a_rank.get(block_id),
                stage_b_rank=stage_b_rank.get(block_id),
                final_rank=final_rank.get(block_id),
                labeled_relevant=labeled_relevant,
                matched_relevance_fragments=matched_fragments,
                explanation_hints=(),
                block_summary_metadata=_summary_metadata_from_block(metadata),
                candidate_id=getattr(block, "candidate_id", None),
                block_size=getattr(block, "block_size", None),
                stride=getattr(block, "stride", None),
                block_mode=getattr(block, "block_mode", None),
                parent_block_id=getattr(block, "parent_block_id", None),
                parent_candidate_id=getattr(block, "parent_candidate_id", None),
                parent_token_start=getattr(block, "parent_token_start", None),
                parent_token_end=getattr(block, "parent_token_end", None),
                candidate_role=getattr(block, "candidate_role", "block"),
            )
        )

    selected_relevant = tuple(
        record.block_id
        for record in records
        if record.selected and record.labeled_relevant is True
    )
    selected_irrelevant = tuple(
        record.block_id
        for record in records
        if record.selected and record.labeled_relevant is False
    )
    missed_relevant = tuple(
        record.block_id
        for record in records
        if not record.selected and record.labeled_relevant is True
    )
    missed_set = set(missed_relevant)
    high_scoring_near_miss = tuple(
        record.block_id
        for record in sorted(
            (
                record
                for record in records
                if not record.selected
                and (not relevant_fragments or record.block_id not in missed_set)
            ),
            key=_near_miss_sort_key,
        )[:top_unselected_blocks]
    )
    near_miss_set = set(high_scoring_near_miss)
    records_with_hints = tuple(
        replace(
            record,
            explanation_hints=_explanation_hints(
                record,
                query_terms=query_terms,
                is_high_scoring_near_miss=record.block_id in near_miss_set,
            ),
        )
        for record in records
    )

    groups = QueryKeyComparisonGroups(
        selected_relevant_block_ids=selected_relevant,
        selected_irrelevant_block_ids=selected_irrelevant,
        missed_relevant_block_ids=missed_relevant,
        high_scoring_near_miss_block_ids=high_scoring_near_miss,
    )
    return QueryKeyInspectionBundle(
        prompt_id=prompt_id,
        prompt_name=prompt_name,
        representation_source=representation_source,
        representation_name=representation_name,
        qk_aggregation_strategy=qk_aggregation_strategy,
        rail_setting=rail_setting,
        selected_block_ids=selected_ids,
        missed_relevant_block_ids=missed_relevant,
        relevance_fragments=relevant_fragments,
        query_summary_metadata=_summary_metadata_from_query(query_summary),
        comparison_groups=groups,
        block_records=records_with_hints,
    )


def _summary_metadata_from_query(query_summary: object) -> SummaryInspectionMetadata:
    if isinstance(query_summary, MultiHeadQuerySummary):
        pooled = query_summary.pooled
        head_count = len(query_summary.per_head)
        return SummaryInspectionMetadata(
            summary_dim=len(pooled.values),
            summary_scale=pooled.scale,
            summary_norm=pooled.summary_norm,
            sign_sketch=generate_sign_sketch(pooled),
            quantized_values_preview=pooled.values[:8],
            has_per_head=True,
            head_count=head_count,
            mean_per_head_summary_scale=_mean(head.scale for head in query_summary.per_head),
            mean_per_head_summary_norm=_mean(
                head.summary_norm for head in query_summary.per_head
            ),
        )
    if isinstance(query_summary, SummaryEncoding):
        return SummaryInspectionMetadata(
            summary_dim=len(query_summary.values),
            summary_scale=query_summary.scale,
            summary_norm=query_summary.summary_norm,
            sign_sketch=generate_sign_sketch(query_summary),
            quantized_values_preview=query_summary.values[:8],
        )
    return SummaryInspectionMetadata(
        summary_dim=0,
        summary_scale=None,
        summary_norm=None,
        sign_sketch=None,
        quantized_values_preview=(),
    )


def _summary_metadata_from_block(metadata: BlockMetadata) -> SummaryInspectionMetadata:
    head_count = len(metadata.per_head_summary_fp8)
    return SummaryInspectionMetadata(
        summary_dim=len(metadata.summary_fp8),
        summary_scale=metadata.summary_scale,
        summary_norm=metadata.summary_norm,
        sign_sketch=metadata.sign_sketch,
        quantized_values_preview=metadata.summary_fp8[:8],
        has_per_head=head_count > 0,
        head_count=head_count,
        mean_per_head_summary_scale=_mean(metadata.per_head_summary_scale),
        mean_per_head_summary_norm=_mean(metadata.per_head_summary_norm),
    )


def _matched_fragments(text: str, fragments: Sequence[str]) -> tuple[str, ...]:
    return tuple(fragment for fragment in fragments if fragment in text)


def _explanation_hints(
    record: QueryKeyBlockInspectionRecord,
    *,
    query_terms: set[str],
    is_high_scoring_near_miss: bool,
) -> tuple[str, ...]:
    hints: list[str] = []
    if "recent" in record.selected_reason:
        hints.append("recent rail")
    if "anchor" in record.selected_reason:
        hints.append("anchor rail")
    if "semantic" in record.selected_reason:
        hints.append("semantic high-score")
    if record.labeled_relevant is True and not record.selected:
        hints.append("missed despite relevance")
    if record.labeled_relevant is False and record.selected:
        hints.append("selected without label match")
    if is_high_scoring_near_miss:
        hints.append("high-scoring near-miss")
    if query_terms and len(query_terms & _text_terms(record.preview_text)) >= 2:
        hints.append("question-adjacent")
    return tuple(hints)


def _near_miss_sort_key(record: QueryKeyBlockInspectionRecord) -> tuple[int, float, int]:
    rank = record.stage_a_rank if record.stage_a_rank is not None else 10**9
    score = record.final_score if record.final_score is not None else -1.0
    return (rank, -score, record.block_id)


def _question_terms(prompt_text: str) -> set[str]:
    tail = prompt_text[-300:]
    if "?" in tail:
        tail = tail.rsplit("?", 1)[0]
        tail = tail[tail.rfind("\n") + 1 :]
    return _text_terms(tail)


def _text_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{3,}", text)
        if token.lower() not in _STOP_TERMS
    }


def _mean(values: Sequence[float] | Any) -> float | None:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        return None
    return sum(materialized) / len(materialized)


_STOP_TERMS = {
    "about",
    "after",
    "also",
    "from",
    "have",
    "into",
    "that",
    "the",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "your",
}
