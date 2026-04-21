"""Stage C final rail-preserving selection for the V1 selector skeleton."""

from __future__ import annotations

from typing import Iterable, Sequence

from kvblock.kv.block_types import BlockId
from kvblock.selector.base import FinalSelection, ScoredBlock
from kvblock.selector.policies import StageCPolicy


class StageCSelector:
    """Preserve rails first, then add semantic blocks from the refined ranking."""

    def __init__(self, policy: StageCPolicy | None = None) -> None:
        self.policy = policy or StageCPolicy()

    def select(
        self,
        candidates: Sequence[ScoredBlock],
        *,
        anchor_block_ids: Iterable[BlockId | int] = (),
        keep_recent_blocks: int | None = None,
        semantic_top_k: int | None = None,
        context_tokens: int | None = None,
    ) -> FinalSelection:
        """Build the final selection set with recent and anchor rails preserved."""

        anchor_ids = _normalize_anchor_ids(anchor_block_ids)
        recent_budget = (
            self.policy.keep_recent_blocks
            if keep_recent_blocks is None
            else keep_recent_blocks
        )
        semantic_budget = (
            self.policy.semantic_top_k_for_context(context_tokens)
            if semantic_top_k is None
            else semantic_top_k
        )

        by_id = {candidate.block_id: candidate for candidate in candidates}
        preserved_ids: set[BlockId] = set()

        recent_blocks = tuple(
            self._pick_recent_blocks(candidates, recent_budget, preserved_ids)
        )
        anchor_blocks = tuple(self._pick_anchor_blocks(by_id, anchor_ids, preserved_ids))
        semantic_blocks = tuple(
            self._pick_semantic_blocks(candidates, semantic_budget, preserved_ids)
        )

        selected = recent_blocks + anchor_blocks + semantic_blocks
        return FinalSelection(
            selected_blocks=selected,
            recent_blocks=recent_blocks,
            anchor_blocks=anchor_blocks,
            semantic_blocks=semantic_blocks,
        )

    def _pick_recent_blocks(
        self,
        candidates: Sequence[ScoredBlock],
        budget: int,
        preserved_ids: set[BlockId],
    ) -> list[ScoredBlock]:
        if budget <= 0:
            return []
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.metadata.last_access_step,
                candidate.metadata.token_start,
                candidate.final_score,
            ),
            reverse=True,
        )
        selected: list[ScoredBlock] = []
        for candidate in ranked:
            if candidate.block_id in preserved_ids:
                continue
            selected.append(candidate)
            preserved_ids.add(candidate.block_id)
            if len(selected) >= budget:
                break
        return selected

    def _pick_anchor_blocks(
        self,
        by_id: dict[BlockId, ScoredBlock],
        anchor_ids: Sequence[BlockId],
        preserved_ids: set[BlockId],
    ) -> list[ScoredBlock]:
        selected: list[ScoredBlock] = []
        for anchor_id in anchor_ids:
            if len(selected) >= self.policy.keep_anchor_blocks:
                break
            candidate = by_id.get(anchor_id)
            if candidate is None or anchor_id in preserved_ids:
                continue
            selected.append(candidate)
            preserved_ids.add(anchor_id)
        return selected

    def _pick_semantic_blocks(
        self,
        candidates: Sequence[ScoredBlock],
        budget: int,
        preserved_ids: set[BlockId],
    ) -> list[ScoredBlock]:
        if budget <= 0:
            return []
        selected: list[ScoredBlock] = []
        ranked = sorted(candidates, key=lambda candidate: candidate.final_score, reverse=True)
        for candidate in ranked:
            if candidate.block_id in preserved_ids:
                continue
            selected.append(candidate)
            preserved_ids.add(candidate.block_id)
            if len(selected) >= budget:
                break
        return selected


def _normalize_anchor_ids(anchor_block_ids: Iterable[BlockId | int]) -> tuple[BlockId, ...]:
    normalized: list[BlockId] = []
    seen: set[BlockId] = set()
    for anchor_id in anchor_block_ids:
        block_id = anchor_id if isinstance(anchor_id, BlockId) else BlockId(int(anchor_id))
        if block_id in seen:
            continue
        seen.add(block_id)
        normalized.append(block_id)
    return tuple(normalized)
