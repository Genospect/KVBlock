"""Simple selectors used for package tests, debugging, and baselines."""

from __future__ import annotations

import random
from typing import Any

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan
from kvblock.policies import KVBlockPolicy


class DenseSelector:
    """Select every logical block."""

    name = "dense"

    def select(
        self,
        query_state: Any,
        kv_metadata: Any,
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        block_ids = [span.block_id for span in layout.spans]
        return _plan_from_ids(
            block_ids,
            layout=layout,
            selector_name=self.name,
            policy_name=policy.name,
            confidence=1.0,
            metadata={"mode": "dense"},
        )


class RecentOnlySelector:
    """Select only the recent rail blocks."""

    name = "recent_only"

    def select(
        self,
        query_state: Any,
        kv_metadata: Any,
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        recent = list(layout.get_recent_blocks(policy.keep_recent_blocks))
        return _plan_from_ids(
            recent,
            layout=layout,
            selector_name=self.name,
            policy_name=policy.name,
            recent_block_ids=recent,
            confidence=1.0,
            metadata={"mode": "recent_only"},
        )


class RandomSparseSelector:
    """Deterministic random sparse selector for debugging."""

    name = "random_sparse"

    def select(
        self,
        query_state: Any,
        kv_metadata: Any,
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        seed = int(policy.metadata.get("seed", 0))
        rng = random.Random(seed)
        all_ids = [span.block_id for span in layout.spans]
        budget = min(len(all_ids), policy.semantic_k)
        sampled = sorted(rng.sample(all_ids, budget)) if budget else []
        return _plan_from_ids(
            sampled,
            layout=layout,
            selector_name=self.name,
            policy_name=policy.name,
            confidence=1.0,
            metadata={"mode": "random_sparse", "seed": seed},
        )


def _plan_from_ids(
    block_ids: list[int],
    *,
    layout: BlockLayout,
    selector_name: str,
    policy_name: str,
    recent_block_ids: list[int] | None = None,
    anchor_block_ids: list[int] | None = None,
    halo_block_ids: list[int] | None = None,
    confidence: float,
    fallback_triggered: bool = False,
    fallback_reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> SelectedKVPlan:
    deduped = list(dict.fromkeys(int(block_id) for block_id in block_ids))
    token_ranges = layout.token_ranges_for_blocks(deduped)
    selected_tokens = sum(end - start for start, end in token_ranges)
    return SelectedKVPlan(
        logical_block_ids=deduped,
        selected_token_ranges=token_ranges,
        recent_block_ids=list(recent_block_ids or []),
        anchor_block_ids=list(anchor_block_ids or []),
        halo_block_ids=list(halo_block_ids or []),
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        fallback_reason=fallback_reason,
        selector_name=selector_name,
        policy_name=policy_name,
        total_blocks=layout.block_count,
        selected_blocks=len(deduped),
        total_tokens=layout.total_tokens,
        selected_tokens=selected_tokens,
        metadata=dict(metadata or {}),
    )
