"""Plan expansion helpers for rails, halo, caps, and stable ordering."""

from __future__ import annotations

from dataclasses import replace
from math import floor
from typing import Iterable

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


def apply_recent_rail(plan: SelectedKVPlan, layout: BlockLayout, n: int) -> SelectedKVPlan:
    """Return a plan that includes the layout's recent rail blocks."""

    recent = list(layout.get_recent_blocks(n))
    return _replace_blocks(
        plan,
        layout,
        block_ids=[*plan.logical_block_ids, *recent],
        recent_block_ids=recent,
    )


def apply_anchor_rail(
    plan: SelectedKVPlan,
    layout: BlockLayout,
    anchor_ids: Iterable[int],
) -> SelectedKVPlan:
    """Return a plan that includes anchor block ids."""

    anchors = [int(block_id) for block_id in anchor_ids]
    for block_id in anchors:
        layout.get_block(block_id)
    return _replace_blocks(
        plan,
        layout,
        block_ids=[*plan.logical_block_ids, *anchors],
        anchor_block_ids=anchors,
    )


def apply_halo(plan: SelectedKVPlan, layout: BlockLayout, halo: int) -> SelectedKVPlan:
    """Return a plan expanded with neighboring halo blocks."""

    halo_ids = list(layout.get_halo(plan.logical_block_ids, halo))
    new_halo_ids = [
        block_id for block_id in halo_ids if block_id not in set(plan.logical_block_ids)
    ]
    return _replace_blocks(
        plan,
        layout,
        block_ids=halo_ids,
        halo_block_ids=new_halo_ids,
    )


def cap_selected_fraction(
    plan: SelectedKVPlan,
    layout: BlockLayout,
    max_fraction: float | None,
) -> SelectedKVPlan:
    """Cap selected blocks while preserving rails first."""

    if max_fraction is None or plan.total_blocks == 0:
        return plan
    if not 0.0 < max_fraction <= 1.0:
        raise ValueError("max_fraction must be in (0, 1]")
    limit = floor(plan.total_blocks * max_fraction)
    if limit <= 0 and plan.logical_block_ids:
        limit = 1
    if plan.selected_blocks <= limit:
        return plan

    protected = list(
        dict.fromkeys(
            [*plan.recent_block_ids, *plan.anchor_block_ids, *plan.linked_block_ids]
        )
    )
    capped: list[int] = []
    for block_id in [*protected, *plan.logical_block_ids]:
        if block_id in capped:
            continue
        capped.append(block_id)
        if len(capped) >= limit:
            break
    return _replace_blocks(plan, layout, block_ids=capped)


def deduplicate_and_sort(plan: SelectedKVPlan, layout: BlockLayout) -> SelectedKVPlan:
    """Return a plan with unique logical block ids sorted ascending."""

    return _replace_blocks(plan, layout, block_ids=sorted(set(plan.logical_block_ids)))


def _replace_blocks(
    plan: SelectedKVPlan,
    layout: BlockLayout,
    *,
    block_ids: Iterable[int],
    recent_block_ids: Iterable[int] | None = None,
    anchor_block_ids: Iterable[int] | None = None,
    halo_block_ids: Iterable[int] | None = None,
) -> SelectedKVPlan:
    deduped = list(dict.fromkeys(int(block_id) for block_id in block_ids))
    token_ranges = layout.token_ranges_for_blocks(deduped)
    physical_page_ids = None
    if plan.physical_page_ids is not None:
        physical_page_ids = None
    return replace(
        plan,
        logical_block_ids=deduped,
        physical_page_ids=physical_page_ids,
        selected_token_ranges=token_ranges,
        recent_block_ids=(
            plan.recent_block_ids
            if recent_block_ids is None
            else list(dict.fromkeys(int(block_id) for block_id in recent_block_ids))
        ),
        anchor_block_ids=(
            plan.anchor_block_ids
            if anchor_block_ids is None
            else list(dict.fromkeys(int(block_id) for block_id in anchor_block_ids))
        ),
        halo_block_ids=(
            plan.halo_block_ids
            if halo_block_ids is None
            else list(dict.fromkeys(int(block_id) for block_id in halo_block_ids))
        ),
        selected_blocks=len(deduped),
        selected_tokens=sum(end - start for start, end in token_ranges),
    )
