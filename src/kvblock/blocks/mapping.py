"""Logical-to-physical page mapping helpers."""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from kvblock.plans import SelectedKVPlan


class LogicalToPhysicalMapper:
    """Map package-level logical block decisions to runtime physical page ids."""

    def __init__(self, logical_to_physical: Mapping[int, int]) -> None:
        self.logical_to_physical = {
            int(logical): int(physical)
            for logical, physical in logical_to_physical.items()
        }
        if any(key < 0 for key in self.logical_to_physical):
            raise ValueError("logical block ids must be >= 0")
        if any(value < 0 for value in self.logical_to_physical.values()):
            raise ValueError("physical page ids must be >= 0")

    @classmethod
    def identity(cls, total_blocks: int) -> "LogicalToPhysicalMapper":
        """Create the simple ``logical_id -> physical_id`` mapping."""

        if total_blocks < 0:
            raise ValueError("total_blocks must be >= 0")
        return cls({block_id: block_id for block_id in range(total_blocks)})

    def map_block_ids(self, logical_block_ids: list[int] | tuple[int, ...]) -> list[int]:
        """Map logical block ids to physical page ids."""

        physical: list[int] = []
        for block_id in logical_block_ids:
            normalized = int(block_id)
            try:
                physical.append(self.logical_to_physical[normalized])
            except KeyError as exc:
                raise KeyError(f"missing physical mapping for block {normalized}") from exc
        return physical

    def map_plan(self, plan: SelectedKVPlan) -> SelectedKVPlan:
        """Return a copy of ``plan`` with ``physical_page_ids`` populated."""

        return replace(
            plan,
            physical_page_ids=self.map_block_ids(plan.logical_block_ids),
        )
