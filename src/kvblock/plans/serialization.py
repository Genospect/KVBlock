"""Small serialization helpers for selected KV plans."""

from __future__ import annotations

from kvblock.plans.selected_kv_plan import SelectedKVPlan


def plan_to_json(plan: SelectedKVPlan, *, indent: int | None = None) -> str:
    """Serialize a selected KV plan to JSON."""

    return plan.to_json(indent=indent)


def plan_from_json(payload: str) -> SelectedKVPlan:
    """Deserialize a selected KV plan from JSON."""

    return SelectedKVPlan.from_json(payload)
