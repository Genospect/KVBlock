import pytest

from kvblock.plans import SelectedKVPlan


def test_selected_kv_plan_serialization_round_trip() -> None:
    plan = SelectedKVPlan(
        request_id="req",
        layer_id=1,
        step_id=2,
        logical_block_ids=[1, 3],
        physical_page_ids=[10, 30],
        selected_token_ranges=[(32, 64), (96, 128)],
        recent_block_ids=[3],
        anchor_block_ids=[1],
        confidence=0.75,
        selector_name="selector",
        policy_name="policy",
        total_blocks=4,
        total_tokens=128,
        metadata={"trace_id": "abc"},
    )

    round_tripped = SelectedKVPlan.from_json(plan.to_json())

    assert round_tripped == plan
    assert round_tripped.has_physical_mapping is True
    assert round_tripped.selected_block_fraction == 0.5
    assert round_tripped.selected_token_fraction == 0.5


def test_selected_kv_plan_allows_empty_physical_mapping() -> None:
    plan = SelectedKVPlan(
        logical_block_ids=[0],
        selected_token_ranges=[(0, 32)],
        total_blocks=2,
        total_tokens=64,
    )

    assert plan.physical_page_ids is None
    assert plan.has_physical_mapping is False


def test_selected_kv_plan_rejects_selected_blocks_above_total() -> None:
    with pytest.raises(ValueError, match="selected_blocks must be <= total_blocks"):
        SelectedKVPlan(
            logical_block_ids=[0, 1, 2],
            selected_blocks=3,
            total_blocks=2,
            total_tokens=96,
        )


def test_selected_kv_plan_requires_fallback_reason_when_triggered() -> None:
    with pytest.raises(ValueError, match="fallback_reason is required"):
        SelectedKVPlan(
            logical_block_ids=[],
            selected_blocks=0,
            total_blocks=0,
            fallback_triggered=True,
        )
