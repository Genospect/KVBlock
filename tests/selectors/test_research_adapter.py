from types import SimpleNamespace

import pytest

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan
from kvblock.policies import KVBlockPolicy
from kvblock.selectors.research_adapter import (
    ExistingSelectorSelectedPlanAdapter,
    MixedGlobalRefineSelectedPlanAdapter,
    selected_kv_plan_from_legacy_output,
    selected_kv_plan_from_longbench_selector_row,
)


def test_adapter_returns_valid_selected_kv_plan_from_real_block_like_result() -> None:
    result = SimpleNamespace(
        selected_block_ids=(1, 3),
        selected_to_semantic_k_ratio=0.25,
        fallback_mode="sparse",
        confidence=SimpleNamespace(raw_margin=0.4, to_dict=lambda: {"raw_margin": 0.4}),
        run_summary=SimpleNamespace(
            block_count=4,
            token_count=128,
            block_mode="fixed_32",
            to_dict=lambda: {"block_count": 4, "token_count": 128},
        ),
        latency=SimpleNamespace(
            selector_sec=0.01,
            to_dict=lambda: {"selector_sec": 0.01},
        ),
        trace=SimpleNamespace(
            final_selection=SimpleNamespace(
                recent_block_ids=(3,),
                anchor_block_ids=(0,),
            ),
            to_dict=lambda: {"trace": "ok"},
        ),
        selected_block_inspections=(
            SimpleNamespace(
                block_id=1,
                token_start=32,
                token_end=64,
                parent_block_id=None,
            ),
            SimpleNamespace(
                block_id=3,
                token_start=96,
                token_end=128,
                parent_block_id=None,
            ),
        ),
    )

    plan = selected_kv_plan_from_legacy_output(
        result,
        selector_name="fixed_32",
        policy_name="quality_guarded_static",
    )

    assert plan.validate() is plan
    assert plan.logical_block_ids == [1, 3]
    assert plan.selected_token_ranges == [(32, 64), (96, 128)]
    assert plan.recent_block_ids == [3]
    assert plan.anchor_block_ids == [0]
    assert plan.confidence == pytest.approx(0.4)
    assert plan.fallback_triggered is False
    assert plan.selector_name == "fixed_32"
    assert plan.policy_name == "quality_guarded_static"


def test_adapter_maps_selected_ranges_to_block_ids_when_ids_missing() -> None:
    layout = BlockLayout.from_token_count(total_tokens=128, block_size=32)
    adapter = ExistingSelectorSelectedPlanAdapter()
    policy = KVBlockPolicy(name="range_only")

    plan = adapter.select(
        {"selected_spans": ("40:70",), "total_tokens": 128},
        None,
        layout,
        policy,
    )

    assert plan.logical_block_ids == [1, 2]
    assert plan.selected_token_ranges == [(40, 70)]
    assert plan.selected_tokens == 30


def test_adapter_handles_missing_confidence_and_fallback_metadata_safely() -> None:
    plan = selected_kv_plan_from_legacy_output(
        {"logical_block_ids": (0,), "selected_spans": ("0:32",), "total_tokens": 32},
        policy_name="debug",
    )

    assert plan.confidence == 1.0
    assert plan.fallback_triggered is False
    assert plan.fallback_reason is None
    assert plan.metadata["confidence_default"] == "missing_confidence_assumed_1.0"


def test_mixed_global_refine_adapter_preserves_selected_ids_and_ranges() -> None:
    layout = BlockLayout.from_token_count(total_tokens=160, block_size=40)
    adapter = MixedGlobalRefineSelectedPlanAdapter()
    policy = KVBlockPolicy(name="quality_guarded_static")

    plan = adapter.select(
        {
            "selected_ids": (2, 1, 2),
            "selected_spans": ("80:120", "40:80"),
            "total_blocks": 4,
            "total_tokens": 160,
            "confidence": 0.2,
            "fallback_triggered": True,
            "fallback_reason": "low_margin",
        },
        None,
        layout,
        policy,
    )

    assert plan.logical_block_ids == [2, 1]
    assert plan.selected_token_ranges == [(80, 120), (40, 80)]
    assert plan.fallback_triggered is True
    assert plan.fallback_reason == "low_margin"
    assert plan.selector_name == "mixed_global_refine_40_16_stride_8"


def test_longbench_selector_row_adapter_uses_filtered_output_selection() -> None:
    row = SimpleNamespace(
        selected_block_ids=(0, 1, 2),
        selected_spans=("0:32", "32:64", "64:96"),
        selected_blocks=(
            {"block_id": 0, "token_start": 0, "token_end": 32},
            {"block_id": 1, "token_start": 32, "token_end": 64},
            {"block_id": 2, "token_start": 64, "token_end": 96},
        ),
        semantic_selected_block_ids=(0, 1),
        candidate_block_count=4,
        tokens=128,
        raw_margin=0.3,
        mixed_fallback_used=False,
        block_mode="mixed_global_refine_40_16_stride_8",
        selector_latency_sec=0.01,
        target_recall=0.5,
        evidence_window_recall=1.0,
        selected_precision=0.25,
        dataset_name="hotpotqa",
        sample_id="sample-1",
    )

    plan = selected_kv_plan_from_longbench_selector_row(
        row,
        selected_block_ids=(0, 2),
        selected_spans=("0:32", "64:96"),
        selected_blocks=(
            {"block_id": 0, "token_start": 0, "token_end": 32},
            {"block_id": 2, "token_start": 64, "token_end": 96},
        ),
        policy_name="quality_guarded_static",
    )

    assert isinstance(plan, SelectedKVPlan)
    assert plan.logical_block_ids == [0, 2]
    assert plan.selected_token_ranges == [(0, 32), (64, 96)]
    assert plan.selected_block_fraction == pytest.approx(0.5)
    assert plan.selected_token_fraction == pytest.approx(0.5)
    assert plan.metadata["evidence_recall"] == 0.5
    assert plan.metadata["evidence_window_recall"] == 1.0
