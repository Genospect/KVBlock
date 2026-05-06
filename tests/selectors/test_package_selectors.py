from kvblock.blocks import BlockLayout
from kvblock.policies import KVBlockPolicy
from kvblock.selectors.expansion import (
    apply_anchor_rail,
    apply_halo,
    apply_recent_rail,
    cap_selected_fraction,
    deduplicate_and_sort,
)
from kvblock.selectors.simple import DenseSelector, RandomSparseSelector, RecentOnlySelector


def test_dense_selector_returns_valid_plan() -> None:
    layout = BlockLayout.from_token_count(total_tokens=96, block_size=32)
    policy = KVBlockPolicy(name="dense_policy", selector="dense")

    plan = DenseSelector().select(None, None, layout, policy)

    assert plan.logical_block_ids == [0, 1, 2]
    assert plan.selected_token_fraction == 1.0
    assert plan.validate() is plan


def test_recent_selector_includes_recent_rail() -> None:
    layout = BlockLayout.from_token_count(total_tokens=128, block_size=32)
    policy = KVBlockPolicy(name="recent", selector="recent_only", keep_recent_blocks=2)

    plan = RecentOnlySelector().select(None, None, layout, policy)

    assert plan.logical_block_ids == [2, 3]
    assert plan.recent_block_ids == [2, 3]


def test_random_sparse_selector_is_deterministic() -> None:
    layout = BlockLayout.from_token_count(total_tokens=320, block_size=32)
    policy = KVBlockPolicy(
        name="random",
        selector="random_sparse",
        semantic_k=3,
        metadata={"seed": 7},
    )

    first = RandomSparseSelector().select(None, None, layout, policy)
    second = RandomSparseSelector().select(None, None, layout, policy)

    assert first.logical_block_ids == second.logical_block_ids
    assert first.selected_blocks == 3


def test_expansion_utilities_update_plan() -> None:
    layout = BlockLayout.from_token_count(total_tokens=192, block_size=32)
    policy = KVBlockPolicy(name="recent", selector="recent_only", keep_recent_blocks=1)
    plan = RecentOnlySelector().select(None, None, layout, policy)

    plan = apply_anchor_rail(plan, layout, [0])
    plan = apply_recent_rail(plan, layout, 2)
    plan = apply_halo(plan, layout, 1)
    plan = cap_selected_fraction(plan, layout, 0.5)
    plan = deduplicate_and_sort(plan, layout)

    assert plan.selected_blocks <= 3
    assert plan.logical_block_ids == sorted(set(plan.logical_block_ids))
    assert plan.recent_block_ids == [4, 5]
    assert plan.anchor_block_ids == [0]
