import pytest

from kvblock.blocks import LogicalToPhysicalMapper
from kvblock.plans import SelectedKVPlan


def test_logical_to_physical_mapper_populates_plan() -> None:
    plan = SelectedKVPlan(
        logical_block_ids=[0, 2],
        selected_token_ranges=[(0, 32), (64, 96)],
        total_blocks=3,
        total_tokens=96,
    )
    mapped = LogicalToPhysicalMapper({0: 10, 2: 12}).map_plan(plan)

    assert mapped.physical_page_ids == [10, 12]
    assert plan.physical_page_ids is None


def test_identity_mapper() -> None:
    mapper = LogicalToPhysicalMapper.identity(3)

    assert mapper.map_block_ids([0, 1, 2]) == [0, 1, 2]


def test_mapper_rejects_missing_mapping() -> None:
    mapper = LogicalToPhysicalMapper({0: 0})

    with pytest.raises(KeyError, match="missing physical mapping"):
        mapper.map_block_ids([1])
