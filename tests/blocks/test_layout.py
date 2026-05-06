import pytest

from kvblock.blocks import BlockLayout


def test_block_layout_fixed_blocks_and_lookup() -> None:
    layout = BlockLayout.from_token_count(total_tokens=100, block_size=32)

    assert layout.block_count == 4
    assert layout.get_block(3).start_token == 96
    assert layout.get_block(3).end_token == 100
    assert layout.token_range_to_block_ids(31, 65) == (0, 1, 2)
    assert layout.get_recent_blocks(2) == (2, 3)


def test_block_layout_strided_blocks_and_halo() -> None:
    layout = BlockLayout.from_token_count(total_tokens=100, block_size=40, stride=20)

    assert layout.block_count == 5
    assert layout.token_range_to_block_ids(35, 41) == (0, 1, 2)
    assert layout.get_halo([2], halo=1) == (1, 2, 3)


def test_block_layout_rejects_invalid_range() -> None:
    layout = BlockLayout.from_token_count(total_tokens=64, block_size=32)

    with pytest.raises(ValueError, match="end_token must be <= total_tokens"):
        layout.token_range_to_block_ids(0, 65)
