import torch

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.oracle import (
    SyntheticDenseOracle,
    SyntheticDenseOracleConfig,
    compare_block_sets,
    dense_reference_block_set,
    sparse_selected_block_set,
)


def _metadata(block_id: int, summary: tuple[int, ...], *, last_access_step: int = 0) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 32,
        token_len=32,
        summary_fp8=summary,
        summary_scale=0.25,
        summary_norm=1.0,
        sign_sketch=block_id,
        last_access_step=last_access_step,
        attn_ema=0.1 * (block_id + 1),
    )


def test_sparse_vs_dense_overlap_math() -> None:
    dense = dense_reference_block_set([0, 1, 2])
    sparse = sparse_selected_block_set([1, 2, 4])

    comparison = compare_block_sets(dense, sparse)

    assert [int(block_id) for block_id in comparison.overlap_block_ids] == [1, 2]
    assert comparison.overlap_count == 2


def test_recall_and_precision_computation() -> None:
    dense = dense_reference_block_set([0, 1, 2, 3])
    sparse = sparse_selected_block_set([1, 3, 4])

    comparison = compare_block_sets(dense, sparse)

    assert comparison.recall_rate == 0.5
    assert comparison.precision_rate == 2 / 3


def test_missing_block_and_extra_block_reporting() -> None:
    dense = dense_reference_block_set([0, 1, 2])
    sparse = sparse_selected_block_set([1, 4])

    comparison = compare_block_sets(dense, sparse)

    assert [int(block_id) for block_id in comparison.missed_important_block_ids] == [0, 2]
    assert [int(block_id) for block_id in comparison.extra_selected_block_ids] == [4]


def test_synthetic_dense_oracle_produces_reference_set() -> None:
    oracle = SyntheticDenseOracle(SyntheticDenseOracleConfig(top_k=2))
    blocks = [
        _metadata(0, (8, 0, 0, 0)),
        _metadata(1, (-8, 0, 0, 0)),
        _metadata(2, (0, 8, 0, 0)),
    ]

    reference = oracle.reference_blocks(torch.tensor([1.0, 0.0, 0.0, 0.0]), blocks)

    assert reference.size == 2
    assert int(reference.block_ids[0]) == 0

