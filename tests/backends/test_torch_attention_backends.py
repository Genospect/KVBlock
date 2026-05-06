import torch

from kvblock.backends import TorchDenseAttentionBackend, TorchSparseReferenceBackend
from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


def test_sparse_reference_matches_dense_when_all_tokens_selected() -> None:
    torch.manual_seed(0)
    layout = BlockLayout.from_token_count(total_tokens=64, block_size=32)
    plan = SelectedKVPlan(
        logical_block_ids=[0, 1],
        selected_token_ranges=[(0, 32), (32, 64)],
        total_blocks=2,
        total_tokens=64,
    )
    query = torch.randn(2, 8)
    key_cache = torch.randn(64, 2, 8)
    value_cache = torch.randn(64, 2, 8)

    dense = TorchDenseAttentionBackend().run_decode(
        query,
        key_cache,
        value_cache,
        plan,
        layout,
    )
    sparse = TorchSparseReferenceBackend().run_decode(
        query,
        key_cache,
        value_cache,
        plan,
        layout,
    )

    assert torch.allclose(dense, sparse)


def test_sparse_reference_gathers_subset() -> None:
    torch.manual_seed(0)
    layout = BlockLayout.from_token_count(total_tokens=64, block_size=32)
    plan = SelectedKVPlan(
        logical_block_ids=[1],
        selected_token_ranges=[(32, 64)],
        total_blocks=2,
        total_tokens=64,
    )
    query = torch.randn(2, 8)
    key_cache = torch.randn(64, 2, 8)
    value_cache = torch.randn(64, 2, 8)

    sparse = TorchSparseReferenceBackend().run_decode(
        query,
        key_cache,
        value_cache,
        plan,
        layout,
    )

    assert sparse.shape == query.shape
