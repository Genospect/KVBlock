import torch

from kvblock.bench.indexed_gather_decode_benchmark import (
    IndexedGatherBenchmarkConfig,
    block_id_runs_to_token_ranges,
    gather_selected_kv_blocks,
    run_benchmark,
    select_block_ids,
    token_count_for_blocks,
)


def test_selection_modes_return_expected_block_counts() -> None:
    generator = torch.Generator().manual_seed(0)

    random_ids = select_block_ids(
        total_blocks=10,
        selected_fraction=0.2,
        selection_mode="random_blocks",
        generator=generator,
    )
    contiguous_ids = select_block_ids(
        total_blocks=10,
        selected_fraction=0.3,
        selection_mode="contiguous_blocks",
        generator=generator,
    )
    recent_ids = select_block_ids(
        total_blocks=10,
        selected_fraction=0.2,
        selection_mode="recent_blocks",
        generator=generator,
    )

    assert len(random_ids) == 2
    assert len(contiguous_ids) == 3
    assert contiguous_ids == tuple(range(contiguous_ids[0], contiguous_ids[0] + 3))
    assert recent_ids == (8, 9)


def test_block_runs_to_token_ranges_merges_contiguous_blocks() -> None:
    assert block_id_runs_to_token_ranges(
        [0, 1, 3],
        total_tokens=100,
        block_size=32,
    ) == ((0, 64), (96, 100))
    assert token_count_for_blocks([0, 1, 3], total_tokens=100, block_size=32) == 68


def test_gather_selected_kv_blocks_compacts_tokens() -> None:
    key = torch.arange(1 * 8 * 1 * 1, dtype=torch.float32).reshape(1, 8, 1, 1)
    value = key + 100

    compact_key, compact_value = gather_selected_kv_blocks(
        key,
        value,
        [1, 3],
        total_tokens=8,
        block_size=2,
    )

    assert compact_key.flatten().tolist() == [2.0, 3.0, 6.0, 7.0]
    assert compact_value.flatten().tolist() == [102.0, 103.0, 106.0, 107.0]


def test_run_benchmark_cpu_all_selected_is_close() -> None:
    rows = run_benchmark(
        IndexedGatherBenchmarkConfig(
            batch_size=1,
            num_heads=2,
            head_dim=4,
            total_tokens=(32,),
            block_sizes=(16,),
            selected_fractions=(1.0,),
            selection_modes=("recent_blocks",),
            dtype="float32",
            device="cpu",
            iters=1,
            warmup=0,
            seed=0,
        )
    )

    assert len(rows) == 1
    assert rows[0].selected_tokens == 32
    assert rows[0].selected_fraction == 1.0
    assert rows[0].output_l2_diff == 0.0
    assert rows[0].max_abs_diff == 0.0
