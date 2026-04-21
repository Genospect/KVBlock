from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata


def test_block_metadata_round_trip_dict() -> None:
    metadata = BlockMetadata(
        block_id=BlockId(7),
        pool_id=1,
        token_start=96,
        token_len=32,
        precision_tier="fp8",
        flags=3,
        summary_fp8=tuple(range(-16, 16)),
        summary_scale=0.125,
        sign_sketch=0xA5,
        summary_norm=1.25,
        attn_ema=0.8,
        attn_var=0.1,
        last_access_step=42,
        hit_count=9,
        priority=2.0,
        rope_bucket=4,
        fallback_miss_count=1,
    )

    round_tripped = BlockMetadata.from_dict(metadata.to_dict())

    assert round_tripped == metadata
    assert metadata.token_span.token_end == 128
    assert metadata.block_ref.pool_id == 1
    assert int(metadata.block_ref.block_id) == 7
    assert metadata.to_dict()["summary_scale"] == 0.125


def test_block_metadata_rejects_invalid_summary_value() -> None:
    try:
        BlockMetadata(
            block_id=BlockId(0),
            pool_id=0,
            token_start=0,
            token_len=32,
            summary_fp8=(128,),
        )
    except ValueError as exc:
        assert "summary_fp8 values" in str(exc)
    else:
        raise AssertionError("Expected ValueError for out-of-range summary value")


def test_block_metadata_dequantize_summary_uses_scale() -> None:
    metadata = BlockMetadata(
        block_id=BlockId(1),
        pool_id=0,
        token_start=0,
        token_len=32,
        summary_fp8=(8, -4, 2, 0),
        summary_scale=0.25,
    )

    restored = metadata.dequantize_summary()

    assert restored.tolist() == [2.0, -1.0, 0.5, 0.0]


def test_block_metadata_round_trips_per_head_summaries() -> None:
    metadata = BlockMetadata(
        block_id=BlockId(2),
        pool_id=0,
        token_start=0,
        token_len=16,
        summary_fp8=(4, -4),
        summary_scale=0.5,
        per_head_summary_fp8=((4, 0), (0, -4)),
        per_head_summary_scale=(0.25, 0.5),
        per_head_summary_norm=(1.0, 2.0),
    )

    round_tripped = BlockMetadata.from_dict(metadata.to_dict())
    per_head = round_tripped.dequantize_per_head_summary()

    assert round_tripped == metadata
    assert per_head.tolist() == [[1.0, 0.0], [0.0, -2.0]]


def test_block_metadata_rejects_mismatched_per_head_summary_shape() -> None:
    try:
        BlockMetadata(
            block_id=BlockId(3),
            pool_id=0,
            token_start=0,
            token_len=16,
            summary_fp8=(1, 2),
            per_head_summary_fp8=((1,),),
            per_head_summary_scale=(1.0,),
            per_head_summary_norm=(1.0,),
        )
    except ValueError as exc:
        assert "per-head summaries" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched per-head summary dim")
