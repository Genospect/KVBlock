from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import ScoredBlock
from kvblock.selector.policies import StageCPolicy
from kvblock.selector.stage_c import StageCSelector


def _candidate(
    block_id: int,
    *,
    final_score: float,
    last_access_step: int,
) -> ScoredBlock:
    return ScoredBlock(
        metadata=BlockMetadata(
            block_id=BlockId(block_id),
            pool_id=0,
            token_start=block_id * 32,
            token_len=32,
            summary_fp8=(1, 0, 0, 0),
            summary_scale=1.0,
            sign_sketch=block_id,
            summary_norm=1.0,
            last_access_step=last_access_step,
        ),
        approx_similarity_score=final_score,
        stage_a_score=final_score,
        stage_b_score=final_score,
        final_score=final_score,
    )


def test_stage_c_preserves_recent_and_anchor_then_adds_semantic_blocks() -> None:
    selector = StageCSelector(
        StageCPolicy(keep_recent_blocks=1, keep_anchor_blocks=1, semantic_top_k=2)
    )
    candidates = [
        _candidate(0, final_score=0.99, last_access_step=2),
        _candidate(1, final_score=0.95, last_access_step=3),
        _candidate(2, final_score=0.4, last_access_step=100),
        _candidate(3, final_score=0.2, last_access_step=1),
    ]

    selection = selector.select(candidates, anchor_block_ids=[3])

    assert [int(block.block_id) for block in selection.recent_blocks] == [2]
    assert [int(block.block_id) for block in selection.anchor_blocks] == [3]
    assert [int(block.block_id) for block in selection.semantic_blocks] == [0, 1]
    assert [int(block.block_id) for block in selection.selected_blocks] == [2, 3, 0, 1]
