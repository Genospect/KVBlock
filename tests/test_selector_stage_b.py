import torch

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import ScoredBlock
from kvblock.selector.policies import StageBPolicy
from kvblock.selector.stage_b import StageBRefiner
from kvblock.summaries.sign_sketch import generate_sign_sketch


def _candidate(block_id: int, summary: torch.Tensor, stage_a_score: float) -> ScoredBlock:
    return ScoredBlock(
        metadata=BlockMetadata(
            block_id=BlockId(block_id),
            pool_id=0,
            token_start=block_id * 32,
            token_len=32,
            summary_fp8=tuple(int(value) for value in summary.tolist()),
            summary_scale=1.0,
            sign_sketch=generate_sign_sketch(summary),
            summary_norm=1.0,
            last_access_step=0,
        ),
        stage_a_score=stage_a_score,
        approx_similarity_score=stage_a_score,
        final_score=stage_a_score,
    )


def test_stage_b_hamming_refinement_reorders_close_stage_a_scores() -> None:
    query = torch.tensor([5.0, -4.0, 3.0, 1.0], dtype=torch.float32)
    query_sketch = generate_sign_sketch(query)
    candidates = [
        _candidate(0, torch.tensor([-5.0, 4.0, -3.0, -1.0]), stage_a_score=0.81),
        _candidate(1, torch.tensor([5.0, -4.0, 3.0, 1.0]), stage_a_score=0.8),
    ]
    refiner = StageBRefiner(
        StageBPolicy(hamming_weight=0.5, base_score_weight=1.0, sketch_bits=64)
    )

    refined = refiner.refine(candidates, query_sketch)

    assert int(refined[0].block_id) == 1
    assert refined[0].hamming_similarity > refined[1].hamming_similarity
    assert refined[0].final_score > refined[1].final_score
