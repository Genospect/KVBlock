"""Stage B Hamming-based refinement for the V1 selector skeleton."""

from __future__ import annotations

from typing import Sequence

from kvblock.selector.base import ScoredBlock
from kvblock.selector.policies import StageBPolicy
from kvblock.summaries.sign_sketch import hamming_similarity


class StageBRefiner:
    """Refine Stage A candidates with 64-bit sign-sketch similarity."""

    def __init__(self, policy: StageBPolicy | None = None) -> None:
        self.policy = policy or StageBPolicy()

    def refine(
        self,
        candidates: Sequence[ScoredBlock],
        query_sign_sketch: int,
        *,
        top_n: int | None = None,
    ) -> list[ScoredBlock]:
        """Return Stage B candidates sorted by the refined score."""

        refined: list[ScoredBlock] = []
        for candidate in candidates:
            sketch_similarity = hamming_similarity(
                candidate.metadata.sign_sketch,
                query_sign_sketch,
                bits=self.policy.sketch_bits,
            )
            stage_b_score = sketch_similarity
            final_score = (
                self.policy.base_score_weight * candidate.stage_a_score
                + self.policy.hamming_weight * stage_b_score
            )
            refined.append(
                ScoredBlock(
                    metadata=candidate.metadata,
                    approx_similarity_score=candidate.approx_similarity_score,
                    recency_score=candidate.recency_score,
                    attn_score=candidate.attn_score,
                    priority_score=candidate.priority_score,
                    stage_a_score=candidate.stage_a_score,
                    hamming_similarity=sketch_similarity,
                    stage_b_score=stage_b_score,
                    final_score=final_score,
                )
            )

        refined.sort(key=_rank_key, reverse=True)
        return refined if top_n is None else refined[:top_n]


def _rank_key(candidate: ScoredBlock) -> tuple[float, int, int]:
    return (
        candidate.final_score,
        candidate.metadata.last_access_step,
        candidate.metadata.token_start,
    )
