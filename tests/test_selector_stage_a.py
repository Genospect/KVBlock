import torch

from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import coerce_query_summary
from kvblock.selector.policies import StageAPolicy, StageAWeights
from kvblock.selector.stage_a import StageAScorer, approx_cosine_similarity
from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding


def _metadata(
    block_id: int,
    *,
    summary: tuple[int, ...],
    last_access_step: int,
    attn_ema: float = 0.0,
    priority: float = 0.0,
) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 32,
        token_len=32,
        summary_fp8=summary,
        summary_scale=0.25,
        sign_sketch=block_id,
        summary_norm=1.0,
        attn_ema=attn_ema,
        last_access_step=last_access_step,
        priority=priority,
    )


def _per_head_metadata(
    block_id: int,
    *,
    per_head_summary: tuple[tuple[int, ...], ...],
) -> BlockMetadata:
    return BlockMetadata(
        block_id=BlockId(block_id),
        pool_id=0,
        token_start=block_id * 16,
        token_len=16,
        summary_fp8=(0, 0),
        summary_scale=1.0,
        sign_sketch=block_id,
        summary_norm=0.0,
        last_access_step=block_id,
        per_head_summary_fp8=per_head_summary,
        per_head_summary_scale=(1.0,) * len(per_head_summary),
        per_head_summary_norm=(1.0,) * len(per_head_summary),
    )


def test_stage_a_prefers_summary_similarity_when_weighted() -> None:
    scorer = StageAScorer(
        StageAPolicy(
            weights=StageAWeights(
                summary_similarity=1.0, recency=0.0, attn_ema=0.0, priority=0.0
            ),
            shortlist_size=2,
        )
    )
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    blocks = [
        _metadata(0, summary=(12, 0, 0, 0), last_access_step=5),
        _metadata(1, summary=(-12, 0, 0, 0), last_access_step=99),
        _metadata(2, summary=(0, 12, 0, 0), last_access_step=99),
    ]

    scored = scorer.score(blocks, query, current_step=100)

    assert [int(item.block_id) for item in scored] == [0, 2]
    assert scored[0].approx_similarity_score > scored[1].approx_similarity_score


def test_stage_a_combines_recency_attn_and_priority() -> None:
    scorer = StageAScorer(
        StageAPolicy(
            weights=StageAWeights(
                summary_similarity=0.0, recency=0.4, attn_ema=0.3, priority=0.3
            ),
            shortlist_size=3,
        )
    )
    query = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    blocks = [
        _metadata(0, summary=(0, 0, 0, 0), last_access_step=90, attn_ema=0.1, priority=0.2),
        _metadata(1, summary=(0, 0, 0, 0), last_access_step=99, attn_ema=0.9, priority=0.8),
        _metadata(2, summary=(0, 0, 0, 0), last_access_step=70, attn_ema=0.2, priority=1.0),
    ]

    scored = scorer.score(blocks, query, current_step=100)

    assert int(scored[0].block_id) == 1
    assert scored[0].recency_score > scored[2].recency_score
    assert scored[0].attn_score > scored[2].attn_score


def test_stage_a_approx_cosine_similarity_uses_summary_encoding_scale() -> None:
    block = _metadata(0, summary=(8, 0, 0, 0), last_access_step=0)
    query = SummaryEncoding(values=(4, 0, 0, 0), scale=0.5, summary_norm=2.0)

    score = approx_cosine_similarity(query, block)

    assert abs(score - 1.0) < 1e-6
    assert torch.allclose(
        coerce_query_summary(query), torch.tensor([2.0, 0.0, 0.0, 0.0])
    )


def test_stage_a_vectorized_scores_match_reference_loop() -> None:
    policy = StageAPolicy(
        weights=StageAWeights(
            summary_similarity=0.5,
            recency=0.2,
            attn_ema=0.2,
            priority=0.1,
        ),
        shortlist_size=4,
    )
    scorer = StageAScorer(policy)
    query = torch.tensor([0.9, 0.2, -0.1, 0.0], dtype=torch.float32)
    blocks = [
        _metadata(0, summary=(8, 1, 0, 0), last_access_step=7, attn_ema=0.1, priority=0.9),
        _metadata(1, summary=(6, 2, -1, 0), last_access_step=9, attn_ema=0.6, priority=0.2),
        _metadata(2, summary=(-8, 0, 0, 0), last_access_step=3, attn_ema=0.3, priority=0.6),
        _metadata(3, summary=(0, 8, 0, 0), last_access_step=10, attn_ema=0.9, priority=0.1),
    ]

    scored = scorer.score_all(blocks, query, current_step=10)
    reference = _reference_stage_a_scores(blocks, query, current_step=10, policy=policy)

    assert [int(item.block_id) for item in scored] == [item[0] for item in reference]
    for scored_item, reference_item in zip(scored, reference):
        assert abs(scored_item.approx_similarity_score - reference_item[1]) < 1e-6
        assert abs(scored_item.stage_a_score - reference_item[2]) < 1e-6


def _reference_stage_a_scores(
    blocks: list[BlockMetadata],
    query: torch.Tensor,
    *,
    current_step: int,
    policy: StageAPolicy,
) -> list[tuple[int, float, float]]:
    attn_values = [block.attn_ema for block in blocks]
    priority_values = [block.priority for block in blocks]
    attn_scores = _normalize(attn_values)
    priority_scores = _normalize(priority_values)
    rows: list[tuple[int, float, float, int, int]] = []
    for index, block in enumerate(blocks):
        similarity = approx_cosine_similarity(query, block)
        recency = 1.0 / (1.0 + float(current_step - block.last_access_step))
        score = (
            policy.weights.summary_similarity * similarity
            + policy.weights.recency * recency
            + policy.weights.attn_ema * attn_scores[index]
            + policy.weights.priority * priority_scores[index]
        )
        rows.append((int(block.block_id), similarity, score, block.last_access_step, block.token_start))
    rows.sort(key=lambda item: (item[2], item[3], item[4]), reverse=True)
    return [(item[0], item[1], item[2]) for item in rows]


def _normalize(values: list[float]) -> list[float]:
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def test_stage_a_supports_per_head_max_and_topk_modes() -> None:
    query = MultiHeadQuerySummary(
        pooled=SummaryEncoding(values=(1, 1), scale=1.0, summary_norm=2**0.5),
        per_head=(
            SummaryEncoding(values=(1, 0), scale=1.0, summary_norm=1.0),
            SummaryEncoding(values=(0, 1), scale=1.0, summary_norm=1.0),
        ),
    )
    blocks = [
        _per_head_metadata(0, per_head_summary=((8, 1), (1, 8))),
        _per_head_metadata(1, per_head_summary=((8, 0), (8, 0))),
    ]

    max_head_scores = StageAScorer(
        StageAPolicy(
            weights=StageAWeights(
                summary_similarity=1.0,
                recency=0.0,
                attn_ema=0.0,
                priority=0.0,
            ),
            shortlist_size=2,
            head_scoring_mode="max_head_score",
        )
    ).score_all(blocks, query, current_step=2)
    topk_scores = StageAScorer(
        StageAPolicy(
            weights=StageAWeights(
                summary_similarity=1.0,
                recency=0.0,
                attn_ema=0.0,
                priority=0.0,
            ),
            shortlist_size=2,
            head_scoring_mode="topk_head_mean",
            head_top_k=2,
        )
    ).score_all(blocks, query, current_step=2)

    assert int(max_head_scores[0].block_id) == 1
    assert int(topk_scores[0].block_id) == 0


def test_stage_a_supports_static_head_weights() -> None:
    query = MultiHeadQuerySummary(
        pooled=SummaryEncoding(values=(1, 1), scale=1.0, summary_norm=2**0.5),
        per_head=(
            SummaryEncoding(values=(1, 0), scale=1.0, summary_norm=1.0),
            SummaryEncoding(values=(0, 1), scale=1.0, summary_norm=1.0),
        ),
    )
    blocks = [
        _per_head_metadata(0, per_head_summary=((8, 0), (0, -8))),
        _per_head_metadata(1, per_head_summary=((0, -8), (0, 8))),
    ]
    scorer = StageAScorer(
        StageAPolicy(
            weights=StageAWeights(
                summary_similarity=1.0,
                recency=0.0,
                attn_ema=0.0,
                priority=0.0,
            ),
            shortlist_size=2,
            head_scoring_mode="weighted_head_mean",
            head_weights=(0.0, 1.0),
        )
    )

    scored = scorer.score_all(blocks, query, current_step=2)

    assert int(scored[0].block_id) == 1
