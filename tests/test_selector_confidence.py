from kvblock.kv.block_types import BlockId
from kvblock.kv.metadata import BlockMetadata
from kvblock.selector.base import ScoredBlock
from kvblock.selector.confidence import (
    ConfidenceEvaluator,
    normalized_score_margin,
    normalized_selected_mass,
    score_margin,
)
from kvblock.selector.policies import ConfidencePolicy


def _candidate(block_id: int, score: float) -> ScoredBlock:
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
        ),
        approx_similarity_score=score,
        final_score=score,
        stage_a_score=score,
    )


def test_confidence_uses_score_margin_threshold() -> None:
    ranked = [_candidate(0, 0.9), _candidate(1, 0.7), _candidate(2, 0.64), _candidate(3, 0.6)]
    evaluator = ConfidenceEvaluator(ConfidencePolicy(margin_threshold=0.05))

    assessment = evaluator.assess(ranked, included_count=3)

    assert abs(score_margin(ranked, included_count=3) - 0.04) < 1e-9
    assert assessment.is_confident is False


def test_confidence_optional_normalized_mass_helper() -> None:
    ranked = [_candidate(0, 1.0), _candidate(1, 0.95), _candidate(2, 0.08), _candidate(3, 0.02)]
    evaluator = ConfidenceEvaluator(
        ConfidencePolicy(margin_threshold=0.01, min_normalized_mass=0.8)
    )

    assessment = evaluator.assess(ranked, included_count=2)

    assert normalized_selected_mass(ranked, included_count=2) > 0.8
    assert assessment.normalized_mass is not None
    assert assessment.is_confident is True


def test_confidence_supports_normalized_margin_threshold() -> None:
    ranked = [_candidate(0, 10.0), _candidate(1, 9.0), _candidate(2, 8.7)]
    evaluator = ConfidenceEvaluator(
        ConfidencePolicy(
            margin_threshold=0.5,
            normalized_margin_threshold=0.02,
        )
    )

    assessment = evaluator.assess(ranked, included_count=2)

    assert score_margin(ranked, included_count=2) == 0.3000000000000007
    assert normalized_score_margin(ranked, included_count=2) is not None
    assert assessment.normalized_margin is not None
    assert assessment.normalized_margin > 0.02
    assert assessment.is_confident is False


def test_confidence_margin_tuning_can_relax_fallback_trigger() -> None:
    ranked = [_candidate(0, 0.90), _candidate(1, 0.86), _candidate(2, 0.83)]

    strict = ConfidenceEvaluator(ConfidencePolicy(margin_threshold=0.05)).assess(
        ranked, included_count=1
    )
    loose = ConfidenceEvaluator(ConfidencePolicy(margin_threshold=0.03)).assess(
        ranked, included_count=1
    )

    assert strict.is_confident is False
    assert loose.is_confident is True
