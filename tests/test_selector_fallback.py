from kvblock.selector.confidence import ConfidenceAssessment
from kvblock.selector.fallback import GradedFallbackController
from kvblock.selector.policies import FallbackPolicy


def test_fallback_escalates_in_required_order() -> None:
    controller = GradedFallbackController(
        FallbackPolicy(widen_top_k_by=4, add_recent_blocks_by=2, allow_dense_fallback=True)
    )
    weak = ConfidenceAssessment(is_confident=False, score_margin=0.01)

    first = controller.decide(
        weak,
        current_top_k=8,
        current_keep_recent_blocks=4,
        base_top_k=8,
        base_keep_recent_blocks=4,
    )
    second = controller.decide(
        weak,
        current_top_k=first.next_top_k,
        current_keep_recent_blocks=4,
        base_top_k=8,
        base_keep_recent_blocks=4,
    )
    third = controller.decide(
        weak,
        current_top_k=first.next_top_k,
        current_keep_recent_blocks=second.next_keep_recent_blocks,
        base_top_k=8,
        base_keep_recent_blocks=4,
    )

    assert first.action == "widen_k"
    assert first.next_top_k == 12
    assert second.action == "add_recent_blocks"
    assert second.next_keep_recent_blocks == 6
    assert third.action == "dense_fallback"
    assert third.use_dense_fallback is True


def test_fallback_keeps_sparse_when_confident() -> None:
    controller = GradedFallbackController()
    strong = ConfidenceAssessment(is_confident=True, score_margin=0.2)

    decision = controller.decide(
        strong,
        current_top_k=8,
        current_keep_recent_blocks=4,
        base_top_k=8,
        base_keep_recent_blocks=4,
    )

    assert decision.action == "keep_sparse"
    assert decision.use_dense_fallback is False
