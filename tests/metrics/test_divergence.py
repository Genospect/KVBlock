import torch

from kvblock.metrics import (
    kl_divergence_logits,
    max_abs_logit_diff,
    topk_agreement,
    topk_overlap,
)


def test_divergence_metrics_identical_logits() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]])

    assert kl_divergence_logits(logits, logits) == 0.0
    assert topk_agreement(logits, logits, k=1) == 1.0
    assert topk_overlap(logits, logits, k=2) == 1.0
    assert max_abs_logit_diff(logits, logits) == 0.0
