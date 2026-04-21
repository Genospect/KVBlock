import torch

from kvblock.summaries.base import MultiHeadQuerySummary, SummaryEncoding
from kvblock.summaries.sign_sketch import generate_sign_sketch, hamming_similarity


def test_generate_sign_sketch_is_deterministic() -> None:
    summary = torch.tensor([0.1, -0.4, 0.8, 0.2], dtype=torch.float32)

    left = generate_sign_sketch(summary)
    right = generate_sign_sketch(summary)

    assert left == right
    assert hamming_similarity(left, right) == 1.0


def test_hamming_similarity_orders_nearby_vectors_above_opposites() -> None:
    anchor = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float32)
    nearby = torch.tensor([0.9, -1.8, 0.55, 2.8], dtype=torch.float32)
    opposite = -anchor

    anchor_sketch = generate_sign_sketch(anchor)
    nearby_sketch = generate_sign_sketch(nearby)
    opposite_sketch = generate_sign_sketch(opposite)

    assert hamming_similarity(anchor_sketch, nearby_sketch) > hamming_similarity(
        anchor_sketch, opposite_sketch
    )


def test_generate_sign_sketch_uses_pooled_multi_head_summary() -> None:
    pooled = SummaryEncoding(values=(1, -2, 3), scale=0.5, summary_norm=2.0)
    multi_head = MultiHeadQuerySummary(
        pooled=pooled,
        per_head=(
            SummaryEncoding(values=(1, 0, 0), scale=1.0, summary_norm=1.0),
            SummaryEncoding(values=(0, 1, 0), scale=1.0, summary_norm=1.0),
        ),
    )

    assert generate_sign_sketch(multi_head) == generate_sign_sketch(pooled)
