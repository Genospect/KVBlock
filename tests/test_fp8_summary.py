import torch

from kvblock.summaries.fp8_summary import FP8SummaryBuilder


def test_fp8_summary_builder_is_deterministic_and_fixed_width() -> None:
    builder = FP8SummaryBuilder(summary_dim=8)
    block_states = torch.arange(0, 48, dtype=torch.float32).reshape(6, 8)

    first = builder.build(block_states)
    second = builder.build(block_states)

    assert first == second
    assert len(first.values) == 8
    assert first.scale > 0
    assert first.summary_norm > 0


def test_fp8_summary_builder_handles_hidden_dim_mismatch() -> None:
    builder = FP8SummaryBuilder(summary_dim=5)
    block_states = torch.arange(0, 24, dtype=torch.float32).reshape(3, 8)

    encoding = builder.build(block_states)
    restored = builder.dequantize(encoding)

    assert len(encoding.values) == 5
    assert restored.shape == (5,)


def test_fp8_summary_builder_zero_input_stays_zero() -> None:
    builder = FP8SummaryBuilder(summary_dim=4)
    encoding = builder.build(torch.zeros(2, 16, dtype=torch.float32))

    assert encoding.values == (0, 0, 0, 0)
    assert encoding.scale > 0
    assert encoding.summary_norm == 0.0


def test_fp8_summary_builder_scale_reconstructs_expected_values() -> None:
    builder = FP8SummaryBuilder(summary_dim=4)
    block_states = torch.tensor([[1.0, -2.0, 0.5, 0.0]], dtype=torch.float32)

    encoding = builder.build(block_states)
    restored = builder.dequantize(encoding)
    expected = torch.tensor(encoding.values, dtype=torch.float32) * encoding.scale

    assert torch.allclose(restored, expected)
