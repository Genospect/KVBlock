"""Deterministic FP8-style summary generation for the V1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from kvblock.summaries.base import SummaryBuilder, SummaryEncoding

FP8_EMULATION_MAX = 127.0


@dataclass(slots=True)
class FP8SummaryBuilder(SummaryBuilder):
    """Build a small deterministic summary with conservative FP8-style quantization.

    The implementation intentionally uses simple mean pooling and adaptive pooling so
    the scaffold stays easy to inspect and easy to swap out later.
    """

    summary_dim: int = 32
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.summary_dim <= 0:
            raise ValueError(f"summary_dim must be > 0, got {self.summary_dim!r}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps!r}")

    def build(self, block_states: torch.Tensor) -> SummaryEncoding:
        """Produce a fixed-width summary for one block.

        ``block_states`` may be 1D or have arbitrary leading dimensions, but the last
        dimension is always interpreted as the feature dimension.
        """

        summary = self._project(block_states)
        summary_norm = float(torch.linalg.vector_norm(summary).item())
        max_abs = float(summary.abs().max().item())
        scale = max(max_abs / FP8_EMULATION_MAX, self.eps)
        quantized = torch.round(summary / scale).clamp(
            -FP8_EMULATION_MAX, FP8_EMULATION_MAX
        )

        return SummaryEncoding(
            values=tuple(int(value) for value in quantized.to(torch.int8).tolist()),
            scale=scale,
            summary_norm=summary_norm,
        )

    def _project(self, block_states: torch.Tensor) -> torch.Tensor:
        if not isinstance(block_states, torch.Tensor):
            raise TypeError("block_states must be a torch.Tensor")
        if block_states.ndim == 0:
            raise ValueError("block_states must have at least one dimension")

        features = block_states.detach().to(dtype=torch.float32)
        if features.ndim == 1:
            pooled = features
        else:
            pooled = features.reshape(-1, features.shape[-1]).mean(dim=0)

        projected = F.adaptive_avg_pool1d(
            pooled.reshape(1, 1, -1), self.summary_dim
        ).reshape(-1)
        return projected.contiguous()

    def dequantize(
        self,
        encoding: SummaryEncoding,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper for reconstructing an approximate float summary."""

        return encoding.dequantize(dtype=dtype, device=device)
