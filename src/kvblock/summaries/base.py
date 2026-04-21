"""Base summary and sketch protocols for the V1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import torch


@dataclass(frozen=True, slots=True)
class SummaryEncoding:
    """Quantized summary representation used by metadata and sketch builders."""

    values: tuple[int, ...]
    scale: float
    summary_norm: float

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("SummaryEncoding.values must not be empty")
        if self.scale <= 0:
            raise ValueError(f"scale must be > 0, got {self.scale!r}")
        if self.summary_norm < 0:
            raise ValueError(
                f"summary_norm must be >= 0, got {self.summary_norm!r}"
            )
        if any(value < -127 or value > 127 for value in self.values):
            raise ValueError("values must fit in the conservative FP8 emulation range")

    def dequantize(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Convert the quantized summary back into an approximate float tensor."""

        tensor = torch.tensor(self.values, dtype=torch.float32, device=device)
        return tensor.mul(self.scale).to(dtype=dtype)


@dataclass(frozen=True, slots=True)
class MultiHeadQuerySummary:
    """Pooled query summary plus optional per-head query summaries."""

    pooled: SummaryEncoding
    per_head: tuple[SummaryEncoding, ...]

    def __post_init__(self) -> None:
        if not self.per_head:
            raise ValueError("MultiHeadQuerySummary.per_head must not be empty")
        expected_dim = len(self.pooled.values)
        if any(len(head.values) != expected_dim for head in self.per_head):
            raise ValueError("all per-head query summaries must match pooled dimension")

    def dequantize(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return the pooled query summary for compatibility with existing code."""

        return self.pooled.dequantize(dtype=dtype, device=device)

    def dequantize_heads(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return per-head query summaries as ``[heads, summary_dim]``."""

        return torch.stack(
            [head.dequantize(dtype=dtype, device=device) for head in self.per_head]
        )


class SummaryBuilder(Protocol):
    """Protocol for deterministic block summary builders."""

    summary_dim: int

    def build(self, block_states: torch.Tensor) -> SummaryEncoding:
        """Build a quantized summary for one block."""


class SketchBuilder(Protocol):
    """Protocol for sign-sketch builders."""

    sketch_bits: int

    def build(self, summary: SummaryEncoding | Sequence[float] | torch.Tensor) -> int:
        """Build an integer sketch from a block summary."""
