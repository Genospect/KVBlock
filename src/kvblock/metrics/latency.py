"""Latency helpers for decode-step timing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class DecodeLatencyBreakdown:
    """Named decode-step latency buckets in milliseconds."""

    selector_ms: float = 0.0
    mapping_ms: float = 0.0
    backend_attention_ms: float = 0.0
    total_ms: float = 0.0


def time_cuda_callable(fn: Callable[[], T]) -> tuple[T, float]:
    """Time a callable with CUDA events and return ``(result, elapsed_ms)``."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA event timing requires torch.cuda.is_available()")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))
