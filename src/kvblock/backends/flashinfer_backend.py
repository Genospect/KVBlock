"""Experimental FlashInfer backend placeholder."""

from __future__ import annotations

from typing import Any

import torch

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


class FlashInferBackend:
    """Lazy FlashInfer backend stub for physical sparse decode experiments."""

    name = "flashinfer"

    def __init__(self) -> None:
        self.flashinfer = _import_flashinfer()

    def run_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: SelectedKVPlan,
        layout: BlockLayout,
        **kwargs: Any,
    ) -> torch.Tensor:
        raise NotImplementedError(
            "FlashInfer sparse decode wiring is experimental. Use the POC script "
            "to check availability before adding API-specific kernels."
        )


def is_flashinfer_available() -> bool:
    """Return whether FlashInfer can be imported in this environment."""

    try:
        _import_flashinfer()
    except ImportError:
        return False
    return True


def _import_flashinfer() -> Any:
    try:
        import flashinfer
    except ImportError as exc:
        raise ImportError(
            "FlashInfer is optional. Install KVBlock with the flashinfer extra "
            "or install flashinfer-python in an environment that supports it."
        ) from exc
    return flashinfer
