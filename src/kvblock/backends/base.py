"""Backend protocol for dense and sparse decode attention."""

from __future__ import annotations

from typing import Any, Protocol

import torch

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan


class KVAttentionBackend(Protocol):
    """Runtime-independent backend interface for one decode attention step."""

    name: str

    def run_decode(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        plan: SelectedKVPlan,
        layout: BlockLayout,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run one decode attention step."""
