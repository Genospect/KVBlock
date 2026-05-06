"""Package-facing selector protocol."""

from __future__ import annotations

from typing import Any, Protocol

from kvblock.blocks import BlockLayout
from kvblock.plans import SelectedKVPlan
from kvblock.policies import KVBlockPolicy


class BaseSelector(Protocol):
    """Selector interface that returns a standardized selected KV plan."""

    name: str

    def select(
        self,
        query_state: Any,
        kv_metadata: Any,
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        """Select logical KV blocks for one decode step."""
