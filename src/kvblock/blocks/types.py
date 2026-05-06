"""Public block span types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class BlockSpan:
    """Half-open token span for one logical KV block."""

    block_id: int
    start_token: int
    end_token: int

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise ValueError("block_id must be >= 0")
        if self.start_token < 0:
            raise ValueError("start_token must be >= 0")
        if self.end_token < self.start_token:
            raise ValueError("end_token must be >= start_token")

    @property
    def token_len(self) -> int:
        """Number of tokens covered by this block."""

        return self.end_token - self.start_token

    def overlaps(self, start_token: int, end_token: int) -> bool:
        """Return true when this block overlaps a half-open token range."""

        return self.start_token < end_token and start_token < self.end_token

    def contains(self, token_index: int) -> bool:
        """Return true when ``token_index`` falls inside the span."""

        return self.start_token <= token_index < self.end_token
