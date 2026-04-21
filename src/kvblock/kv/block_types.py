"""Small typed structures for V1 KV block bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, order=True)
class BlockId:
    """Stable block identifier within a pool."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError(f"BlockId must be >= 0, got {self.value!r}")

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class TokenSpan:
    """Inclusive token start plus span length for one block."""

    token_start: int
    token_len: int

    def __post_init__(self) -> None:
        if self.token_start < 0:
            raise ValueError(f"token_start must be >= 0, got {self.token_start!r}")
        if self.token_len <= 0:
            raise ValueError(f"token_len must be > 0, got {self.token_len!r}")

    @property
    def token_end(self) -> int:
        """Exclusive end offset for the covered token range."""

        return self.token_start + self.token_len

    def contains(self, token_index: int) -> bool:
        """Return True when ``token_index`` falls inside the span."""

        return self.token_start <= token_index < self.token_end


@dataclass(frozen=True, slots=True)
class BlockReference:
    """Minimal reference to a block and its token coverage."""

    block_id: BlockId
    pool_id: int
    token_span: TokenSpan

    def __post_init__(self) -> None:
        if self.pool_id < 0:
            raise ValueError(f"pool_id must be >= 0, got {self.pool_id!r}")


BlockRef = BlockReference
