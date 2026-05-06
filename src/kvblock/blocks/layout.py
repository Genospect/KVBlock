"""Logical block layout utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from kvblock.blocks.types import BlockSpan


@dataclass(frozen=True, slots=True)
class BlockLayout:
    """Token-to-logical-block layout independent of runtime page tables."""

    block_size: int
    stride: int
    spans: tuple[BlockSpan, ...]
    total_tokens: int

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        seen: set[int] = set()
        for span in self.spans:
            if span.block_id in seen:
                raise ValueError("block ids must be unique")
            if span.end_token > self.total_tokens:
                raise ValueError("block spans must not exceed total_tokens")
            seen.add(span.block_id)

    @classmethod
    def from_token_count(
        cls,
        *,
        total_tokens: int,
        block_size: int,
        stride: int | None = None,
    ) -> "BlockLayout":
        """Create fixed-size or strided logical blocks for a token count."""

        if total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if block_size <= 0:
            raise ValueError("block_size must be > 0")
        resolved_stride = block_size if stride is None else stride
        if resolved_stride <= 0:
            raise ValueError("stride must be > 0")

        spans: list[BlockSpan] = []
        start = 0
        block_id = 0
        while start < total_tokens:
            spans.append(
                BlockSpan(
                    block_id=block_id,
                    start_token=start,
                    end_token=min(start + block_size, total_tokens),
                )
            )
            block_id += 1
            start += resolved_stride
        return cls(
            block_size=block_size,
            stride=resolved_stride,
            spans=tuple(spans),
            total_tokens=total_tokens,
        )

    @property
    def block_count(self) -> int:
        """Number of logical blocks in this layout."""

        return len(self.spans)

    def get_block(self, block_id: int) -> BlockSpan:
        """Return one block span by id."""

        normalized = int(block_id)
        for span in self.spans:
            if span.block_id == normalized:
                return span
        raise KeyError(f"unknown block_id: {block_id!r}")

    def token_range_to_block_ids(self, start_token: int, end_token: int) -> tuple[int, ...]:
        """Return ids for blocks overlapping a half-open token range."""

        if start_token < 0:
            raise ValueError("start_token must be >= 0")
        if end_token < start_token:
            raise ValueError("end_token must be >= start_token")
        if end_token > self.total_tokens:
            raise ValueError("end_token must be <= total_tokens")
        if start_token == end_token:
            return ()
        return tuple(
            span.block_id for span in self.spans if span.overlaps(start_token, end_token)
        )

    def get_recent_blocks(self, n: int) -> tuple[int, ...]:
        """Return the last ``n`` logical blocks by token order."""

        if n <= 0:
            return ()
        return tuple(span.block_id for span in self.spans[-n:])

    def get_halo(self, block_ids: Iterable[int], halo: int) -> tuple[int, ...]:
        """Return selected block ids expanded by neighboring block ids."""

        if halo < 0:
            raise ValueError("halo must be >= 0")
        valid_ids = {span.block_id for span in self.spans}
        expanded: set[int] = set()
        for block_id in block_ids:
            normalized = int(block_id)
            if normalized not in valid_ids:
                raise KeyError(f"unknown block_id: {block_id!r}")
            for candidate in range(normalized - halo, normalized + halo + 1):
                if candidate in valid_ids:
                    expanded.add(candidate)
        return tuple(sorted(expanded))

    def token_ranges_for_blocks(self, block_ids: Iterable[int]) -> list[tuple[int, int]]:
        """Return token ranges for logical block ids in input order."""

        return [
            (span.start_token, span.end_token)
            for span in (self.get_block(int(block_id)) for block_id in block_ids)
        ]
