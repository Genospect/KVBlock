"""KV block metadata and typed block identifiers."""

from kvblock.kv.block_manager import (
    BlockIngestConfig,
    BlockIngestResult,
    build_block_metadata_from_representations,
    split_token_blocks,
)
from kvblock.kv.block_types import BlockId, BlockRef, BlockReference, TokenSpan
from kvblock.kv.metadata import BlockMetadata

__all__ = [
    "BlockId",
    "BlockIngestConfig",
    "BlockIngestResult",
    "BlockMetadata",
    "BlockRef",
    "BlockReference",
    "TokenSpan",
    "build_block_metadata_from_representations",
    "split_token_blocks",
]
