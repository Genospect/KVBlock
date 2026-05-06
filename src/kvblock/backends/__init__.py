"""Attention backends for correctness and experimental acceleration."""

from kvblock.backends.base import KVAttentionBackend
from kvblock.backends.torch_dense import TorchDenseAttentionBackend, dense_decode_attention
from kvblock.backends.torch_sparse_reference import (
    TorchSparseReferenceBackend,
    sparse_reference_decode_attention,
)

__all__ = [
    "KVAttentionBackend",
    "TorchDenseAttentionBackend",
    "TorchSparseReferenceBackend",
    "dense_decode_attention",
    "sparse_reference_decode_attention",
]
