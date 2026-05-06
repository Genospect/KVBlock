"""Experimental Triton backend placeholder."""

from __future__ import annotations


class TritonSparseGatherBackend:
    """Placeholder for indexed sparse K/V block gathering experiments."""

    name = "triton_sparse_gather"

    def __init__(self) -> None:
        try:
            import triton  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Triton is optional. Install KVBlock with the triton extra to use "
                "experimental Triton kernels."
            ) from exc
