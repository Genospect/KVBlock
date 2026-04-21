"""Block metadata structures for the initial V1 scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from kvblock.kv.block_types import BlockId, BlockReference, TokenSpan

DEFAULT_SUMMARY_DIM = 32


@dataclass(slots=True)
class BlockMetadata:
    """Lightweight, inspectable metadata for one KV block.

    Attributes:
        block_id: Stable identifier for the block within its pool.
        pool_id: Backing pool or page-table identifier that owns the payload.
        token_start: Inclusive token index covered by the block.
        token_len: Number of tokens covered by the block.
        precision_tier: Storage precision label for the payload, such as ``fp16`` or ``fp8``.
        flags: Integer bitfield reserved for runtime state and selector hints.
        summary_fp8: Quantized summary values for the block. V1 defaults to 32 elements.
        summary_scale: Dequantization scale for the stored low-precision summary values.
        sign_sketch: 64-bit sign sketch derived from the summary for cheap refinement.
        summary_norm: L2 norm of the unquantized summary before FP8-style emulation.
        attn_ema: Exponential moving average of observed attention mass for the block.
        attn_var: Variance estimate associated with the attention EMA.
        last_access_step: Decode step index of the most recent access.
        hit_count: Number of times this block was selected or touched.
        priority: Runtime priority score used for heuristic retention or promotion decisions.
        rope_bucket: Coarse position bucket used for RoPE-aware handling.
        fallback_miss_count: Number of sparse misses that later required fallback coverage.
    """

    block_id: BlockId = field(metadata={"doc": "Stable identifier for the block."})
    pool_id: int = field(metadata={"doc": "Pool/page-table identifier for the block payload."})
    token_start: int = field(metadata={"doc": "Inclusive token index of the block start."})
    token_len: int = field(metadata={"doc": "Number of tokens covered by the block."})
    precision_tier: str = field(
        default="fp16",
        metadata={"doc": "Payload precision tier such as fp16 or fp8."},
    )
    flags: int = field(
        default=0,
        metadata={"doc": "Integer bitfield reserved for runtime state flags."},
    )
    summary_fp8: tuple[int, ...] = field(
        default_factory=lambda: (0,) * DEFAULT_SUMMARY_DIM,
        metadata={"doc": "Quantized FP8-style summary vector for this block."},
    )
    summary_scale: float = field(
        default=1.0,
        metadata={"doc": "Per-summary scale for reconstructing approximate float values."},
    )
    sign_sketch: int = field(
        default=0,
        metadata={"doc": "64-bit sign sketch derived from the summary vector."},
    )
    summary_norm: float = field(
        default=0.0,
        metadata={"doc": "L2 norm of the pre-quantized summary vector."},
    )
    attn_ema: float = field(
        default=0.0,
        metadata={"doc": "Exponential moving average of block attention usage."},
    )
    attn_var: float = field(
        default=0.0,
        metadata={"doc": "Variance estimate for the attention EMA."},
    )
    last_access_step: int = field(
        default=-1,
        metadata={"doc": "Most recent decode step that accessed this block."},
    )
    hit_count: int = field(
        default=0,
        metadata={"doc": "Number of times the block was selected or accessed."},
    )
    priority: float = field(
        default=0.0,
        metadata={"doc": "Heuristic priority score for retention or routing."},
    )
    rope_bucket: int = field(
        default=0,
        metadata={"doc": "Coarse RoPE bucket used for position-aware handling."},
    )
    fallback_miss_count: int = field(
        default=0,
        metadata={"doc": "Number of sparse misses repaired by fallback."},
    )
    per_head_summary_fp8: tuple[tuple[int, ...], ...] = field(
        default_factory=tuple,
        metadata={"doc": "Optional per-head quantized summary vectors."},
    )
    per_head_summary_scale: tuple[float, ...] = field(
        default_factory=tuple,
        metadata={"doc": "Optional per-head summary dequantization scales."},
    )
    per_head_summary_norm: tuple[float, ...] = field(
        default_factory=tuple,
        metadata={"doc": "Optional per-head pre-quantized summary norms."},
    )

    def __post_init__(self) -> None:
        if self.pool_id < 0:
            raise ValueError(f"pool_id must be >= 0, got {self.pool_id!r}")
        if self.token_start < 0:
            raise ValueError(f"token_start must be >= 0, got {self.token_start!r}")
        if self.token_len <= 0:
            raise ValueError(f"token_len must be > 0, got {self.token_len!r}")
        if not self.precision_tier.strip():
            raise ValueError("precision_tier must be non-empty")
        if self.summary_scale <= 0:
            raise ValueError(f"summary_scale must be > 0, got {self.summary_scale!r}")
        if self.summary_norm < 0:
            raise ValueError(f"summary_norm must be >= 0, got {self.summary_norm!r}")
        if self.attn_var < 0:
            raise ValueError(f"attn_var must be >= 0, got {self.attn_var!r}")
        if self.hit_count < 0:
            raise ValueError(f"hit_count must be >= 0, got {self.hit_count!r}")
        if self.fallback_miss_count < 0:
            raise ValueError(
                "fallback_miss_count must be >= 0, "
                f"got {self.fallback_miss_count!r}"
            )
        if not self.summary_fp8:
            raise ValueError("summary_fp8 must not be empty")
        if any(value < -127 or value > 127 for value in self.summary_fp8):
            raise ValueError("summary_fp8 values must fit in the conservative FP8 emulation range")
        if self.per_head_summary_fp8 or self.per_head_summary_scale or self.per_head_summary_norm:
            head_count = len(self.per_head_summary_fp8)
            if head_count == 0:
                raise ValueError("per_head_summary_fp8 must be set when per-head scales/norms are set")
            if len(self.per_head_summary_scale) != head_count:
                raise ValueError("per_head_summary_scale length must match per_head_summary_fp8")
            if len(self.per_head_summary_norm) != head_count:
                raise ValueError("per_head_summary_norm length must match per_head_summary_fp8")
            expected_dim = len(self.summary_fp8)
            for values in self.per_head_summary_fp8:
                if len(values) != expected_dim:
                    raise ValueError("per-head summaries must match summary_fp8 dimension")
                if any(value < -127 or value > 127 for value in values):
                    raise ValueError("per_head_summary_fp8 values must fit in the conservative FP8 emulation range")
            if any(scale <= 0 for scale in self.per_head_summary_scale):
                raise ValueError("per_head_summary_scale values must be > 0")
            if any(norm < 0 for norm in self.per_head_summary_norm):
                raise ValueError("per_head_summary_norm values must be >= 0")

    @property
    def token_span(self) -> TokenSpan:
        """Return the block's token coverage as a typed span."""

        return TokenSpan(token_start=self.token_start, token_len=self.token_len)

    @property
    def block_ref(self) -> BlockReference:
        """Return a minimal typed reference to this block."""

        return BlockReference(
            block_id=self.block_id,
            pool_id=self.pool_id,
            token_span=self.token_span,
        )

    def dequantize_summary(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Reconstruct the stored low-precision summary as an approximate float tensor."""

        summary = torch.tensor(self.summary_fp8, dtype=torch.float32, device=device)
        return summary.mul(self.summary_scale).to(dtype=dtype)

    def dequantize_per_head_summary(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Return optional per-head summaries as ``[heads, summary_dim]``."""

        if not self.per_head_summary_fp8:
            raise ValueError("per-head summaries are not available for this block")
        summaries = torch.tensor(
            self.per_head_summary_fp8,
            dtype=torch.float32,
            device=device,
        )
        scales = torch.tensor(
            self.per_head_summary_scale,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(1)
        return summaries.mul(scales).to(dtype=dtype)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata into JSON-friendly Python primitives."""

        return {
            "block_id": int(self.block_id),
            "pool_id": self.pool_id,
            "token_start": self.token_start,
            "token_len": self.token_len,
            "precision_tier": self.precision_tier,
            "flags": self.flags,
            "summary_fp8": list(self.summary_fp8),
            "summary_scale": self.summary_scale,
            "sign_sketch": self.sign_sketch,
            "summary_norm": self.summary_norm,
            "attn_ema": self.attn_ema,
            "attn_var": self.attn_var,
            "last_access_step": self.last_access_step,
            "hit_count": self.hit_count,
            "priority": self.priority,
            "rope_bucket": self.rope_bucket,
            "fallback_miss_count": self.fallback_miss_count,
            "per_head_summary_fp8": [
                list(values) for values in self.per_head_summary_fp8
            ],
            "per_head_summary_scale": list(self.per_head_summary_scale),
            "per_head_summary_norm": list(self.per_head_summary_norm),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BlockMetadata":
        """Rebuild metadata from a dict previously produced by :meth:`to_dict`."""

        return cls(
            block_id=BlockId(int(data["block_id"])),
            pool_id=int(data["pool_id"]),
            token_start=int(data["token_start"]),
            token_len=int(data["token_len"]),
            precision_tier=str(data.get("precision_tier", "fp16")),
            flags=int(data.get("flags", 0)),
            summary_fp8=tuple(
                int(value)
                for value in data.get("summary_fp8", (0,) * DEFAULT_SUMMARY_DIM)
            ),
            summary_scale=float(data.get("summary_scale", 1.0)),
            sign_sketch=int(data.get("sign_sketch", 0)),
            summary_norm=float(data.get("summary_norm", 0.0)),
            attn_ema=float(data.get("attn_ema", 0.0)),
            attn_var=float(data.get("attn_var", 0.0)),
            last_access_step=int(data.get("last_access_step", -1)),
            hit_count=int(data.get("hit_count", 0)),
            priority=float(data.get("priority", 0.0)),
            rope_bucket=int(data.get("rope_bucket", 0)),
            fallback_miss_count=int(data.get("fallback_miss_count", 0)),
            per_head_summary_fp8=tuple(
                tuple(int(value) for value in values)
                for values in data.get("per_head_summary_fp8", ())
            ),
            per_head_summary_scale=tuple(
                float(value) for value in data.get("per_head_summary_scale", ())
            ),
            per_head_summary_norm=tuple(
                float(value) for value in data.get("per_head_summary_norm", ())
            ),
        )
