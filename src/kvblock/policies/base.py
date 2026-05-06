"""Package-facing KVBlock policy model."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class KVBlockPolicy:
    """User-facing policy configuration for KV block selection."""

    name: str = "default"
    block_size: int = 32
    stride: int | None = None
    selector: str = "qk_blockmax"
    representation_source: str = "query_mean_last_layer"
    qk_aggregation: str = "block_max"
    shortlist_m: int = 24
    semantic_k: int = 8
    halo: int = 2
    keep_recent_blocks: int = 4
    keep_anchor_blocks: bool = True
    fallback_mode: str = "confidence_guarded"
    fallback_margin: float = 0.05
    max_selected_fraction: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be > 0 when set")
        if not self.selector.strip():
            raise ValueError("selector must be non-empty")
        if self.shortlist_m <= 0:
            raise ValueError("shortlist_m must be > 0")
        if self.semantic_k < 0:
            raise ValueError("semantic_k must be >= 0")
        if self.halo < 0:
            raise ValueError("halo must be >= 0")
        if self.keep_recent_blocks < 0:
            raise ValueError("keep_recent_blocks must be >= 0")
        if self.fallback_margin < 0:
            raise ValueError("fallback_margin must be >= 0")
        if self.max_selected_fraction is not None and not (
            0.0 < self.max_selected_fraction <= 1.0
        ):
            raise ValueError("max_selected_fraction must be in (0, 1] when set")

    def resolve(self) -> "KVBlockPolicy":
        """Return a fully resolved policy with implicit defaults made explicit."""

        return replace(
            self,
            stride=self.block_size if self.stride is None else self.stride,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this policy into JSON/YAML-friendly primitives."""

        return {
            "name": self.name,
            "block_size": self.block_size,
            "stride": self.stride,
            "selector": self.selector,
            "representation_source": self.representation_source,
            "qk_aggregation": self.qk_aggregation,
            "shortlist_m": self.shortlist_m,
            "semantic_k": self.semantic_k,
            "halo": self.halo,
            "keep_recent_blocks": self.keep_recent_blocks,
            "keep_anchor_blocks": self.keep_anchor_blocks,
            "fallback_mode": self.fallback_mode,
            "fallback_margin": self.fallback_margin,
            "max_selected_fraction": self.max_selected_fraction,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "KVBlockPolicy":
        """Build a policy from a mapping."""

        return cls(
            name=str(data.get("name", "default")),
            block_size=int(data.get("block_size", 32)),
            stride=None if data.get("stride") is None else int(data["stride"]),
            selector=str(data.get("selector", "qk_blockmax")),
            representation_source=str(
                data.get("representation_source", "query_mean_last_layer")
            ),
            qk_aggregation=str(data.get("qk_aggregation", "block_max")),
            shortlist_m=int(data.get("shortlist_m", 24)),
            semantic_k=int(data.get("semantic_k", 8)),
            halo=int(data.get("halo", 2)),
            keep_recent_blocks=int(data.get("keep_recent_blocks", 4)),
            keep_anchor_blocks=bool(data.get("keep_anchor_blocks", True)),
            fallback_mode=str(data.get("fallback_mode", "confidence_guarded")),
            fallback_margin=float(data.get("fallback_margin", 0.05)),
            max_selected_fraction=(
                None
                if data.get("max_selected_fraction") is None
                else float(data["max_selected_fraction"])
            ),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "KVBlockPolicy":
        """Load a policy from a YAML file."""

        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is optional at runtime
            raise ImportError("Install KVBlock with the yaml extra to load policies") from exc

        payload = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("policy YAML must contain a mapping")
        return cls.from_dict(payload)
