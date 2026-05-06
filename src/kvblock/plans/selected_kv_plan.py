"""Backend-consumable sparse KV selection plan."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping


TokenRange = tuple[int, int]


@dataclass(slots=True)
class SelectedKVPlan:
    """Standard selector output consumed by mapping and backend layers.

    Logical block ids describe the policy decision. Physical page ids are optional
    because package-level selectors should not depend on a serving runtime page table.
    Token ranges use half-open ``[start, end)`` offsets.
    """

    request_id: str | None = None
    layer_id: int | None = None
    step_id: int | None = None
    logical_block_ids: list[int] = field(default_factory=list)
    physical_page_ids: list[int] | None = None
    selected_token_ranges: list[TokenRange] = field(default_factory=list)
    recent_block_ids: list[int] = field(default_factory=list)
    anchor_block_ids: list[int] = field(default_factory=list)
    halo_block_ids: list[int] = field(default_factory=list)
    linked_block_ids: list[int] = field(default_factory=list)
    confidence: float = 1.0
    fallback_triggered: bool = False
    fallback_reason: str | None = None
    selector_name: str = "unknown"
    policy_name: str = "unknown"
    total_blocks: int = 0
    selected_blocks: int = 0
    total_tokens: int = 0
    selected_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.logical_block_ids = _int_list(self.logical_block_ids, "logical_block_ids")
        self.physical_page_ids = (
            None
            if self.physical_page_ids is None
            else _int_list(self.physical_page_ids, "physical_page_ids")
        )
        self.selected_token_ranges = [
            _coerce_token_range(item) for item in self.selected_token_ranges
        ]
        self.recent_block_ids = _int_list(self.recent_block_ids, "recent_block_ids")
        self.anchor_block_ids = _int_list(self.anchor_block_ids, "anchor_block_ids")
        self.halo_block_ids = _int_list(self.halo_block_ids, "halo_block_ids")
        self.linked_block_ids = _int_list(self.linked_block_ids, "linked_block_ids")
        if self.selected_blocks == 0 and self.logical_block_ids:
            self.selected_blocks = len(self.logical_block_ids)
        if self.selected_tokens == 0 and self.selected_token_ranges:
            self.selected_tokens = sum(
                end - start for start, end in self.selected_token_ranges
            )
        self.validate()

    @property
    def selected_block_fraction(self) -> float:
        """Fraction of logical blocks selected by this plan."""

        if self.total_blocks == 0:
            return 0.0
        return self.selected_blocks / self.total_blocks

    @property
    def selected_token_fraction(self) -> float:
        """Fraction of logical tokens covered by this plan."""

        if self.total_tokens == 0:
            return 0.0
        return self.selected_tokens / self.total_tokens

    @property
    def has_physical_mapping(self) -> bool:
        """Return true when logical blocks have been mapped to physical pages."""

        return self.physical_page_ids is not None

    def validate(self) -> "SelectedKVPlan":
        """Validate internal consistency and return ``self`` for chaining."""

        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be >= 0 when set")
        if self.step_id is not None and self.step_id < 0:
            raise ValueError("step_id must be >= 0 when set")
        if self.total_blocks < 0:
            raise ValueError("total_blocks must be >= 0")
        if self.selected_blocks < 0:
            raise ValueError("selected_blocks must be >= 0")
        if self.selected_blocks > self.total_blocks:
            raise ValueError("selected_blocks must be <= total_blocks")
        if self.selected_blocks != len(self.logical_block_ids):
            raise ValueError("selected_blocks must match logical_block_ids length")
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        if self.selected_tokens < 0:
            raise ValueError("selected_tokens must be >= 0")
        if self.selected_tokens > self.total_tokens:
            raise ValueError("selected_tokens must be <= total_tokens")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.physical_page_ids is not None and len(self.physical_page_ids) != len(
            self.logical_block_ids
        ):
            raise ValueError("physical_page_ids must match logical_block_ids length")
        if self.fallback_triggered and not self.fallback_reason:
            raise ValueError("fallback_reason is required when fallback_triggered is true")
        if self.fallback_reason and not self.fallback_reason.strip():
            raise ValueError("fallback_reason must be non-empty when set")
        for start, end in self.selected_token_ranges:
            if start < 0:
                raise ValueError("selected token range start must be >= 0")
            if end < start:
                raise ValueError("selected token range end must be >= start")
            if end > self.total_tokens:
                raise ValueError("selected token range end must be <= total_tokens")
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan into JSON-friendly primitives."""

        return {
            "request_id": self.request_id,
            "layer_id": self.layer_id,
            "step_id": self.step_id,
            "logical_block_ids": list(self.logical_block_ids),
            "physical_page_ids": (
                None if self.physical_page_ids is None else list(self.physical_page_ids)
            ),
            "selected_token_ranges": [
                [start, end] for start, end in self.selected_token_ranges
            ],
            "recent_block_ids": list(self.recent_block_ids),
            "anchor_block_ids": list(self.anchor_block_ids),
            "halo_block_ids": list(self.halo_block_ids),
            "linked_block_ids": list(self.linked_block_ids),
            "confidence": self.confidence,
            "fallback_triggered": self.fallback_triggered,
            "fallback_reason": self.fallback_reason,
            "selector_name": self.selector_name,
            "policy_name": self.policy_name,
            "total_blocks": self.total_blocks,
            "selected_blocks": self.selected_blocks,
            "total_tokens": self.total_tokens,
            "selected_tokens": self.selected_tokens,
            "metadata": dict(self.metadata),
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the plan as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SelectedKVPlan":
        """Rebuild a plan from :meth:`to_dict` output."""

        return cls(
            request_id=_optional_str(data.get("request_id")),
            layer_id=_optional_int(data.get("layer_id")),
            step_id=_optional_int(data.get("step_id")),
            logical_block_ids=list(data.get("logical_block_ids", ())),
            physical_page_ids=(
                None
                if data.get("physical_page_ids") is None
                else list(data.get("physical_page_ids", ()))
            ),
            selected_token_ranges=[
                _coerce_token_range(item)
                for item in data.get("selected_token_ranges", ())
            ],
            recent_block_ids=list(data.get("recent_block_ids", ())),
            anchor_block_ids=list(data.get("anchor_block_ids", ())),
            halo_block_ids=list(data.get("halo_block_ids", ())),
            linked_block_ids=list(data.get("linked_block_ids", ())),
            confidence=float(data.get("confidence", 1.0)),
            fallback_triggered=bool(data.get("fallback_triggered", False)),
            fallback_reason=_optional_str(data.get("fallback_reason")),
            selector_name=str(data.get("selector_name", "unknown")),
            policy_name=str(data.get("policy_name", "unknown")),
            total_blocks=int(data.get("total_blocks", 0)),
            selected_blocks=int(
                data.get("selected_blocks", len(data.get("logical_block_ids", ())))
            ),
            total_tokens=int(data.get("total_tokens", 0)),
            selected_tokens=int(
                data.get(
                    "selected_tokens",
                    sum(
                        _coerce_token_range(item)[1] - _coerce_token_range(item)[0]
                        for item in data.get("selected_token_ranges", ())
                    ),
                )
            ),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> "SelectedKVPlan":
        """Rebuild a plan from JSON produced by :meth:`to_json`."""

        return cls.from_dict(json.loads(payload))


def _int_list(values: list[int] | tuple[int, ...], name: str) -> list[int]:
    converted = [int(value) for value in values]
    if any(value < 0 for value in converted):
        raise ValueError(f"{name} must contain only values >= 0")
    return converted


def _coerce_token_range(value: Any) -> TokenRange:
    if len(value) != 2:
        raise ValueError("selected token ranges must contain two values")
    start, end = value
    return int(start), int(end)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
