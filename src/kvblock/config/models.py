"""Typed configuration models for the initial V1 scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Self

VALID_PAYLOAD_PRECISIONS = {"fp16", "fp8"}
VALID_RUNTIME_BACKENDS = {"flashinfer", "vllm", "mock"}


def _ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")


def _ensure_non_negative(name: str, value: int | float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _ensure_keys_match(model_cls: type[Any], data: Mapping[str, Any]) -> None:
    valid_names = {item.name for item in fields(model_cls)}
    unknown = sorted(set(data) - valid_names)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown fields for {model_cls.__name__}: {joined}")


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Static model-side settings used by the V1 scaffold."""

    model_name: str = "unknown"
    block_size: int = 32
    summary_dim: int = 32
    payload_precision: str = "fp16"
    max_context_tokens: int = 32768
    rope_aware_summaries: bool = True

    def __post_init__(self) -> None:
        _ensure_positive("block_size", self.block_size)
        _ensure_positive("summary_dim", self.summary_dim)
        _ensure_positive("max_context_tokens", self.max_context_tokens)
        if self.payload_precision not in VALID_PAYLOAD_PRECISIONS:
            raise ValueError(
                "payload_precision must be one of "
                f"{sorted(VALID_PAYLOAD_PRECISIONS)}, got {self.payload_precision!r}"
            )
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        _ensure_keys_match(cls, data)
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SelectorConfig:
    """Heuristic selector defaults for the V1 research baseline."""

    keep_recent_blocks: int = 4
    keep_anchor_blocks: int = 2
    stage_a_shortlist: int = 24
    stage_a_long_context_shortlist: int = 48
    final_top_k: int = 8
    final_top_k_long_context: int = 16
    long_context_threshold: int = 32768
    confidence_margin: float = 0.05
    confidence_normalized_margin: float | None = None
    sign_sketch_bits: int = 64
    allow_dense_fallback: bool = True

    def __post_init__(self) -> None:
        _ensure_non_negative("keep_recent_blocks", self.keep_recent_blocks)
        _ensure_non_negative("keep_anchor_blocks", self.keep_anchor_blocks)
        _ensure_positive("stage_a_shortlist", self.stage_a_shortlist)
        _ensure_positive(
            "stage_a_long_context_shortlist", self.stage_a_long_context_shortlist
        )
        _ensure_positive("final_top_k", self.final_top_k)
        _ensure_positive("final_top_k_long_context", self.final_top_k_long_context)
        _ensure_positive("long_context_threshold", self.long_context_threshold)
        _ensure_non_negative("confidence_margin", self.confidence_margin)
        if self.confidence_normalized_margin is not None:
            _ensure_non_negative(
                "confidence_normalized_margin", self.confidence_normalized_margin
            )
        _ensure_positive("sign_sketch_bits", self.sign_sketch_bits)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        _ensure_keys_match(cls, data)
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class BenchmarkConfig:
    """Instrumentation-oriented defaults for early benchmarking."""

    dense_refresh_interval: int = 32
    record_access_traces: bool = True
    record_runtime_metrics: bool = True
    record_hardware_metrics: bool = True
    output_dir: str = "artifacts/benchmarks"

    def __post_init__(self) -> None:
        _ensure_positive("dense_refresh_interval", self.dense_refresh_interval)
        if not self.output_dir.strip():
            raise ValueError("output_dir must be non-empty")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        _ensure_keys_match(cls, data)
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """Execution/runtime settings for adapters and reproducibility."""

    backend: str = "flashinfer"
    device: str = "cpu"
    seed: int = 0
    torch_dtype: str = "float16"
    enable_profiling: bool = False

    def __post_init__(self) -> None:
        if self.backend not in VALID_RUNTIME_BACKENDS:
            raise ValueError(
                f"backend must be one of {sorted(VALID_RUNTIME_BACKENDS)}, "
                f"got {self.backend!r}"
            )
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not self.torch_dtype.strip():
            raise ValueError("torch_dtype must be non-empty")
        _ensure_non_negative("seed", self.seed)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        _ensure_keys_match(cls, data)
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class KVBlockConfig:
    """Top-level configuration bundle used by loaders and tests."""

    model: ModelConfig = field(default_factory=ModelConfig)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> Self:
        valid_names = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - valid_names)
        if unknown:
            joined = ", ".join(unknown)
            raise ValueError(f"Unknown top-level config sections: {joined}")

        return cls(
            model=ModelConfig.from_mapping(data.get("model", {})),
            selector=SelectorConfig.from_mapping(data.get("selector", {})),
            benchmark=BenchmarkConfig.from_mapping(data.get("benchmark", {})),
            runtime=RuntimeConfig.from_mapping(data.get("runtime", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
