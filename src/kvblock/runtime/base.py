"""Runtime backend contracts for dense-only V1 block-ingest experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class RuntimeLoadConfig:
    """Configuration needed to load a small local causal LM backend."""

    model_name: str = "sshleifer/tiny-gpt2"
    device: str = "cpu"
    torch_dtype: str = "float32"
    device_map: str | None = None
    local_files_only: bool = False
    trust_remote_code: bool = False
    max_length: int | None = None

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not self.torch_dtype.strip():
            raise ValueError("torch_dtype must be non-empty")
        if self.device_map is not None and not self.device_map.strip():
            raise ValueError("device_map must be non-empty when provided")
        if self.max_length is not None and self.max_length <= 0:
            raise ValueError("max_length must be > 0 when provided")


@dataclass(frozen=True, slots=True)
class TokenizedPrompt:
    """Tokenized prompt payload with JSON-friendly token id storage."""

    prompt: str
    token_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.attention_mask) != len(self.token_ids):
            raise ValueError("attention_mask and token_ids must have matching length")
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token_ids must be non-negative")

    @property
    def token_count(self) -> int:
        """Return the number of prompt tokens."""

        return len(self.token_ids)


@dataclass(frozen=True, slots=True)
class ModelPrefillOutput:
    """Dense prefill output used by the first real-block metadata bridge.

    ``token_representations`` is intentionally generic. The local Hugging Face
    backend can provide hidden-state streams or K/V-adjacent key streams. That
    keeps this bridge CPU-safe while preserving the metadata and selector path
    shape needed for later sparse KV-backed runtimes.
    """

    prompt: str
    token_ids: tuple[int, ...]
    token_representations: torch.Tensor
    query_representation: torch.Tensor
    representation_name: str
    runtime_name: str
    per_head_token_representations: torch.Tensor | None = None
    per_head_query_representation: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("token_ids must not be empty")
        if self.token_representations.ndim != 2:
            raise ValueError("token_representations must have shape [tokens, features]")
        if self.token_representations.shape[0] != len(self.token_ids):
            raise ValueError("token_representations rows must match token_ids length")
        if self.query_representation.ndim != 1:
            raise ValueError("query_representation must be a 1D tensor")
        if self.query_representation.numel() != self.token_representations.shape[1]:
            raise ValueError("query_representation dim must match token representation dim")
        if self.per_head_token_representations is not None:
            if self.per_head_token_representations.ndim != 3:
                raise ValueError(
                    "per_head_token_representations must have shape [heads, tokens, features]"
                )
            if self.per_head_token_representations.shape[1] != len(self.token_ids):
                raise ValueError("per_head_token_representations tokens must match token_ids")
            if (
                self.per_head_token_representations.shape[2]
                != self.token_representations.shape[1]
            ):
                raise ValueError(
                    "per_head_token_representations feature dim must match token representations"
                )
        if self.per_head_query_representation is not None:
            if self.per_head_query_representation.ndim != 2:
                raise ValueError(
                    "per_head_query_representation must have shape [heads, features]"
                )
            if self.per_head_token_representations is None:
                raise ValueError(
                    "per_head_token_representations are required with per_head_query_representation"
                )
            if (
                self.per_head_query_representation.shape[0]
                != self.per_head_token_representations.shape[0]
            ):
                raise ValueError("per-head query count must match per-head token representations")
            if (
                self.per_head_query_representation.shape[1]
                != self.per_head_token_representations.shape[2]
            ):
                raise ValueError(
                    "per_head_query_representation feature dim must match per-head token representations"
                )
        if not self.representation_name.strip():
            raise ValueError("representation_name must be non-empty")
        if not self.runtime_name.strip():
            raise ValueError("runtime_name must be non-empty")

    @property
    def token_count(self) -> int:
        """Return the number of represented prompt tokens."""

        return len(self.token_ids)


@runtime_checkable
class RuntimeBackend(Protocol):
    """Minimal backend interface needed by the dense-only V1 ingest bridge."""

    @property
    def name(self) -> str:
        """Human-readable backend name."""

    def load_model(self) -> None:
        """Load tokenizer/model state. Implementations should be idempotent."""

    def tokenize(self, prompt: str) -> TokenizedPrompt:
        """Tokenize a prompt into model token ids."""

    def prefill(self, prompt: str) -> ModelPrefillOutput:
        """Run dense prefill and return per-token model-side representations."""

    def decode_token_ids(self, token_ids: tuple[int, ...]) -> str:
        """Decode token ids into text for inspection output."""
