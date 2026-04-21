"""Local Hugging Face/PyTorch runtime for dense-only V1 ingest smoke runs."""

from __future__ import annotations

from typing import Any

import torch

from kvblock.runtime.base import ModelPrefillOutput, RuntimeBackend, RuntimeLoadConfig, TokenizedPrompt
from kvblock.runtime.hooks import (
    HiddenStateCaptureConfig,
    is_query_source,
    select_model_prefill_representations_with_name,
)


class LocalHfRuntime(RuntimeBackend):
    """Small CPU-safe Hugging Face causal LM backend.

    This backend performs dense prefill and extracts configured model-side
    representations for metadata construction. It can expose hidden states or
    attention-key-derived vectors, but intentionally does not expose sparse
    execution or runtime-specific K/V page structures yet.
    """

    def __init__(
        self,
        config: RuntimeLoadConfig | None = None,
        *,
        capture_config: HiddenStateCaptureConfig | None = None,
    ) -> None:
        self.config = config or RuntimeLoadConfig()
        self.capture_config = capture_config or HiddenStateCaptureConfig()
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def name(self) -> str:
        """Return a stable backend label."""

        return f"local_hf:{self.config.model_name}"

    def load_model(self) -> None:
        """Load the tokenizer/model pair if it is not already loaded."""

        if self._tokenizer is not None and self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised only without optional dep
            raise ImportError(
                "LocalHfRuntime requires the optional 'transformers' dependency. "
                "Install it separately or use the mocked runtime in tests."
            ) from exc

        dtype = _resolve_torch_dtype(self.config.torch_dtype)
        model_kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "trust_remote_code": self.config.trust_remote_code,
        }
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            local_files_only=self.config.local_files_only,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs,
        )
        self._model.to(self.config.device)
        self._model.eval()

    def tokenize(self, prompt: str) -> TokenizedPrompt:
        """Tokenize ``prompt`` with the loaded HF tokenizer."""

        if not prompt:
            raise ValueError("prompt must be non-empty")
        self.load_model()
        tokenizer = self._require_tokenizer()
        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=self.config.max_length is not None,
            max_length=self.config.max_length,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", torch.ones_like(input_ids))
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("local HF bridge expects tokenizer batch size 1")

        token_ids = tuple(int(value) for value in input_ids.squeeze(0).tolist())
        if not token_ids:
            raise ValueError("tokenized prompt must contain at least one token")
        return TokenizedPrompt(
            prompt=prompt,
            token_ids=token_ids,
            attention_mask=tuple(int(value) for value in attention_mask.squeeze(0).tolist()),
        )

    def prefill(self, prompt: str) -> ModelPrefillOutput:
        """Run dense prefill and return selected per-token model-side vectors."""

        tokenized = self.tokenize(prompt)
        model = self._require_model()
        input_ids = torch.tensor(
            [tokenized.token_ids],
            dtype=torch.long,
            device=self.config.device,
        )
        attention_mask = torch.tensor(
            [tokenized.attention_mask],
            dtype=torch.long,
            device=self.config.device,
        )

        query_capture = (
            _Gpt2QueryProjectionCapture(model)
            if is_query_source(self.capture_config.representation_source)
            else None
        )
        if query_capture is None:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True,
                    return_dict=True,
                )
            query_states = None
        else:
            with query_capture:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=True,
                        return_dict=True,
                    )
            query_states = query_capture.query_states

        (
            token_representations,
            query_representation,
            representation_name,
            per_head_token_representations,
            per_head_query_representation,
        ) = select_model_prefill_representations_with_name(
            outputs.hidden_states,
            outputs.past_key_values,
            query_states,
            config=self.capture_config,
        )
        return ModelPrefillOutput(
            prompt=prompt,
            token_ids=tokenized.token_ids,
            token_representations=token_representations,
            query_representation=query_representation,
            representation_name=representation_name,
            runtime_name=self.name,
            per_head_token_representations=per_head_token_representations,
            per_head_query_representation=per_head_query_representation,
        )

    def decode_token_ids(self, token_ids: tuple[int, ...]) -> str:
        """Decode token ids with the loaded HF tokenizer for block inspection."""

        self.load_model()
        tokenizer = self._require_tokenizer()
        return str(tokenizer.decode(list(token_ids)))

    def _require_tokenizer(self) -> Any:
        if self._tokenizer is None:
            raise RuntimeError("tokenizer is not loaded")
        return self._tokenizer

    def _require_model(self) -> Any:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        return self._model


def _resolve_torch_dtype(value: str) -> torch.dtype | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(
            "torch_dtype must be one of auto, float32, float16, or bfloat16"
        ) from exc


class _Gpt2QueryProjectionCapture:
    """Temporarily capture GPT-2-style attention query projections.

    GPT-2-family HF modules use a fused ``c_attn`` projection that returns
    concatenated Q/K/V. This hook splits the Q slice and reshapes it to match
    the cache convention ``[batch, heads, tokens, head_dim]``.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._handles: list[Any] = []
        self._query_states: list[torch.Tensor] = []

    @property
    def query_states(self) -> tuple[torch.Tensor, ...]:
        """Return captured query tensors in layer order."""

        return tuple(self._query_states)

    def __enter__(self) -> "_Gpt2QueryProjectionCapture":
        self._query_states.clear()
        n_head = _gpt2_num_heads(self._model)
        for module in _iter_gpt2_c_attn_modules(self._model):
            self._handles.append(module.register_forward_hook(_make_query_hook(self, n_head)))
        if not self._handles:
            raise RuntimeError(
                "query representation sources currently require GPT-2-style "
                "transformer.h[*].attn.c_attn modules"
            )
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def append(self, query_state: torch.Tensor) -> None:
        self._query_states.append(query_state.detach())


def _make_query_hook(capture: _Gpt2QueryProjectionCapture, n_head: int):
    def hook(module: Any, _inputs: Any, output: Any) -> None:
        if isinstance(output, tuple):
            projected = output[0]
        else:
            projected = output
        if not isinstance(projected, torch.Tensor):
            raise TypeError("GPT-2 c_attn output must be a torch.Tensor")
        if projected.ndim != 3:
            raise ValueError("GPT-2 c_attn output must have shape [batch, tokens, 3 * hidden]")
        if projected.shape[-1] % 3 != 0:
            raise ValueError("GPT-2 c_attn output feature dim must be divisible by 3")
        hidden_dim = projected.shape[-1] // 3
        if hidden_dim % n_head != 0:
            raise ValueError("GPT-2 query hidden dim must be divisible by n_head")
        query = projected[..., :hidden_dim]
        head_dim = hidden_dim // n_head
        query = query.reshape(query.shape[0], query.shape[1], n_head, head_dim)
        capture.append(query.permute(0, 2, 1, 3).contiguous())

    return hook


def _iter_gpt2_c_attn_modules(model: Any):
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        return ()
    modules = []
    for block in blocks:
        attn = getattr(block, "attn", None)
        c_attn = getattr(attn, "c_attn", None)
        if c_attn is None:
            continue
        modules.append(c_attn)
    return tuple(modules)


def _gpt2_num_heads(model: Any) -> int:
    config = getattr(model, "config", None)
    n_head = getattr(config, "n_head", None) or getattr(config, "num_attention_heads", None)
    if n_head is None or int(n_head) <= 0:
        raise RuntimeError("model config must expose a positive n_head for query capture")
    return int(n_head)
