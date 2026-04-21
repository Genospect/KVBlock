"""Small representation-selection helpers for local dense runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import torch

RepresentationSource = Literal[
    "final_hidden",
    "hidden_layer_index",
    "middle_hidden",
    "avg_last4_hidden",
    "avg_mid4_hidden",
    "key_mean_last_layer",
    "key_mean_mid_layer",
    "key_avg_last4",
    "query_mean_last_layer",
    "query_mean_mid_layer",
    "query_avg_last4",
]


@dataclass(frozen=True, slots=True)
class HiddenStateCaptureConfig:
    """Configuration for selecting a hidden-state tensor from model outputs."""

    representation_source: RepresentationSource = "avg_mid4_hidden"
    layer_index: int = -1
    representation_name: str | None = None

    def __post_init__(self) -> None:
        valid_sources = {
            "final_hidden",
            "hidden_layer_index",
            "middle_hidden",
            "avg_last4_hidden",
            "avg_mid4_hidden",
            "key_mean_last_layer",
            "key_mean_mid_layer",
            "key_avg_last4",
            "query_mean_last_layer",
            "query_mean_mid_layer",
            "query_avg_last4",
        }
        if self.representation_source not in valid_sources:
            raise ValueError(f"unsupported representation_source: {self.representation_source!r}")
        if self.representation_name is not None and not self.representation_name.strip():
            raise ValueError("representation_name must be non-empty")


def select_hidden_state(
    hidden_states: Sequence[torch.Tensor],
    config: HiddenStateCaptureConfig | None = None,
) -> torch.Tensor:
    """Select one HF hidden-state tensor and return shape ``[tokens, hidden]``.

    Hugging Face causal LM outputs typically expose ``hidden_states`` as a tuple
    containing embedding output plus per-layer activations. V1 uses the selected
    dense hidden-state stream as an interim metadata source until true K/V
    tensors are wired through runtime adapters.
    """

    selected, _ = select_hidden_state_with_name(hidden_states, config)
    return selected


def select_model_representation_with_name(
    hidden_states: Sequence[torch.Tensor] | None,
    past_key_values: Any | None,
    config: HiddenStateCaptureConfig | None = None,
) -> tuple[torch.Tensor, str]:
    """Select the configured model-side representation stream.

    Hidden-state sources return the existing dense hidden stream. Key sources
    return a K/V-adjacent stream built by mean-pooling attention keys across
    heads for a selected layer or layer window. The output contract stays
    source-agnostic: shape ``[tokens, features]``.
    """

    resolved = config or HiddenStateCaptureConfig()
    if _is_key_source(resolved.representation_source):
        return select_key_state_with_name(past_key_values, resolved)
    if hidden_states is None:
        raise ValueError("hidden_states are required for hidden-state representation sources")
    return select_hidden_state_with_name(hidden_states, resolved)


def select_model_prefill_representations_with_name(
    hidden_states: Sequence[torch.Tensor] | None,
    past_key_values: Any | None,
    query_states: Sequence[torch.Tensor] | None,
    config: HiddenStateCaptureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor | None, torch.Tensor | None]:
    """Select block metadata vectors plus the current-query vector.

    For hidden and key proxy modes, the query vector remains the latest token in
    the same selected stream. For query-key modes, block vectors come from keys
    and the query vector comes from the matching real attention query
    projection, enabling the existing Stage-A path to approximate ``Q · K``.
    """

    resolved = config or HiddenStateCaptureConfig()
    if is_query_source(resolved.representation_source):
        return select_query_key_states_with_name(
            past_key_values,
            query_states,
            resolved,
        )
    selected, name = select_model_representation_with_name(
        hidden_states,
        past_key_values,
        resolved,
    )
    return selected, latest_token_state(selected), name, None, None


def select_hidden_state_with_name(
    hidden_states: Sequence[torch.Tensor],
    config: HiddenStateCaptureConfig | None = None,
) -> tuple[torch.Tensor, str]:
    """Select hidden-state representations and return the selected name."""

    if not hidden_states:
        raise ValueError("hidden_states must not be empty")
    resolved = config or HiddenStateCaptureConfig()
    selected = _select_raw_hidden_state(hidden_states, resolved)
    if selected.ndim != 3:
        raise ValueError("selected hidden state must have shape [batch, tokens, hidden]")
    if selected.shape[0] != 1:
        raise ValueError("V1 local bridge expects batch size 1")

    return (
        selected.detach().to(dtype=torch.float32, device="cpu").squeeze(0).contiguous(),
        _representation_name(resolved, hidden_state_count=len(hidden_states)),
    )


def select_key_state_with_name(
    past_key_values: Any,
    config: HiddenStateCaptureConfig | None = None,
) -> tuple[torch.Tensor, str]:
    """Select per-token key representations and return shape ``[tokens, head_dim]``.

    Hugging Face causal LMs expose cache keys as one tensor per layer, usually
    with shape ``[batch, heads, tokens, head_dim]``. V1 starts with a cheap
    head-mean representation rather than a per-head metadata redesign.
    """

    if past_key_values is None:
        raise ValueError("past_key_values are required for key representation sources")
    resolved = config or HiddenStateCaptureConfig(representation_source="key_mean_last_layer")
    if not _is_key_source(resolved.representation_source):
        raise ValueError(
            f"unsupported key representation_source: {resolved.representation_source!r}"
        )

    layer_count = _key_layer_count(past_key_values)
    if layer_count <= 0:
        raise ValueError("past_key_values must contain at least one key layer")

    selected = _select_raw_key_state(past_key_values, resolved, layer_count=layer_count)
    if selected.ndim != 2:
        raise ValueError("selected key representation must have shape [tokens, features]")
    if selected.shape[0] == 0 or selected.shape[1] == 0:
        raise ValueError("selected key representation must be non-empty")

    return (
        selected.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        _key_representation_name(resolved, layer_count=layer_count),
    )


def select_query_key_states_with_name(
    past_key_values: Any,
    query_states: Sequence[torch.Tensor] | None,
    config: HiddenStateCaptureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, str, torch.Tensor, torch.Tensor]:
    """Select key block vectors and matching latest-token query vector."""

    if past_key_values is None:
        raise ValueError("past_key_values are required for query-key sources")
    if query_states is None:
        raise ValueError("query_states are required for query-key sources")
    resolved = config or HiddenStateCaptureConfig(representation_source="query_mean_last_layer")
    if not is_query_source(resolved.representation_source):
        raise ValueError(
            f"unsupported query representation_source: {resolved.representation_source!r}"
        )

    layer_count = _key_layer_count(past_key_values)
    if len(query_states) != layer_count:
        raise ValueError("query_states layer count must match past_key_values")

    per_head_key_representations = _select_raw_key_heads(
        past_key_values,
        _query_source_as_key_source(resolved),
        layer_count=layer_count,
    )
    per_head_query_representations = _select_raw_query_heads(
        query_states,
        resolved,
        layer_count=layer_count,
    )
    per_head_key_representations = _align_key_heads_to_query_heads(
        per_head_key_representations,
        query_head_count=per_head_query_representations.shape[0],
    )
    key_representations = _head_mean_attention_tensor(
        per_head_key_representations.unsqueeze(0)
    )
    query_representations = _head_mean_attention_tensor(
        per_head_query_representations.unsqueeze(0)
    )
    return (
        key_representations.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        latest_token_state(query_representations),
        _query_representation_name(resolved, layer_count=layer_count),
        per_head_key_representations.detach().to(dtype=torch.float32, device="cpu").contiguous(),
        per_head_query_representations[:, -1, :]
        .detach()
        .to(dtype=torch.float32, device="cpu")
        .contiguous(),
    )


def latest_token_state(token_representations: torch.Tensor) -> torch.Tensor:
    """Return the representation for the latest prompt token."""

    if token_representations.ndim != 2:
        raise ValueError("token_representations must have shape [tokens, features]")
    if token_representations.shape[0] == 0:
        raise ValueError("token_representations must not be empty")
    return token_representations[-1].detach().to(dtype=torch.float32, device="cpu").contiguous()


def is_key_source(source: RepresentationSource) -> bool:
    """Return whether a representation source reads attention keys."""

    return source in {
        "key_mean_last_layer",
        "key_mean_mid_layer",
        "key_avg_last4",
    }


def is_query_source(source: RepresentationSource) -> bool:
    """Return whether a representation source needs attention query capture."""

    return source in {
        "query_mean_last_layer",
        "query_mean_mid_layer",
        "query_avg_last4",
    }


def _is_key_source(source: RepresentationSource) -> bool:
    return is_key_source(source)


def _select_raw_hidden_state(
    hidden_states: Sequence[torch.Tensor],
    config: HiddenStateCaptureConfig,
) -> torch.Tensor:
    source = config.representation_source
    if source == "final_hidden":
        return hidden_states[-1]
    if source == "hidden_layer_index":
        return hidden_states[config.layer_index]
    if source == "middle_hidden":
        return hidden_states[len(hidden_states) // 2]
    if source == "avg_last4_hidden":
        return _average_hidden_states(hidden_states[-min(4, len(hidden_states)) :])
    if source == "avg_mid4_hidden":
        return _average_hidden_states(_middle_window(hidden_states, window_size=4))
    raise ValueError(f"unsupported representation_source: {source!r}")


def _select_raw_key_state(
    past_key_values: Any,
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> torch.Tensor:
    source = config.representation_source
    if source == "key_mean_last_layer":
        return _head_mean_key_tensor(_key_tensor_at(past_key_values, layer_count - 1))
    if source == "key_mean_mid_layer":
        return _head_mean_key_tensor(_key_tensor_at(past_key_values, layer_count // 2))
    if source == "key_avg_last4":
        layer_indices = range(max(0, layer_count - 4), layer_count)
        return torch.stack(
            [
                _head_mean_key_tensor(_key_tensor_at(past_key_values, index))
                for index in layer_indices
            ]
        ).mean(dim=0)
    raise ValueError(f"unsupported key representation_source: {source!r}")


def _select_raw_key_heads(
    past_key_values: Any,
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> torch.Tensor:
    source = config.representation_source
    if source == "key_mean_last_layer":
        return _attention_heads_tensor(_key_tensor_at(past_key_values, layer_count - 1))
    if source == "key_mean_mid_layer":
        return _attention_heads_tensor(_key_tensor_at(past_key_values, layer_count // 2))
    if source == "key_avg_last4":
        layer_indices = range(max(0, layer_count - 4), layer_count)
        return torch.stack(
            [
                _attention_heads_tensor(_key_tensor_at(past_key_values, index))
                for index in layer_indices
            ]
        ).mean(dim=0)
    raise ValueError(f"unsupported key representation_source: {source!r}")


def _select_raw_query_state(
    query_states: Sequence[torch.Tensor],
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> torch.Tensor:
    source = config.representation_source
    if source == "query_mean_last_layer":
        return _head_mean_attention_tensor(query_states[layer_count - 1])
    if source == "query_mean_mid_layer":
        return _head_mean_attention_tensor(query_states[layer_count // 2])
    if source == "query_avg_last4":
        layer_indices = range(max(0, layer_count - 4), layer_count)
        return torch.stack(
            [
                _head_mean_attention_tensor(query_states[index])
                for index in layer_indices
            ]
        ).mean(dim=0)
    raise ValueError(f"unsupported query representation_source: {source!r}")


def _select_raw_query_heads(
    query_states: Sequence[torch.Tensor],
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> torch.Tensor:
    source = config.representation_source
    if source == "query_mean_last_layer":
        return _attention_heads_tensor(query_states[layer_count - 1])
    if source == "query_mean_mid_layer":
        return _attention_heads_tensor(query_states[layer_count // 2])
    if source == "query_avg_last4":
        layer_indices = range(max(0, layer_count - 4), layer_count)
        return torch.stack(
            [
                _attention_heads_tensor(query_states[index])
                for index in layer_indices
            ]
        ).mean(dim=0)
    raise ValueError(f"unsupported query representation_source: {source!r}")


def _key_layer_count(past_key_values: Any) -> int:
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        return len(key_cache)
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        return len(layers)
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy_cache = past_key_values.to_legacy_cache()
        return len(legacy_cache)
    return len(past_key_values)


def _key_tensor_at(past_key_values: Any, layer_index: int) -> torch.Tensor:
    key_cache = getattr(past_key_values, "key_cache", None)
    if key_cache is not None:
        key = key_cache[layer_index]
    else:
        layers = getattr(past_key_values, "layers", None)
        if layers is not None:
            key = _key_tensor_from_cache_layer(layers[layer_index])
        elif hasattr(past_key_values, "to_legacy_cache"):
            key = past_key_values.to_legacy_cache()[layer_index][0]
        else:
            key = past_key_values[layer_index][0]
    if not isinstance(key, torch.Tensor):
        raise TypeError("attention key cache entry must be a torch.Tensor")
    return key


def _key_tensor_from_cache_layer(layer: Any) -> torch.Tensor:
    """Extract keys from newer HF cache layer objects without importing HF types."""

    if isinstance(layer, torch.Tensor):
        return layer
    if isinstance(layer, (tuple, list)) and layer:
        key = layer[0]
        if isinstance(key, torch.Tensor):
            return key
    for attr_name in ("keys", "key_cache", "key_states", "key"):
        key = getattr(layer, attr_name, None)
        if isinstance(key, torch.Tensor):
            return key
    raise TypeError(
        "unsupported DynamicCache layer shape; expected a tensor, a legacy "
        "(key, value) pair, or an object exposing keys/key_cache/key_states/key"
    )


def _head_mean_key_tensor(key_tensor: torch.Tensor) -> torch.Tensor:
    return _head_mean_attention_tensor(key_tensor)


def _head_mean_attention_tensor(attention_tensor: torch.Tensor) -> torch.Tensor:
    return _attention_heads_tensor(attention_tensor).mean(dim=0)


def _attention_heads_tensor(attention_tensor: torch.Tensor) -> torch.Tensor:
    if attention_tensor.ndim != 4:
        raise ValueError(
            "attention tensor must have shape [batch, heads, tokens, head_dim]"
        )
    if attention_tensor.shape[0] != 1:
        raise ValueError("V1 local bridge expects batch size 1")
    if (
        attention_tensor.shape[1] <= 0
        or attention_tensor.shape[2] <= 0
        or attention_tensor.shape[3] <= 0
    ):
        raise ValueError("attention tensor dimensions must be non-empty")

    # HF GPT-style caches use [batch, heads, tokens, head_dim]. Mean-pooling
    # heads gives a compact K/V-adjacent per-token stream without changing
    # BlockMetadata for this first key-ingest pass.
    return attention_tensor.detach().to(dtype=torch.float32).squeeze(0)


def _align_key_heads_to_query_heads(
    key_heads: torch.Tensor,
    *,
    query_head_count: int,
) -> torch.Tensor:
    """Repeat grouped-query key heads to match query-head count when needed."""

    if key_heads.ndim != 3:
        raise ValueError("key_heads must have shape [heads, tokens, head_dim]")
    if query_head_count <= 0:
        raise ValueError("query_head_count must be > 0")
    key_head_count = key_heads.shape[0]
    if key_head_count == query_head_count:
        return key_heads
    if query_head_count % key_head_count != 0:
        raise ValueError(
            "query/key head mismatch is only supported when query heads are an "
            "integer multiple of key heads"
        )
    return key_heads.repeat_interleave(query_head_count // key_head_count, dim=0)


def _average_hidden_states(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    if not hidden_states:
        raise ValueError("hidden_states must not be empty")
    return torch.stack([state.detach().to(dtype=torch.float32) for state in hidden_states]).mean(
        dim=0
    )


def _middle_window(
    hidden_states: Sequence[torch.Tensor],
    *,
    window_size: int,
) -> Sequence[torch.Tensor]:
    if window_size <= 0:
        raise ValueError("window_size must be > 0")
    count = len(hidden_states)
    if count <= window_size:
        return hidden_states
    center = count // 2
    start = max(0, center - window_size // 2)
    end = min(count, start + window_size)
    start = max(0, end - window_size)
    return hidden_states[start:end]


def _representation_name(
    config: HiddenStateCaptureConfig,
    *,
    hidden_state_count: int,
) -> str:
    if config.representation_name is not None:
        return config.representation_name
    source = config.representation_source
    if source == "hidden_layer_index":
        return f"hidden_layer_{config.layer_index}"
    if source == "middle_hidden":
        return f"middle_hidden_{hidden_state_count // 2}"
    if source == "avg_last4_hidden":
        return f"avg_last{min(4, hidden_state_count)}_hidden"
    if source == "avg_mid4_hidden":
        window = _middle_window(tuple(range(hidden_state_count)), window_size=4)
        return f"avg_mid_hidden_{window[0]}_{window[-1]}"
    return "final_hidden"


def _key_representation_name(
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> str:
    if config.representation_name is not None:
        return config.representation_name
    source = config.representation_source
    if source == "key_mean_last_layer":
        return f"key_mean_layer_{layer_count - 1}"
    if source == "key_mean_mid_layer":
        return f"key_mean_layer_{layer_count // 2}"
    if source == "key_avg_last4":
        start = max(0, layer_count - 4)
        return f"key_avg_layers_{start}_{layer_count - 1}"
    raise ValueError(f"unsupported key representation_source: {source!r}")


def _query_source_as_key_source(
    config: HiddenStateCaptureConfig,
) -> HiddenStateCaptureConfig:
    source_map: dict[str, RepresentationSource] = {
        "query_mean_last_layer": "key_mean_last_layer",
        "query_mean_mid_layer": "key_mean_mid_layer",
        "query_avg_last4": "key_avg_last4",
    }
    try:
        key_source = source_map[config.representation_source]
    except KeyError as exc:
        raise ValueError(
            f"unsupported query representation_source: {config.representation_source!r}"
        ) from exc
    return HiddenStateCaptureConfig(representation_source=key_source)


def _query_representation_name(
    config: HiddenStateCaptureConfig,
    *,
    layer_count: int,
) -> str:
    if config.representation_name is not None:
        return config.representation_name
    source = config.representation_source
    if source == "query_mean_last_layer":
        layer = layer_count - 1
        return f"query_mean_layer_{layer}_key_mean_layer_{layer}"
    if source == "query_mean_mid_layer":
        layer = layer_count // 2
        return f"query_mean_layer_{layer}_key_mean_layer_{layer}"
    if source == "query_avg_last4":
        start = max(0, layer_count - 4)
        return f"query_avg_layers_{start}_{layer_count - 1}_key_avg_layers_{start}_{layer_count - 1}"
    raise ValueError(f"unsupported query representation_source: {source!r}")
