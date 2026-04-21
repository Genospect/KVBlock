"""Runtime backend construction helpers."""

from __future__ import annotations

from kvblock.runtime.base import RuntimeBackend, RuntimeLoadConfig
from kvblock.runtime.hooks import HiddenStateCaptureConfig
from kvblock.runtime.local_hf_runtime import LocalHfRuntime


def create_runtime_backend(
    config: RuntimeLoadConfig | None = None,
    *,
    backend: str = "local_hf",
    capture_config: HiddenStateCaptureConfig | None = None,
) -> RuntimeBackend:
    """Create a runtime backend without loading model weights yet."""

    normalized = backend.strip().lower()
    resolved_config = config or RuntimeLoadConfig()
    if normalized in {"local_hf", "hf", "huggingface"}:
        return LocalHfRuntime(resolved_config, capture_config=capture_config)
    if normalized in {"flashinfer", "vllm"}:
        raise NotImplementedError(
            f"{backend} runtime integration is intentionally deferred beyond this bridge"
        )
    raise ValueError(f"unsupported runtime backend: {backend!r}")


def load_runtime_backend(
    config: RuntimeLoadConfig | None = None,
    *,
    backend: str = "local_hf",
    capture_config: HiddenStateCaptureConfig | None = None,
) -> RuntimeBackend:
    """Create and load a runtime backend."""

    runtime = create_runtime_backend(
        config,
        backend=backend,
        capture_config=capture_config,
    )
    runtime.load_model()
    return runtime
