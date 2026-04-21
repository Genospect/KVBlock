"""Dense-only runtime bridges for V1 block-ingest experiments."""

from kvblock.runtime.base import (
    ModelPrefillOutput,
    RuntimeBackend,
    RuntimeLoadConfig,
    TokenizedPrompt,
)
from kvblock.runtime.hooks import HiddenStateCaptureConfig, RepresentationSource
from kvblock.runtime.local_hf_runtime import LocalHfRuntime
from kvblock.runtime.model_loader import create_runtime_backend, load_runtime_backend
from kvblock.runtime.real_block_eval import (
    BlockInspectionRecord,
    RealBlockLatencySummary,
    RealBlockRunSummary,
    RealBlockSelectorConfig,
    RealBlockSelectorResult,
    run_real_block_selector,
)

__all__ = [
    "LocalHfRuntime",
    "ModelPrefillOutput",
    "BlockInspectionRecord",
    "HiddenStateCaptureConfig",
    "RealBlockLatencySummary",
    "RealBlockRunSummary",
    "RealBlockSelectorConfig",
    "RealBlockSelectorResult",
    "RepresentationSource",
    "RuntimeBackend",
    "RuntimeLoadConfig",
    "TokenizedPrompt",
    "create_runtime_backend",
    "load_runtime_backend",
    "run_real_block_selector",
]
