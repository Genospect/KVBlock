# KVBlock Package Architecture

KVBlock is the policy/control layer for deciding which logical KV cache blocks
should be read during decode. It does not replace PagedAttention storage, vLLM
scheduling, or FlashInfer/Triton kernels.

## Stable API Surface

- `KVBlockPolicy` describes selector budgets, rails, fallback mode, and caps.
- `BlockLayout` describes logical block spans independent of runtime page tables.
- `SelectedKVPlan` is the standardized selector output consumed by mappers and
  backends.
- `LogicalToPhysicalMapper` fills optional physical page ids once a runtime page
  table is available.

## Current Package Boundaries

- `kvblock.blocks`: logical spans, layouts, and mapping.
- `kvblock.policies`: public policy object and named presets.
- `kvblock.selectors`: selector protocol, simple debug selectors, and adapters
  around existing research selectors.
- `kvblock.plans`: selected KV plan serialization and validation.
- `kvblock.backends`: dense and sparse PyTorch references plus guarded optional
  FlashInfer/Triton placeholders.
- `kvblock.metrics`: divergence, quality, latency, bandwidth, and recall helpers.
- `kvblock.bench`: package-facing benchmark records and promoted harnesses.

## Design Constraint

Masking dense attention is not a physical sparse speedup proof. A backend only
supports acceleration claims when it avoids reading unselected K/V blocks through
indexed gathering or sparse attention kernels.
