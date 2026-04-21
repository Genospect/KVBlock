# Experiment Status

This document classifies current KVBlock research paths so the next GPU/larger-model phase can start from a clean baseline without losing experimental history.

## Active Baseline

The active local CPU/Mac baseline is:

- `query_mean_last_layer`
- `block_max`
- `fixed_40`
- pooled `mean_heads`
- reduced/no rails for routing-quality evaluation

Use `docs/current-best-path.md` as the source of truth for the current runnable baseline command.

## Promising But Unproven

These are worth revisiting after GPU and larger-model validation:

- `top_token_mean` for exact identifier and needle-style retrieval
- query/key qualitative inspection for missed evidence and distractor analysis
- longer prompt suites with real Q/K capture
- fixed block sizes near 32-48 tokens
- dynamic block selection only if larger-scale evidence shows fixed sizes fail

## Archived Or Not Current Priority

These paths produced useful evidence but should not drive the next phase unless new results change the picture:

- hidden-state-only bridge as the main routing source
- per-head routing modes beyond pooled `mean_heads`
- static head-weight validation
- head 9 or specialist-head weighting as a default
- naive multi-scale block pooling
- overlap suppression as the main multi-scale fix
- coarse-to-fine replacement of parent blocks
- coarse-to-fine parent retention

The code remains in place where tests and scripts depend on it. These paths are labeled as benchmark or diagnostic history rather than moved into a new namespace.

## Result Organization

Result artifacts are grouped by experiment family:

- `results/baselines/` for current and early selector baselines
- `results/representation/` for hidden-state, key, and query-source sweeps
- `results/aggregation/` for query/key aggregation ablations
- `results/heads/` for head diagnostics, ablation, and static weighting
- `results/dynamic_blocks/` for fixed, multi-scale, suppression, and hierarchy block experiments
- `results/inspection/` for prompt-level qualitative inspection JSON/text
- `results/plots/` for generated figures

## Future Ideas

Defer these until the active baseline is validated on GPU/larger models:

- FlashInfer/vLLM sparse execution integration
- real page-list construction against backend KV caches
- measured DRAM bytes/token via Nsight/CUPTI/NVML
- real dense-runtime oracle traces
- learned routing from dense access traces
- adaptive block sizing or query classification
- stronger RoPE-aware summary construction
