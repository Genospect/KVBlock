# Experiment Status

This document classifies current KVBlock research paths so the next GPU/larger-model phase can start from a clean baseline without losing experimental history.

## Active Baseline

The active LongBench control is `fixed40_modern_control`:

- `query_only_last_layer`
- `block_max`
- `fixed_40`
- `dense_qk_token_refine`
- `softmax_mass`
- `semantic_refined_mix`
- `halo_radius = 1`
- `max_selected_blocks = 16`
- `semantic_k = 8`
- `shortlist_m = 32`
- `confidence_margin = 0.05`

Use `docs/current-best-path.md` as the source of truth for the current runnable baseline command.

## Promising But Unproven

These are the next ordered tests:

- Representation sweep against `fixed40_modern_control`.
- Existing coarse-to-fine modes with the same modern Stage-B/C stack.
- Broader retrieval-friendly task families after the control-vs-candidate decision is clearer.
- Adaptive budget controller after representation and coarse-to-fine outcomes are known.

## Archived Or Not Current Priority

These paths produced useful evidence but should not drive the next phase unless new results change the picture:

- hidden-state-only bridge as the main routing source
- per-head routing modes beyond pooled `mean_heads`
- static head-weight validation
- head 9 or specialist-head weighting as a default
- naive multi-scale block pooling
- overlap suppression as the main multi-scale fix
- coarse-to-fine modes before retesting under the current Stage-B/C stack
- fixed_24 under the current LongBench stack

The code remains in place where tests and scripts depend on it. These paths are labeled as benchmark or diagnostic history rather than moved into a new namespace.

## Result Organization

Result artifacts are grouped by experiment family:

- `results/baselines/` for current and early selector baselines
- `results/representation/` for hidden-state, key, and query-source sweeps
- `results/aggregation/` for query/key aggregation ablations
- `results/heads/` for head diagnostics, ablation, and static weighting
- `results/dynamic_blocks/` for fixed, multi-scale, suppression, and hierarchy block experiments
- `results/longbench/` for LongBench selector benchmark JSON/text outputs and comparisons
- `results/inspection/` for prompt-level qualitative inspection JSON/text
- `results/plots/` for generated figures

## Future Ideas

Defer these until the active baseline is validated on GPU/larger models:

- FlashInfer/vLLM sparse execution integration
- real page-list construction against backend KV caches
- measured DRAM bytes/token via Nsight/CUPTI/NVML
- learned routing from dense access traces
- adaptive block sizing or query classification
- stronger RoPE-aware summary construction
