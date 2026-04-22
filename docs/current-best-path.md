# Current Best Path

This document records the active frozen control for LongBench selector work.
It is a research baseline, not a production default and not sparse runtime
execution.

## Frozen Control

Use `fixed40_modern_control` as the named control for the next LongBench
comparison batch.

- `representation_source = query_only_last_layer`
- `qk_aggregation = block_max`
- `rerank_mode = dense_qk_token_refine`
- `refine_score_mode = softmax_mass`
- `stage_c_policy = semantic_refined_mix`
- `refine_top_n_tokens = 4`
- `exclude_scaffold_blocks = true`
- `halo_radius = 1`
- `max_selected_blocks = 16`
- `evidence_window_radius = 2`
- `shortlist_m = 32`
- `semantic_k = 8`
- `confidence_margin = 0.05`
- `block_mode = fixed_40`

Current small-slice interpretation:

- Budget 16 is the operating knee for this stack.
- Budget 8 is too tight for HotpotQA.
- Budget 24 mostly buys extra evidence-window coverage while hurting precision.
- `fixed_40` remains the best simple production-style fixed-block baseline.
- `fixed_24` currently fragments evidence and increases candidate latency.

## Active Modules

The active path for the current baseline uses:

- `src/kvblock/runtime/local_hf_runtime.py` for local dense Hugging Face prefill and Q/K capture
- `src/kvblock/runtime/real_block_eval.py` for real-block selector bridge runs
- `src/kvblock/kv/block_manager.py` for metadata construction from model-side vectors
- `src/kvblock/kv/qk_aggregation.py` for constrained query/key aggregation strategies
- `src/kvblock/selector/` for the unchanged V1 Stage A/B/C selector pipeline
- `src/kvblock/benchmark/dynamic_block_benchmark.py` for fixed-block validation against dynamic-block alternatives
- `scripts/run_dynamic_block_benchmark.py` for current fixed-vs-dynamic local runs
- `scripts/run_longbench_benchmark.py` for LongBench selector validation
- `scripts/compare_longbench_runs.py` for comparing benchmark JSON outputs

## Stable Findings

- Query-only routing is viable on the current LongBench slice.
- Dense QK token refinement with `softmax_mass` is the main current unlock.
- `semantic_refined_mix` is the robust Stage-C default for the next batch.
- Halo is a continuity rail, not the primary retrieval engine.
- Budget is a policy lever; more selected blocks are not automatically better.
- Smaller fixed blocks are not automatically better; `fixed_24` lost to `fixed_40` under the modern stack.
- No single fixed block size should be treated as the endgame, but `fixed_40` is the frozen control before representation and coarse-to-fine tests.

## Experiments That Did Not Win

These remain useful research history but are not the current path:

- Hidden-state-only bridge defaults such as `avg_mid4_hidden`
- Naive per-head routing modes such as `max_head_score` and `topk_head_mean`
- Static head weighting schemes such as `head9_heavy`, `retrieval_mix`, and `code_mix`
- Naive multi-scale block pooling such as `multiscale_16_32` and `multiscale_16_24_32`
- Overlap suppression as a standalone fix for multi-scale crowding
- Earlier coarse-to-fine tests without the current Stage-B/C stack
- `fixed_24` under the current LongBench stack

## Recommended Evaluation Command

Run the frozen control as:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/run_longbench_benchmark.py \
  --models Qwen/Qwen2.5-1.5B-Instruct \
  --longbench-datasets hotpotqa,musique \
  --limit 5 \
  --length-bucket 0-4k \
  --representation-source query_only_last_layer \
  --qk-aggregation block_max \
  --rerank-mode dense_qk_token_refine \
  --refine-score-mode softmax_mass \
  --stage-c-policy semantic_refined_mix \
  --refine-top-n-tokens 4 \
  --exclude-scaffold-blocks \
  --halo-radius 1 \
  --max-selected-blocks 16 \
  --evidence-window-radius 2 \
  --shortlist-m 32 \
  --semantic-k 8 \
  --confidence-margin 0.05 \
  --oracle-mode none \
  --block-modes fixed_40 \
  --device cuda \
  --out-dir results/longbench \
  --name fixed40_modern_control
```

Compare later runs against it with:

```bash
python scripts/compare_longbench_runs.py \
  fixed40_modern_control=results/longbench/fixed40_modern_control.json \
  candidate=results/longbench/candidate.json \
  --control-label fixed40_modern_control \
  --scope both
```

## Next Phase

The next phase is ordered:

1. Use the comparison utility for all benchmark JSON outputs.
2. Run the representation sweep against `fixed40_modern_control`.
3. Run existing coarse-to-fine modes against the best representation using the same Stage-B/C stack.
4. Design a new adaptive multi-scale router only if current coarse-to-fine still loses.
5. Add an adaptive budget controller after the representation/coarse-to-fine decision is clear.

Do not add learned routing, CUDA integration, or framework rewrites before these
measurements settle the next selector direction.
