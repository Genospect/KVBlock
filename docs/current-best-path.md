# Current Best Path

This document records the current local CPU/Mac baseline for KVBlock selector research. It is a research baseline, not a production default and not sparse runtime execution.

## Baseline

The current best local benchmark path is:

- `representation_source = query_mean_last_layer`
- `qk_aggregation = block_max`
- `block_mode = fixed_40`
- `head_scoring_mode = mean_heads`
- `keep_recent_blocks = 0` and `keep_anchor_blocks = 0` for reduced-rail evaluation
- `shortlist_m = 16`
- `semantic_k = 4`
- `confidence_margin = 0.0`

For identifier-heavy needle prompts, `top_token_mean` remains useful as a prompt-specific aggregation override. The general baseline should still be evaluated with `block_max` unless the experiment explicitly isolates identifier retrieval.

## Active Modules

The active path for the current baseline uses:

- `src/kvblock/runtime/local_hf_runtime.py` for local dense Hugging Face prefill and Q/K capture
- `src/kvblock/runtime/real_block_eval.py` for real-block selector bridge runs
- `src/kvblock/kv/block_manager.py` for metadata construction from model-side vectors
- `src/kvblock/kv/qk_aggregation.py` for constrained query/key aggregation strategies
- `src/kvblock/selector/` for the unchanged V1 Stage A/B/C selector pipeline
- `src/kvblock/benchmark/dynamic_block_benchmark.py` for fixed-block validation against dynamic-block alternatives
- `scripts/run_dynamic_block_benchmark.py` for current fixed-vs-dynamic local runs

## Stable Findings

- Hidden-state bridge experiments were useful for bootstrapping, but attention-native query/key routing is stronger.
- `query_mean_last_layer` is the strongest current attention-native routing source.
- Pooled per-head scoring with `mean_heads` remains best or tied against tested per-head alternatives.
- Static head weighting did not beat pooled routing strongly enough to justify becoming a default.
- `block_max` is the strongest general query/key aggregation candidate so far.
- `top_token_mean` helps exact identifier and needle-like retrieval.
- No single fixed block size is universally optimal, but `fixed_40` is the best current single-size compromise.

## Experiments That Did Not Win

These remain useful research history but are not the current path:

- Hidden-state-only bridge defaults such as `avg_mid4_hidden`
- Naive per-head routing modes such as `max_head_score` and `topk_head_mean`
- Static head weighting schemes such as `head9_heavy`, `retrieval_mix`, and `code_mix`
- Naive multi-scale block pooling such as `multiscale_16_32` and `multiscale_16_24_32`
- Overlap suppression as a standalone fix for multi-scale crowding
- `coarse_to_fine_40_16`
- `coarse_to_fine_40_16_keep_parent`

## Recommended Evaluation Command

Use a cached/local model first:

```bash
python scripts/run_dynamic_block_benchmark.py \
  --models gpt2 \
  --representation-source query_mean_last_layer \
  --qk-aggregation block_max \
  --needle-qk-aggregation top_token_mean \
  --block-modes fixed_40 \
  --prompts needle,long_reference,code_context,repeated_reference \
  --block-size 16 \
  --shortlist-m 16 \
  --semantic-k 4 \
  --confidence-margin 0.0 \
  --keep-recent-blocks 0 \
  --keep-anchor-blocks 0 \
  --local-files-only \
  --out-dir results/baselines \
  --name fixed40_query_key_baseline
```

## Next Phase

The next phase should validate this dense-only selector baseline on GPU and larger cached models before adding sparse execution. The goal is to confirm that the same routing behavior holds when model depth, context length, and prompt scale increase.

Do not treat GPU validation as FlashInfer integration yet. First validate:

- larger model Q/K capture
- selector latency under longer contexts
- selected block quality under fixed_40
- whether `block_max` remains the best general aggregation
- whether `top_token_mean` remains a targeted needle override
