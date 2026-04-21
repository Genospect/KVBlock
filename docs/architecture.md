# Architecture Overview

## Project objective

KVBlock aims to reduce **decode-time KV bandwidth** in long-context transformer inference.

The target is not just smaller stored KV.
The target is **less KV touched per output token**.

Conceptually:

- Dense decode: `alpha ~= 1`
- Sparse decode: `alpha << 1`

where `alpha` is the fraction of dense KV actually read during one decode step.

## Resource model

Useful conceptual formulas:

```text
KV capacity bytes ~= 2 * L * n_kv * d_h * T * b
KV read bytes / output token ~= alpha * 2 * L * n_kv * d_h * T * b
```

Where:
- `L` = number of layers
- `n_kv` = KV heads
- `d_h` = head dimension
- `T` = cached tokens
- `b` = bytes per stored element
- `alpha` = fraction of dense KV actually read

The V1 project mainly targets `alpha`, secondarily `b`, and uses paging to improve allocator efficiency.

## V1 system diagram

```text
Prompt / Prefill
    ->
Paged KV blocks created
    ->
Per-block metadata + summaries stored
    ->
Decode step begins
    ->
Current query summary generated
    ->
Stage A coarse block scoring
    ->
Stage B shortlist refinement
    ->
Stage C final block selection
    ->
Sparse page list built
    ->
Block-sparse attention executes
    ->
Metadata updated
    ->
Confidence evaluated
    ->
If weak: widen K / add recent / dense fallback
```

## V1 architecture principles

### 1. Standard transformer only
No architecture changes in V1.

### 2. Heuristic selector first
Do not train a learned router until heuristic behavior is profiled.

### 3. Access trace collection matters
The sparse selector should be comparable to a dense oracle so later versions can learn from real traces.

### 4. Confidence control matters
If confidence is wrong:
- sparse stays on too long and hurts quality
- or dense fallback triggers too often and kills throughput

### 5. RoPE awareness matters
Summaries should ideally be built from a RoPE-free or pre-RoPE side channel.
Naive post-RoPE pooling is risky.

## V1 block structure

### Payload
Each block/page contains standard KV payload for a token span.

Default span:
- **32 tokens**

### Metadata
Each block also carries lightweight metadata:
- identity / token range
- precision tier
- flags
- low-precision summary vector
- sign sketch
- statistics like `attn_ema`, `attn_var`, hit count, recency, priority, rope bucket

Metadata should stay tiny compared with block payload.

## Summary design

V1 default summary:
- `summary_fp8[32]`
- `summary_scale`
- `sign_sketch` (64-bit)

Rationale:
- low-precision summary gives coarse semantic/query-aligned score
- sign sketch gives a nearly free refinement signal
- combined design is small and hardware-friendly

For V1, this low-precision summary may be implemented as an `int8 + scale` approximation rather than a true hardware-native FP8 datatype. The summary is metadata for selection, so determinism, compactness, and stable scoring are more important in V1 than exact FP8 storage semantics.

In the current V1 scaffold, Stage A should treat this representation as an approximate, scale-aware similarity signal rather than a true cosine over original full-precision summaries. That is acceptable for V1 because the selector is heuristic and instrumentation-first.

### Current dense-only ingest bridge

Before sparse runtime execution is integrated, the repository includes a local Hugging Face/PyTorch bridge for real-block selector smoke tests. The bridge runs dense prefill on a small causal LM and summarizes a configured per-token model-side representation into `BlockMetadata`.

Current sources include hidden-state streams, K/V-adjacent key streams built by mean-pooling attention keys across heads for selected layers, and query-key modes that compare latest-token attention query projections against key-derived block summaries. This is still an interim metadata source, not a replacement for true sparse runtime integration. Future runtime adapters still need to expose actual K/V cache payloads, page tables, sparse page lists, and dense fallback execution behind backend-specific adapters.

Future variants:
- mean-pooled keys
- PCA / low-rank summary
- learned summary embedding
- latent projection summary

## Selector pipeline

### Stage A — coarse score
Use a cheap weighted combination of:
- approximate low-precision summary similarity
- recency
- attention EMA
- priority

### Stage B — shortlist refinement
Refine a shortlist using:
- Hamming similarity on sign sketch
- optional later mini-dot refinement

### Stage C — final selection
Final set should always include:
- recent blocks
- anchor blocks

Then add:
- semantic top-K blocks

## Fallback policy

Fallback must be **graded**:

1. widen K
2. add more recent blocks
3. dense fallback

Fallback should trigger from selector uncertainty, not only from obviously bad final outputs.

## Sparse execution path

Selected blocks are converted into a page list compatible with the backend sparse execution path.

Requirements:
- selected pages should be sorted for better locality
- sparse path should have a clean dense fallback
- cold/prefetched/offloaded tiers can be stubbed in V1 and expanded in V2

## Dense oracle path

The architecture should support a dense path that can answer:

- what would dense attention have used?
- what blocks did sparse miss?
- what is selector recall rate?
- how often did sparse selection disagree with dense importance patterns?

This is central to future V3 learned routing.

## Immediate V1 success checks

A V1 build is not “working” just because it runs.

A useful V1 should show:
- logical KV bytes/token down
- measured DRAM bytes/token down
- decode throughput up or TPOT down
- acceptable quality retention
- manageable selector overhead
- informative access traces for future learning
