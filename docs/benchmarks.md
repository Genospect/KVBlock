# Benchmark Protocol

## Benchmark objective

The benchmark suite exists to answer one question:

**Does block-sparse selection reduce real decode-time memory traffic enough to produce runtime benefit without unacceptable quality loss?**

We care about both:
- intended sparse behavior
- actual hardware effects

## Metric categories

### Logical metrics
These describe what the algorithm intended to do.

- **KV capacity bytes**
- **logical KV read bytes/output token**
- **selected pages/output token**
- **estimated alpha**

### Runtime metrics
These describe actual serving behavior.

- **TTFT**
- **TPOT**
- **tokens/sec**
- **selector latency**
- **sparse execution latency**
- **fallback frequency**

### Hardware metrics
These describe actual memory movement.

- **DRAM bytes read/output token**
- **DRAM bytes write/output token**
- **occupancy** where available

### Energy / cost metrics
- **joules/token**
- **relative $/token**
- optional actual dollar estimate if instance price is known

## Primary success metrics

The most important V1 metrics are:

1. logical KV read bytes/output token
2. measured DRAM bytes/output token
3. TPOT / decode throughput
4. quality retention

If logical bytes fall but DRAM bytes do not, the sparse design has not yet translated to a real hardware win.

## Workload buckets

### 1. Synthetic needle-in-a-haystack
Purpose:
- verify selector can recover a distant exact block from noise
- good early sanity check for summaries and Stage-A weights

Suggested first test:
- insert a unique enzyme or identifier into a long noisy context
- force selector to find the containing block

### 2. Long-context reasoning / retrieval
Purpose:
- test whether sparse decode retains useful far-context access
- measure quality-bandwidth tradeoff on realistic long prompts

### 3. Code tasks
Purpose:
- strong practical target
- mixes locality with occasional long-range reference

### 4. Multi-turn repeated reference
Purpose:
- test if older blocks become important again after several turns
- stress recency bias vs memory persistence

### 5. Adversarial / semantically misleading prompts
Purpose:
- test false positives in summary-driven selection
- measure robustness against misleading semantic similarity

### 6. RoPE-sensitive position tests
Purpose:
- test whether block summaries break when meaning depends on position
- useful when comparing pre-RoPE vs post-RoPE summary strategies

### 7. Head-diverse prompts
Purpose:
- test whether one global selector misses blocks needed by different heads/layers

## Baselines

Compare against:

- dense paged attention at native precision
- dense paged + FP8 KV
- dense paged + 4-bit KV if supported
- sliding-window / cyclic KV
- H2O-style eviction
- FIER-style retrieval if feasible
- ClusterAttn-style prompt compression if feasible
- BLASST-like threshold sparse baseline if feasible

Keep MLA as a reference point, not a direct V1 serving baseline.

## Microbench suite

### A. Selector microbench
What it tests:
- Stage A/B/C cost as block count scales
- effect of summary dimension
- effect of shortlist size M

Why it matters:
- selector overhead must stay small enough to preserve bandwidth gains

### B. Sparse decode kernel microbench
What it tests:
- effect of controlled alpha
- context length vs selected blocks
- sparse execution locality and runtime

Why it matters:
- isolates hardware benefit of sparse selection

### C. End-to-end prefill + decode benchmark
What it tests:
- full serving behavior under real workloads
- interaction among selector cost, sparse execution, fallback, and model quality

Why it matters:
- final practical evaluation

### D. Real-block selector smoke test
What it tests:
- local dense prefill on a small causal LM
- metadata creation from real model-side token representations
- selector behavior on real block records before sparse execution exists

Why it matters:
- bridges synthetic selector microbenchmarks to real model-derived metadata
- keeps early validation CPU/Mac-safe
- makes representation and trace assumptions inspectable before K/V runtime integration

## Current local benchmark path

The current best local dense-only selector baseline is tracked in `docs/current-best-path.md`.

Use `scripts/run_dynamic_block_benchmark.py` for the active fixed-block validation path. By default, new dynamic-block benchmark outputs should be written under `results/dynamic_blocks/`; current best baseline runs can be written under `results/baselines/`.

Current active settings for local CPU/Mac validation are:

- `representation_source = query_mean_last_layer`
- `qk_aggregation = block_max`
- `block_mode = fixed_40`
- pooled `mean_heads`
- reduced/no rails when isolating routing quality

Historical ablations for representation sources, head weighting, query/key aggregation, and dynamic block modes remain useful but are not the current baseline.

## Ablations

Run these across versions where possible:

- block size: 16 / 32 / 64 / 128
- summary type
- shortlist M
- final K
- recent-block count
- anchor count
- precision tier
- recency weighting
- per-layer vs shared selector
- confidence policy
- fallback policy

Each ablation should report:
- quality
- logical bytes/token
- DRAM bytes/token
- TPOT / throughput

## Dense oracle evaluation

The dense oracle benchmark is critical.

Purpose:
- compare sparse block selection to dense reference behavior
- compute selector recall rate
- generate training data for a future learned router

This is one of the highest-leverage components of V1.

## Profiling tools

Recommended:
- **Nsight Compute**
- **CUPTI** if needed later
- **NVML** for power sampling
- runtime allocator logs from backend engine

## Suggested first-week benchmark order

1. selector microbench
2. synthetic needle test
3. dense baseline vs sparse baseline on small context
4. Nsight DRAM-byte profiling on one controlled workload
5. long-context and code workloads
6. multi-turn repeated-reference tests
