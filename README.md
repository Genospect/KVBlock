# KVBlock

Block-sparse KV cache research project focused on **reducing decode-time bandwidth** in long-context LLM inference.

## What this project is

KVBlock is a systems research project exploring whether a **selector-driven block-sparse KV cache** can reduce the amount of key/value memory touched during autoregressive decode while preserving model quality.

The project is grounded in one practical goal:

- reduce **logical KV read bytes per output token**
- reduce **measured DRAM bytes per output token**
- improve **TPOT / decode throughput**
- reduce relative **$/token** in decode-bound settings

This project is **not** starting from a new model architecture. V1 targets a **standard transformer** plus a better runtime memory access policy.

## Core hypothesis

In long-context inference, the main bottleneck is often no longer just storing the KV cache. The bottleneck is **reading enough KV quickly during decode**.

That means the highest-leverage variable is often not just KV capacity, but the fraction of dense KV actually touched per decode step.

We represent that fraction conceptually as:

```text
alpha in (0, 1]
```

Where:
- `alpha = 1` means dense decode reads all relevant KV
- `alpha << 1` means sparse decode reads only a selected subset of KV blocks

V1 exists to test whether we can drive `alpha` down in real workloads **without collapsing quality**.

## Current research baseline

The current best local dense-only selector baseline is documented in [`docs/current-best-path.md`](docs/current-best-path.md), with broader experiment triage in [`docs/experiment-status.md`](docs/experiment-status.md).

Current local CPU/Mac findings point to:

- `representation_source = query_mean_last_layer`
- `qk_aggregation = block_max`
- `block_mode = fixed_40`
- pooled `mean_heads` scoring

This is not yet sparse runtime execution. It is the baseline to validate next on GPU and larger cached models before adding FlashInfer/vLLM sparse execution.

## V1 scope

V1 is intentionally:

- **heuristic**
- **model-agnostic**
- **bandwidth-first**
- **benchmark-heavy**
- **instrumentation-first**

We are not starting with MLA, Mamba, or a learned router.

We are starting with:

- paged/block KV cache
- FP16 or FP8 KV payloads
- 32-token blocks by default
- lightweight per-block metadata
- a hybrid selector
- block-sparse decode
- graded fallback when selector confidence is weak

## V1 default design

Initial proposed defaults:

- **Block size:** 32 tokens
- **Summary vector:** 32-dimensional low-precision block summary (initially implemented as `int8 + scale` emulation of an FP8-style summary)
- **Sketch:** 64-bit sign sketch
- **Always-kept recent blocks:** 4
- **Always-kept anchors:** up to 2
- **Stage-A shortlist:** 24 blocks up to 32k context, 48 above 32k
- **Final semantic blocks:** 8 up to 32k, 16 above 32k
- **Confidence margin:** 0.05 normalized score gap
- **Dense refresh period (benchmarking):** every 32 decode steps
- **Payload precision:** FP16 or FP8 in V1

These are starting defaults, not permanent truths.

## Why V1 matters

The main value of V1 is not just a speedup.

The main value is **instrumentation** and **access traces**.

By logging:
- which blocks the heuristic selector chose
- which blocks dense attention would have effectively needed
- where the selector missed
- where fallback triggered
- and how logical bytes compare to actual DRAM bytes

...V1 creates the empirical foundation for:
- V2 adaptive precision
- better summary designs
- a V3 learned router
- and later architecture-level memory experiments

Without V1, later “intelligent” memory routing is mostly guessing.

## V1 metadata schema

Each block should have small metadata, roughly 96–128 bytes total.

Candidate fields:

- `block_id`
- `pool_id`
- `token_start`
- `token_len`
- `precision_tier`
- `flags`
- `summary_fp8[32]`
- `summary_scale`
- `sign_sketch`
- `summary_norm`
- `attn_ema`
- `attn_var`
- `last_access_step`
- `hit_count`
- `priority`
- `rope_bucket`
- `fallback_miss_count`

Important intuition:
- `attn_ema` tells us whether a block tends to matter
- `attn_var` tells us whether a block is stable or jittery
- `sign_sketch` provides a nearly free refinement signal
- `rope_bucket` protects against position-related summary drift

> Note: the current V1 implementation may use an `int8 + scale` representation as a practical stand-in for a true FP8 summary vector. This is acceptable for V1 because the summary is selector metadata, not the full KV payload. A true FP8 summary path can be added later as an ablation once selector behavior and benchmarking are in place.

## Current dense-only real-block bridge

The V1 scaffold now includes a CPU-safe local Hugging Face/PyTorch bridge that can:

- load a small causal LM locally
- run dense prompt prefill
- extract configured per-token model-side representations
- use hidden-state streams or mean-pooled attention-key streams as metadata sources
- optionally score key-derived block summaries with real attention query projections
- split those representations into token blocks
- build `BlockMetadata` using the existing low-precision summary and sign-sketch path
- run the existing selector pipeline over those real model-derived blocks

This bridge is intentionally **not** sparse runtime execution. It does not yet build backend page lists or call FlashInfer/vLLM kernels. Its purpose is to connect the synthetic selector lab to real model-side representations while keeping the current V1 path Mac-safe and inspectable.

## Selector design

V1 uses a three-stage selection pipeline.

### Stage A — coarse score
Cheap block-level score using:
- approximate low-precision summary similarity
- recency
- historical attention
- priority

In the current V1 scaffold, Stage A uses a scale-aware approximate similarity score derived from the stored `int8 + scale` summary metadata. It should be treated as a heuristic ranking signal for selector use, not as an exact cosine similarity over original full-precision summaries.

### Stage B — shortlist refinement
Refine shortlisted blocks with:
- Hamming similarity on sign sketch
- optional later mini-dot refinement

### Stage C — final selection
Always include:
- recent blocks
- anchor blocks

Then add:
- top-K semantic blocks

### Fallback
Fallback is **graded**, not binary.

Order of escalation:
1. widen K
2. add more recent blocks
3. dense fallback

This is critical. If confidence logic is wrong, either:
- sparse mode stays on too long and hurts quality
- or dense fallback happens too often and kills bandwidth savings

## Immediate build priorities

1. Establish backend/runtime adapter
2. Implement paged KV block table plumbing
3. Implement block metadata schema
4. Implement low-precision summary + 64-bit sign sketch
5. Implement Stage A + Stage B + Stage C selector
6. Implement sparse execution path
7. Implement dense oracle / dense baseline comparison
8. Implement full benchmark + profiling harness

## Success criteria

V1 is only successful if it demonstrates real runtime improvement.

Primary success metrics:

- lower **logical KV read bytes/token**
- lower **measured DRAM bytes/token**
- lower **TPOT**
- higher decode tokens/sec
- acceptable quality retention

Secondary success metrics:

- selector overhead remains small
- metadata overhead remains negligible
- fallback is useful but not overly frequent
- recent/anchor rails prevent catastrophic misses

If logical reads go down but measured DRAM bytes do not, then the design has not yet produced a real hardware win.

## Benchmark categories

At minimum, benchmark:

- long-context reasoning / retrieval
- synthetic needle-in-a-haystack
- code tasks
- multi-turn repeated reference
- synthetic adversarial prompts

Also include:
- selector microbench
- sparse kernel microbench
- full prefill+decode end-to-end benchmark

## Recommended stack

- **Python** for orchestration and experiments
- **PyTorch** for tensor/model integration
- **FlashInfer** preferred as V1 substrate
- **vLLM** optional comparison backend
- **Nsight / CUPTI / NVML** for profiling
- **DuckDB / Parquet** for run logging and analysis

## Version roadmap

### V1
Heuristic, bandwidth-first sparse KV selector on a standard transformer.

### V2
Adaptive precision by block hotness, better summaries, stronger RoPE-aware handling, reuse/prefetch/promotion.

### V3
Learned router, SALS/SWAN-style experiments, and only later architecture-native directions if justified by V1/V2 evidence.

## Non-goals for V1

V1 is not trying to solve:
- general reasoning brittleness
- tokenization or character-count failures
- commonsense reasoning failures
- full model architecture redesign

This is a **systems + memory bandwidth project**, not a universal model-quality fix.

## Suggested docs

- `AGENTS.md` — implementation instructions for Codex/agents
- `docs/architecture.md` — technical architecture
- `docs/benchmarks.md` — benchmark protocol
- `docs/roadmap.md` — V1/V2/V3 roadmap
- `context/v1plan.md` — working context and discussion notes

## Current recommendation

Start with V1.
Do not overcomplicate V1.
Get the instrumentation working.
Measure logical bytes/token and real DRAM bytes/token.
Only then move toward more advanced routing, latent summaries, or architecture changes.
