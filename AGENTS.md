# AGENTS.md

This repository is a research/prototyping codebase for **block-sparse KV cache systems** focused on **decode-bandwidth reduction**.

## Mission

Build a V1 system that reduces **decode-time KV reads** on a standard transformer using:

- paged/block KV cache
- FP16 or FP8 payloads
- per-block metadata
- hybrid selector
- sparse attention execution
- graded widening / fallback
- strong benchmarking and profiling

## First principle

Optimize **bytes touched during decode**, not just bytes stored.

A change is only a V1 win if it helps at least one of:

- logical KV read bytes/token
- measured DRAM bytes/token
- TPOT / decode throughput
- quality-bandwidth tradeoff

If a change adds complexity without helping those, defer it.

## V1 guardrails

### V1 must remain:
- standard transformer based
- heuristic
- instrumentation-heavy
- benchmarkable
- easy to inspect/debug

### Avoid in V1:
- learned router training
- MLA/Mamba architecture changes
- speculative custom memory object redesign
- heavy custom CUDA work before selector correctness is proven
- bloated abstractions that slow iteration

## Preferred substrate

Prefer **FlashInfer** for V1 because it exposes paged KV and sparse primitives more directly.

vLLM can still be useful as:
- comparison backend
- dense baseline engine
- reference integration path

But the repo should be designed so backend-specific logic lives behind adapters.

## Core V1 defaults

Unless an experiment explicitly overrides them, assume:

- block size = 32
- summary = 32-d FP8
- sign sketch = 64-bit
- keep recent blocks = 4
- keep anchors = up to 2
- stage A shortlist = 24 blocks up to 32k, 48 above 32k
- final semantic K = 8 up to 32k, 16 above 32k
- confidence margin = 0.05
- payload precision = FP16 or FP8

## Selector invariants

The selector must always support:

- recent blocks rail
- anchor blocks rail
- stage A coarse scoring
- stage B shortlist refinement
- stage C final selection
- graded fallback

If confidence is weak:
1. widen K
2. add more recent blocks
3. dense fallback

Do not silently stay sparse when confidence is low.

## Dense oracle requirement

A major purpose of V1 is to generate **access traces**.

The codebase should support comparison against a **dense oracle** so that later versions can learn from:
- selector choices
- misses
- block usefulness patterns
- fallback frequency
- disagreement between sparse and dense selection

This is a critical V1 design principle.

## Metadata rules

The V1 metadata record should be lightweight and explicit.

Target fields include:
- block_id
- pool_id
- token_start
- token_len
- precision_tier
- flags
- summary_fp8[32]
- sign_sketch
- summary_norm
- attn_ema
- attn_var
- last_access_step
- hit_count
- priority
- rope_bucket
- fallback_miss_count

`attn_ema` and `attn_var` are important and should not be dropped casually.

## RoPE handling rules

Do not assume post-RoPE pooled keys are safe universal summaries.

If possible:
- compute summaries in a pre-RoPE or RoPE-free side channel

If not possible:
- use coarse RoPE-aware correction or buckets
- avoid pretending all pooled keys are position invariant

## Benchmark-first mindset

No major design decision should be accepted without measurement.

At minimum, benchmark logic should expose:
- logical KV read bytes/token
- selected pages/token
- TTFT
- TPOT
- tokens/s
- selector latency
- hardware DRAM bytes/token
- correctness / quality metrics

## Implementation order

### Phase 1 — plumbing
- config loading
- runtime adapter
- block metadata
- summary generation
- sign sketch generation

### Phase 2 — selector
- stage A
- stage B
- stage C
- confidence scoring
- graded fallback

### Phase 3 — sparse execution
- selected page list builder
- sparse execution wrapper
- dense fallback path

### Phase 4 — instrumentation
- benchmark harness
- logical bytes calculator
- runtime log capture
- NVML / Nsight integration

### Phase 5 — workloads and analysis
- needle workload
- code workload
- repeated reference workload
- plotting / result tables

## Code quality rules

- prefer small, testable modules
- keep runtime-specific code isolated
- keep configs explicit and reproducible
- avoid giant notebook-style files in `src/`
- every selector decision should be inspectable
- every metric should have a clear source of truth

## Testing rules

Write tests for:
- metadata schema correctness
- summary generation
- sign sketch determinism
- stage A scoring
- stage B refinement
- stage C selection
- fallback logic
- logical bytes/token calculations
- page list building
- benchmark harness outputs

## Decision rule for new features

A new feature belongs in V1 only if it satisfies at least one:

1. It is required to make V1 runnable
2. It improves correctness/debuggability
3. It improves instrumentation or profiling quality
4. It improves the quality-bandwidth tradeoff
5. It is necessary scaffolding for expected V2 work

Otherwise defer it.

## V2/V3 context

V2:
- adaptive precision by block hotness
- better summaries
- stronger RoPE-aware handling
- stronger refinement
- reuse / prefetch / promotion

V3:
- learned router from dense traces
- SALS-like latent summaries
- SWAN-style direct compressed-use experiments
- architecture-native memory only if justified by data

Do not implement V2/V3 ideas prematurely unless behind clean extension points.

## Expected output from Codex

When building:
- prioritize a runnable baseline over speculative sophistication
- make benchmarkability a first-class concern
- leave clear extension points for V2/V3
- prefer concrete modules over theoretical scaffolding
