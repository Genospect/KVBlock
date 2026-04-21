# V1 Spec — Block-Sparse KV Decode Bandwidth Reduction

## Purpose

V1 exists to answer one question:

**Can a selector-driven block-sparse KV system reduce real decode-time memory traffic enough to improve decode performance without unacceptable quality loss?**

V1 is:
- heuristic
- instrumentation-first
- standard-transformer-based
- benchmark-heavy
- designed to generate access traces for later learned routing

V1 is **not**:
- a new architecture
- a learned router
- MLA/Mamba
- a general model-quality project

The immediate recommended V1 stack is:
- paged/block KV
- FP16 or FP8 payloads
- low-precision block summaries
- 64-bit sign sketch
- tiny RoPE-aware metadata
- block-sparse decode
- conservative fallback :contentReference[oaicite:0]{index=0}

---

## Primary success criteria

V1 is successful only if it demonstrates all or most of:

1. lower **logical KV read bytes / token**
2. lower **measured DRAM bytes / token**
3. lower **TPOT** and/or higher decode tokens/sec
4. acceptable quality retention on target workloads
5. selector overhead remains small relative to decode savings
6. generated access traces are useful for future V2/V3 work

If logical sparse reads go down but hardware DRAM bytes do not, V1 is not yet a real systems win. :contentReference[oaicite:1]{index=1}

---

## Resource model

Useful conceptual equations:

```text
KV capacity bytes ~= 2 * L * n_kv * d_h * T * b
KV read bytes / output token ~= alpha * 2 * L * n_kv * d_h * T * b
```

## Summary construction requirements

### V1 required implementations
- low-precision block summary
- 64-bit sign sketch
- optional mean-pooled summary for ablation only

### V1 note on precision
The initial V1 implementation may represent the block summary as an `int8 + scale` approximation of an FP8-style low-precision summary rather than a true FP8 storage format.

This is acceptable for V1 because:
- the summary is selector metadata, not the full KV payload
- V1 prioritizes deterministic behavior, compactness, and benchmarkability
- true FP8 summary storage can be added later as a targeted ablation
- selector scoring can use a cheap approximate similarity path over the stored low-precision summaries without blocking the broader V1 plan

### V1 summary rules
- summary must be cheap
- summary must be deterministic
- summary overhead must stay negligible vs KV block payload
- summary should ideally be built in a RoPE-free side channel
- if that is not possible, use coarse RoPE-aware correction such as `rope_bucket`

## Current real-block ingest bridge

The initial real-runtime bridge is dense-only and Mac-safe. It may use a local Hugging Face/PyTorch causal LM to run prompt prefill, extract a configured per-token model-side representation, and summarize that representation into V1 `BlockMetadata`.

Supported bridge sources include hidden-state streams, K/V-adjacent key streams that mean-pool attention keys across heads for selected layers, and query-key modes that build block summaries from keys while using the latest real attention query projection as the query summary. This bridge exists to connect the selector lab to real model-derived representations. It is not yet sparse page-list construction, FlashInfer/vLLM execution, or a dense runtime oracle.
