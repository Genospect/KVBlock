# Physical Sparse Decode Roadmap

## v0.1: Policy and Correctness

- Stabilize `SelectedKVPlan` as the selector/backend contract.
- Compare dense attention with `torch_sparse_reference` on synthetic Q/K/V.
- Report selected token fraction and logit divergence.
- Keep all acceleration language qualified as estimated until a physical sparse
  backend skips reads.

## v0.2: Kernel-Level POC

- Map logical block ids to physical page ids.
- Use FlashInfer where possible for paged/sparse primitives.
- If FlashInfer APIs are unavailable or insufficient, prototype indexed K/V
  gather in Triton before attempting a direct sparse decode attention kernel.
- Measure dense attention time, gather time, sparse attention time, and total
  decode-step time with CUDA events.

## v0.3: Runtime Adapter Preview

- Observe runtime block/page tables without invasive integration.
- Emit multi-layer selected plans.
- Compare selected pages with dense-oracle access traces.
- Defer vLLM/SGLang integration until the synthetic backend shows a real kernel
  path that reduces K/V reads.

## v1.0 Criteria

KVBlock should only claim real acceleration when measured decode latency improves,
quality stays within threshold, fallback works, multiple context lengths/models
are tested, and the backend physically skips unselected KV reads.
