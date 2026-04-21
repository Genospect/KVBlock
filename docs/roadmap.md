# Roadmap

## Guiding principle

Build **instrumentation before intelligence**.

The highest-leverage output of V1 is not just a speedup.  
It is an **access trace** and a measured understanding of:

- which blocks matter
- when they matter
- how stable that usefulness is
- how sparse selection maps to real DRAM traffic

That data makes V2 and V3 intelligent instead of speculative.

---

## V1 — heuristic, bandwidth-first

### Goal
Prove real decode-bandwidth reduction on a standard transformer.

### Build
- standard transformer backend
- local dense-only real-block ingest bridge
- paged KV
- FP16 or FP8 payload
- 32-token blocks
- metadata schema
- 3-stage heuristic selector
- recent/anchor rails
- widening + fallback
- full profiling harness
- dense oracle comparison path

### Key success criteria
- lower logical KV bytes/token
- lower measured DRAM bytes/token
- lower TPOT or higher decode throughput
- acceptable quality retention
- useful access traces collected

### Key open questions
- is block size 32 best?
- is FP8 summary + 64-bit sign sketch the best V1 combo?
- what confidence signal works best?
- shared vs per-layer selector?
- how often does fallback trigger?

### Current transition status

The current local dense-only selector baseline is `query_mean_last_layer` + `block_max` + `fixed_40`, documented in `docs/current-best-path.md`.

The next V1 phase is GPU and larger-model validation of this baseline before sparse runtime execution. FlashInfer/vLLM integration, page-list execution, and hardware-byte profiling remain intentionally deferred until the selector baseline is validated beyond local CPU/Mac experiments.

---

## V2 — adaptive precision and stronger summaries

### Goal
Improve the quality-bandwidth tradeoff without changing the base model architecture.

### Planned upgrades
- adaptive precision by block hotness
- stronger summary variants
- compare true FP8 summary storage against the initial int8+scale summary approximation
- compare richer similarity refinements against the initial scale-aware heuristic similarity path
- better RoPE-aware summary handling
- optional mini-dot or richer refinement
- temporal reuse logic
- prefetch / promotion hooks
- task-specific tuning if needed

### Key success criteria
- lower DRAM bytes/token than V1
- same quality at lower alpha, or better quality at same alpha
- fewer unnecessary fallback events
- improved repeated-reference behavior

### Key open questions
- how much does adaptive precision still help after sparsity?
- can RoPE-free side-channel summaries be added cleanly?
- how stable are block-use patterns across decode steps?
- is Hamming enough, or is richer refinement needed?

---

## V3 — learned or compression-native memory improvements

### Goal
Move beyond pure heuristic selection into learned or compressed/native memory systems.

### Planned directions
- learned router trained from dense traces
- SALS-like latent summary experiments
- SWAN-style direct compressed-use experiments
- maybe architecture-adjacent memory work later if justified

### Key success criteria
- selector or latent routing overhead stays low
- further bandwidth reduction beyond V2
- no major recall collapse
- enough measured evidence to justify more ambitious redesigns

### Key open questions
- when does a learned router beat the heuristic enough to justify complexity?
- is SALS-style latent selection worth its integration cost?
- can SWAN-like direct compressed-use memory complement sparse selection?
- when would MLA-like native memory become more attractive than explicit KV optimization?

---

## Suggested near-term build schedule

### Week 1
- backend/runtime adapter
- local dense-only Hugging Face/PyTorch block-ingest bridge
- paged KV block table
- metadata schema

### Week 2
- summary generation
- sign sketch generation
- Stage A + Stage B selector

### Week 3
- Stage C selection
- graded fallback
- dense oracle benchmark path

### Week 4
- Nsight / NVML instrumentation
- first DRAM-byte profiling pass
- synthetic needle and small-context validation

### Week 5+
- end-to-end workloads
- ablations
- V2 design decisions based on real traces
