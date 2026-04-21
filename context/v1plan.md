# V1 Plan (Historical / Reasoning Context)

This document captures the original reasoning and planning behind V1.

For the current authoritative design, see:
- docs/architecture.md
- docs/benchmarks.md
- docs/roadmap.md



Yep. For **V1**, I would make this a **Python-first research codebase** with a **thin systems layer** for profiling/integration, and only add custom CUDA/Triton once the selector logic is proven. That matches the report’s recommendation to keep V1 heuristic, bandwidth-first, and built on a standard transformer with paged KV, FP16/FP8 payloads, hybrid selector logic, and strong benchmarking. 

## Recommended stack

### Core language choices

**Python**
for:

* orchestration
* experiments
* model/runtime integration
* benchmarking
* selector logic
* analysis and plotting

**PyTorch**
for:

* model access
* tensor ops
* hooks if you need pre-RoPE or query-summary side channels
* prototyping summary generation

**vLLM or FlashInfer-backed runtime**
for:

* paged KV engine
* sparse/paged infrastructure
* realistic decode path

**Triton or CUDA later**
for:

* optimized block-sparse kernels
* custom refinement kernels
* reducing selector overhead if Python becomes a bottleneck

**JSON / YAML / TOML**
for experiment configs

**DuckDB or Parquet**
for logging benchmark runs and easy analysis

## Best practical choice

For V1, I’d structure it as:

* **Python**
* **PyTorch**
* **FlashInfer or vLLM integration**
* **Nsight / CUPTI / NVML tooling wrappers**
* **no custom training code required initially**

That fits the report’s point that current engines already expose enough block tables and sparse/paged hooks to prototype a block-sparse design, and that the V1 success metric is lower **hardware DRAM bytes/output token** plus decode speedup at similar quality. 

---

# V1 codebase outline

## Top-level repo structure

```text
kv-scout/
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
├── scripts/
├── src/
├── tests/
├── benchmarks/
├── notebooks/
├── docs/
├── data/
└── results/
```

---

## `README.md`

Contains:

* project goal
* V1 scope
* current architecture
* setup steps
* quick benchmark example
* roadmap V1/V2/V3

---

## `pyproject.toml`

Use this if you want a clean Python package setup.

Include:

* package name
* dependencies
* linting/formatting
* test config

Likely deps:

* `torch`
* `transformers`
* `vllm` and/or `flashinfer`
* `numpy`
* `pydantic`
* `typer` or `click`
* `rich`
* `pandas`
* `duckdb`
* `pyyaml`
* `matplotlib`
* optional: `triton`

---

# `configs/`

```text
configs/
├── model/
│   ├── llama3_8b.yaml
│   ├── mistral_7b.yaml
│   └── qwen2_7b.yaml
├── selector/
│   ├── v1_fp8_sign32.yaml
│   ├── v1_fp16_dense_baseline.yaml
│   ├── v1_hamming_only.yaml
│   └── v1_mini_dot_refine.yaml
├── benchmark/
│   ├── longbench.yaml
│   ├── needle.yaml
│   ├── code.yaml
│   ├── multiturn_reuse.yaml
│   └── synthetic_adversarial.yaml
└── runtime/
    ├── flashinfer.yaml
    └── vllm.yaml
```

Purpose:

* keep experiments reproducible
* let you sweep block size, K, summary type, precision, fallback settings

The report explicitly recommends ablations over block size, summary type, shortlist M, final K, recent/anchor counts, precision tier, and selector style, so configs should make those easy to vary. 

---

# `scripts/`

```text
scripts/
├── run_benchmark.py
├── run_selector_microbench.py
├── run_kernel_microbench.py
├── build_summary_cache.py
├── replay_dense_attention.py
├── profile_run.py
├── compare_results.py
└── export_report.py
```

### What each does

`run_benchmark.py`

* end-to-end prefill + decode benchmark
* main CLI entrypoint

`run_selector_microbench.py`

* measures selector latency as block count, summary dim, shortlist M scale

`run_kernel_microbench.py`

* synthetic controlled-alpha sparse decode tests

`build_summary_cache.py`

* generates block summaries / sign sketches during prefill or from saved KV traces

`replay_dense_attention.py`

* dense baseline runs for oracle comparison
* useful for future learned routing targets

`profile_run.py`

* wraps Nsight / NVML / runtime logging

`compare_results.py`

* aggregates baseline vs selector runs

`export_report.py`

* turns stored results into CSV/Markdown/plots

This mirrors the report’s recommended three-piece benchmark setup:

* decode-kernel microbench
* selector microbench
* end-to-end prefill+decode bench. 

---

# `src/`

## Core package layout

```text
src/kv_scout/
├── __init__.py
├── config/
├── runtime/
├── kv/
├── summaries/
├── selector/
├── sparse_exec/
├── benchmark/
├── profiling/
├── analysis/
├── datasets/
├── utils/
└── cli/
```

---

## `src/kv_scout/config/`

```text
config/
├── models.py
├── selector.py
├── benchmark.py
├── runtime.py
└── loader.py
```

Purpose:

* typed config parsing
* validation of experiment parameters

Use:

* `pydantic` or dataclasses

Key classes:

* `ModelConfig`
* `SelectorConfig`
* `BenchmarkConfig`
* `RuntimeConfig`

---

## `src/kv_scout/runtime/`

```text
runtime/
├── base.py
├── flashinfer_runtime.py
├── vllm_runtime.py
├── model_loader.py
├── paged_kv_adapter.py
└── hooks.py
```

Purpose:

* abstract over the backend
* expose paged KV access
* get current query state
* register summary-building hooks
* call sparse or dense attention path

### Important design point

You want a backend interface like:

```python
class RuntimeBackend:
    def prefill(...)
    def decode_step(...)
    def get_block_table(...)
    def get_query_summary(...)
    def run_dense_attention(...)
    def run_sparse_attention(...)
```

This keeps the selector independent of backend choice.

The report points to FlashInfer as the most practical sparse-kernel substrate, while also grounding the design in paged runtimes generally. 

---

## `src/kv_scout/kv/`

```text
kv/
├── block_types.py
├── metadata.py
├── block_table.py
├── block_manager.py
├── hotness.py
├── precision_tiers.py
└── rope_utils.py
```

This is one of the most important modules.

### `block_types.py`

Define typed structures for:

* block IDs
* token ranges
* block/page references
* layer-group references

### `metadata.py`

Implements the report’s proposed metadata schema.

Suggested dataclass:

```python
@dataclass
class BlockMetadata:
    block_id: int
    pool_id: int
    token_start: int
    token_len: int
    precision_tier: int
    flags: int
    summary_fp8: np.ndarray  # shape [32]
    sign_sketch: np.uint64
    summary_norm: np.float16
    attn_ema: np.float16
    attn_var: np.float16
    last_access_step: int
    hit_count: int
    priority: int
    rope_bucket: int
    fallback_miss_count: int
```

That is directly aligned with the V1 schema proposed in the report. 

### `block_table.py`

Maps:

* logical token ranges
* block IDs
* runtime page indices
* layer/head-window pools

### `block_manager.py`

Handles:

* creation
* updates
* summary attachment
* page lookup
* recent/anchor tagging

### `hotness.py`

For V1:

* update EMA
* update hit count
* recency scores
  For V2:
* precision promotion / demotion logic

### `precision_tiers.py`

Start simple:

* FP16
* FP8
  Later:
* 4-bit
* offloaded / cold

### `rope_utils.py`

Contains:

* rope bucket helpers
* inverse-RoPE approximations if attempted
* utilities for any RoPE-free side channel summary generation

---

## `src/kv_scout/summaries/`

```text
summaries/
├── base.py
├── fp8_summary.py
├── sign_sketch.py
├── mean_pool.py
├── pca_summary.py
├── latent_summary.py
└── summary_builder.py
```

Purpose:

* create block summaries at prefill or block creation time

### V1

Implement only:

* `fp8_summary.py`
* `sign_sketch.py`
* maybe `mean_pool.py` for ablation

### V2

Add:

* `pca_summary.py`
* `latent_summary.py`
* maybe learned summary embedding

The report recommends the practical V1 hybrid:

* **32-d FP8 vector**
* **64-bit sign sketch**
  and treats more advanced summary types as ablation or later-stage options. 

---

## `src/kv_scout/selector/`

```text
selector/
├── base.py
├── stage_a.py
├── stage_b.py
├── stage_c.py
├── confidence.py
├── fallback.py
├── policies.py
└── oracle.py
```

This is the heart of V1.

### `stage_a.py`

Implements cheap coarse scoring:

[
score_A(b)=w_q \cdot \cos(q_s,s_b)+w_r \cdot recency+w_h \cdot attn_ema+w_p \cdot priority
]

as proposed in the report. 

### `stage_b.py`

Implements refinement:

* Hamming similarity on sign sketch
* optional mini-dot variant for ablation

### `stage_c.py`

Build final selected set:

* recent blocks
* anchors
* top-K semantic blocks

### `confidence.py`

Implements:

* score margin
* maybe normalized mass
* maybe miss-history penalties

### `fallback.py`

Implements graded fallback:

1. widen K
2. add more recent blocks
3. dense fallback

The report explicitly recommends graded fallback, not binary. 

### `policies.py`

Stores policies like:

* `RecentAnchorHybridPolicy`
* `DenseBaselinePolicy`
* `HammingOnlyPolicy`
* `MiniDotRefinePolicy`

### `oracle.py`

Very useful.
Lets you compare selector choices with:

* dense attention traces
* dense-selected “important” pages
* later learned routing targets

---

## `src/kv_scout/sparse_exec/`

```text
sparse_exec/
├── base.py
├── page_list_builder.py
├── flashinfer_sparse.py
├── dense_fallback.py
├── prefetch.py
└── layout.py
```

Purpose:

* convert selector output into runtime/kernel-friendly sparse execution

### `page_list_builder.py`

Builds:

* selected page indices
* position-sorted page lists
* layout required by backend sparse kernel

### `flashinfer_sparse.py`

Wraps:

* FlashInfer paged/block-sparse attention path

### `dense_fallback.py`

Standard dense attention path for:

* comparison
* fallback
* dense refresh

### `prefetch.py`

For V1 this can be a stub or simple placeholder.
For V2 it becomes important for cold-page prefetch / promotion.

The report specifically recommends position-sorted selected pages and optional prefetch/promotion when cold pages or remote tiers are used. 

---

## `src/kv_scout/benchmark/`

```text
benchmark/
├── harness.py
├── workloads.py
├── metrics.py
├── baselines.py
├── ablations.py
├── kernel_microbench.py
├── selector_microbench.py
└── e2e.py
```

### `harness.py`

Implements the benchmark structure the report sketches:

* model
* workload
* context length
* block size
* selector config
* profiling on/off

### `workloads.py`

Contains workload interfaces for:

* long-context retrieval/reasoning
* needle-in-haystack
* code
* repeated-reference multiturn
* synthetic adversarial

### `metrics.py`

Compute:

* allocator bytes
* logical KV read bytes/token
* tokens/sec
* TTFT
* TPOT
* correctness / task metrics
* maybe alpha estimate

### `baselines.py`

Implements comparisons against:

* dense paged attention
* FP8 dense
* sliding window / cyclic KV if feasible
* maybe H2O/FIER/BLASST-inspired modes later

### `ablations.py`

Sweeps:

* block size
* summary type
* shortlist M
* K
* recency weighting
* anchor count
* precision tier
* selector style

The report recommends all of these. 

### `kernel_microbench.py`

Synthetic alpha-controlled test

### `selector_microbench.py`

Selector-only scaling test

### `e2e.py`

Full prefill+decode run

---

## `src/kv_scout/profiling/`

```text
profiling/
├── nsight.py
├── cupti.py
├── nvml.py
├── runtime_logs.py
└── trace.py
```

Purpose:

* wrap the hardware and runtime measurement path

### `nsight.py`

Helpers to invoke / parse Nsight Compute runs

### `cupti.py`

Optional if you want lower-level counters later

### `nvml.py`

Energy / power sampling

### `runtime_logs.py`

Parse allocator-side information from backend runtime logs

The report explicitly calls out:

* `dram__bytes_read.sum`
* `dram__bytes_write.sum`
* NVML / TokenPowerBench style energy measurement
* allocator-side bytes and page counts. 

---

## `src/kv_scout/analysis/`

```text
analysis/
├── aggregate.py
├── plots.py
├── tables.py
├── regressions.py
└── report.py
```

Purpose:

* convert raw runs into useful engineering conclusions

Outputs:

* bytes/token vs quality plots
* TTFT/TPOT comparisons
* selector overhead scaling
* best config tables
* alpha vs quality tradeoff

---

## `src/kv_scout/datasets/`

```text
datasets/
├── base.py
├── longbench.py
├── needle.py
├── code_tasks.py
├── multiturn_reuse.py
└── synthetic_adversarial.py
```

### `multiturn_reuse.py`

Probably synthetic at first, because the report notes this is not neatly packaged in public datasets and may need to be created. 

---

## `src/kv_scout/utils/`

```text
utils/
├── logging.py
├── timers.py
├── seed.py
├── serialization.py
└── math_utils.py
```

Boring but necessary.

---

## `src/kv_scout/cli/`

```text
cli/
├── main.py
├── benchmark.py
├── profile.py
├── summarize.py
└── inspect.py
```

Suggested commands:

```bash
kv-scout benchmark --config configs/benchmark/needle.yaml
kv-scout profile --run-id run_001
kv-scout summarize --model llama3_8b --selector v1_fp8_sign32
kv-scout inspect-blocks --run-id run_001
```

---

# `benchmarks/`

```text
benchmarks/
├── dense_baseline/
├── fp8_dense/
├── sparse_v1/
├── sparse_v1_hamming_only/
├── sparse_v1_mini_dot/
└── ablations/
```

This is where benchmark configs and reusable run definitions live.

---

# `tests/`

```text
tests/
├── test_metadata.py
├── test_sign_sketch.py
├── test_selector_stage_a.py
├── test_selector_stage_b.py
├── test_fallback.py
├── test_logical_bytes.py
├── test_page_list_builder.py
├── test_runtime_adapter.py
└── test_benchmark_harness.py
```

## What to test first

For V1, the most important tests are:

### Unit tests

* metadata schema serialization
* sign sketch determinism
* FP8 summary creation
* selector scoring stability
* fallback triggering
* logical bytes/token calculation

### Integration tests

* selected pages map correctly into backend page list
* dense fallback path matches baseline path wiring
* benchmark harness logs required metrics
* sparse run never silently drops recent/anchor rails

---

# `notebooks/`

```text
notebooks/
├── 01_metadata_exploration.ipynb
├── 02_selector_costs.ipynb
├── 03_quality_vs_alpha.ipynb
├── 04_block_size_sweep.ipynb
└── 05_v1_summary_ablation.ipynb
```

Use notebooks for:

* fast analysis
* plots
* sanity checks
* not production logic

---

# `results/`

```text
results/
├── raw/
├── processed/
├── plots/
└── reports/
```

Store:

* raw benchmark logs
* summarized tables
* plot artifacts
* markdown reports

---

# Build order for the codebase

If I were implementing this, I’d do it in this order:

## Phase 1 — plumbing

1. `config/`
2. `runtime/`
3. `kv/metadata.py`
4. `summaries/fp8_summary.py`
5. `summaries/sign_sketch.py`

## Phase 2 — selector

6. `selector/stage_a.py`
7. `selector/stage_b.py`
8. `selector/stage_c.py`
9. `selector/confidence.py`
10. `selector/fallback.py`

## Phase 3 — execution path

11. `sparse_exec/page_list_builder.py`
12. `sparse_exec/dense_fallback.py`
13. `sparse_exec/flashinfer_sparse.py`

## Phase 4 — metrics and harness

14. `benchmark/metrics.py`
15. `benchmark/harness.py`
16. `profiling/runtime_logs.py`
17. `profiling/nvml.py`
18. `profiling/nsight.py`

## Phase 5 — workloads and analysis

19. `datasets/needle.py`
20. `datasets/code_tasks.py`
21. `datasets/multiturn_reuse.py`
22. `analysis/plots.py`
23. `analysis/tables.py`

That order matches the report’s logic:

* build the heuristic V1 first
* measure real DRAM bytes/token
* benchmark before adding fancy routing or architecture changes. 

---

# Final recommendation on language / architecture

For **V1**, don’t over-engineer this into a giant mixed-language systems repo yet.

## Best V1 setup

* **Python**
* **PyTorch**
* **FlashInfer or vLLM backend**
* **Nsight/NVML profiling wrappers**
* **minimal Triton/CUDA only if proven necessary**

## Why

Because the report’s whole point is that V1 should be:

* heuristic
* bandwidth-first
* standard transformer based
* measured by real DRAM bytes/output token and decode speedup
* with adaptive precision and learned routing coming only later 

So the codebase should reflect that:

* **fast to iterate**
* **easy to ablate**
* **easy to profile**
* **not locked into a speculative architecture too early**

If you want, next I can turn this into a **file-by-file outline** with each file’s purpose, inputs, outputs, and the first functions/classes to implement.
