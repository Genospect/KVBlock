# Current Architecture Inventory

This inventory captures the pre-package research modules that should remain
available while KVBlock grows a stable package-facing API.

| Path | Main classes/functions | Current role | Recommendation |
| --- | --- | --- | --- |
| `src/kvblock/kv/metadata.py` | `BlockMetadata` | V1 metadata record with summaries, sketch, EMA/variance, and debug counters. | Keep; re-export through `kvblock.blocks.metadata`. |
| `src/kvblock/kv/block_types.py` | `BlockId`, `TokenSpan`, `BlockReference` | Internal typed ids/spans used by existing selector tests. | Keep; avoid forcing these into the public API too early. |
| `src/kvblock/kv/qk_aggregation.py` | QK aggregation helpers | Research path for block-level query/key scoring. | Keep; later wrap behind package selectors. |
| `src/kvblock/selector/pipeline.py` | `SelectorPipeline`, `SelectorPipelineConfig` | Inspectable Stage A/B/C selector with confidence and fallback. | Keep; expose through package adapter rather than rewrite. |
| `src/kvblock/selector/stage_a.py` | `StageAScorer` | Coarse shortlist scoring. | Keep; benchmark and trace source. |
| `src/kvblock/selector/stage_b.py` | `StageBRefiner` | Sign-sketch/Hamming refinement. | Keep; benchmark and trace source. |
| `src/kvblock/selector/stage_c.py` | `StageCSelector` | Recent/anchor rails plus semantic final selection. | Keep; package expansion utilities mirror this behavior. |
| `src/kvblock/selector/confidence.py` | `ConfidenceEvaluator` | Confidence margin and mass checks. | Keep; align package fallback helpers with it. |
| `src/kvblock/selector/fallback.py` | `GradedFallbackController` | Widen/add-recent/dense fallback control. | Keep; do not silently stay sparse on low confidence. |
| `src/kvblock/selector/oracle.py` | Dense-oracle helpers | Access-trace and selector quality comparison path. | Keep; important for V1 traces and later learned-router data. |
| `src/kvblock/selector/trace.py` | `SelectorDecisionTrace` and split/score traces | Source of inspectable selector decisions. | Keep; embed in `SelectedKVPlan.metadata` when adapting. |
| `src/kvblock/summaries/*` | FP8 summary and sign-sketch generation | Summary/scoring inputs for selector experiments. | Keep; package selectors should reuse this. |
| `src/kvblock/benchmark/longbench_output.py` | LongBench output benchmark harness | Output-quality benchmark path. | Keep; promote stable pieces into `kvblock.bench` gradually. |
| `src/kvblock/benchmark/answer_metrics.py` | Answer metrics | Current LongBench-oriented answer scoring. | Keep; reconcile with `kvblock.metrics.quality`. |
| `src/kvblock/benchmark/selector_microbench.py` | Synthetic selector microbenchmarks | Selector latency/quality microbenchmarks. | Keep; package CLI can call it later. |
| `scripts/run_*benchmark*.py` | Benchmark entry scripts | Research and report generation entrypoints. | Keep; avoid invasive churn until package API stabilizes. |
| `results/` | Historical experiment outputs | Reproducibility and comparison artifacts. | Keep; do not rewrite. |

Package-facing additions should consume these modules through adapters and stable
objects like `KVBlockPolicy`, `BlockLayout`, and `SelectedKVPlan`.
