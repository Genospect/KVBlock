from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import kvblock.benchmark.qk_aggregation_benchmark as qk_bench
from kvblock.benchmark.qk_aggregation_benchmark import (
    QKAggregationBenchmarkResult,
    QKAggregationRunRow,
    QKAggregationSummary,
    format_qk_aggregation_report,
    run_qk_aggregation_benchmark,
    write_qk_aggregation_benchmark_outputs,
)
from kvblock.benchmark.real_block_representation_sweep import (
    PromptRetrievalCase,
    RetrievalQuality,
)


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_qk_aggregation_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_qk_aggregation_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality(recall: float, precision: float) -> RetrievalQuality:
    return RetrievalQuality(
        expected_block_ids=(1,),
        selected_expected_block_ids=(1,) if recall > 0 else (),
        missed_expected_block_ids=() if recall > 0 else (1,),
        extra_selected_block_ids=(0,),
        target_recall=recall,
        selected_precision=precision,
        target_hit=recall > 0,
    )


def _row(strategy: str, *, recall: float, precision: float) -> QKAggregationRunRow:
    return QKAggregationRunRow(
        model_name="model",
        prompt_name="needle",
        prompt_file="prompts/needle.txt",
        representation_source="query_mean_last_layer",
        representation_name=f"query_mean_layer_0_key_mean_layer_0_qkagg_{strategy}",
        qk_aggregation_strategy=strategy,
        rail_setting="reduced",
        keep_recent_blocks=1,
        keep_anchor_blocks=0,
        tokens=32,
        blocks=2,
        selected_ids=(1, 0),
        selected_reasons={1: "semantic", 0: "recent"},
        selected_count=2,
        selected_to_semantic_k_ratio=1.0,
        selector_latency_sec=0.001,
        total_latency_sec=0.01,
        prefill_latency_sec=0.006,
        metadata_latency_sec=0.002,
        inspection_latency_sec=0.001,
        fallback_mode="sparse",
        raw_margin=0.1,
        retrieval_quality=_quality(recall, precision),
    )


def test_qk_aggregation_benchmark_output_schema_and_report(tmp_path) -> None:
    result = QKAggregationBenchmarkResult(
        rows=(
            _row("mean_pool", recall=0.0, precision=0.0),
            _row("top_token_mean", recall=1.0, precision=0.5),
        ),
        aggregate_summaries=(
            QKAggregationSummary(
                qk_aggregation_strategy="mean_pool",
                run_count=1,
                mean_recall=0.0,
                mean_precision=0.0,
                mean_selected_count=2.0,
                mean_selected_to_semantic_k_ratio=1.0,
                mean_selector_latency_sec=0.001,
            ),
        ),
        ranked_summaries=(),
        prompt_breakdowns=(),
        model_load_seconds={"model": 0.1},
    )
    json_path = tmp_path / "qk.json"
    text_path = tmp_path / "qk.txt"

    write_qk_aggregation_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["qk_aggregation_strategy"] == "mean_pool"
    assert "QUERY/KEY AGGREGATION BENCHMARK" in text_path.read_text(encoding="utf-8")
    assert "QUERY/KEY AGGREGATION BENCHMARK" in format_qk_aggregation_report(result)


def test_run_qk_aggregation_benchmark_uses_strategy_config(monkeypatch, tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("answer TOKEN", encoding="utf-8")
    seen_strategies: list[str] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

    class FakeResult:
        selected_block_ids = (0,)
        selected_to_semantic_k_ratio = 0.25
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)
        run_summary = SimpleNamespace(
            representation_name="query_mean_layer_0_key_mean_layer_0",
            token_count=4,
            block_count=1,
        )
        latency = SimpleNamespace(
            selector_sec=0.001,
            total_sec=0.002,
            prefill_sec=0.0005,
            metadata_sec=0.0003,
            inspection_sec=0.0002,
        )
        block_inspections = (
            SimpleNamespace(
                block_id=0,
                selected=True,
                selected_reason="semantic",
                final_score=0.9,
                block_text="answer TOKEN",
                preview_text="answer TOKEN",
            ),
        )

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def fake_run_real_block_selector(runtime, prompt, config):
        seen_strategies.append(config.qk_aggregation_strategy)
        assert config.representation_source == "query_mean_last_layer"
        assert config.prompt_name == "case"
        return FakeResult()

    monkeypatch.setattr(qk_bench, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(qk_bench, "run_real_block_selector", fake_run_real_block_selector)

    result = run_qk_aggregation_benchmark(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="case",
                path=prompt_path,
                target_fragments=("TOKEN",),
            ),
        ),
        qk_aggregation_strategies=("mean_pool", "top_token_mean"),
    )

    assert seen_strategies == ["mean_pool", "top_token_mean"]
    assert [row.qk_aggregation_strategy for row in result.rows] == [
        "mean_pool",
        "top_token_mean",
    ]
    assert result.aggregate_summaries
    assert result.prompt_breakdowns


def test_qk_aggregation_benchmark_script_parser() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "--models",
            "gpt2",
            "--qk-aggregations",
            "mean_pool,top_token_mean",
            "--top-token-count",
            "2",
            "--local-files-only",
        ]
    )

    assert args.models == "gpt2"
    assert args.qk_aggregations == "mean_pool,top_token_mean"
    assert args.top_token_count == 2
    assert args.local_files_only is True
