from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import kvblock.benchmark.real_block_representation_sweep as sweep
from kvblock.benchmark.real_block_representation_sweep import (
    LayerDifferenceSummary,
    PromptRetrievalCase,
    RailSetting,
    RepresentationAggregateSummary,
    RepresentationRankingSummary,
    RepresentationSweepResult,
    RepresentationSweepRunRow,
    RetrievalQuality,
    default_rail_settings,
    format_representation_sweep_report,
    head_scoring_settings_from_names,
    rank_aggregate_summaries,
    rail_settings_from_presets,
    representation_sources_from_names,
    retrieval_quality_for_result,
    run_representation_sweep,
    write_representation_sweep_outputs,
)


def _load_sweep_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_real_block_representation_sweep.py"
    )
    spec = importlib.util.spec_from_file_location("run_real_block_representation_sweep", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_retrieval_quality_from_answer_fragments() -> None:
    quality = retrieval_quality_for_result(
        selected_ids=(3, 1, 0),
        block_text_by_id={
            0: "distractor",
            1: "answer AES-256-GCM",
            2: "also answer ZXQ-4917-BETA",
            3: "question",
        },
        target_fragments=("AES-256-GCM", "ZXQ-4917-BETA"),
    )

    assert quality.expected_block_ids == (1, 2)
    assert quality.selected_expected_block_ids == (1,)
    assert quality.missed_expected_block_ids == (2,)
    assert quality.extra_selected_block_ids == (3, 0)
    assert quality.target_recall == 0.5
    assert quality.selected_precision == 1 / 3
    assert quality.target_hit is True


def test_prompt_retrieval_case_requires_target_fragments(tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("prompt", encoding="utf-8")

    case = PromptRetrievalCase(
        name="case",
        path=prompt_path,
        target_fragments=("needle",),
    )

    assert case.path == prompt_path


def test_rail_presets_include_no_rail_and_reduced_settings() -> None:
    settings = default_rail_settings()
    by_name = {setting.name: setting for setting in settings}

    assert by_name["no_rails"].keep_recent_blocks == 0
    assert by_name["no_rails"].keep_anchor_blocks == 0
    assert by_name["reduced"].keep_recent_blocks == 1
    assert by_name["reduced"].keep_anchor_blocks == 0
    assert rail_settings_from_presets(("default", "no_rails")) == (
        by_name["default"],
        by_name["no_rails"],
    )


def test_representation_source_parser_supports_query_sources() -> None:
    assert representation_sources_from_names(
        ("avg_mid4_hidden", "key_mean_mid_layer", "query_mean_last_layer")
    ) == ("avg_mid4_hidden", "key_mean_mid_layer", "query_mean_last_layer")


def test_representation_sweep_script_parses_representation_sources() -> None:
    module = _load_sweep_script()
    args = module.build_parser().parse_args(
        [
            "--representation-sources",
            "avg_mid4_hidden,key_mean_mid_layer,query_mean_last_layer",
            "--head-scoring-modes",
            "mean_heads,max_head_score,topk_head_mean",
            "--head-top-k",
            "3",
            "--include-head-diagnostics",
            "--top-heads",
            "4",
        ]
    )

    assert (
        args.representation_sources
        == "avg_mid4_hidden,key_mean_mid_layer,query_mean_last_layer"
    )
    assert args.head_scoring_modes == "mean_heads,max_head_score,topk_head_mean"
    assert args.head_top_k == 3
    assert args.include_head_diagnostics is True
    assert args.top_heads == 4


def test_head_scoring_settings_parser_supports_per_head_modes() -> None:
    settings = head_scoring_settings_from_names(
        ("mean_heads", "max_head_score", "weighted_head_mean"),
        head_top_k=3,
        head_weights=(1.0, 0.5),
    )

    assert [setting.mode for setting in settings] == [
        "mean_heads",
        "max_head_score",
        "weighted_head_mean",
    ]
    assert settings[-1].head_top_k == 3
    assert settings[-1].head_weights == (1.0, 0.5)


def test_representation_sweep_output_writing(tmp_path) -> None:
    quality = RetrievalQuality(
        expected_block_ids=(1,),
        selected_expected_block_ids=(1,),
        missed_expected_block_ids=(),
        extra_selected_block_ids=(0,),
        target_recall=1.0,
        selected_precision=0.5,
        target_hit=True,
    )
    row = RepresentationSweepRunRow(
        model_name="model",
        prompt_name="prompt",
        prompt_file="prompts/prompt.txt",
        representation_source="final_hidden",
        layer_index=None,
        representation_name="final_hidden",
        rail_setting="no_rails",
        keep_recent_blocks=0,
        keep_anchor_blocks=0,
        head_scoring_mode="mean_heads",
        head_top_k=2,
        head_weights=(),
        tokens=10,
        blocks=2,
        selected_ids=(0, 1),
        selected_reasons={0: "anchor", 1: "semantic"},
        selected_scores={0: 0.1, 1: 0.9},
        selected_count=2,
        selected_to_semantic_k_ratio=1.0,
        selector_latency_sec=0.001,
        total_latency_sec=0.01,
        prefill_latency_sec=0.004,
        metadata_latency_sec=0.003,
        inspection_latency_sec=0.002,
        fallback_mode="sparse",
        raw_margin=0.0,
        retrieval_quality=quality,
    )
    result = RepresentationSweepResult(
        rows=(row,),
        layer_differences=(
            LayerDifferenceSummary(
                model_name="model",
                prompt_name="prompt",
                rail_setting="no_rails",
                keep_recent_blocks=0,
                keep_anchor_blocks=0,
                head_scoring_mode="mean_heads",
                baseline_source="final_hidden",
                representation_source="middle_hidden",
                selected_jaccard_vs_baseline=0.5,
                recall_delta_vs_baseline=0.0,
                precision_delta_vs_baseline=-0.1,
                selector_latency_delta_sec=0.001,
            ),
        ),
        aggregate_summaries=(
            RepresentationAggregateSummary(
                model_name="model",
                representation_source="final_hidden",
                rail_setting="no_rails",
                keep_recent_blocks=0,
                keep_anchor_blocks=0,
                head_scoring_mode="mean_heads",
                head_top_k=2,
                head_weights=(),
                run_count=1,
                mean_recall=1.0,
                mean_precision=0.5,
                mean_selected_count=2.0,
                mean_selected_to_semantic_k_ratio=1.0,
                mean_selector_latency_sec=0.001,
            ),
        ),
        model_load_seconds={"model": 0.1},
    )

    json_path = tmp_path / "sweep.json"
    text_path = tmp_path / "sweep.txt"
    write_representation_sweep_outputs(result, json_path=json_path, text_path=text_path)

    assert json_path.exists()
    assert text_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["rail_setting"] == "no_rails"
    assert payload["rows"][0]["keep_recent_blocks"] == 0
    assert payload["aggregate_summaries"][0]["rail_setting"] == "no_rails"
    assert payload["ranking_summaries"][0]["rank"] == 1
    report = format_representation_sweep_report(result)
    assert "REAL-BLOCK REPRESENTATION SWEEP" in report
    assert "model | prompt | final_hidden" in report
    assert "rail=no_rails(recent=0,anchors=0)" in report
    assert "AGGREGATES BY MODEL / REPRESENTATION / RAILS" in report
    assert "SUMMARY RANKING BY MODEL / RAILS" in report


def test_ranking_sorts_by_recall_precision_sparsity_then_latency() -> None:
    aggregates = (
        RepresentationAggregateSummary(
            model_name="model",
            representation_source="slow",
            rail_setting="default",
            keep_recent_blocks=4,
            keep_anchor_blocks=2,
            head_scoring_mode="mean_heads",
            head_top_k=2,
            head_weights=(),
            run_count=5,
            mean_recall=1.0,
            mean_precision=0.5,
            mean_selected_count=5.0,
            mean_selected_to_semantic_k_ratio=1.25,
            mean_selector_latency_sec=0.2,
        ),
        RepresentationAggregateSummary(
            model_name="model",
            representation_source="best",
            rail_setting="default",
            keep_recent_blocks=4,
            keep_anchor_blocks=2,
            head_scoring_mode="mean_heads",
            head_top_k=2,
            head_weights=(),
            run_count=5,
            mean_recall=1.0,
            mean_precision=0.5,
            mean_selected_count=4.0,
            mean_selected_to_semantic_k_ratio=1.0,
            mean_selector_latency_sec=0.3,
        ),
        RepresentationAggregateSummary(
            model_name="model",
            representation_source="lower_recall",
            rail_setting="default",
            keep_recent_blocks=4,
            keep_anchor_blocks=2,
            head_scoring_mode="mean_heads",
            head_top_k=2,
            head_weights=(),
            run_count=5,
            mean_recall=0.8,
            mean_precision=1.0,
            mean_selected_count=4.0,
            mean_selected_to_semantic_k_ratio=1.0,
            mean_selector_latency_sec=0.1,
        ),
    )

    rankings = rank_aggregate_summaries(aggregates)

    assert isinstance(rankings[0], RepresentationRankingSummary)
    assert [ranking.representation_source for ranking in rankings] == [
        "best",
        "slow",
        "lower_recall",
    ]
    assert [ranking.rank for ranking in rankings] == [1, 2, 3]


def test_run_representation_sweep_accepts_no_rails(monkeypatch, tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("answer AES-256-GCM", encoding="utf-8")

    class FakeRuntime:
        def __init__(self, load_config):
            self.load_config = load_config
            self.capture_config = None

        def load_model(self):
            return None

    class FakeResult:
        selected_block_ids = (0,)
        selected_to_semantic_k_ratio = 0.25
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)
        run_summary = SimpleNamespace(
            representation_name="final_hidden",
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
                block_text="answer AES-256-GCM",
                preview_text="answer AES-256-GCM",
            ),
        )

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def fake_run_real_block_selector(runtime, prompt, config):
        assert config.keep_recent_blocks == 0
        assert config.keep_anchor_blocks == 0
        assert config.head_scoring_mode == "max_head_score"
        return FakeResult()

    monkeypatch.setattr(sweep, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(sweep, "run_real_block_selector", fake_run_real_block_selector)

    result = run_representation_sweep(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="case",
                path=prompt_path,
                target_fragments=("AES-256-GCM",),
            ),
        ),
        representation_sources=("final_hidden",),
        rail_settings=(RailSetting("no_rails", 0, 0),),
        head_scoring_settings=head_scoring_settings_from_names(("max_head_score",)),
    )

    assert result.rows[0].rail_setting == "no_rails"
    assert result.rows[0].keep_recent_blocks == 0
    assert result.rows[0].keep_anchor_blocks == 0
    assert result.rows[0].head_scoring_mode == "max_head_score"
    assert result.aggregate_summaries[0].rail_setting == "no_rails"


def test_run_representation_sweep_can_include_head_diagnostic_summary(
    monkeypatch,
    tmp_path,
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("answer AES-256-GCM", encoding="utf-8")

    class FakeRuntime:
        def __init__(self, load_config):
            self.load_config = load_config
            self.capture_config = None

        def load_model(self):
            return None

    class FakeHeadSummary:
        def to_dict(self):
            return {
                "prompt_name": "case",
                "selected_top_head_counts": [
                    {"head_index": 3, "count": 2, "fraction": 1.0}
                ],
            }

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
        head_diagnostic_summary = FakeHeadSummary()
        block_inspections = (
            SimpleNamespace(
                block_id=0,
                selected=True,
                selected_reason="semantic",
                final_score=0.9,
                block_text="answer AES-256-GCM",
                preview_text="answer AES-256-GCM",
            ),
        )

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def fake_run_real_block_selector(runtime, prompt, config):
        assert config.emit_head_diagnostics is True
        assert config.top_heads == 4
        assert config.rail_setting == "no_rails"
        assert config.representation_source == "query_mean_last_layer"
        assert config.prompt_name == "case"
        return FakeResult()

    monkeypatch.setattr(sweep, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(sweep, "run_real_block_selector", fake_run_real_block_selector)

    result = run_representation_sweep(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="case",
                path=prompt_path,
                target_fragments=("AES-256-GCM",),
            ),
        ),
        representation_sources=("query_mean_last_layer",),
        rail_settings=(RailSetting("no_rails", 0, 0),),
        include_head_diagnostics=True,
        diagnostic_top_heads=4,
    )

    assert result.rows[0].head_diagnostic_summary is not None
    assert result.rows[0].head_diagnostic_summary["selected_top_head_counts"][0][
        "head_index"
    ] == 3
    assert "selected_top_heads=h3:2(1.00)" in format_representation_sweep_report(result)
