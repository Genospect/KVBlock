from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import kvblock.benchmark.dynamic_block_benchmark as dynamic_bench
from kvblock.benchmark.dynamic_block_benchmark import (
    DynamicBlockBenchmarkResult,
    DynamicBlockRunRow,
    DynamicBlockAggregateSummary,
    default_dynamic_prompt_cases,
    format_dynamic_block_report,
    run_dynamic_block_benchmark,
    write_dynamic_block_benchmark_outputs,
)
from kvblock.benchmark.real_block_representation_sweep import (
    PromptRetrievalCase,
    RetrievalQuality,
)


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_dynamic_block_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_dynamic_block_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality() -> RetrievalQuality:
    return RetrievalQuality(
        expected_block_ids=(0,),
        selected_expected_block_ids=(0,),
        missed_expected_block_ids=(),
        extra_selected_block_ids=(1,),
        target_recall=1.0,
        selected_precision=0.5,
        target_hit=True,
    )


def _row(block_mode: str) -> DynamicBlockRunRow:
    return DynamicBlockRunRow(
        model_name="model",
        prompt_name="needle",
        prompt_file="prompts/needle.txt",
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        qk_aggregation_strategy="top_token_mean",
        block_mode=block_mode,
        suppression_mode="none",
        suppression_threshold=0.75,
        keep_recent_blocks=0,
        keep_anchor_blocks=0,
        tokens=32,
        candidate_block_count=2,
        candidate_count_before_suppression=2,
        candidate_count_after_suppression=2,
        selected_ids=(0, 1),
        selected_candidate_ids=("s16_stride16_t0_16", "s16_stride16_t16_32"),
        selected_spans=("0:16", "16:32"),
        selected_block_sizes=(16, 16),
        selected_candidate_roles=("block", "block"),
        suppression_decisions=(
            {
                "block_id": 0,
                "candidate_id": "s16_stride16_t0_16",
                "survived": True,
            },
        ),
        selected_count=2,
        selected_to_semantic_k_ratio=0.5,
        selector_latency_sec=0.001,
        total_latency_sec=0.01,
        prefill_latency_sec=0.006,
        metadata_latency_sec=0.002,
        inspection_latency_sec=0.001,
        fallback_mode="sparse",
        raw_margin=0.1,
        retrieval_quality=_quality(),
    )


def test_dynamic_block_benchmark_output_schema_and_report(tmp_path) -> None:
    result = DynamicBlockBenchmarkResult(
        rows=(_row("fixed_16"),),
        aggregate_summaries=(
            DynamicBlockAggregateSummary(
                block_mode="fixed_16",
                suppression_mode="none",
                suppression_threshold=0.75,
                run_count=1,
                mean_recall=1.0,
                mean_precision=0.5,
                mean_selected_count=2.0,
                mean_selected_to_semantic_k_ratio=0.5,
                mean_candidate_block_count=2.0,
                mean_candidate_count_after_suppression=2.0,
                mean_selector_latency_sec=0.001,
            ),
        ),
        ranked_summaries=(),
        prompt_breakdowns=(),
        model_load_seconds={"model": 0.1},
    )
    json_path = tmp_path / "dynamic.json"
    text_path = tmp_path / "dynamic.txt"

    write_dynamic_block_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["block_mode"] == "fixed_16"
    assert payload["rows"][0]["selected_candidate_ids"] == [
        "s16_stride16_t0_16",
        "s16_stride16_t16_32",
    ]
    assert payload["rows"][0]["candidate_count_after_suppression"] == 2
    assert payload["rows"][0]["suppression_decisions"][0]["survived"] is True
    assert "DYNAMIC BLOCK BENCHMARK" in text_path.read_text(encoding="utf-8")
    assert "DYNAMIC BLOCK BENCHMARK" in format_dynamic_block_report(result)


def test_run_dynamic_block_benchmark_uses_modes_and_prompt_strategy(monkeypatch, tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("CONTEXT:\nanswer TOKEN\n\nINPUT:\nWhere is TOKEN?", encoding="utf-8")
    seen: list[tuple[str, str, str | None, int]] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

        def tokenize(self, prompt):
            return SimpleNamespace(token_count=4)

    class FakeResult:
        selected_block_ids = (0,)
        selected_to_semantic_k_ratio = 0.25
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)
        run_summary = SimpleNamespace(
            representation_name="query_mean_layer_0_key_mean_layer_0",
            token_count=4,
            block_count=1,
            block_mode="fixed_16",
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
                candidate_id="s16_stride16_t0_4",
                token_start=0,
                token_end=4,
                token_count=4,
                block_size=16,
            ),
        )

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def fake_run_real_block_selector(runtime, prompt, config):
        seen.append(
            (
                config.block_mode,
                config.qk_aggregation_strategy,
                config.query_prompt,
                len(config.block_candidates),
            )
        )
        assert config.keep_recent_blocks == 0
        assert config.keep_anchor_blocks == 0
        return FakeResult()

    monkeypatch.setattr(dynamic_bench, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(dynamic_bench, "run_real_block_selector", fake_run_real_block_selector)

    result = run_dynamic_block_benchmark(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="needle",
                path=prompt_path,
                target_fragments=("TOKEN",),
            ),
        ),
        block_modes=("fixed_16", "multiscale_16_32"),
        representation_source="query_only_last_layer",
        qk_aggregation_strategy="block_max",
        needle_qk_aggregation_strategy="top_token_mean",
    )

    assert seen == [
        ("fixed_16", "top_token_mean", "Where is TOKEN?", 1),
        ("multiscale_16_32", "top_token_mean", "Where is TOKEN?", 2),
    ]
    assert [row.block_mode for row in result.rows] == ["fixed_16", "multiscale_16_32"]
    assert result.rows[0].selected_candidate_ids == ("s16_stride16_t0_4",)
    assert result.aggregate_summaries
    assert result.prompt_breakdowns


def test_query_only_prompt_override_extracts_input_tail() -> None:
    prompt = "DATASET: x\n\nCONTEXT:\nalpha\n\nINPUT:\nWhat is alpha?"

    assert dynamic_bench.query_prompt_override_for_representation(
        prompt,
        representation_source="query_only_last_layer",
    ) == "What is alpha?"
    assert dynamic_bench.query_prompt_override_for_representation(
        prompt,
        representation_source="query_mean_last_layer",
    ) is None


def test_dynamic_block_cli_parser_supports_query_only_source() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "--models",
            "gpt2",
            "--representation-source",
            "query_only_last_layer",
        ]
    )

    assert args.representation_source == "query_only_last_layer"


def test_fragment_quality_credits_adjacent_selected_boundary_spans() -> None:
    block_by_id = {
        0: SimpleNamespace(block_id=0, token_start=0, token_end=40),
        1: SimpleNamespace(block_id=1, token_start=40, token_end=80),
        2: SimpleNamespace(block_id=2, token_start=80, token_end=83),
    }
    quality = dynamic_bench._fragment_quality_for_result(
        selected_ids=(0, 1, 2),
        block_text_by_id={
            0: "prefix ",
            1: "needle ZXQ-4917-",
            2: "BETA suffix",
        },
        block_by_id=block_by_id,
        target_fragments=("ZXQ-4917-BETA",),
    )

    assert quality.target_recall == 1.0
    assert quality.target_hit is True
    assert quality.selected_expected_block_ids == (1, 2)
    assert quality.missed_expected_block_ids == ()
    assert quality.extra_selected_block_ids == (0,)


def test_run_dynamic_block_benchmark_coarse_to_fine_mode(monkeypatch, tmp_path) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("alpha TOKEN beta gamma", encoding="utf-8")
    seen: list[tuple[str, int, int]] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

    class FakeResult:
        selected_to_semantic_k_ratio = 0.5
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)

        def __init__(self, *, block_mode, blocks, selected_ids):
            self.selected_block_ids = selected_ids
            self.run_summary = SimpleNamespace(
                representation_name="query_mean_layer_0_key_mean_layer_0",
                token_count=80,
                block_count=len(blocks),
                block_mode=block_mode,
            )
            self.latency = SimpleNamespace(
                selector_sec=0.001,
                total_sec=0.002,
                prefill_sec=0.0005,
                metadata_sec=0.0003,
                inspection_sec=0.0002,
            )
            self.block_inspections = tuple(blocks)

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def block(
        block_id,
        start,
        end,
        *,
        candidate_id,
        mode,
        score,
        selected=False,
        text="",
        parent_candidate_id=None,
        candidate_role="block",
    ):
        return SimpleNamespace(
            block_id=block_id,
            selected=selected,
            selected_reason="semantic" if selected else "unselected",
            final_score=score,
            block_text=text or f"span {start}:{end}",
            preview_text=text or f"span {start}:{end}",
            candidate_id=candidate_id,
            token_start=start,
            token_end=end,
            token_count=end - start,
            block_size=end - start,
            stride=end - start,
            block_mode=mode,
            parent_candidate_id=parent_candidate_id,
            candidate_role=candidate_role,
        )

    def fake_run_real_block_selector(runtime, prompt, config):
        seen.append((config.block_mode, config.block_size, len(config.block_candidates)))
        if config.block_mode == "fixed_40":
            return FakeResult(
                block_mode="fixed_40",
                selected_ids=(0,),
                blocks=(
                    block(
                        0,
                        0,
                        40,
                        candidate_id="s40_stride40_t0_40",
                        mode="fixed_40",
                        score=0.9,
                        selected=True,
                        text="alpha TOKEN",
                    ),
                    block(
                        1,
                        40,
                        80,
                        candidate_id="s40_stride40_t40_80",
                        mode="fixed_40",
                        score=0.1,
                    ),
                ),
            )
        assert config.block_mode == "coarse_to_fine_40_16"
        assert len(config.block_candidates) == 3
        return FakeResult(
            block_mode="coarse_to_fine_40_16",
            selected_ids=(0,),
            blocks=tuple(
                block(
                    candidate.block_id,
                    candidate.token_start,
                    candidate.token_end,
                    candidate_id=candidate.candidate_id,
                    mode=candidate.block_mode,
                    score=1.0 - candidate.block_id * 0.1,
                    selected=candidate.block_id == 0,
                    text="alpha TOKEN" if candidate.block_id == 0 else "other",
                    parent_candidate_id=candidate.parent_candidate_id,
                    candidate_role=getattr(candidate, "candidate_role", "block"),
                )
                for candidate in config.block_candidates
            ),
        )

    monkeypatch.setattr(dynamic_bench, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(dynamic_bench, "run_real_block_selector", fake_run_real_block_selector)

    result = run_dynamic_block_benchmark(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="needle",
                path=prompt_path,
                target_fragments=("TOKEN",),
            ),
        ),
        block_modes=("coarse_to_fine_40_16",),
        qk_aggregation_strategy="block_max",
        needle_qk_aggregation_strategy="top_token_mean",
        coarse_top_k=1,
    )

    assert seen == [
        ("fixed_40", 40, 0),
        ("coarse_to_fine_40_16", 16, 3),
    ]
    row = result.rows[0]
    assert row.block_mode == "coarse_to_fine_40_16"
    assert row.coarse_candidate_count == 2
    assert row.fine_candidate_count_after_drilldown == 3
    assert row.coarse_selected_candidate_ids == ("s40_stride40_t0_40",)
    assert row.selected_candidate_roles == ("child", "child", "child")
    assert row.suppression_decisions[0]["parent_candidate_id"] == "s40_stride40_t0_40"
    assert row.suppression_decisions[0]["candidate_role"] == "child"


def test_run_dynamic_block_benchmark_coarse_to_fine_keep_parent_mode(
    monkeypatch, tmp_path
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("alpha TOKEN beta gamma", encoding="utf-8")
    seen: list[tuple[str, int, int]] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

    class FakeResult:
        selected_to_semantic_k_ratio = 0.5
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)

        def __init__(self, *, block_mode, blocks, selected_ids):
            self.selected_block_ids = selected_ids
            self.run_summary = SimpleNamespace(
                representation_name="query_mean_layer_0_key_mean_layer_0",
                token_count=80,
                block_count=len(blocks),
                block_mode=block_mode,
            )
            self.latency = SimpleNamespace(
                selector_sec=0.001,
                total_sec=0.002,
                prefill_sec=0.0005,
                metadata_sec=0.0003,
                inspection_sec=0.0002,
            )
            self.block_inspections = tuple(blocks)

        @property
        def selected_block_inspections(self):
            return tuple(block for block in self.block_inspections if block.selected)

    def block(candidate, *, selected=False, text="other"):
        return SimpleNamespace(
            block_id=candidate.block_id,
            selected=selected,
            selected_reason="semantic" if selected else "unselected",
            final_score=1.0 - candidate.block_id * 0.1,
            block_text=text,
            preview_text=text,
            candidate_id=candidate.candidate_id,
            token_start=candidate.token_start,
            token_end=candidate.token_end,
            token_count=candidate.token_len,
            block_size=candidate.block_size,
            stride=candidate.stride,
            block_mode=candidate.block_mode,
            parent_candidate_id=candidate.parent_candidate_id,
            candidate_role=candidate.candidate_role,
        )

    def fake_run_real_block_selector(runtime, prompt, config):
        seen.append((config.block_mode, config.block_size, len(config.block_candidates)))
        if config.block_mode == "fixed_40":
            parent_candidates = config.block_candidates or ()
            if not parent_candidates:
                parent_candidates = (
                    SimpleNamespace(
                        block_id=0,
                        candidate_id="s40_stride40_t0_40",
                        block_mode="fixed_40",
                        block_size=40,
                        stride=40,
                        token_start=0,
                        token_len=40,
                        token_end=40,
                        candidate_role="block",
                        parent_candidate_id=None,
                    ),
                    SimpleNamespace(
                        block_id=1,
                        candidate_id="s40_stride40_t40_80",
                        block_mode="fixed_40",
                        block_size=40,
                        stride=40,
                        token_start=40,
                        token_len=40,
                        token_end=80,
                        candidate_role="block",
                        parent_candidate_id=None,
                    ),
                )
            return FakeResult(
                block_mode="fixed_40",
                selected_ids=(0,),
                blocks=tuple(
                    block(candidate, selected=candidate.block_id == 0, text="alpha TOKEN")
                    for candidate in parent_candidates
                ),
            )
        assert config.block_mode == "coarse_to_fine_40_16_keep_parent"
        assert len(config.block_candidates) == 4
        return FakeResult(
            block_mode="coarse_to_fine_40_16_keep_parent",
            selected_ids=(0,),
            blocks=tuple(
                block(
                    candidate,
                    selected=candidate.block_id == 0,
                    text="alpha TOKEN" if candidate.block_id == 0 else "other",
                )
                for candidate in config.block_candidates
            ),
        )

    monkeypatch.setattr(dynamic_bench, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(dynamic_bench, "run_real_block_selector", fake_run_real_block_selector)

    result = run_dynamic_block_benchmark(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="needle",
                path=prompt_path,
                target_fragments=("TOKEN",),
            ),
        ),
        block_modes=("coarse_to_fine_40_16_keep_parent",),
        qk_aggregation_strategy="block_max",
        needle_qk_aggregation_strategy="top_token_mean",
        coarse_top_k=1,
    )

    assert seen == [
        ("fixed_40", 40, 0),
        ("coarse_to_fine_40_16_keep_parent", 16, 4),
    ]
    row = result.rows[0]
    assert row.retained_parent_count == 1
    assert row.fine_candidate_count_after_drilldown == 3
    assert row.selected_candidate_roles == ("parent", "child", "child", "child")
    assert row.suppression_decisions[0]["candidate_role"] == "parent"


def test_dynamic_block_benchmark_prompt_filter_and_script_parser() -> None:
    cases = default_dynamic_prompt_cases(("needle", "code_context"))
    assert [case.name for case in cases] == ["needle", "code_context"]

    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "--models",
            "gpt2",
            "--block-modes",
            "fixed_16,multiscale_16_32,coarse_to_fine_40_16_keep_parent",
            "--qk-aggregation",
            "block_max",
            "--needle-qk-aggregation",
            "top_token_mean",
            "--suppression-modes",
            "none,overlap_threshold",
            "--suppression-threshold",
            "0.8",
            "--coarse-top-k",
            "3",
            "--device-map",
            "auto",
            "--local-files-only",
        ]
    )

    assert args.models == "gpt2"
    assert args.block_modes == "fixed_16,multiscale_16_32,coarse_to_fine_40_16_keep_parent"
    assert args.qk_aggregation == "block_max"
    assert args.needle_qk_aggregation == "top_token_mean"
    assert args.suppression_modes == "none,overlap_threshold"
    assert args.suppression_threshold == 0.8
    assert args.coarse_top_k == 3
    assert args.device_map == "auto"
    assert args.local_files_only is True
