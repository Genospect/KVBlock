from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch

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
        rerank_mode="none",
        rerank_weight=0.3,
        block_mode=block_mode,
        suppression_mode="none",
        suppression_threshold=0.75,
        keep_recent_blocks=0,
        keep_anchor_blocks=0,
        tokens=32,
        candidate_block_count=2,
        candidate_count_before_suppression=2,
        candidate_count_after_suppression=2,
        neighbor_expansion=0,
        halo_radius=0,
        max_selected_blocks=None,
        semantic_selected_ids=(0, 1),
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


def test_rerank_candidates_can_promote_lexical_entity_match() -> None:
    ranked = (
        dynamic_bench.RankedCandidateSpan(
            block_id=0,
            candidate_id="a",
            token_start=0,
            token_end=40,
            score=1.0,
            rank=1,
        ),
        dynamic_bench.RankedCandidateSpan(
            block_id=1,
            candidate_id="b",
            token_start=40,
            token_end=80,
            score=0.2,
            rank=2,
        ),
    )
    result = SimpleNamespace(
        block_inspections=(
            SimpleNamespace(
                block_id=0,
                block_text="general music background and unrelated credits",
                preview_text="general music background",
            ),
            SimpleNamespace(
                block_id=1,
                block_text="The Rebirth released an album with featured vocals",
                preview_text="The Rebirth released an album",
            ),
        )
    )

    reranked = dynamic_bench._rerank_candidates(
        ranked,
        result=result,
        query_prompt="Who was in The Rebirth?",
        mode="semantic_plus_tokenmax",
        weight=1.0,
    )

    assert [candidate.block_id for candidate in reranked] == [1, 0]
    assert reranked[0].rank == 1


def test_dense_qk_token_refine_promotes_exact_qk_match() -> None:
    ranked = (
        dynamic_bench.RankedCandidateSpan(
            block_id=0,
            candidate_id="a",
            token_start=0,
            token_end=2,
            score=1.0,
            rank=1,
        ),
        dynamic_bench.RankedCandidateSpan(
            block_id=1,
            candidate_id="b",
            token_start=2,
            token_end=4,
            score=0.2,
            rank=2,
        ),
    )
    result = SimpleNamespace(
        per_head_token_representations=torch.tensor(
            [
                [
                    [0.1, 0.0],
                    [0.0, 0.1],
                    [3.0, 0.0],
                    [2.0, 0.0],
                ]
            ],
            dtype=torch.float32,
        ),
        per_head_query_representation=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    reranked = dynamic_bench._rerank_candidates(
        ranked,
        result=result,
        query_prompt="unused for dense qk",
        mode="dense_qk_token_refine",
        weight=0.3,
        refine_top_n_tokens=1,
        refine_candidate_limit=2,
    )

    assert [candidate.block_id for candidate in reranked] == [1, 0]
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2
    assert reranked[0].score > reranked[1].score


def test_dense_qk_cosine_refine_reduces_norm_bias() -> None:
    ranked = (
        dynamic_bench.RankedCandidateSpan(
            block_id=0,
            candidate_id="a",
            token_start=0,
            token_end=1,
            score=1.0,
            rank=1,
        ),
        dynamic_bench.RankedCandidateSpan(
            block_id=1,
            candidate_id="b",
            token_start=1,
            token_end=2,
            score=0.2,
            rank=2,
        ),
    )
    result = SimpleNamespace(
        per_head_token_representations=torch.tensor(
            [[[10.0, 10.0], [1.0, 0.0]]],
            dtype=torch.float32,
        ),
        per_head_query_representation=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
    )

    raw = dynamic_bench._rerank_candidates(
        ranked,
        result=result,
        query_prompt="unused",
        mode="dense_qk_token_refine",
        weight=0.3,
        refine_top_n_tokens=1,
        refine_score_mode="raw_topn_mean",
        refine_candidate_limit=2,
    )
    cosine = dynamic_bench._rerank_candidates(
        ranked,
        result=result,
        query_prompt="unused",
        mode="dense_qk_token_refine",
        weight=0.3,
        refine_top_n_tokens=1,
        refine_score_mode="cosine_topn_mean",
        refine_candidate_limit=2,
    )

    assert [candidate.block_id for candidate in raw] == [0, 1]
    assert [candidate.block_id for candidate in cosine] == [1, 0]


def test_scaffold_exclusion_removes_metadata_only_blocks_from_rerank() -> None:
    ranked = (
        dynamic_bench.RankedCandidateSpan(
            block_id=0,
            candidate_id="scaffold",
            token_start=0,
            token_end=40,
            score=1.0,
            rank=1,
        ),
        dynamic_bench.RankedCandidateSpan(
            block_id=1,
            candidate_id="context",
            token_start=40,
            token_end=80,
            score=0.5,
            rank=2,
        ),
    )
    result = SimpleNamespace(
        block_inspections=(
            SimpleNamespace(
                block_id=0,
                block_text="DATASET: hotpotqa SAMPLE_ID: abc LENGTH: 6021",
                preview_text="DATASET: hotpotqa SAMPLE_ID: abc",
            ),
            SimpleNamespace(
                block_id=1,
                block_text="LENGTH: 6021 CONTEXT: Passage 1: useful evidence",
                preview_text="CONTEXT: Passage 1: useful evidence",
            ),
        )
    )

    excluded = dynamic_bench._scaffold_block_ids_from_result(result)
    reranked = dynamic_bench._rerank_candidates(
        ranked,
        result=result,
        query_prompt="unused",
        mode="none",
        weight=0.3,
        excluded_block_ids=excluded,
    )

    assert excluded == (0,)
    assert [candidate.block_id for candidate in reranked] == [1]
    assert reranked[0].rank == 1


def test_neighbor_expansion_does_not_reintroduce_excluded_scaffold() -> None:
    block_by_id = {
        block_id: SimpleNamespace(
            block_id=block_id,
            token_start=block_id * 40,
            token_end=(block_id + 1) * 40,
        )
        for block_id in range(3)
    }

    expanded = dynamic_bench._expand_selected_ids_by_neighbors(
        (1,),
        block_by_id=block_by_id,
        radius=1,
        excluded_block_ids=(0,),
    )

    assert expanded == (1, 2)


def test_stage_c_semantic_refined_mix_keeps_both_ranking_rails() -> None:
    selected = dynamic_bench._stage_c_semantic_ids(
        refined_selected_ids=(10, 11, 12, 13),
        original_ranked_block_ids=(1, 2, 10, 3),
        semantic_k=4,
        policy="semantic_refined_mix",
    )

    assert selected == (1, 2, 10, 11)


def test_stage_c_refined_only_preserves_existing_selection_policy() -> None:
    selected = dynamic_bench._stage_c_semantic_ids(
        refined_selected_ids=(10, 11, 12, 13),
        original_ranked_block_ids=(1, 2, 10, 3),
        semantic_k=4,
        policy="refined_only",
    )

    assert selected == (10, 11, 12, 13)


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


def test_neighbor_expansion_adds_adjacent_token_blocks() -> None:
    block_by_id = {
        0: SimpleNamespace(block_id=0, token_start=0, token_end=40),
        1: SimpleNamespace(block_id=1, token_start=40, token_end=80),
        2: SimpleNamespace(block_id=2, token_start=80, token_end=120),
        3: SimpleNamespace(block_id=3, token_start=120, token_end=160),
    }

    assert dynamic_bench._expand_selected_ids_by_neighbors(
        (1,),
        block_by_id=block_by_id,
        radius=1,
    ) == (1, 0, 2)
    assert dynamic_bench._expand_selected_ids_by_neighbors(
        (1,),
        block_by_id=block_by_id,
        radius=2,
    ) == (1, 0, 2, 3)


def test_budgeted_halo_preserves_anchors_and_caps_neighbors() -> None:
    block_by_id = {
        0: SimpleNamespace(block_id=0, token_start=0, token_end=40),
        1: SimpleNamespace(block_id=1, token_start=40, token_end=80),
        2: SimpleNamespace(block_id=2, token_start=80, token_end=120),
        3: SimpleNamespace(block_id=3, token_start=120, token_end=160),
        4: SimpleNamespace(block_id=4, token_start=160, token_end=200),
    }

    expanded = dynamic_bench._expand_selected_ids_by_neighbors(
        (1, 3),
        block_by_id=block_by_id,
        radius=1,
        max_selected_blocks=4,
    )

    assert expanded == (1, 3, 0, 2)


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


def test_query_only_coarse_to_fine_restricts_coarse_candidates_to_context(
    monkeypatch, tmp_path
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    context = "CONTEXT:\nalpha TOKEN beta gamma delta epsilon"
    prompt_path.write_text(
        f"{context}\n\nINPUT:\nWhich question tail mentions TOKEN?",
        encoding="utf-8",
    )
    context_token_count = len(context.split())
    seen: list[tuple[str, tuple[tuple[int, int], ...], str | None]] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

        def tokenize(self, prompt):
            return SimpleNamespace(token_count=len(prompt.split()))

    class FakeResult:
        selected_to_semantic_k_ratio = 0.5
        fallback_mode = "sparse"
        confidence = SimpleNamespace(raw_margin=0.1)

        def __init__(self, *, block_mode, token_count, blocks, selected_ids):
            self.selected_block_ids = selected_ids
            self.run_summary = SimpleNamespace(
                representation_name="query_only_last_layer",
                token_count=token_count,
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

    def block(candidate, *, selected=False):
        return SimpleNamespace(
            block_id=candidate.block_id,
            selected=selected,
            selected_reason="semantic" if selected else "unselected",
            final_score=1.0 - candidate.block_id * 0.1,
            block_text="alpha TOKEN beta" if selected else "other",
            preview_text="alpha TOKEN beta" if selected else "other",
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
        spans = tuple(
            (candidate.token_start, candidate.token_end)
            for candidate in config.block_candidates
        )
        seen.append((config.block_mode, spans, config.query_prompt))
        assert config.block_candidates
        assert all(
            candidate.block_mode == config.block_mode
            for candidate in config.block_candidates
        )
        assert all(
            candidate.token_end <= context_token_count
            for candidate in config.block_candidates
        )
        return FakeResult(
            block_mode=config.block_mode,
            token_count=len(prompt.split()),
            selected_ids=(0,),
            blocks=tuple(
                block(candidate, selected=candidate.block_id == 0)
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
        representation_source="query_only_last_layer",
        qk_aggregation_strategy="block_max",
        coarse_top_k=1,
    )

    assert seen[0] == (
        "fixed_40",
        ((0, context_token_count),),
        "Which question tail mentions TOKEN?",
    )
    assert seen[1][0] == "coarse_to_fine_40_16"
    assert seen[1][1] == ((0, context_token_count),)
    assert result.rows[0].coarse_selected_spans == (f"0:{context_token_count}",)


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


def test_run_dynamic_block_benchmark_mixed_global_refine_mode(
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

        def __init__(self, *, block_mode, blocks, selected_ids, raw_margin=0.1):
            self.selected_block_ids = selected_ids
            self.confidence = SimpleNamespace(raw_margin=raw_margin)
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

    def block(candidate, *, selected=False, text="other", score=0.5):
        return SimpleNamespace(
            block_id=candidate.block_id,
            selected=selected,
            selected_reason="semantic" if selected else "unselected",
            final_score=score,
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
                    block(
                        candidate,
                        selected=candidate.block_id == 0,
                        text="alpha TOKEN" if candidate.block_id == 0 else "other",
                        score=1.0 - candidate.block_id * 0.1,
                    )
                    for candidate in parent_candidates
                ),
            )
        assert config.block_mode == "mixed_global_refine_40_16"
        assert len(config.block_candidates) == 5
        return FakeResult(
            block_mode="mixed_global_refine_40_16",
            selected_ids=(0, 1, 2, 3),
            blocks=tuple(
                block(
                    candidate,
                    selected=candidate.block_id < 4,
                    text="alpha TOKEN" if candidate.block_id in {0, 2} else "other",
                    score=1.0 - candidate.block_id * 0.1,
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
        block_modes=("mixed_global_refine_40_16",),
        qk_aggregation_strategy="block_max",
        needle_qk_aggregation_strategy="top_token_mean",
        mixed_refine_parent_k=1,
        mixed_global_anchor_k=2,
        mixed_fallback_margin=0.05,
    )

    assert seen == [
        ("fixed_40", 40, 0),
        ("mixed_global_refine_40_16", 16, 5),
    ]
    row = result.rows[0]
    assert row.block_mode == "mixed_global_refine_40_16"
    assert row.coarse_candidate_count == 2
    assert row.retained_parent_count == 2
    assert row.fine_candidate_count_after_drilldown == 3
    assert row.coarse_selected_candidate_ids == (
        "s40_stride40_t0_40",
        "s40_stride40_t40_80",
    )
    assert row.mixed_fallback_used is False
    assert row.selected_candidate_roles == ("parent", "parent", "child", "child")
    assert row.suppression_decisions[0]["candidate_role"] == "parent"
    assert row.suppression_decisions[2]["candidate_role"] == "child"


def test_mixed_global_refine_selects_parents_after_rerank(
    monkeypatch, tmp_path
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("alpha TOKEN beta gamma", encoding="utf-8")
    final_candidate_ids: tuple[str, ...] = ()
    rerank_calls: list[tuple[str, str]] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

    class FakeResult:
        selected_to_semantic_k_ratio = 0.5
        fallback_mode = "sparse"

        def __init__(self, *, block_mode, blocks, selected_ids, raw_margin=0.1):
            self.selected_block_ids = selected_ids
            self.confidence = SimpleNamespace(raw_margin=raw_margin)
            self.run_summary = SimpleNamespace(
                representation_name="query_mean_layer_0_key_mean_layer_0",
                token_count=120,
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

    def block(candidate, *, selected=False, text="other", score=0.5):
        return SimpleNamespace(
            block_id=candidate.block_id,
            selected=selected,
            selected_reason="semantic" if selected else "unselected",
            final_score=score,
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
        nonlocal final_candidate_ids
        if config.block_mode == "fixed_40":
            parent_candidates = tuple(
                SimpleNamespace(
                    block_id=block_id,
                    candidate_id=(
                        f"s40_stride40_t{block_id * 40}_{block_id * 40 + 40}"
                    ),
                    block_mode="fixed_40",
                    block_size=40,
                    stride=40,
                    token_start=block_id * 40,
                    token_len=40,
                    token_end=block_id * 40 + 40,
                    parent_candidate_id=None,
                    candidate_role="block",
                )
                for block_id in range(3)
            )
            return FakeResult(
                block_mode="fixed_40",
                selected_ids=(0,),
                blocks=tuple(
                    block(
                        candidate,
                        selected=candidate.block_id == 0,
                        text="promoted TOKEN" if candidate.block_id == 2 else "other",
                        score=1.0 - candidate.block_id * 0.1,
                    )
                    for candidate in parent_candidates
                ),
            )

        assert config.block_mode == "mixed_global_refine_40_16"
        final_candidate_ids = tuple(
            candidate.candidate_id for candidate in config.block_candidates
        )
        return FakeResult(
            block_mode="mixed_global_refine_40_16",
            selected_ids=tuple(
                candidate.block_id for candidate in config.block_candidates
            ),
            blocks=tuple(
                block(candidate, selected=True)
                for candidate in config.block_candidates
            ),
        )

    def fake_rerank_candidates(ranked_candidates, *, result, mode, **kwargs):
        rerank_calls.append((result.run_summary.block_mode, mode))
        if result.run_summary.block_mode != "fixed_40":
            return tuple(ranked_candidates)
        order = {2: 0, 0: 1, 1: 2}
        reranked = sorted(ranked_candidates, key=lambda item: order[item.block_id])
        return tuple(
            dynamic_bench.RankedCandidateSpan(
                block_id=candidate.block_id,
                candidate_id=candidate.candidate_id,
                token_start=candidate.token_start,
                token_end=candidate.token_end,
                score=candidate.score,
                rank=rank,
                block_size=candidate.block_size,
                block_mode=candidate.block_mode,
            )
            for rank, candidate in enumerate(reranked, start=1)
        )

    monkeypatch.setattr(dynamic_bench, "LocalHfRuntime", FakeRuntime)
    monkeypatch.setattr(
        dynamic_bench, "run_real_block_selector", fake_run_real_block_selector
    )
    monkeypatch.setattr(dynamic_bench, "_rerank_candidates", fake_rerank_candidates)

    result = run_dynamic_block_benchmark(
        model_names=("fake-model",),
        prompt_cases=(
            PromptRetrievalCase(
                name="needle",
                path=prompt_path,
                target_fragments=("TOKEN",),
            ),
        ),
        block_modes=("mixed_global_refine_40_16",),
        qk_aggregation_strategy="block_max",
        rerank_mode="dense_qk_token_refine",
        mixed_refine_parent_k=1,
        mixed_global_anchor_k=2,
        mixed_fallback_margin=0.0,
    )

    assert ("fixed_40", "dense_qk_token_refine") in rerank_calls
    assert result.rows[0].coarse_selected_candidate_ids == (
        "s40_stride40_t80_120",
        "s40_stride40_t0_40",
    )
    assert "s40_stride40_t80_120__parent" in final_candidate_ids
    assert "s40_stride40_t40_80__parent" not in final_candidate_ids
    assert any(
        candidate_id.startswith(
            "s40_stride40_t80_120__child_s16_stride16_t80_"
        )
        for candidate_id in final_candidate_ids
    )


def test_mixed_global_refine_falls_back_to_fixed40_on_weak_margin(
    monkeypatch, tmp_path
) -> None:
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("alpha TOKEN beta gamma", encoding="utf-8")
    seen: list[str] = []

    class FakeRuntime:
        def __init__(self, load_config, *, capture_config=None):
            self.load_config = load_config
            self.capture_config = capture_config

        def load_model(self):
            return None

    class FakeResult:
        selected_to_semantic_k_ratio = 0.5
        fallback_mode = "sparse"

        def __init__(self, *, block_mode, blocks):
            self.selected_block_ids = (0,)
            self.confidence = SimpleNamespace(raw_margin=0.01)
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

    def fake_run_real_block_selector(runtime, prompt, config):
        seen.append(config.block_mode)
        assert config.block_mode == "fixed_40"
        blocks = (
            SimpleNamespace(
                block_id=0,
                selected=True,
                selected_reason="semantic",
                final_score=1.0,
                block_text="alpha TOKEN",
                preview_text="alpha TOKEN",
                candidate_id="s40_stride40_t0_40",
                token_start=0,
                token_end=40,
                token_count=40,
                block_size=40,
                stride=40,
                block_mode="fixed_40",
                parent_candidate_id=None,
                candidate_role="block",
            ),
            SimpleNamespace(
                block_id=1,
                selected=False,
                selected_reason="unselected",
                final_score=0.1,
                block_text="other",
                preview_text="other",
                candidate_id="s40_stride40_t40_80",
                token_start=40,
                token_end=80,
                token_count=40,
                block_size=40,
                stride=40,
                block_mode="fixed_40",
                parent_candidate_id=None,
                candidate_role="block",
            ),
        )
        return FakeResult(block_mode="fixed_40", blocks=blocks)

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
        block_modes=("mixed_global_refine_40_16",),
        qk_aggregation_strategy="block_max",
        mixed_fallback_margin=0.05,
    )

    assert seen == ["fixed_40"]
    row = result.rows[0]
    assert row.block_mode == "mixed_global_refine_40_16"
    assert row.coarse_candidate_count == 2
    assert row.fine_candidate_count_after_drilldown == 0
    assert row.retained_parent_count == 0
    assert row.mixed_fallback_used is True
    assert row.selected_candidate_roles == ("block", "block")


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
            "--mixed-refine-parent-k",
            "6",
            "--mixed-global-anchor-k",
            "10",
            "--mixed-fallback-margin",
            "0.02",
            "--rerank-mode",
            "dense_qk_token_refine",
            "--rerank-weight",
            "0.4",
            "--refine-top-n-tokens",
            "3",
            "--refine-score-mode",
            "cosine_topn_mean",
            "--stage-c-policy",
            "semantic_refined_mix",
            "--neighbor-expansion",
            "1",
            "--halo-radius",
            "0",
            "--max-selected-blocks",
            "8",
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
    assert args.mixed_refine_parent_k == 6
    assert args.mixed_global_anchor_k == 10
    assert args.mixed_fallback_margin == 0.02
    assert args.rerank_mode == "dense_qk_token_refine"
    assert args.rerank_weight == 0.4
    assert args.refine_top_n_tokens == 3
    assert args.refine_score_mode == "cosine_topn_mean"
    assert args.stage_c_policy == "semantic_refined_mix"
    assert args.neighbor_expansion == 1
    assert args.halo_radius == 0
    assert args.max_selected_blocks == 8
    assert args.device_map == "auto"
    assert args.local_files_only is True
