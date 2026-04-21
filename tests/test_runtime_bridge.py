from __future__ import annotations

import torch

from kvblock.kv.block_manager import (
    BlockIngestConfig,
    build_block_metadata_from_representations,
    split_token_blocks,
)
from kvblock.runtime.base import ModelPrefillOutput, RuntimeBackend, TokenizedPrompt
from kvblock.runtime.hooks import (
    HiddenStateCaptureConfig,
    latest_token_state,
    select_hidden_state,
    select_key_state_with_name,
    select_model_representation_with_name,
    select_model_prefill_representations_with_name,
    select_query_key_states_with_name,
)
from kvblock.runtime.local_hf_runtime import _Gpt2QueryProjectionCapture
from kvblock.runtime.real_block_eval import RealBlockSelectorConfig, run_real_block_selector
from kvblock.summaries.base import MultiHeadQuerySummary


class MockRuntime:
    def __init__(self, token_count: int = 10, hidden_dim: int = 6) -> None:
        self.loaded = False
        self.token_count = token_count
        self.hidden_dim = hidden_dim

    @property
    def name(self) -> str:
        return "mock-runtime"

    def load_model(self) -> None:
        self.loaded = True

    def tokenize(self, prompt: str) -> TokenizedPrompt:
        return TokenizedPrompt(
            prompt=prompt,
            token_ids=tuple(range(self.token_count)),
            attention_mask=(1,) * self.token_count,
        )

    def prefill(self, prompt: str) -> ModelPrefillOutput:
        if not self.loaded:
            raise RuntimeError("mock runtime was not loaded")
        tokenized = self.tokenize(prompt)
        values = torch.arange(
            self.token_count * self.hidden_dim,
            dtype=torch.float32,
        ).reshape(self.token_count, self.hidden_dim)
        token_representations = values / values.max().clamp_min(1.0)
        return ModelPrefillOutput(
            prompt=prompt,
            token_ids=tokenized.token_ids,
            token_representations=token_representations,
            query_representation=token_representations[-1],
            representation_name="mock_final_hidden_state",
            runtime_name=self.name,
        )

    def decode_token_ids(self, token_ids: tuple[int, ...]) -> str:
        return " ".join(f"tok{token_id}" for token_id in token_ids)


class PerHeadMockRuntime(MockRuntime):
    def prefill(self, prompt: str) -> ModelPrefillOutput:
        if not self.loaded:
            raise RuntimeError("mock runtime was not loaded")
        tokenized = self.tokenize(prompt)
        token_representations = torch.ones(self.token_count, self.hidden_dim)
        first_head = torch.zeros(self.token_count, self.hidden_dim)
        first_head[:, 0] = 1.0
        second_head = torch.zeros(self.token_count, self.hidden_dim)
        second_head[:, 1] = 1.0
        per_head = torch.stack((first_head, second_head))
        return ModelPrefillOutput(
            prompt=prompt,
            token_ids=tokenized.token_ids,
            token_representations=token_representations,
            query_representation=token_representations[-1],
            representation_name="query_mean_layer_0_key_mean_layer_0",
            runtime_name=self.name,
            per_head_token_representations=per_head,
            per_head_query_representation=torch.stack((first_head[-1], second_head[-1])),
        )


def test_runtime_backend_protocol_shape() -> None:
    runtime = MockRuntime()

    assert isinstance(runtime, RuntimeBackend)
    runtime.load_model()
    output = runtime.prefill("hello")

    assert output.token_count == 10
    assert output.token_representations.shape == (10, 6)
    assert output.representation_name == "mock_final_hidden_state"


def test_hidden_state_hook_selects_final_layer_and_latest_token() -> None:
    first = torch.zeros(1, 3, 4)
    final = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)

    selected = select_hidden_state(
        (first, final),
        HiddenStateCaptureConfig(
            representation_source="final_hidden",
            layer_index=-1,
            representation_name="final",
        ),
    )

    assert selected.shape == (3, 4)
    assert torch.equal(latest_token_state(selected), selected[-1])


def test_hidden_state_hook_supports_representation_sources() -> None:
    hidden_states = tuple(
        torch.full((1, 2, 3), float(index), dtype=torch.float32)
        for index in range(6)
    )

    assert torch.equal(
        select_hidden_state(
            hidden_states,
            HiddenStateCaptureConfig(representation_source="final_hidden"),
        ),
        torch.full((2, 3), 5.0),
    )
    assert torch.equal(
        select_hidden_state(
            hidden_states,
            HiddenStateCaptureConfig(
                representation_source="hidden_layer_index",
                layer_index=2,
            ),
        ),
        torch.full((2, 3), 2.0),
    )
    assert torch.equal(
        select_hidden_state(
            hidden_states,
            HiddenStateCaptureConfig(representation_source="middle_hidden"),
        ),
        torch.full((2, 3), 3.0),
    )
    assert torch.equal(
        select_hidden_state(
            hidden_states,
            HiddenStateCaptureConfig(representation_source="avg_last4_hidden"),
        ),
        torch.full((2, 3), 3.5),
    )
    assert torch.equal(
        select_hidden_state(
            hidden_states,
            HiddenStateCaptureConfig(representation_source="avg_mid4_hidden"),
        ),
        torch.full((2, 3), 2.5),
    )


def test_key_hook_supports_key_representation_sources() -> None:
    key_layers = tuple(
        (
            torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
            + (layer_index * 100),
            torch.zeros(1, 2, 3, 4),
        )
        for layer_index in range(6)
    )

    selected, name = select_key_state_with_name(
        key_layers,
        HiddenStateCaptureConfig(representation_source="key_mean_last_layer"),
    )
    expected_last = key_layers[-1][0].mean(dim=1).squeeze(0)

    assert name == "key_mean_layer_5"
    assert selected.shape == (3, 4)
    assert torch.equal(selected, expected_last)

    mid, mid_name = select_key_state_with_name(
        key_layers,
        HiddenStateCaptureConfig(representation_source="key_mean_mid_layer"),
    )

    assert mid_name == "key_mean_layer_3"
    assert torch.equal(mid, key_layers[3][0].mean(dim=1).squeeze(0))

    averaged, avg_name = select_key_state_with_name(
        key_layers,
        HiddenStateCaptureConfig(representation_source="key_avg_last4"),
    )
    expected_avg = torch.stack(
        [layer[0].mean(dim=1).squeeze(0) for layer in key_layers[-4:]]
    ).mean(dim=0)

    assert avg_name == "key_avg_layers_2_5"
    assert torch.equal(averaged, expected_avg)


def test_query_key_hook_supports_query_representation_sources() -> None:
    key_layers = tuple(
        (
            torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
            + (layer_index * 100),
            torch.zeros(1, 2, 3, 4),
        )
        for layer_index in range(6)
    )
    query_layers = tuple(
        torch.full((1, 2, 3, 4), float(layer_index), dtype=torch.float32)
        for layer_index in range(6)
    )

    keys, query, name, per_head_keys, per_head_query = select_query_key_states_with_name(
        key_layers,
        query_layers,
        HiddenStateCaptureConfig(representation_source="query_mean_mid_layer"),
    )

    assert name == "query_mean_layer_3_key_mean_layer_3"
    assert torch.equal(keys, key_layers[3][0].mean(dim=1).squeeze(0))
    assert torch.equal(query, torch.full((4,), 3.0))
    assert per_head_keys.shape == (2, 3, 4)
    assert per_head_query.shape == (2, 4)
    assert torch.equal(per_head_keys, key_layers[3][0].squeeze(0))
    assert torch.equal(per_head_query, torch.full((2, 4), 3.0))

    (
        avg_keys,
        avg_query,
        avg_name,
        avg_per_head_keys,
        avg_per_head_query,
    ) = select_query_key_states_with_name(
        key_layers,
        query_layers,
        HiddenStateCaptureConfig(representation_source="query_avg_last4"),
    )

    assert avg_name == "query_avg_layers_2_5_key_avg_layers_2_5"
    assert avg_keys.shape == (3, 4)
    assert torch.equal(avg_query, torch.full((4,), 3.5))
    assert avg_per_head_keys.shape == (2, 3, 4)
    assert avg_per_head_query.shape == (2, 4)


def test_model_representation_selector_dispatches_to_key_cache() -> None:
    hidden_states = (torch.zeros(1, 2, 4), torch.ones(1, 2, 4))
    key_layers = (
        (torch.ones(1, 2, 2, 3), torch.zeros(1, 2, 2, 3)),
    )

    selected, name = select_model_representation_with_name(
        hidden_states,
        key_layers,
        HiddenStateCaptureConfig(representation_source="key_mean_last_layer"),
    )

    assert name == "key_mean_layer_0"
    assert selected.shape == (2, 3)
    assert torch.equal(selected, torch.ones(2, 3))


def test_model_prefill_selector_uses_query_vector_for_query_sources() -> None:
    key_layers = (
        (torch.ones(1, 2, 2, 3), torch.zeros(1, 2, 2, 3)),
    )
    query_layers = (
        torch.full((1, 2, 2, 3), 7.0, dtype=torch.float32),
    )

    (
        keys,
        query,
        name,
        per_head_keys,
        per_head_query,
    ) = select_model_prefill_representations_with_name(
        hidden_states=None,
        past_key_values=key_layers,
        query_states=query_layers,
        config=HiddenStateCaptureConfig(representation_source="query_mean_last_layer"),
    )

    assert name == "query_mean_layer_0_key_mean_layer_0"
    assert torch.equal(keys, torch.ones(2, 3))
    assert torch.equal(query, torch.full((3,), 7.0))
    assert per_head_keys is not None
    assert per_head_query is not None
    assert per_head_keys.shape == (2, 2, 3)
    assert per_head_query.shape == (2, 3)


def test_gpt2_query_projection_capture_splits_fused_qkv() -> None:
    class FakeCAttn(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.offset = offset

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            batch, tokens, hidden = values.shape
            query = (
                torch.arange(batch * tokens * hidden, dtype=torch.float32)
                .reshape(batch, tokens, hidden)
                + self.offset
            )
            key = torch.zeros_like(query)
            value = torch.zeros_like(query)
            return torch.cat((query, key, value), dim=-1)

    class FakeAttn(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.c_attn = FakeCAttn(offset)

    class FakeBlock(torch.nn.Module):
        def __init__(self, offset: float) -> None:
            super().__init__()
            self.attn = FakeAttn(offset)

    class FakeTransformer(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.h = torch.nn.ModuleList((FakeBlock(0.0), FakeBlock(100.0)))

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = type("Config", (), {"n_head": 2})()
            self.transformer = FakeTransformer()

        def forward(self, values: torch.Tensor) -> None:
            for block in self.transformer.h:
                block.attn.c_attn(values)

    model = FakeModel()
    with _Gpt2QueryProjectionCapture(model) as capture:
        model(torch.zeros(1, 3, 4))

    assert len(capture.query_states) == 2
    assert capture.query_states[0].shape == (1, 2, 3, 2)
    assert torch.equal(
        capture.query_states[1].permute(0, 2, 1, 3).reshape(1, 3, 4),
        torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + 100.0,
    )


def test_block_splitting_is_deterministic() -> None:
    spans = split_token_blocks(token_count=7, block_size=3)

    assert [(span.token_start, span.token_len) for span in spans] == [
        (0, 3),
        (3, 3),
        (6, 1),
    ]


def test_block_metadata_creation_from_model_representations_is_deterministic() -> None:
    token_ids = tuple(range(7))
    representations = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    config = BlockIngestConfig(
        block_size=3,
        summary_dim=4,
        representation_name="test_hidden_state",
    )

    first = build_block_metadata_from_representations(representations, token_ids, config)
    second = build_block_metadata_from_representations(representations, token_ids, config)

    assert [block.to_dict() for block in first.metadata_blocks] == [
        block.to_dict() for block in second.metadata_blocks
    ]
    assert first.representation_name == "test_hidden_state"
    assert first.token_count == 7
    assert [block.token_len for block in first.metadata_blocks] == [3, 3, 1]
    assert all(block.summary_scale > 0 for block in first.metadata_blocks)
    assert all(isinstance(block.sign_sketch, int) for block in first.metadata_blocks)
    assert len(first.query_summary.values) == 4


def test_block_metadata_creation_supports_multiscale_candidates() -> None:
    token_ids = tuple(range(40))
    representations = torch.arange(160, dtype=torch.float32).reshape(40, 4)

    result = build_block_metadata_from_representations(
        representations,
        token_ids,
        BlockIngestConfig(
            block_size=16,
            summary_dim=4,
            representation_name="test_hidden_state",
            block_mode="multiscale_16_32",
        ),
    )

    assert result.block_mode == "multiscale_16_32"
    assert len(result.metadata_blocks) == len(result.block_candidates)
    assert result.block_candidates[0].candidate_id == "s16_stride16_t0_16"
    assert any(candidate.block_size == 32 for candidate in result.block_candidates)
    assert [int(block.block_id) for block in result.metadata_blocks] == [
        candidate.block_id for candidate in result.block_candidates
    ]


def test_block_metadata_creation_from_key_representations_is_deterministic() -> None:
    token_ids = tuple(range(6))
    key_layers = (
        (torch.arange(1 * 2 * 6 * 4, dtype=torch.float32).reshape(1, 2, 6, 4), None),
    )
    key_representations, representation_name = select_key_state_with_name(
        key_layers,
        HiddenStateCaptureConfig(representation_source="key_mean_last_layer"),
    )

    first = build_block_metadata_from_representations(
        key_representations,
        token_ids,
        BlockIngestConfig(
            block_size=2,
            summary_dim=4,
            representation_name=representation_name,
        ),
    )
    second = build_block_metadata_from_representations(
        key_representations,
        token_ids,
        BlockIngestConfig(
            block_size=2,
            summary_dim=4,
            representation_name=representation_name,
        ),
    )

    assert first.representation_name == "key_mean_layer_0"
    assert [block.to_dict() for block in first.metadata_blocks] == [
        block.to_dict() for block in second.metadata_blocks
    ]
    assert all(block.summary_scale > 0 for block in first.metadata_blocks)
    assert all(block.summary_norm >= 0 for block in first.metadata_blocks)


def test_block_metadata_uses_explicit_query_representation() -> None:
    token_ids = tuple(range(4))
    representations = torch.zeros(4, 4)
    query = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = build_block_metadata_from_representations(
        representations,
        token_ids,
        BlockIngestConfig(block_size=2, summary_dim=4, representation_name="query_test"),
        query_representation=query,
    )

    assert result.query_summary.summary_norm > 0
    assert any(value != 0 for value in result.query_summary.values)


def test_block_metadata_creation_preserves_per_head_query_key_summaries() -> None:
    token_ids = tuple(range(4))
    pooled_keys = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    pooled_query = torch.tensor([1.0, 0.0, 0.0, 0.0])
    per_head_keys = torch.stack((pooled_keys, pooled_keys + 10.0))
    per_head_query = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    result = build_block_metadata_from_representations(
        pooled_keys,
        token_ids,
        BlockIngestConfig(
            block_size=2,
            summary_dim=4,
            representation_name="query_mean_layer_0_key_mean_layer_0",
        ),
        query_representation=pooled_query,
        per_head_token_representations=per_head_keys,
        per_head_query_representation=per_head_query,
    )

    assert isinstance(result.query_summary, MultiHeadQuerySummary)
    assert all(block.per_head_summary_fp8 for block in result.metadata_blocks)
    assert len(result.metadata_blocks[0].per_head_summary_fp8) == 2
    assert result.query_summary.dequantize_heads().shape == (2, 4)


def test_block_metadata_creation_supports_qk_aggregation_strategies() -> None:
    token_ids = tuple(range(4))
    pooled_keys = torch.zeros(4, 4)
    pooled_query = torch.zeros(4)
    per_head_keys = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0]] * 4,
            [[0.0, 2.0, 0.0, 0.0]] * 4,
        ],
        dtype=torch.float32,
    )
    per_head_query = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        dtype=torch.float32,
    )

    first = build_block_metadata_from_representations(
        pooled_keys,
        token_ids,
        BlockIngestConfig(
            block_size=2,
            summary_dim=4,
            representation_name="query_mean_layer_0_key_mean_layer_0",
            qk_aggregation_strategy="norm_weighted_mean",
        ),
        query_representation=pooled_query,
        per_head_token_representations=per_head_keys,
        per_head_query_representation=per_head_query,
    )
    second = build_block_metadata_from_representations(
        pooled_keys,
        token_ids,
        BlockIngestConfig(
            block_size=2,
            summary_dim=4,
            representation_name="query_mean_layer_0_key_mean_layer_0",
            qk_aggregation_strategy="norm_weighted_mean",
        ),
        query_representation=pooled_query,
        per_head_token_representations=per_head_keys,
        per_head_query_representation=per_head_query,
    )

    assert [block.to_dict() for block in first.metadata_blocks] == [
        block.to_dict() for block in second.metadata_blocks
    ]
    assert isinstance(first.query_summary, MultiHeadQuerySummary)
    assert torch.linalg.vector_norm(first.query_summary.dequantize()) > 0
    assert any(value != 0 for value in first.metadata_blocks[0].summary_fp8)


def test_real_block_selector_bridge_runs_on_mock_runtime() -> None:
    runtime = MockRuntime(token_count=12, hidden_dim=8)

    result = run_real_block_selector(
        runtime,
        "prompt text",
        RealBlockSelectorConfig(
            block_size=3,
            summary_dim=4,
            shortlist_m=4,
            semantic_k=2,
            confidence_margin=0.0,
            keep_recent_blocks=1,
            keep_anchor_blocks=1,
        ),
    )

    assert runtime.loaded is True
    assert result.run_summary.token_count == 12
    assert result.run_summary.block_count == 4
    assert result.run_summary.representation_name == "mock_final_hidden_state"
    assert result.selected_block_ids
    assert result.selected_to_semantic_k_ratio == len(result.selected_block_ids) / 2
    assert result.fallback_mode in {"sparse", "widen_k", "add_recent", "dense"}
    assert result.trace.stage_a_shortlist_block_ids
    assert result.latency.total_sec >= result.latency.selector_sec
    assert result.head_diagnostics == ()
    assert result.head_diagnostic_summary is None


def test_real_block_selector_bridge_builds_inspection_records() -> None:
    runtime = MockRuntime(token_count=9, hidden_dim=8)

    result = run_real_block_selector(
        runtime,
        "prompt text",
        RealBlockSelectorConfig(
            block_size=3,
            summary_dim=4,
            shortlist_m=4,
            semantic_k=1,
            confidence_margin=0.0,
            keep_recent_blocks=1,
            keep_anchor_blocks=1,
            preview_chars=12,
        ),
    )

    assert [(block.token_start, block.token_end) for block in result.block_inspections] == [
        (0, 3),
        (3, 6),
        (6, 9),
    ]
    assert result.selected_block_inspections
    assert result.unselected_block_inspections or len(result.block_inspections) == len(
        result.selected_block_inspections
    )
    assert all(len(block.preview_text) <= 12 for block in result.block_inspections)
    assert any("recent" in block.selected_reason for block in result.block_inspections)
    assert any("anchor" in block.selected_reason for block in result.block_inspections)
    assert any("semantic" in block.selected_reason for block in result.block_inspections)
    assert all(
        block.stage_a_score is not None
        for block in result.block_inspections
    )
    assert result.to_dict()["block_inspections"][0]["preview_text"].startswith("tok")


def test_real_block_selector_bridge_exposes_multiscale_inspection_metadata() -> None:
    runtime = MockRuntime(token_count=40, hidden_dim=8)

    result = run_real_block_selector(
        runtime,
        "prompt text",
        RealBlockSelectorConfig(
            block_size=16,
            block_mode="multiscale_16_32",
            summary_dim=4,
            shortlist_m=8,
            semantic_k=2,
            confidence_margin=0.0,
            keep_recent_blocks=0,
            keep_anchor_blocks=0,
            preview_chars=20,
        ),
    )

    assert result.run_summary.block_mode == "multiscale_16_32"
    assert result.block_inspections[0].candidate_id == "s16_stride16_t0_16"
    assert any(block.block_size == 32 for block in result.block_inspections)
    assert result.to_dict()["block_inspections"][0]["candidate_id"] == "s16_stride16_t0_16"


def test_real_block_selector_bridge_can_emit_head_diagnostics() -> None:
    runtime = PerHeadMockRuntime(token_count=6, hidden_dim=4)

    result = run_real_block_selector(
        runtime,
        "prompt text",
        RealBlockSelectorConfig(
            block_size=2,
            summary_dim=4,
            shortlist_m=3,
            semantic_k=1,
            confidence_margin=0.0,
            keep_recent_blocks=0,
            keep_anchor_blocks=0,
            emit_head_diagnostics=True,
            top_heads=2,
            representation_source="query_mean_last_layer",
            rail_setting="no_rails",
            prompt_name="mock_prompt",
        ),
    )

    assert result.head_diagnostics
    assert result.head_diagnostic_summary is not None
    assert result.head_diagnostics[0].head_scores
    assert result.head_diagnostics[0].top_contributing_heads
    assert result.to_dict()["head_diagnostic_summary"]["prompt_name"] == "mock_prompt"
