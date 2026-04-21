from __future__ import annotations

from argparse import Namespace
import json

from kvblock.cli import real_block_selector as cli
from kvblock.cli.real_block_selector import build_parser, read_prompt_from_args
from tests.test_runtime_bridge import MockRuntime


def test_real_block_selector_cli_defaults_to_candidate_selector_config() -> None:
    args = build_parser().parse_args([])

    assert args.model == "sshleifer/tiny-gpt2"
    assert args.shortlist_m == 16
    assert args.semantic_k == 10
    assert args.confidence_margin == 0.0
    assert args.device == "cpu"
    assert args.preview_chars == 120
    assert args.keep_recent_blocks == 4
    assert args.keep_anchor_blocks == 2
    assert args.representation_source == "avg_mid4_hidden"
    assert args.layer_index == -1
    assert args.block_mode == "fixed"


def test_real_block_selector_cli_parses_representation_source() -> None:
    args = build_parser().parse_args(
        ["--representation-source", "hidden_layer_index", "--layer-index", "1"]
    )

    assert args.representation_source == "hidden_layer_index"
    assert args.layer_index == 1


def test_real_block_selector_cli_parses_key_representation_source() -> None:
    args = build_parser().parse_args(["--representation-source", "key_mean_last_layer"])

    assert args.representation_source == "key_mean_last_layer"


def test_real_block_selector_cli_parses_query_representation_source() -> None:
    args = build_parser().parse_args(["--representation-source", "query_mean_mid_layer"])

    assert args.representation_source == "query_mean_mid_layer"


def test_real_block_selector_cli_parses_head_scoring_options() -> None:
    args = build_parser().parse_args(
        [
            "--head-scoring-mode",
            "topk_head_mean",
            "--head-top-k",
            "3",
            "--head-weights",
            "1,0.5,2",
            "--qk-aggregation",
            "top_token_mean",
            "--top-token-count",
            "2",
            "--show-head-diagnostics",
            "--top-heads",
            "4",
        ]
    )

    assert args.head_scoring_mode == "topk_head_mean"
    assert args.head_top_k == 3
    assert cli._parse_head_weights(args.head_weights) == (1.0, 0.5, 2.0)
    assert args.qk_aggregation == "top_token_mean"
    assert args.top_token_count == 2
    assert args.show_head_diagnostics is True
    assert args.top_heads == 4


def test_real_block_selector_cli_parses_query_key_inspection_options() -> None:
    args = build_parser().parse_args(
        [
            "--show-query-key-inspection",
            "--show-missed-blocks",
            "--show-top-unselected",
            "3",
            "--relevance-fragments",
            "needle,target",
            "--inspection-json-out",
            "inspection.json",
        ]
    )

    assert args.show_query_key_inspection is True
    assert args.show_missed_blocks is True
    assert args.show_top_unselected == 3
    assert cli._parse_relevance_fragments(args.relevance_fragments) == (
        "needle",
        "target",
    )
    assert args.inspection_json_out == "inspection.json"


def test_real_block_selector_cli_parses_rail_overrides() -> None:
    args = build_parser().parse_args(
        ["--keep-recent-blocks", "0", "--keep-anchor-blocks", "0"]
    )

    assert args.keep_recent_blocks == 0
    assert args.keep_anchor_blocks == 0


def test_real_block_selector_cli_parses_dynamic_block_options() -> None:
    args = build_parser().parse_args(
        ["--block-mode", "multiscale_16_24_32", "--overlap-stride", "8"]
    )

    assert args.block_mode == "multiscale_16_24_32"
    assert args.overlap_stride == 8


def test_real_block_selector_cli_reads_prompt_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("A small prompt for real-block ingest.", encoding="utf-8")
    args = Namespace(prompt=None, prompt_file=str(prompt_file))

    assert read_prompt_from_args(args) == "A small prompt for real-block ingest."


def test_real_block_selector_cli_prompt_arg_wins_over_file(tmp_path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("file prompt", encoding="utf-8")
    args = Namespace(prompt="inline prompt", prompt_file=str(prompt_file))

    assert read_prompt_from_args(args) == "inline prompt"


def test_real_block_selector_cli_writes_inspection_json(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "inspection.json"
    head_output_path = tmp_path / "head_diagnostics.json"

    monkeypatch.setattr(
        cli,
        "create_runtime_backend",
        lambda _config, **_kwargs: MockRuntime(token_count=6, hidden_dim=8),
    )

    cli_result = cli.run_real_block_selector_cli(
        [
            "--prompt",
            "inspect this prompt",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
            "--keep-recent-blocks",
            "1",
            "--keep-anchor-blocks",
            "1",
            "--preview-chars",
            "9",
            "--show-selected-blocks",
            "--show-stage-scores",
            "--json-out",
            str(output_path),
            "--head-diagnostics-json-out",
            str(head_output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    head_payload = json.loads(head_output_path.read_text(encoding="utf-8"))
    assert cli_result.show_selected_blocks is True
    assert cli_result.show_stage_scores is True
    assert cli_result.head_diagnostics_json_out_path == head_output_path
    assert payload["run_summary"]["block_count"] == 3
    assert "latency" in payload
    assert len(payload["block_inspections"]) == 3
    assert payload["selected_block_inspections"]
    assert "preview_text" in payload["block_inspections"][0]
    assert "selected_reason" in payload["block_inspections"][0]
    assert head_payload["head_diagnostics"] == []


def test_real_block_selector_cli_writes_query_key_inspection_json(monkeypatch, tmp_path) -> None:
    output_path = tmp_path / "query_key_inspection.json"

    monkeypatch.setattr(
        cli,
        "create_runtime_backend",
        lambda _config, **_kwargs: MockRuntime(token_count=6, hidden_dim=8),
    )

    cli_result = cli.run_real_block_selector_cli(
        [
            "--prompt",
            "Where is tok2 evidence?",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
            "--keep-recent-blocks",
            "0",
            "--keep-anchor-blocks",
            "0",
            "--preview-chars",
            "40",
            "--show-query-key-inspection",
            "--show-missed-blocks",
            "--show-top-unselected",
            "1",
            "--relevance-fragments",
            "tok2",
            "--inspection-json-out",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    inspection = payload["query_key_inspection"]
    assert cli_result.show_query_key_inspection is True
    assert cli_result.show_missed_blocks is True
    assert cli_result.inspection_json_out_path == output_path
    assert inspection["prompt_id"] == "inline_prompt"
    assert inspection["relevance_fragments"] == ["tok2"]
    assert "query_summary_metadata" in inspection
    assert inspection["block_records"]


def test_real_block_selector_cli_keeps_query_key_inspection_off_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "create_runtime_backend",
        lambda _config, **_kwargs: MockRuntime(token_count=5, hidden_dim=8),
    )

    cli_result = cli.run_real_block_selector_cli(
        [
            "--prompt",
            "no inspection",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
        ]
    )

    assert cli_result.show_query_key_inspection is False
    assert cli_result.result.query_key_inspection is None


def test_real_block_selector_cli_passes_query_representation_source(monkeypatch) -> None:
    captured = {}

    def fake_create_runtime_backend(_config, **kwargs):
        captured["capture_config"] = kwargs["capture_config"]
        return MockRuntime(token_count=6, hidden_dim=8)

    monkeypatch.setattr(cli, "create_runtime_backend", fake_create_runtime_backend)

    cli.run_real_block_selector_cli(
        [
            "--prompt",
            "query source prompt",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
            "--representation-source",
            "query_mean_mid_layer",
            "--head-scoring-mode",
            "max_head_score",
        ]
    )

    assert captured["capture_config"].representation_source == "query_mean_mid_layer"


def test_real_block_selector_summary_can_print_all_blocks(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "create_runtime_backend",
        lambda _config, **_kwargs: MockRuntime(token_count=5, hidden_dim=8),
    )
    cli_result = cli.run_real_block_selector_cli(
        [
            "--prompt",
            "inspect this prompt",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
            "--show-all-blocks",
            "--show-stage-scores",
        ]
    )

    summary = cli.format_real_block_selector_summary(
        cli_result.result,
        show_all_blocks=cli_result.show_all_blocks,
        show_stage_scores=cli_result.show_stage_scores,
        show_head_diagnostics=True,
    )

    assert "blocks:" in summary
    assert "block=0" in summary
    assert "stage_a=" in summary
    assert "head_diagnostics:" in summary


def test_real_block_selector_summary_can_print_query_key_inspection(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "create_runtime_backend",
        lambda _config, **_kwargs: MockRuntime(token_count=6, hidden_dim=8),
    )
    cli_result = cli.run_real_block_selector_cli(
        [
            "--prompt",
            "Where is tok2 evidence?",
            "--block-size",
            "2",
            "--summary-dim",
            "4",
            "--shortlist-m",
            "3",
            "--semantic-k",
            "1",
            "--show-query-key-inspection",
            "--show-missed-blocks",
            "--relevance-fragments",
            "tok2",
        ]
    )

    summary = cli.format_real_block_selector_summary(
        cli_result.result,
        show_query_key_inspection=cli_result.show_query_key_inspection,
        show_missed_blocks=cli_result.show_missed_blocks,
        show_top_unselected=cli_result.show_top_unselected,
    )

    assert "query_key_inspection:" in summary
    assert "selected_relevant=" in summary
    assert "high_scoring_near_miss_blocks:" in summary
