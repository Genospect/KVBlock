from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from kvblock.benchmark.head_ablation import (
    OfflineHeadScoreBlock,
    evaluate_prompt_head_ablation,
    format_head_ablation_report,
    load_blocks_from_head_diagnostic_json,
    summarize_head_ablation_results,
)


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_head_ablation_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("run_head_ablation_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_head_and_leave_one_out_metrics() -> None:
    result = evaluate_prompt_head_ablation(
        model_name="model",
        prompt_name="long_reference",
        prompt_file="prompt.txt",
        representation_source="query_mean_last_layer",
        representation_name="query_mean_layer_0_key_mean_layer_0",
        blocks=(
            OfflineHeadScoreBlock(0, (0.9, 0.0), preview_text="target AES-256"),
            OfflineHeadScoreBlock(1, (0.1, 1.0), preview_text="distractor"),
        ),
        target_fragments=("AES-256",),
        top_k=1,
    )

    assert result.pooled_baseline.selected_block_ids == (1,)
    assert result.pooled_baseline.recall == 0.0
    assert result.single_head_results[0].selected_block_ids == (0,)
    assert result.single_head_results[0].recall == 1.0
    assert result.single_head_results[0].recall_delta_vs_pooled == 1.0
    assert result.leave_one_out_results[1].selected_block_ids == (0,)
    assert result.leave_one_out_results[1].recall_delta_vs_pooled == 1.0
    assert result.leave_one_out_results[0].recall_delta_vs_pooled == 0.0


def test_head_ablation_summary_tables_and_prompt_specialists() -> None:
    long_result = evaluate_prompt_head_ablation(
        model_name="model",
        prompt_name="long_reference",
        prompt_file="long.txt",
        representation_source="query_mean_last_layer",
        representation_name="query",
        blocks=(
            OfflineHeadScoreBlock(0, (0.9, 0.0), preview_text="target AES-256"),
            OfflineHeadScoreBlock(1, (0.1, 1.0), preview_text="distractor"),
        ),
        target_fragments=("AES-256",),
        top_k=1,
    )
    code_result = evaluate_prompt_head_ablation(
        model_name="model",
        prompt_name="code_context",
        prompt_file="code.txt",
        representation_source="query_mean_last_layer",
        representation_name="query",
        blocks=(
            OfflineHeadScoreBlock(0, (0.0, 0.9), preview_text="def calculate_total"),
            OfflineHeadScoreBlock(1, (1.0, 0.1), preview_text="distractor"),
        ),
        target_fragments=("calculate_total",),
        top_k=1,
    )

    summary = summarize_head_ablation_results((long_result, code_result), top_n=1)
    report = format_head_ablation_report(summary)

    assert summary.best_single_heads_overall[0].mean_recall == 0.5
    assert {row.prompt_name: row.head_index for row in summary.prompt_specialist_heads} == {
        "code_context": 1,
        "long_reference": 0,
    }
    assert "BEST SINGLE HEADS OVERALL" in report
    assert "PROMPT-SPECIALIST HEADS" in report


def test_load_blocks_from_head_diagnostic_json(tmp_path) -> None:
    path = tmp_path / "head_diag.json"
    path.write_text(
        json.dumps(
            {
                "head_diagnostics": [
                    {
                        "block_id": 2,
                        "head_scores": [0.1, 0.2],
                        "preview_text": "target",
                        "token_start": 4,
                        "token_end": 6,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    blocks = load_blocks_from_head_diagnostic_json(path)

    assert blocks[0].block_id == 2
    assert blocks[0].head_scores == (0.1, 0.2)
    assert blocks[0].token_start == 4


def test_head_ablation_script_parser() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "--models",
            "gpt2",
            "--representation-source",
            "query_mean_mid_layer",
            "--semantic-k",
            "8",
            "--top-n",
            "4",
            "--local-files-only",
        ]
    )

    assert args.models == "gpt2"
    assert args.representation_source == "query_mean_mid_layer"
    assert args.semantic_k == 8
    assert args.top_n == 4
    assert args.local_files_only is True
