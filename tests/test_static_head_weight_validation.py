from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from kvblock.benchmark.real_block_representation_sweep import RetrievalQuality
from kvblock.benchmark.static_head_weight_validation import (
    StaticHeadWeightRunRow,
    default_static_head_weight_schemes,
    format_static_head_weight_report,
    schemes_from_names,
    summarize_static_head_weight_rows,
    write_static_head_weight_validation_outputs,
)


def _load_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / (
        "run_static_head_weight_validation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_static_head_weight_validation",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality(recall: float, precision: float) -> RetrievalQuality:
    return RetrievalQuality(
        expected_block_ids=(1, 2),
        selected_expected_block_ids=(1,),
        missed_expected_block_ids=(2,),
        extra_selected_block_ids=(3,),
        target_recall=recall,
        selected_precision=precision,
        target_hit=recall > 0,
    )


def _row(
    *,
    prompt_name: str,
    scheme_name: str,
    recall: float,
    precision: float,
    selected_to_k: float,
    latency: float,
) -> StaticHeadWeightRunRow:
    return StaticHeadWeightRunRow(
        model_name="gpt2",
        prompt_name=prompt_name,
        prompt_file=f"prompts/{prompt_name}.txt",
        representation_source="query_mean_last_layer",
        scheme_name=scheme_name,
        scheme_description=f"{scheme_name} description",
        head_scoring_mode=(
            "mean_heads" if scheme_name == "pooled_mean_heads" else "weighted_head_mean"
        ),
        head_weights=() if scheme_name == "pooled_mean_heads" else (1.0, 2.0),
        tokens=64,
        blocks=4,
        selected_ids=(1, 3),
        selected_reasons={1: "semantic", 3: "recent"},
        selected_count=2,
        selected_to_semantic_k_ratio=selected_to_k,
        selector_latency_sec=latency,
        total_latency_sec=0.1,
        prefill_latency_sec=0.08,
        metadata_latency_sec=0.01,
        inspection_latency_sec=0.001,
        fallback_mode="sparse",
        raw_margin=0.1,
        retrieval_quality=_quality(recall, precision),
    )


def test_default_schemes_use_explicit_gpt2_class_weights() -> None:
    schemes = {scheme.name: scheme for scheme in default_static_head_weight_schemes()}

    assert tuple(schemes) == (
        "pooled_mean_heads",
        "head9_only",
        "head9_heavy",
        "retrieval_mix",
        "code_mix",
    )
    assert schemes["pooled_mean_heads"].weights_for_head_count(12) == ()
    head9_only = schemes["head9_only"].weights_for_head_count(12)
    assert head9_only[9] == 1.0
    assert sum(head9_only) == 1.0
    assert schemes["head9_heavy"].weights_for_head_count(12)[9] == 3.0
    assert schemes["retrieval_mix"].weights_for_head_count(12)[4] == 2.0
    assert schemes["code_mix"].weights_for_head_count(12)[2] == 3.0

    with pytest.raises(ValueError, match="requires head 9"):
        schemes["head9_only"].weights_for_head_count(4)


def test_scheme_name_resolution_validates_input() -> None:
    schemes = schemes_from_names(("pooled_mean_heads", "code_mix"))

    assert [scheme.name for scheme in schemes] == ["pooled_mean_heads", "code_mix"]
    with pytest.raises(ValueError, match="unknown static head scheme"):
        schemes_from_names(("missing",))


def test_summary_ranks_and_deltas_against_pooled_baseline() -> None:
    rows = (
        _row(
            prompt_name="long_reference",
            scheme_name="pooled_mean_heads",
            recall=0.5,
            precision=0.25,
            selected_to_k=1.0,
            latency=0.01,
        ),
        _row(
            prompt_name="needle",
            scheme_name="pooled_mean_heads",
            recall=0.5,
            precision=0.25,
            selected_to_k=1.0,
            latency=0.01,
        ),
        _row(
            prompt_name="long_reference",
            scheme_name="head9_heavy",
            recall=1.0,
            precision=0.5,
            selected_to_k=1.0,
            latency=0.02,
        ),
        _row(
            prompt_name="needle",
            scheme_name="head9_heavy",
            recall=1.0,
            precision=0.5,
            selected_to_k=1.0,
            latency=0.02,
        ),
    )

    result = summarize_static_head_weight_rows(rows)

    assert result.ranked_summaries[0].scheme_name == "head9_heavy"
    by_scheme = {row.scheme_name: row for row in result.aggregate_summaries}
    assert by_scheme["head9_heavy"].mean_recall == 1.0
    assert by_scheme["head9_heavy"].recall_delta_vs_pooled == 0.5
    assert by_scheme["head9_heavy"].precision_delta_vs_pooled == 0.25
    assert by_scheme["head9_heavy"].selector_latency_delta_vs_pooled_sec == 0.01


def test_prompt_breakdowns_and_report_are_stable() -> None:
    rows = (
        _row(
            prompt_name="code_context",
            scheme_name="pooled_mean_heads",
            recall=0.0,
            precision=0.0,
            selected_to_k=1.0,
            latency=0.01,
        ),
        _row(
            prompt_name="code_context",
            scheme_name="code_mix",
            recall=1.0,
            precision=0.5,
            selected_to_k=1.0,
            latency=0.02,
        ),
    )

    result = summarize_static_head_weight_rows(rows)
    report = format_static_head_weight_report(result)

    assert [(row.prompt_name, row.scheme_name) for row in result.prompt_breakdowns] == [
        ("code_context", "code_mix"),
        ("code_context", "pooled_mean_heads"),
    ]
    assert "STATIC HEAD-WEIGHT VALIDATION" in report
    assert "PER-PROMPT BREAKDOWN" in report
    assert "code_context | code_mix" in report


def test_write_static_head_weight_outputs(tmp_path) -> None:
    result = summarize_static_head_weight_rows(
        (
            _row(
                prompt_name="needle",
                scheme_name="pooled_mean_heads",
                recall=1.0,
                precision=0.5,
                selected_to_k=1.0,
                latency=0.01,
            ),
        )
    )
    json_path = tmp_path / "validation.json"
    text_path = tmp_path / "validation.txt"

    write_static_head_weight_validation_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["scheme_name"] == "pooled_mean_heads"
    assert "STATIC HEAD-WEIGHT VALIDATION" in text_path.read_text(encoding="utf-8")


def test_static_head_weight_script_parser() -> None:
    module = _load_script()
    args = module.build_parser().parse_args(
        [
            "--models",
            "gpt2",
            "--schemes",
            "pooled_mean_heads,head9_only",
            "--semantic-k",
            "8",
            "--keep-recent-blocks",
            "0",
            "--local-files-only",
        ]
    )

    assert args.models == "gpt2"
    assert args.schemes == "pooled_mean_heads,head9_only"
    assert args.semantic_k == 8
    assert args.keep_recent_blocks == 0
    assert args.local_files_only is True
