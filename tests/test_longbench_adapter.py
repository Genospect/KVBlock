from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import zipfile

import pytest

from kvblock.benchmark import longbench_adapter as longbench
from kvblock.benchmark.dynamic_block_benchmark import (
    DynamicBlockBenchmarkResult,
    DynamicBlockRunRow,
)
from kvblock.benchmark.real_block_representation_sweep import RetrievalQuality


def _fake_longbench_rows(repo: str, dataset_name: str, split: str):
    assert repo == "THUDM/LongBench"
    assert split == "test"
    return (
        {
            "_id": f"{dataset_name}-short",
            "dataset": dataset_name,
            "context": "Short context containing alpha evidence.",
            "input": "What evidence is present?",
            "answers": ["alpha evidence"],
            "length": 1200,
        },
        {
            "_id": f"{dataset_name}-mid",
            "dataset": dataset_name,
            "context": "Longer context containing beta evidence.",
            "input": "What evidence is present?",
            "answers": ["beta evidence"],
            "length": 5200,
        },
    )


def test_longbench_record_conversion_preserves_fields() -> None:
    record = longbench.longbench_record_from_mapping(
        {
            "_id": "sample-1",
            "dataset": "narrativeqa",
            "context": "The answer is Mercury.",
            "input": "Which planet?",
            "answers": [["Mercury"], "Mercury"],
            "length": "4096",
        },
        dataset_name="narrativeqa",
        index=0,
    )

    assert record.sample_id == "sample-1"
    assert record.dataset_name == "narrativeqa"
    assert record.answers == ("Mercury",)
    assert record.length == 4096
    assert "CONTEXT:" in record.prompt_text
    assert "INPUT:" in record.prompt_text
    assert "ANSWERS:" not in record.prompt_text
    assert record.prompt_text.rstrip().endswith("Which planet?")


def test_load_longbench_records_filters_length_and_limit() -> None:
    records = longbench.load_longbench_records(
        dataset_names=("narrativeqa", "hotpotqa"),
        length_bucket="4k-8k",
        limit_per_dataset=1,
        dataset_loader=_fake_longbench_rows,
    )

    assert [record.dataset_name for record in records] == ["narrativeqa", "hotpotqa"]
    assert all(record.length == 5200 for record in records)


def test_materialize_longbench_prompt_cases(tmp_path: Path) -> None:
    record = longbench.LongBenchRecord(
        dataset_name="qasper",
        sample_id="paper/42",
        context="A paper mentions sparse attention.",
        input_text="What mechanism is mentioned?",
        answers=("sparse attention",),
        length=6000,
    )

    cases, metadata = longbench.materialize_longbench_prompt_cases(
        (record,),
        prompt_dir=tmp_path,
    )

    assert len(cases) == 1
    assert cases[0].target_fragments == ("sparse attention",)
    assert cases[0].path.exists()
    assert "sparse attention" in cases[0].path.read_text(encoding="utf-8")
    assert metadata[0].dataset_name == "qasper"
    assert metadata[0].answer_labels == ("sparse attention",)
    assert metadata[0].answer_present_count == 1
    assert metadata[0].answer_missing_count == 0
    assert metadata[0].answer_presence_rate == 1.0


def test_run_longbench_benchmark_wraps_dynamic_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_dynamic_benchmark(**kwargs):
        case = kwargs["prompt_cases"][0]
        quality = RetrievalQuality(
            expected_block_ids=(1,),
            selected_expected_block_ids=(1,),
            missed_expected_block_ids=(),
            extra_selected_block_ids=(2,),
            target_recall=1.0,
            selected_precision=0.5,
            target_hit=True,
        )
        row = DynamicBlockRunRow(
            model_name="fake/model",
            prompt_name=case.name,
            prompt_file=str(case.path),
            representation_source="query_mean_last_layer",
            representation_name="query_mean_last_layer",
            qk_aggregation_strategy="block_max",
            block_mode="fixed_40",
            suppression_mode="none",
            suppression_threshold=0.75,
            keep_recent_blocks=0,
            keep_anchor_blocks=0,
            tokens=5120,
            candidate_block_count=128,
            candidate_count_before_suppression=128,
            candidate_count_after_suppression=128,
            selected_ids=(1, 2),
            selected_candidate_ids=("s40_t0_40", "s40_t40_80"),
            selected_spans=("0:40", "40:80"),
            selected_block_sizes=(40, 40),
            selected_candidate_roles=("block", "block"),
            suppression_decisions=(),
            selected_count=2,
            selected_to_semantic_k_ratio=0.5,
            selector_latency_sec=0.002,
            total_latency_sec=0.25,
            prefill_latency_sec=0.2,
            metadata_latency_sec=0.04,
            inspection_latency_sec=0.0,
            fallback_mode="sparse",
            raw_margin=0.1,
            retrieval_quality=quality,
        )
        return DynamicBlockBenchmarkResult(
            rows=(row,),
            aggregate_summaries=(),
            ranked_summaries=(),
            prompt_breakdowns=(),
            model_load_seconds={"fake/model": 0.01},
        )

    monkeypatch.setattr(longbench, "run_dynamic_block_benchmark", fake_dynamic_benchmark)

    result = longbench.run_longbench_selector_benchmark(
        model_names=("fake/model",),
        dataset_names=("narrativeqa",),
        limit_per_dataset=1,
        prompt_cache_dir=tmp_path,
        dataset_loader=_fake_longbench_rows,
    )

    assert result.samples[0].dataset_name == "narrativeqa"
    assert result.rows[0].dataset_name == "narrativeqa"
    assert result.rows[0].candidate_block_count == 128
    assert result.rows[0].answer_present_count == 1
    assert result.rows[0].expected_block_count == 1
    assert result.rows[0].selected_expected_block_count == 1
    assert result.rows[0].expected_block_ids == (1,)
    assert result.rows[0].selected_block_ids == (1, 2)
    assert result.rows[0].expected_block_distance == 0
    assert result.rows[0].target_recall == 1.0
    assert result.dataset_summaries[0].mean_precision == 0.5
    assert result.dataset_summaries[0].scoreable_run_count == 1
    assert result.to_dict()["rows"][0]["selected_to_semantic_k_ratio"] == 0.5


def test_parse_helpers_validate_inputs() -> None:
    assert longbench.parse_dataset_names("narrativeqa,lcc") == ("narrativeqa", "lcc")
    assert longbench.parse_length_bucket("8k+").contains(9000)
    assert not longbench.parse_length_bucket("8k+").contains(7999)

    with pytest.raises(ValueError):
        longbench.parse_dataset_names("unknown")

    with pytest.raises(ValueError):
        longbench.parse_length_bucket("bad")


def test_direct_longbench_zip_loader_reads_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    zip_path = tmp_path / "data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "data/narrativeqa.jsonl",
            (
                '{"input":"Question?","context":"Context.",'
                '"answers":["Answer"],"length":5120,"dataset":"narrativeqa",'
                '"language":"en","_id":"row-1","all_classes":[]}\n'
            ),
        )
    fake_hub = SimpleNamespace(
        hf_hub_download=lambda repo_id, filename, repo_type: str(zip_path)
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    rows = longbench._load_longbench_jsonl_from_hub(
        "THUDM/LongBench",
        "narrativeqa",
    )

    assert len(rows) == 1
    assert rows[0]["_id"] == "row-1"
    assert rows[0]["answers"] == ["Answer"]


def test_longbench_cli_parser_accepts_baseline_flags() -> None:
    script_path = Path("scripts/run_longbench_benchmark.py")
    spec = importlib.util.spec_from_file_location("run_longbench_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.build_parser().parse_args(
        [
            "--models",
            "Qwen/Qwen2.5-3B-Instruct",
            "--longbench-datasets",
            "narrativeqa,musique",
            "--length-bucket",
            "4k-8k",
            "--limit",
            "2",
            "--device-map",
            "auto",
        ]
    )

    assert args.models == "Qwen/Qwen2.5-3B-Instruct"
    assert args.longbench_datasets == "narrativeqa,musique"
    assert args.length_bucket == "4k-8k"
    assert args.limit == 2
    assert args.device_map == "auto"
