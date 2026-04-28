#!/usr/bin/env python3
"""Run output-based LongBench QA evaluation over KVBlock-selected context."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated HF model names.")
    parser.add_argument(
        "--longbench-datasets",
        default="hotpotqa,musique",
        help="Comma-separated LongBench dataset names.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-repo", default="THUDM/LongBench")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--length-bucket",
        default="all",
        help="Length bucket such as all, 0-4k, 4k-8k, 8k-16k.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        default="results/longbench_output/prompts",
        help="Directory for materialized LongBench prompts.",
    )
    parser.add_argument(
        "--block-modes",
        default="fixed_40",
        help="Comma-separated block modes.",
    )
    parser.add_argument(
        "--representation-source",
        default="query_mean_last_layer",
        choices=(
            "query_mean_last_layer",
            "query_mean_mid_layer",
            "query_avg_last4",
            "query_only_last_layer",
            "query_only_avg_last4",
            "query_only_attention_masked",
        ),
    )
    parser.add_argument("--qk-aggregation", default="block_max")
    parser.add_argument(
        "--needle-qk-aggregation",
        default="",
        choices=("", "mean", "max", "top_token_mean", "block_max"),
        help="Optional qk aggregation override for the synthetic needle prompt.",
    )
    parser.add_argument("--coarse-top-k", type=int, default=2)
    parser.add_argument("--mixed-refine-parent-k", type=int, default=8)
    parser.add_argument("--mixed-global-anchor-k", type=int, default=8)
    parser.add_argument("--mixed-fallback-margin", type=float, default=0.05)
    parser.add_argument("--mixed-max-children-per-parent", type=int, default=None)
    parser.add_argument("--mixed-child-window-radius", type=int, default=0)
    parser.add_argument("--keep-recent-blocks", type=int, default=0)
    parser.add_argument("--keep-anchor-blocks", type=int, default=0)
    parser.add_argument("--top-token-count", type=int, default=4)
    parser.add_argument(
        "--rerank-mode",
        default="none",
        choices=("none", "semantic_plus_tokenmax", "dense_qk_token_refine"),
    )
    parser.add_argument("--rerank-weight", type=float, default=0.3)
    parser.add_argument("--refine-top-n-tokens", type=int, default=4)
    parser.add_argument(
        "--refine-score-mode",
        default="raw_topn_mean",
        choices=("raw_topn_mean", "cosine_topn_mean", "softmax_mass"),
    )
    parser.add_argument(
        "--stage-c-policy",
        default="refined_only",
        choices=("refined_only", "semantic_refined_mix"),
    )
    parser.add_argument("--exclude-scaffold-blocks", action="store_true")
    parser.add_argument(
        "--neighbor-expansion",
        type=int,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--halo-radius",
        type=int,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument("--max-selected-blocks", type=int, default=None)
    parser.add_argument(
        "--evidence-window-radius",
        type=int,
        default=0,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--oracle-mode",
        default="none",
        choices=("none", "dense_qk"),
    )
    parser.add_argument("--oracle-top-k", default="4,8,16,32")
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--overlap-stride", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--context-reconstruction",
        default="selected_spans",
        choices=("selected_spans", "passage_window"),
        help="How selected token spans are rebuilt into generation context.",
    )
    parser.add_argument(
        "--passage-window-tokens",
        type=int,
        default=120,
        help="Minimum local token window per selected span in passage_window mode.",
    )
    parser.add_argument(
        "--passage-header-tokens",
        type=int,
        default=24,
        help="Passage-leading tokens to preserve in passage_window mode.",
    )
    parser.add_argument(
        "--selection-min-blocks",
        type=int,
        default=0,
        help="Minimum output-context blocks to keep before score-ratio trimming.",
    )
    parser.add_argument(
        "--selection-score-ratio",
        type=float,
        default=None,
        help=(
            "Optional output-only trim: after min blocks, stop when block score is "
            "below this ratio of the top selected score."
        ),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/longbench_output",
        help="Directory for JSON and text output reports.",
    )
    parser.add_argument("--name", default="longbench_output_benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.longbench_adapter import (
        parse_dataset_names,
        parse_length_bucket,
        parse_oracle_top_k,
    )
    from kvblock.benchmark.longbench_output import (
        format_longbench_output_report,
        run_longbench_output_benchmark,
        write_longbench_output_benchmark_outputs,
    )
    from kvblock.kv.block_modes import block_modes_from_names
    from kvblock.kv.qk_aggregation import qk_aggregation_strategy_from_name
    from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

    args = build_parser().parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    needle_strategy = (
        None
        if not args.needle_qk_aggregation.strip()
        else qk_aggregation_strategy_from_name(args.needle_qk_aggregation)
    )
    result = run_longbench_output_benchmark(
        name=args.name,
        model_names=models,
        dataset_names=parse_dataset_names(args.longbench_datasets),
        split=args.split,
        dataset_repo=args.dataset_repo,
        limit_per_dataset=args.limit,
        length_bucket=parse_length_bucket(args.length_bucket),
        prompt_cache_dir=args.prompt_cache_dir,
        block_modes=block_modes_from_names(
            tuple(item.strip() for item in args.block_modes.split(",") if item.strip())
        ),
        representation_source=args.representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy_from_name(args.qk_aggregation),
        needle_qk_aggregation_strategy=needle_strategy,
        coarse_top_k=args.coarse_top_k,
        mixed_refine_parent_k=args.mixed_refine_parent_k,
        mixed_global_anchor_k=args.mixed_global_anchor_k,
        mixed_fallback_margin=args.mixed_fallback_margin,
        mixed_max_children_per_parent=args.mixed_max_children_per_parent,
        mixed_child_window_radius=args.mixed_child_window_radius,
        rerank_mode=args.rerank_mode,
        rerank_weight=args.rerank_weight,
        refine_top_n_tokens=args.refine_top_n_tokens,
        refine_score_mode=args.refine_score_mode,
        stage_c_policy=args.stage_c_policy,
        exclude_scaffold_blocks=args.exclude_scaffold_blocks,
        neighbor_expansion=args.neighbor_expansion,
        halo_radius=args.halo_radius,
        max_selected_blocks=args.max_selected_blocks,
        evidence_window_radius=args.evidence_window_radius,
        oracle_mode=args.oracle_mode,
        oracle_top_k=parse_oracle_top_k(args.oracle_top_k),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        context_reconstruction=args.context_reconstruction,
        passage_window_tokens=args.passage_window_tokens,
        passage_header_tokens=args.passage_header_tokens,
        selection_min_blocks=args.selection_min_blocks,
        selection_score_ratio=args.selection_score_ratio,
        load_config_kwargs={
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "local_files_only": args.local_files_only,
            "trust_remote_code": args.trust_remote_code,
        },
        selector_config=RealBlockSelectorConfig(
            block_size=args.block_size,
            shortlist_m=args.shortlist_m,
            semantic_k=args.semantic_k,
            confidence_margin=args.confidence_margin,
            keep_recent_blocks=args.keep_recent_blocks,
            keep_anchor_blocks=args.keep_anchor_blocks,
            top_token_count=args.top_token_count,
            overlap_stride=args.overlap_stride,
            preview_chars=160,
            include_block_text=True,
            representation_source=args.representation_source,
        ),
    )
    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_longbench_output_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )
    print(format_longbench_output_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
