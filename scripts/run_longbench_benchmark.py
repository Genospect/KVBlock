#!/usr/bin/env python3
"""Run KVBlock selector benchmarks over Hugging Face LongBench samples."""

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
    parser = argparse.ArgumentParser(
        description="Run LongBench samples through the current KVBlock selector path.",
    )
    parser.add_argument(
        "--models",
        default="gpt2",
        help="Comma-separated HF model names or local paths.",
    )
    parser.add_argument(
        "--longbench-datasets",
        default="narrativeqa,hotpotqa,lcc",
        help="Comma-separated LongBench datasets to load.",
    )
    parser.add_argument("--dataset-repo", default="THUDM/LongBench")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum samples per dataset after filtering.",
    )
    parser.add_argument(
        "--length-bucket",
        default="all",
        help="Length bucket: all, 0-4k, 4k-8k, or 8k+.",
    )
    parser.add_argument(
        "--prompt-cache-dir",
        default="results/longbench/prompts",
        help="Directory for materialized LongBench prompt files.",
    )
    parser.add_argument(
        "--block-modes",
        default="fixed_40",
        help="Comma-separated block modes to compare.",
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
        help="Optional top-token override; usually empty for LongBench.",
    )
    parser.add_argument("--block-size", type=int, default=40)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=2,
        help="Number of coarse regions to refine for coarse-to-fine block modes.",
    )
    parser.add_argument(
        "--mixed-refine-parent-k",
        type=int,
        default=8,
        help="Number of global fixed40 parents to refine for mixed global-refine modes.",
    )
    parser.add_argument(
        "--mixed-global-anchor-k",
        type=int,
        default=8,
        help="Number of global fixed40 parents retained as backbone anchors.",
    )
    parser.add_argument(
        "--mixed-fallback-margin",
        type=float,
        default=0.05,
        help="Fallback to fixed40 when the global rail raw margin is below this value.",
    )
    parser.add_argument("--keep-recent-blocks", type=int, default=0)
    parser.add_argument("--keep-anchor-blocks", type=int, default=0)
    parser.add_argument("--top-token-count", type=int, default=4)
    parser.add_argument(
        "--rerank-mode",
        default="none",
        choices=("none", "semantic_plus_tokenmax", "dense_qk_token_refine"),
        help=(
            "Optional benchmark-only rerank over ranked candidates. "
            "semantic_plus_tokenmax blends selector score with lexical token "
            "matches; dense_qk_token_refine uses exact QK token scores over "
            "the shortlist."
        ),
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=0.3,
        help="Weight assigned to tokenmax score when --rerank-mode is enabled.",
    )
    parser.add_argument(
        "--refine-top-n-tokens",
        type=int,
        default=4,
        help="Number of strongest block tokens averaged by dense_qk_token_refine.",
    )
    parser.add_argument(
        "--refine-score-mode",
        default="raw_topn_mean",
        choices=("raw_topn_mean", "cosine_topn_mean", "softmax_mass"),
        help="Dense-QK refinement scorer used by dense_qk_token_refine.",
    )
    parser.add_argument(
        "--stage-c-policy",
        default="refined_only",
        choices=("refined_only", "semantic_refined_mix"),
        help=(
            "Benchmark-only final anchor policy. refined_only preserves the "
            "current reranked top-K path; semantic_refined_mix splits anchors "
            "between original semantic rank and refined rank."
        ),
    )
    parser.add_argument(
        "--exclude-scaffold-blocks",
        action="store_true",
        help="LongBench-only benchmark hygiene: exclude metadata-only prompt blocks from rerank candidates.",
    )
    parser.add_argument(
        "--neighbor-expansion",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Benchmark-only final selection expansion by adjacent token blocks.",
    )
    parser.add_argument(
        "--halo-radius",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help=(
            "Budgeted locality halo around semantic anchors. Mutually exclusive "
            "with --neighbor-expansion."
        ),
    )
    parser.add_argument(
        "--max-selected-blocks",
        type=int,
        default=None,
        help="Maximum final selected blocks when --halo-radius is enabled.",
    )
    parser.add_argument(
        "--evidence-window-radius",
        type=int,
        default=0,
        choices=(0, 1, 2),
        help="Metric-only radius for evidence-window recall/precision reporting.",
    )
    parser.add_argument(
        "--oracle-mode",
        default="none",
        choices=("none", "dense_qk"),
        help="Optional benchmark-only dense QK oracle diagnostic.",
    )
    parser.add_argument(
        "--oracle-top-k",
        default="4,8,16,32",
        help="Comma-separated oracle cutoffs to report, e.g. 4,8,16,32.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/longbench",
        help="Directory for JSON and text LongBench reports.",
    )
    parser.add_argument("--name", default="longbench_selector_benchmark")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.longbench_adapter import (
        format_longbench_benchmark_report,
        parse_dataset_names,
        parse_length_bucket,
        parse_oracle_top_k,
        run_longbench_selector_benchmark,
        write_longbench_benchmark_outputs,
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
    result = run_longbench_selector_benchmark(
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
            preview_chars=160,
            include_block_text=True,
            representation_source=args.representation_source,
            rail_setting=(
                f"longbench_recent{args.keep_recent_blocks}_"
                f"anchor{args.keep_anchor_blocks}"
            ),
        ),
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_longbench_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )
    print(format_longbench_benchmark_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
