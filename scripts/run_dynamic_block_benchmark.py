#!/usr/bin/env python3
"""Run focused fixed-vs-multi-scale block candidate benchmarks."""

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
        description="Compare fixed and multi-scale block candidate modes.",
    )
    parser.add_argument(
        "--models",
        default="gpt2",
        help="Comma-separated HF model names or local paths.",
    )
    parser.add_argument(
        "--prompts",
        default="needle,long_reference,code_context,repeated_reference",
        help="Comma-separated prompt case names to include.",
    )
    parser.add_argument(
        "--block-modes",
        default=(
            "fixed_16,fixed_24,fixed_32,fixed_40,"
            "multiscale_16_32,multiscale_16_24_32,overlap_16_stride_8"
        ),
        help="Comma-separated block modes to compare.",
    )
    parser.add_argument(
        "--representation-source",
        default="query_mean_last_layer",
        choices=(
            "query_mean_last_layer",
            "query_mean_mid_layer",
            "query_avg_last4",
        ),
        help="Query/key representation source to use.",
    )
    parser.add_argument(
        "--qk-aggregation",
        default="block_max",
        help="Default query/key aggregation strategy.",
    )
    parser.add_argument(
        "--needle-qk-aggregation",
        default="top_token_mean",
        help="Optional query/key aggregation override for needle prompts; use empty to disable.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--keep-recent-blocks", type=int, default=0)
    parser.add_argument("--keep-anchor-blocks", type=int, default=0)
    parser.add_argument("--top-token-count", type=int, default=4)
    parser.add_argument("--overlap-stride", type=int, default=None)
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=2,
        help="Number of coarse regions retained by coarse-to-fine block modes.",
    )
    parser.add_argument(
        "--suppression-modes",
        default="none",
        help=(
            "Comma-separated benchmark-only suppression modes: none, "
            "overlap_threshold, interval_iou, keep_highest_score_per_overlap_cluster."
        ),
    )
    parser.add_argument(
        "--suppression-threshold",
        type=float,
        default=0.75,
        help="Overlap/IoU threshold used by non-none suppression modes.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/dynamic_blocks",
        help="Directory for JSON and text benchmark reports.",
    )
    parser.add_argument(
        "--name",
        default="dynamic_block_benchmark",
        help="Output file stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.dynamic_block_benchmark import (
        default_dynamic_prompt_cases,
        format_dynamic_block_report,
        run_dynamic_block_benchmark,
        write_dynamic_block_benchmark_outputs,
    )
    from kvblock.kv.block_modes import block_modes_from_names
    from kvblock.kv.qk_aggregation import qk_aggregation_strategy_from_name
    from kvblock.benchmark.candidate_suppression import suppression_modes_from_names
    from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

    args = build_parser().parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    prompt_names = tuple(item.strip() for item in args.prompts.split(",") if item.strip())
    block_modes = block_modes_from_names(
        tuple(item.strip() for item in args.block_modes.split(",") if item.strip())
    )
    suppression_modes = suppression_modes_from_names(
        tuple(item.strip() for item in args.suppression_modes.split(",") if item.strip())
    )
    needle_strategy = (
        None
        if not args.needle_qk_aggregation.strip()
        else qk_aggregation_strategy_from_name(args.needle_qk_aggregation)
    )
    result = run_dynamic_block_benchmark(
        model_names=models,
        prompt_cases=default_dynamic_prompt_cases(prompt_names),
        block_modes=block_modes,
        representation_source=args.representation_source,
        qk_aggregation_strategy=qk_aggregation_strategy_from_name(args.qk_aggregation),
        needle_qk_aggregation_strategy=needle_strategy,
        suppression_modes=suppression_modes,
        suppression_threshold=args.suppression_threshold,
        coarse_top_k=args.coarse_top_k,
        load_config_kwargs={
            "device": args.device,
            "torch_dtype": args.torch_dtype,
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
            rail_setting=(
                f"dynamic_blocks_recent{args.keep_recent_blocks}_"
                f"anchor{args.keep_anchor_blocks}"
            ),
        ),
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_dynamic_block_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )
    print(format_dynamic_block_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
