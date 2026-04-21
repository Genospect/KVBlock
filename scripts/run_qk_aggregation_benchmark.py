#!/usr/bin/env python3
"""Run a focused query/key aggregation benchmark over prompt files."""

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
        description="Compare constrained query/key aggregation strategies.",
    )
    parser.add_argument(
        "--models",
        default="distilgpt2,gpt2",
        help="Comma-separated HF model names or local paths.",
    )
    parser.add_argument(
        "--qk-aggregations",
        default="mean_pool,max_pool,norm_weighted_mean,top_token_mean,block_max",
        help="Comma-separated aggregation strategies to compare.",
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
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--keep-recent-blocks", type=int, default=1)
    parser.add_argument("--keep-anchor-blocks", type=int, default=0)
    parser.add_argument("--top-token-count", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/aggregation",
        help="Directory for JSON and text benchmark reports.",
    )
    parser.add_argument(
        "--name",
        default="qk_aggregation_benchmark",
        help="Output file stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.qk_aggregation_benchmark import (
        format_qk_aggregation_report,
        run_qk_aggregation_benchmark,
        write_qk_aggregation_benchmark_outputs,
    )
    from kvblock.kv.qk_aggregation import qk_aggregation_strategies_from_names
    from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

    args = build_parser().parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    strategies = qk_aggregation_strategies_from_names(
        tuple(item.strip() for item in args.qk_aggregations.split(",") if item.strip())
    )
    result = run_qk_aggregation_benchmark(
        model_names=models,
        qk_aggregation_strategies=strategies,
        representation_source=args.representation_source,
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
            preview_chars=160,
            include_block_text=True,
            representation_source=args.representation_source,
            rail_setting=(
                f"qk_aggregation_recent{args.keep_recent_blocks}_"
                f"anchor{args.keep_anchor_blocks}"
            ),
        ),
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_qk_aggregation_benchmark_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )
    print(format_qk_aggregation_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
