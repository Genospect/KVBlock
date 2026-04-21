#!/usr/bin/env python3
"""Run constrained static head-weight validation for query_mean_last_layer."""

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
        description=(
            "Validate explicit static head-weight schemes against pooled "
            "query_mean_last_layer routing."
        ),
    )
    parser.add_argument(
        "--models",
        default="distilgpt2,gpt2",
        help="Comma-separated HF model names or local paths.",
    )
    parser.add_argument(
        "--schemes",
        default="pooled_mean_heads,head9_only,head9_heavy,retrieval_mix,code_mix",
        help="Comma-separated static schemes to run.",
    )
    parser.add_argument(
        "--representation-source",
        default="query_mean_last_layer",
        choices=(
            "query_mean_last_layer",
            "query_mean_mid_layer",
            "query_avg_last4",
        ),
        help="Query/key representation source to validate.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument(
        "--keep-recent-blocks",
        type=int,
        default=1,
        help="Recent rail count for validation; default keeps rails reduced.",
    )
    parser.add_argument(
        "--keep-anchor-blocks",
        type=int,
        default=0,
        help="Anchor rail count for validation; default keeps rails reduced.",
    )
    parser.add_argument(
        "--head-count",
        type=int,
        default=None,
        help="Optional explicit head count for static weight vectors.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/heads",
        help="Directory for JSON and text validation reports.",
    )
    parser.add_argument(
        "--name",
        default="static_head_weight_validation",
        help="Output file stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.static_head_weight_validation import (
        format_static_head_weight_report,
        run_static_head_weight_validation,
        schemes_from_names,
        write_static_head_weight_validation_outputs,
    )
    from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

    args = build_parser().parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    schemes = schemes_from_names(
        tuple(item.strip() for item in args.schemes.split(",") if item.strip())
    )
    result = run_static_head_weight_validation(
        model_names=models,
        schemes=schemes,
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
            preview_chars=160,
            include_block_text=True,
            representation_source=args.representation_source,
            rail_setting=(
                f"static_head_validation_recent{args.keep_recent_blocks}_"
                f"anchor{args.keep_anchor_blocks}"
            ),
        ),
        head_count=args.head_count,
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_static_head_weight_validation_outputs(
        result,
        json_path=json_path,
        text_path=text_path,
    )
    print(format_static_head_weight_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
