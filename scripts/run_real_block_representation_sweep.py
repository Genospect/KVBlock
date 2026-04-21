#!/usr/bin/env python3
"""Run representation-source sweeps for local Hugging Face real-block metadata."""

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
        description="Run real-block representation-source sweeps over prompt files.",
    )
    parser.add_argument(
        "--models",
        default="distilgpt2,gpt2",
        help="Comma-separated HF model names or local paths.",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shortlist-m", type=int, default=16)
    parser.add_argument("--semantic-k", type=int, default=4)
    parser.add_argument("--confidence-margin", type=float, default=0.0)
    parser.add_argument("--hidden-layer-index", type=int, default=1)
    parser.add_argument(
        "--representation-sources",
        default="avg_mid4_hidden,key_mean_mid_layer,query_mean_last_layer",
        help=(
            "Comma-separated representation sources to compare, e.g. "
            "avg_mid4_hidden,key_mean_mid_layer,query_mean_last_layer."
        ),
    )
    parser.add_argument(
        "--head-scoring-modes",
        default="mean_heads",
        help=(
            "Comma-separated Stage-A head scoring modes: mean_heads,"
            "max_head_score,topk_head_mean,weighted_head_mean."
        ),
    )
    parser.add_argument("--head-top-k", type=int, default=2)
    parser.add_argument(
        "--head-weights",
        default="",
        help="Comma-separated static head weights for weighted_head_mean.",
    )
    parser.add_argument(
        "--include-head-diagnostics",
        action="store_true",
        help="Include per-run selected head contribution summaries in JSON/text output.",
    )
    parser.add_argument(
        "--top-heads",
        type=int,
        default=5,
        help="Number of top contributing heads to keep in diagnostics.",
    )
    parser.add_argument(
        "--rail-presets",
        default="default",
        help=(
            "Comma-separated rail presets: default,no_rails,recent_only,"
            "anchor_only,reduced. Ignored when explicit rail counts are set."
        ),
    )
    parser.add_argument(
        "--keep-recent-blocks",
        type=int,
        default=None,
        help="Override sweep rails with one custom recent-block count.",
    )
    parser.add_argument(
        "--keep-anchor-blocks",
        type=int,
        default=None,
        help="Override sweep rails with one custom anchor-block count.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--out-dir",
        default="results/representation",
        help="Directory for JSON and text sweep reports.",
    )
    parser.add_argument(
        "--name",
        default="deep_hf_representation_sweep",
        help="Output file stem.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.benchmark.real_block_representation_sweep import (
        RailSetting,
        format_representation_sweep_report,
        head_scoring_settings_from_names,
        rail_settings_from_presets,
        representation_sources_from_names,
        run_representation_sweep,
        write_representation_sweep_outputs,
    )
    from kvblock.runtime.real_block_eval import RealBlockSelectorConfig

    args = build_parser().parse_args(argv)
    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    representation_sources = representation_sources_from_names(
        tuple(item.strip() for item in args.representation_sources.split(",") if item.strip())
    )
    head_scoring_settings = head_scoring_settings_from_names(
        tuple(item.strip() for item in args.head_scoring_modes.split(",") if item.strip()),
        head_top_k=args.head_top_k,
        head_weights=tuple(
            float(item.strip())
            for item in args.head_weights.split(",")
            if item.strip()
        ),
    )
    if args.keep_recent_blocks is not None or args.keep_anchor_blocks is not None:
        rail_settings = (
            RailSetting(
                "custom",
                keep_recent_blocks=(
                    4 if args.keep_recent_blocks is None else args.keep_recent_blocks
                ),
                keep_anchor_blocks=(
                    2 if args.keep_anchor_blocks is None else args.keep_anchor_blocks
                ),
            ),
        )
    else:
        rail_settings = rail_settings_from_presets(
            tuple(item.strip() for item in args.rail_presets.split(",") if item.strip())
        )
    result = run_representation_sweep(
        model_names=models,
        representation_sources=representation_sources,
        hidden_layer_index=args.hidden_layer_index,
        rail_settings=rail_settings,
        head_scoring_settings=head_scoring_settings,
        load_config_kwargs={
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "local_files_only": args.local_files_only,
        },
        selector_config=RealBlockSelectorConfig(
            block_size=args.block_size,
            shortlist_m=args.shortlist_m,
            semantic_k=args.semantic_k,
            confidence_margin=args.confidence_margin,
            preview_chars=120,
            include_block_text=True,
        ),
        include_head_diagnostics=args.include_head_diagnostics,
        diagnostic_top_heads=args.top_heads,
    )

    out_dir = Path(args.out_dir)
    json_path = out_dir / f"{args.name}.json"
    text_path = out_dir / f"{args.name}.txt"
    write_representation_sweep_outputs(result, json_path=json_path, text_path=text_path)
    print(format_representation_sweep_report(result))
    print(f"JSON written to {json_path}")
    print(f"text report written to {text_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
