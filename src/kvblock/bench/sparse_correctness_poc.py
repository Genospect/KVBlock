"""Synthetic dense-vs-sparse correctness proof of concept."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

import torch

from kvblock.backends import (
    TorchDenseAttentionBackend,
    TorchSparseReferenceBackend,
)
from kvblock.blocks import BlockLayout
from kvblock.metrics import (
    kl_divergence_logits,
    max_abs_logit_diff,
    topk_agreement,
    topk_overlap,
)
from kvblock.policies import KVBlockPolicy
from kvblock.selectors.simple import RecentOnlySelector


def build_parser() -> argparse.ArgumentParser:
    """Build the sparse correctness POC parser."""

    parser = argparse.ArgumentParser(
        description="Run a synthetic dense-vs-sparse KV selection correctness POC.",
    )
    parser.add_argument("--total-tokens", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--keep-recent-blocks", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default="results/sparse_correctness")
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    """Run the synthetic correctness comparison and return metrics."""

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    layout = BlockLayout.from_token_count(
        total_tokens=args.total_tokens,
        block_size=args.block_size,
    )
    policy = KVBlockPolicy(
        name="synthetic_recent_only",
        block_size=args.block_size,
        selector="recent_only",
        keep_recent_blocks=args.keep_recent_blocks,
        halo=0,
    )
    plan = RecentOnlySelector().select(None, None, layout, policy)

    query = torch.randn(args.num_heads, args.head_dim, device=device)
    key_cache = torch.randn(
        args.total_tokens,
        args.num_heads,
        args.head_dim,
        device=device,
    )
    value_cache = torch.randn_like(key_cache)
    projection = torch.randn(
        args.num_heads * args.head_dim,
        args.vocab_size,
        device=device,
    )

    dense_output = TorchDenseAttentionBackend().run_decode(
        query,
        key_cache,
        value_cache,
        plan,
        layout,
    )
    sparse_output = TorchSparseReferenceBackend().run_decode(
        query,
        key_cache,
        value_cache,
        plan,
        layout,
    )
    dense_logits = dense_output.reshape(1, -1).matmul(projection)
    sparse_logits = sparse_output.reshape(1, -1).matmul(projection)

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "device": str(device),
        "total_tokens": args.total_tokens,
        "block_size": args.block_size,
        "total_blocks": layout.block_count,
        "selected_blocks": plan.selected_blocks,
        "selected_tokens": plan.selected_tokens,
        "selected_block_fraction": plan.selected_block_fraction,
        "selected_token_fraction": plan.selected_token_fraction,
        "selector_name": plan.selector_name,
        "policy_name": plan.policy_name,
        "kl_divergence": kl_divergence_logits(dense_logits, sparse_logits),
        "top1_agreement": topk_agreement(dense_logits, sparse_logits, k=1),
        "top5_overlap": topk_overlap(dense_logits, sparse_logits, k=5),
        "max_abs_logit_diff": max_abs_logit_diff(dense_logits, sparse_logits),
        "note": (
            "torch_sparse_reference gathers selected tokens with PyTorch indexing. "
            "It is a correctness reference, not a physical sparse speedup proof."
        ),
    }


def write_report(metrics: dict[str, object], out_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown reports for a POC run."""

    run_dir = out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    md_path.write_text(_format_markdown(metrics))
    return json_path, md_path


def _format_markdown(metrics: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Sparse Correctness POC",
            "",
            f"- selector: `{metrics['selector_name']}`",
            f"- policy: `{metrics['policy_name']}`",
            f"- selected blocks: {metrics['selected_blocks']} / {metrics['total_blocks']}",
            f"- selected token fraction: {metrics['selected_token_fraction']:.4f}",
            f"- KL(dense || sparse): {metrics['kl_divergence']:.6f}",
            f"- top-1 agreement: {metrics['top1_agreement']:.4f}",
            f"- top-5 overlap: {metrics['top5_overlap']:.4f}",
            f"- max absolute logit diff: {metrics['max_abs_logit_diff']:.6f}",
            "",
            str(metrics["note"]),
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the POC from CLI-style arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    metrics = run(args)
    json_path, md_path = write_report(metrics, Path(args.out_dir))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0
