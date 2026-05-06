"""Experimental FlashInfer sparse decode POC skeleton."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kvblock.backends.flashinfer_backend import FlashInferBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check FlashInfer availability for sparse decode experiments.",
    )
    parser.add_argument("--out-dir", default="results/flashinfer_poc")
    parser.add_argument("--context-len", type=int, default=8192)
    parser.add_argument("--selected-fraction", type=float, default=0.2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = Path(args.out_dir) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"

    try:
        backend = FlashInferBackend()
    except ImportError as exc:
        metrics = {
            "status": "flashinfer_unavailable",
            "context_len": args.context_len,
            "selected_fraction": args.selected_fraction,
            "message": str(exc),
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        report_path.write_text(
            "# FlashInfer Sparse Decode POC\n\n"
            "FlashInfer is not available in this environment. No kernel timing was run.\n"
        )
        print(str(exc))
        print(f"Wrote {metrics_path}")
        return 0

    metrics = {
        "status": "flashinfer_available_backend_not_wired",
        "backend": backend.name,
        "context_len": args.context_len,
        "selected_fraction": args.selected_fraction,
        "message": (
            "FlashInfer imported successfully, but sparse decode API wiring is not "
            "implemented in this package skeleton yet."
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    report_path.write_text(
        "# FlashInfer Sparse Decode POC\n\n"
        "FlashInfer imported successfully. The package backend is still a guarded "
        "placeholder until API-specific sparse page execution is wired.\n"
    )
    print(f"Wrote {metrics_path}")
    print(metrics["message"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
