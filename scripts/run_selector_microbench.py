#!/usr/bin/env python3
"""Run the synthetic selector microbenchmark harness."""

from __future__ import annotations

from pathlib import Path
import sys


def _ensure_repo_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: list[str] | None = None) -> int:
    _ensure_repo_src_on_path()

    from kvblock.cli.benchmark import format_console_summary, run_selector_microbench_cli

    result = run_selector_microbench_cli(argv)
    print(format_console_summary(result.aggregate_rows))
    if result.run_jsonl_path is not None:
        print(f"per-run rows written to {result.run_jsonl_path}")
    if result.aggregate_jsonl_path is not None:
        print(f"aggregate rows written to {result.aggregate_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
