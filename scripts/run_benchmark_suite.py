#!/usr/bin/env python3
"""Run V1 selector benchmark suite presets."""

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

    from kvblock.cli.suite import (
        format_suite_console_summary,
        run_benchmark_suite_cli,
    )

    cli_result = run_benchmark_suite_cli(argv)
    print(format_suite_console_summary(cli_result.result))
    paths = cli_result.output_paths
    if paths.run_jsonl is not None:
        print(f"per-run rows written to {paths.run_jsonl}")
    if paths.aggregate_jsonl is not None:
        print(f"aggregate rows written to {paths.aggregate_jsonl}")
    if paths.cases_jsonl is not None:
        print(f"case definitions written to {paths.cases_jsonl}")
    if paths.summary_json is not None:
        print(f"suite summary written to {paths.summary_json}")
    if paths.run_csv is not None:
        print(f"per-run CSV written to {paths.run_csv}")
    if paths.aggregate_csv is not None:
        print(f"aggregate CSV written to {paths.aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
