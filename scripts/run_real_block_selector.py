#!/usr/bin/env python3
"""Run the dense-only real-block selector bridge."""

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

    from kvblock.cli.real_block_selector import (
        format_real_block_selector_summary,
        run_real_block_selector_cli,
    )

    cli_result = run_real_block_selector_cli(argv)
    print(
        format_real_block_selector_summary(
            cli_result.result,
            show_selected_blocks=cli_result.show_selected_blocks,
            show_all_blocks=cli_result.show_all_blocks,
            show_stage_scores=cli_result.show_stage_scores,
            show_head_diagnostics=cli_result.show_head_diagnostics,
            show_query_key_inspection=cli_result.show_query_key_inspection,
            show_missed_blocks=cli_result.show_missed_blocks,
            show_top_unselected=cli_result.show_top_unselected,
        )
    )
    if cli_result.json_out_path is not None:
        print(f"inspection JSON written to {cli_result.json_out_path}")
    if cli_result.head_diagnostics_json_out_path is not None:
        print(
            "head diagnostics JSON written to "
            f"{cli_result.head_diagnostics_json_out_path}"
        )
    if cli_result.inspection_json_out_path is not None:
        print(f"query/key inspection JSON written to {cli_result.inspection_json_out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
