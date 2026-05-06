"""Top-level KVBlock CLI shell."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from kvblock.policies import get_policy_preset, list_policy_presets


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parser = argparse.ArgumentParser(
        prog="kvblock",
        description="Policy-driven KV cache block selection tools.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_policy = subparsers.add_parser(
        "inspect-policy",
        help="Print a resolved policy preset.",
    )
    inspect_policy.add_argument(
        "policy_name",
        choices=list_policy_presets(),
        help="Policy preset to inspect.",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run a benchmark harness. This CLI path is intentionally minimal.",
    )
    benchmark.add_argument("--policy", default="quality_guarded_static")
    benchmark.add_argument("--out-dir", default="results/package_bench")

    trace = subparsers.add_parser(
        "trace",
        help="Trace selector decisions. Full tracing remains in the research CLI.",
    )
    trace.add_argument("--policy", default="quality_guarded_static")

    sparse_poc = subparsers.add_parser(
        "sparse-poc",
        help="Run the synthetic dense-vs-sparse correctness proof of concept.",
    )
    sparse_poc.add_argument("--out-dir", default="results/sparse_correctness")

    return parser


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Run the top-level CLI."""

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "inspect-policy":
        policy = get_policy_preset(args.policy_name).resolve()
        print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "benchmark":
        print(
            "Benchmark CLI shell is available. Use existing research scripts or "
            "the package benchmark modules as they are promoted."
        )
        print(json.dumps({"policy": args.policy, "out_dir": args.out_dir}, sort_keys=True))
        return 0
    if args.command == "trace":
        print(
            "Trace CLI shell is available. Existing detailed traces live under "
            "kvblock.selector and the research scripts."
        )
        print(json.dumps({"policy": args.policy}, sort_keys=True))
        return 0
    if args.command == "sparse-poc":
        from kvblock.bench.sparse_correctness_poc import main as sparse_poc_main

        return sparse_poc_main(["--out-dir", args.out_dir])
    parser.error(f"unsupported command: {args.command}")
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Console script entrypoint."""

    return run_cli(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
