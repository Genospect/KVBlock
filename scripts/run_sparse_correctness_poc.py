"""Synthetic dense-vs-sparse correctness proof of concept script."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kvblock.bench.sparse_correctness_poc import main as _main


def main(argv: Sequence[str] | None = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
