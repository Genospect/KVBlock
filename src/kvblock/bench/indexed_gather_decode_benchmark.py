"""Synthetic indexed gather + compact decode attention benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from math import ceil, sqrt
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence
import warnings

import torch

from kvblock.plans import SelectedKVPlan

SelectionMode = str
VALID_SELECTION_MODES: tuple[SelectionMode, ...] = (
    "random_blocks",
    "contiguous_blocks",
    "recent_blocks",
)


@dataclass(frozen=True, slots=True)
class IndexedGatherBenchmarkConfig:
    """CLI-resolved benchmark configuration."""

    batch_size: int = 1
    num_heads: int = 16
    head_dim: int = 128
    total_tokens: tuple[int, ...] = (8192, 32768, 65536)
    block_sizes: tuple[int, ...] = (16, 32, 64)
    selected_fractions: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)
    selection_modes: tuple[str, ...] = VALID_SELECTION_MODES
    dtype: str = "float16"
    device: str = "cuda"
    iters: int = 50
    warmup: int = 10
    seed: int = 0
    selected_plan_json: str | None = None


@dataclass(frozen=True, slots=True)
class IndexedGatherBenchmarkRow:
    """One dense-vs-sparse indexed gather benchmark result."""

    total_tokens: int
    block_size: int
    selection_mode: str
    selected_fraction: float
    selected_blocks: int
    selected_tokens: int
    dense_attention_ms: float
    sparse_gather_ms: float
    sparse_attention_ms: float
    sparse_total_ms: float
    speedup_vs_dense: float
    theoretical_kv_read_reduction: float
    output_l2_diff: float
    max_abs_diff: float
    device: str
    dtype: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-friendly row."""

        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the indexed gather decode benchmark."""

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark dense decode attention against naive indexed sparse "
            "K/V gather plus compact attention."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--total-tokens", default="8192,32768,65536")
    parser.add_argument("--block-sizes", default="16,32,64")
    parser.add_argument("--selected-fractions", default="0.05,0.10,0.20,0.30")
    parser.add_argument(
        "--selection-modes",
        default="random_blocks,contiguous_blocks,recent_blocks",
    )
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32", "bfloat16"))
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to cuda when available, otherwise cpu.",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selected-plan-json",
        default=None,
        help="Optional SelectedKVPlan JSON file. Uses its logical_block_ids.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/indexed_gather_benchmark",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> IndexedGatherBenchmarkConfig:
    """Resolve and validate CLI arguments."""

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; falling back to CPU.")
        device = "cpu"
    if device == "cpu":
        warnings.warn(
            "Running indexed gather decode benchmark on CPU. Timings are for "
            "functional dry-run only, not GPU viability.",
        )
    total_tokens = _parse_positive_ints(args.total_tokens, name="total-tokens")
    block_sizes = _parse_positive_ints(args.block_sizes, name="block-sizes")
    selected_fractions = _parse_fractions(
        args.selected_fractions,
        name="selected-fractions",
    )
    selection_modes = _parse_selection_modes(args.selection_modes)
    if args.selected_plan_json is not None:
        selection_modes = ("selected_plan_json",)
        selected_fractions = (0.0,)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be > 0")
    if args.num_heads <= 0:
        raise ValueError("num-heads must be > 0")
    if args.head_dim <= 0:
        raise ValueError("head-dim must be > 0")
    if args.iters <= 0:
        raise ValueError("iters must be > 0")
    if args.warmup < 0:
        raise ValueError("warmup must be >= 0")
    return IndexedGatherBenchmarkConfig(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        total_tokens=total_tokens,
        block_sizes=block_sizes,
        selected_fractions=selected_fractions,
        selection_modes=selection_modes,
        dtype=args.dtype,
        device=device,
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        selected_plan_json=args.selected_plan_json,
    )


def run_benchmark(config: IndexedGatherBenchmarkConfig) -> tuple[IndexedGatherBenchmarkRow, ...]:
    """Run the indexed gather decode benchmark."""

    device = torch.device(config.device)
    dtype = _torch_dtype(config.dtype)
    selected_plan_ids = (
        None
        if config.selected_plan_json is None
        else _load_selected_plan_block_ids(Path(config.selected_plan_json))
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)

    rows: list[IndexedGatherBenchmarkRow] = []
    for total_tokens in config.total_tokens:
        query, key_cache, value_cache = _synthetic_tensors(
            batch_size=config.batch_size,
            total_tokens=total_tokens,
            num_heads=config.num_heads,
            head_dim=config.head_dim,
            dtype=dtype,
            device=device,
            generator=generator,
        )
        dense_output, dense_ms = _time_callable(
            lambda: decode_attention(query, key_cache, value_cache),
            device=device,
            warmup=config.warmup,
            iters=config.iters,
        )
        for block_size in config.block_sizes:
            total_blocks = ceil(total_tokens / block_size)
            for selection_mode in config.selection_modes:
                for requested_fraction in config.selected_fractions:
                    block_ids = select_block_ids(
                        total_blocks=total_blocks,
                        selected_fraction=requested_fraction,
                        selection_mode=selection_mode,
                        generator=generator,
                        selected_plan_block_ids=selected_plan_ids,
                    )
                    selected_tokens = token_count_for_blocks(
                        block_ids,
                        total_tokens=total_tokens,
                        block_size=block_size,
                    )

                    compact_key, compact_value = gather_selected_kv_blocks(
                        key_cache,
                        value_cache,
                        block_ids,
                        total_tokens=total_tokens,
                        block_size=block_size,
                    )
                    _, gather_ms = _time_callable(
                        lambda: gather_selected_kv_blocks(
                            key_cache,
                            value_cache,
                            block_ids,
                            total_tokens=total_tokens,
                            block_size=block_size,
                        ),
                        device=device,
                        warmup=config.warmup,
                        iters=config.iters,
                    )
                    sparse_output, sparse_attention_ms = _time_callable(
                        lambda: decode_attention(query, compact_key, compact_value),
                        device=device,
                        warmup=config.warmup,
                        iters=config.iters,
                    )
                    sparse_total_ms = gather_ms + sparse_attention_ms
                    output_l2_diff, max_abs_diff = output_diffs(
                        dense_output,
                        sparse_output,
                    )
                    if selected_tokens == total_tokens and not torch.allclose(
                        dense_output.float(),
                        sparse_output.float(),
                        rtol=3e-2,
                        atol=3e-2,
                    ):
                        raise RuntimeError(
                            "dense and sparse outputs diverged when all tokens "
                            "were selected"
                        )

                    actual_fraction = (
                        0.0 if total_tokens == 0 else selected_tokens / total_tokens
                    )
                    rows.append(
                        IndexedGatherBenchmarkRow(
                            total_tokens=total_tokens,
                            block_size=block_size,
                            selection_mode=selection_mode,
                            selected_fraction=actual_fraction,
                            selected_blocks=len(block_ids),
                            selected_tokens=selected_tokens,
                            dense_attention_ms=dense_ms,
                            sparse_gather_ms=gather_ms,
                            sparse_attention_ms=sparse_attention_ms,
                            sparse_total_ms=sparse_total_ms,
                            speedup_vs_dense=(
                                0.0
                                if sparse_total_ms <= 0.0
                                else dense_ms / sparse_total_ms
                            ),
                            theoretical_kv_read_reduction=1.0 - actual_fraction,
                            output_l2_diff=output_l2_diff,
                            max_abs_diff=max_abs_diff,
                            device=str(device),
                            dtype=config.dtype,
                        )
                    )
        del query, key_cache, value_cache, dense_output
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return tuple(rows)


def decode_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """Run one-token decode attention over K/V tensors shaped [B, T, H, D]."""

    if query.ndim != 3:
        raise ValueError("query must have shape [batch, heads, head_dim]")
    if key_cache.shape != value_cache.shape or key_cache.ndim != 4:
        raise ValueError("key_cache and value_cache must both have shape [B, T, H, D]")
    if query.shape[0] != key_cache.shape[0]:
        raise ValueError("query/cache batch size mismatch")
    if query.shape[1] != key_cache.shape[2]:
        raise ValueError("query/cache head count mismatch")
    if query.shape[2] != key_cache.shape[3]:
        raise ValueError("query/cache head dimension mismatch")
    if key_cache.shape[1] <= 0:
        raise ValueError("key_cache must contain at least one token")

    scale = 1.0 / sqrt(float(query.shape[-1]))
    if query.device.type == "cpu" and query.dtype in {torch.float16, torch.bfloat16}:
        query = query.float()
        key_cache = key_cache.float()
        value_cache = value_cache.float()
    scores = torch.einsum("bhd,bthd->bht", query, key_cache) * scale
    probs = torch.softmax(scores.float(), dim=-1).to(dtype=value_cache.dtype)
    return torch.einsum("bht,bthd->bhd", probs, value_cache)


def gather_selected_kv_blocks(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_ids: Sequence[int],
    *,
    total_tokens: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather selected logical K/V blocks into compact contiguous tensors."""

    ranges = block_id_runs_to_token_ranges(
        block_ids,
        total_tokens=total_tokens,
        block_size=block_size,
    )
    if not ranges:
        raise ValueError("at least one block must be selected")
    key_chunks = [key_cache[:, start:end, :, :] for start, end in ranges]
    value_chunks = [value_cache[:, start:end, :, :] for start, end in ranges]
    if len(key_chunks) == 1:
        return key_chunks[0].contiguous(), value_chunks[0].contiguous()
    return (
        torch.cat(key_chunks, dim=1).contiguous(),
        torch.cat(value_chunks, dim=1).contiguous(),
    )


def select_block_ids(
    *,
    total_blocks: int,
    selected_fraction: float,
    selection_mode: str,
    generator: torch.Generator,
    selected_plan_block_ids: Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Select logical block ids for one synthetic case."""

    if total_blocks <= 0:
        raise ValueError("total_blocks must be > 0")
    if selected_plan_block_ids is not None:
        ids = tuple(dict.fromkeys(int(block_id) for block_id in selected_plan_block_ids))
        if not ids:
            raise ValueError("selected plan contains no logical_block_ids")
        if min(ids) < 0 or max(ids) >= total_blocks:
            raise ValueError("selected plan block ids do not fit this total/block size")
        return tuple(sorted(ids))
    if selection_mode not in VALID_SELECTION_MODES:
        raise ValueError(f"unknown selection mode: {selection_mode!r}")
    selected_blocks = _selected_block_count(total_blocks, selected_fraction)
    if selected_blocks >= total_blocks:
        return tuple(range(total_blocks))
    if selection_mode == "recent_blocks":
        return tuple(range(total_blocks - selected_blocks, total_blocks))
    if selection_mode == "contiguous_blocks":
        start = _randint(generator, 0, total_blocks - selected_blocks)
        return tuple(range(start, start + selected_blocks))
    indices = torch.randperm(total_blocks, generator=generator)[:selected_blocks]
    return tuple(sorted(int(index) for index in indices.tolist()))


def block_id_runs_to_token_ranges(
    block_ids: Sequence[int],
    *,
    total_tokens: int,
    block_size: int,
) -> tuple[tuple[int, int], ...]:
    """Convert selected block ids into contiguous token ranges."""

    if total_tokens <= 0:
        raise ValueError("total_tokens must be > 0")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    total_blocks = ceil(total_tokens / block_size)
    ids = tuple(sorted(dict.fromkeys(int(block_id) for block_id in block_ids)))
    if not ids:
        return ()
    if ids[0] < 0 or ids[-1] >= total_blocks:
        raise ValueError("block id out of range")
    runs: list[tuple[int, int]] = []
    run_start = ids[0]
    previous = ids[0]
    for block_id in ids[1:]:
        if block_id == previous + 1:
            previous = block_id
            continue
        runs.append(_run_to_token_range(run_start, previous, total_tokens, block_size))
        run_start = previous = block_id
    runs.append(_run_to_token_range(run_start, previous, total_tokens, block_size))
    return tuple(runs)


def token_count_for_blocks(
    block_ids: Sequence[int],
    *,
    total_tokens: int,
    block_size: int,
) -> int:
    """Return exact token coverage for selected block ids."""

    return sum(
        end - start
        for start, end in block_id_runs_to_token_ranges(
            block_ids,
            total_tokens=total_tokens,
            block_size=block_size,
        )
    )


def output_diffs(
    dense_output: torch.Tensor,
    sparse_output: torch.Tensor,
) -> tuple[float, float]:
    """Return L2 and max absolute differences between dense and sparse outputs."""

    diff = dense_output.detach().float() - sparse_output.detach().float()
    return float(torch.linalg.vector_norm(diff).item()), float(diff.abs().max().item())


def write_outputs(
    rows: Sequence[IndexedGatherBenchmarkRow],
    *,
    config: IndexedGatherBenchmarkConfig,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Write metrics JSON and Markdown report."""

    run_dir = out_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "indexed_gather_decode",
        "note": _benchmark_note(),
        "config": asdict(config),
        "rows": [row.to_dict() for row in rows],
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    report_path.write_text(format_report(rows, config=config))
    return metrics_path, report_path


def format_report(
    rows: Sequence[IndexedGatherBenchmarkRow],
    *,
    config: IndexedGatherBenchmarkConfig,
) -> str:
    """Format a compact Markdown report."""

    lines = [
        "# Indexed Gather Decode Benchmark",
        "",
        _benchmark_note(),
        "",
        "This benchmark compares dense one-token decode attention against a naive "
        "sparse path that copies selected K/V blocks into compact tensors before "
        "running ordinary dense attention over the compacted cache.",
        "",
        "## Config",
        "",
        f"- batch size: {config.batch_size}",
        f"- heads: {config.num_heads}",
        f"- head dim: {config.head_dim}",
        f"- dtype: `{config.dtype}`",
        f"- device: `{config.device}`",
        f"- warmup/iters: {config.warmup}/{config.iters}",
        "",
        "## Results",
        "",
        "| tokens | block | mode | selected | dense ms | gather ms | sparse attn ms | sparse total ms | speedup | read reduction | L2 diff | max diff |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.total_tokens} | {row.block_size} | {row.selection_mode} "
            f"| {row.selected_fraction:.3f} | {row.dense_attention_ms:.4f} "
            f"| {row.sparse_gather_ms:.4f} | {row.sparse_attention_ms:.4f} "
            f"| {row.sparse_total_ms:.4f} | {row.speedup_vs_dense:.3f} "
            f"| {row.theoretical_kv_read_reduction:.3f} "
            f"| {row.output_l2_diff:.6f} | {row.max_abs_diff:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark from CLI-style arguments."""

    args = build_parser().parse_args(list(argv) if argv is not None else None)
    config = config_from_args(args)
    rows = run_benchmark(config)
    metrics_path, report_path = write_outputs(
        rows,
        config=config,
        out_dir=Path(args.out_dir),
    )
    print(f"Wrote {metrics_path}")
    print(f"Wrote {report_path}")
    print(json.dumps({"rows": [row.to_dict() for row in rows]}, indent=2, sort_keys=True))
    return 0


def _synthetic_tensors(
    *,
    batch_size: int,
    total_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cpu_query = torch.randn(batch_size, num_heads, head_dim, generator=generator)
    cpu_key = torch.randn(batch_size, total_tokens, num_heads, head_dim, generator=generator)
    cpu_value = torch.randn(batch_size, total_tokens, num_heads, head_dim, generator=generator)
    return (
        cpu_query.to(device=device, dtype=dtype),
        cpu_key.to(device=device, dtype=dtype),
        cpu_value.to(device=device, dtype=dtype),
    )


def _time_callable(
    fn: Callable[[], torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    warmup: int,
    iters: int,
) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], float]:
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None
    with torch.no_grad():
        for _ in range(warmup):
            result = fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iters):
                result = fn()
            end.record()
            torch.cuda.synchronize(device)
            return _require_result(result), float(start.elapsed_time(end) / iters)
        started_at = perf_counter()
        for _ in range(iters):
            result = fn()
        elapsed_ms = (perf_counter() - started_at) * 1000.0 / iters
        return _require_result(result), elapsed_ms


def _require_result(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if result is None:
        raise RuntimeError("timed callable did not run")
    return result


def _run_to_token_range(
    run_start_block: int,
    run_end_block: int,
    total_tokens: int,
    block_size: int,
) -> tuple[int, int]:
    start = run_start_block * block_size
    end = min(total_tokens, (run_end_block + 1) * block_size)
    return start, end


def _selected_block_count(total_blocks: int, selected_fraction: float) -> int:
    if not 0.0 < selected_fraction <= 1.0:
        raise ValueError("selected_fraction must be in (0, 1]")
    return min(total_blocks, max(1, ceil(total_blocks * selected_fraction)))


def _randint(generator: torch.Generator, low: int, high_inclusive: int) -> int:
    if high_inclusive < low:
        raise ValueError("invalid randint range")
    value = torch.randint(low, high_inclusive + 1, (1,), generator=generator)
    return int(value.item())


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name!r}")


def _parse_positive_ints(value: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _parse_fractions(value: str, *, name: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(item <= 0.0 or item > 1.0 for item in values):
        raise ValueError(f"{name} must contain values in (0, 1]")
    return values


def _parse_selection_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip() for part in value.split(",") if part.strip())
    if not modes:
        raise ValueError("selection-modes must not be empty")
    unknown = [mode for mode in modes if mode not in VALID_SELECTION_MODES]
    if unknown:
        raise ValueError(f"unknown selection mode(s): {unknown!r}")
    return modes


def _load_selected_plan_block_ids(path: Path) -> tuple[int, ...]:
    data = json.loads(path.read_text())
    plan = SelectedKVPlan.from_dict(data)
    if not plan.logical_block_ids:
        raise ValueError("SelectedKVPlan JSON has no logical_block_ids")
    return tuple(plan.logical_block_ids)


def _benchmark_note() -> str:
    return (
        "Naive benchmark only: this measures PyTorch indexed gather plus compact "
        "dense attention, not an optimized block-sparse kernel or serving-engine "
        "speedup."
    )
