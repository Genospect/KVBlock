"""Block candidate generation for fixed and constrained multi-scale V1 sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, cast

BlockModeName = Literal[
    "fixed",
    "fixed_16",
    "fixed_24",
    "fixed_32",
    "fixed_40",
    "multiscale_16_32",
    "multiscale_16_24_32",
    "overlap_16_stride_8",
    "coarse_to_fine_40_16",
    "coarse_to_fine_32_16",
    "coarse_to_fine_40_16_keep_parent",
    "mixed_global_refine_40_16",
    "mixed_global_refine_40_16_stride_8",
]

VALID_BLOCK_MODES: tuple[BlockModeName, ...] = (
    "fixed",
    "fixed_16",
    "fixed_24",
    "fixed_32",
    "fixed_40",
    "multiscale_16_32",
    "multiscale_16_24_32",
    "overlap_16_stride_8",
    "coarse_to_fine_40_16",
    "coarse_to_fine_32_16",
    "coarse_to_fine_40_16_keep_parent",
    "mixed_global_refine_40_16",
    "mixed_global_refine_40_16_stride_8",
)


@dataclass(frozen=True, slots=True)
class BlockCandidate:
    """One candidate block span used for selector metadata construction."""

    block_id: int
    candidate_id: str
    block_mode: str
    block_size: int
    stride: int
    token_start: int
    token_len: int
    parent_block_id: int | None = None
    parent_candidate_id: str | None = None
    parent_token_start: int | None = None
    parent_token_len: int | None = None
    candidate_role: str = "block"

    def __post_init__(self) -> None:
        if self.block_id < 0:
            raise ValueError("block_id must be >= 0")
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.block_mode.strip():
            raise ValueError("block_mode must be non-empty")
        if self.block_size <= 0:
            raise ValueError("block_size must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if self.token_start < 0:
            raise ValueError("token_start must be >= 0")
        if self.token_len <= 0:
            raise ValueError("token_len must be > 0")
        if self.token_len > self.block_size:
            raise ValueError("token_len must be <= block_size")
        if self.candidate_role not in {"block", "parent", "child"}:
            raise ValueError("candidate_role must be block, parent, or child")
        parent_values = (
            self.parent_block_id,
            self.parent_candidate_id,
            self.parent_token_start,
            self.parent_token_len,
        )
        if any(value is not None for value in parent_values):
            if any(value is None for value in parent_values):
                raise ValueError("all parent fields must be set together")
            if self.parent_block_id is not None and self.parent_block_id < 0:
                raise ValueError("parent_block_id must be >= 0 when set")
            if self.parent_candidate_id is not None and not self.parent_candidate_id.strip():
                raise ValueError("parent_candidate_id must be non-empty when set")
            if self.parent_token_start is not None and self.parent_token_start < 0:
                raise ValueError("parent_token_start must be >= 0 when set")
            if self.parent_token_len is not None and self.parent_token_len <= 0:
                raise ValueError("parent_token_len must be > 0 when set")

    @property
    def token_end(self) -> int:
        """Exclusive token end for this candidate span."""

        return self.token_start + self.token_len

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-friendly candidate record."""

        payload: dict[str, int | str] = {
            "block_id": self.block_id,
            "candidate_id": self.candidate_id,
            "block_mode": self.block_mode,
            "block_size": self.block_size,
            "stride": self.stride,
            "token_start": self.token_start,
            "token_len": self.token_len,
            "token_end": self.token_end,
            "candidate_role": self.candidate_role,
        }
        if self.parent_block_id is not None:
            payload.update(
                {
                    "parent_block_id": self.parent_block_id,
                    "parent_candidate_id": self.parent_candidate_id or "",
                    "parent_token_start": self.parent_token_start or 0,
                    "parent_token_len": self.parent_token_len or 0,
                    "parent_token_end": (
                        (self.parent_token_start or 0)
                        + (self.parent_token_len or 0)
                    ),
                }
            )
        return payload


def block_mode_from_name(name: str) -> BlockModeName:
    """Validate and normalize one block mode name."""

    normalized = name.strip()
    if normalized not in VALID_BLOCK_MODES:
        valid = ", ".join(VALID_BLOCK_MODES)
        raise ValueError(f"unknown block mode {name!r}; valid: {valid}")
    return cast(BlockModeName, normalized)


def block_modes_from_names(names: Sequence[str]) -> tuple[BlockModeName, ...]:
    """Resolve a non-empty sequence of block mode names."""

    if not names:
        raise ValueError("block mode names must not be empty")
    modes = tuple(block_mode_from_name(name) for name in names if name.strip())
    if not modes:
        raise ValueError("block mode names must not be empty")
    return modes


def generate_block_candidates(
    *,
    token_count: int,
    mode: str = "fixed",
    default_block_size: int = 32,
    overlap_stride: int | None = None,
) -> tuple[BlockCandidate, ...]:
    """Generate deterministic block candidates for fixed or multi-scale sweeps."""

    if token_count <= 0:
        raise ValueError("token_count must be > 0")
    if default_block_size <= 0:
        raise ValueError("default_block_size must be > 0")
    resolved = block_mode_from_name(mode)
    specs = _mode_specs(
        resolved,
        default_block_size=default_block_size,
        overlap_stride=overlap_stride,
    )
    candidates: list[BlockCandidate] = []
    for block_size, stride in specs:
        for token_start in range(0, token_count, stride):
            token_len = min(block_size, token_count - token_start)
            if token_len <= 0:
                break
            candidates.append(
                BlockCandidate(
                    block_id=len(candidates),
                    candidate_id=_candidate_id(
                        block_size=block_size,
                        stride=stride,
                        token_start=token_start,
                        token_len=token_len,
                    ),
                    block_mode=resolved,
                    block_size=block_size,
                    stride=stride,
                    token_start=token_start,
                    token_len=token_len,
                )
            )
            if token_start + token_len >= token_count:
                break
    return tuple(candidates)


def is_coarse_to_fine_mode(mode: str) -> bool:
    """Return whether a block mode is a benchmark-only coarse-to-fine mode."""

    return mode in {
        "coarse_to_fine_40_16",
        "coarse_to_fine_32_16",
        "coarse_to_fine_40_16_keep_parent",
    }


def is_parent_retention_coarse_to_fine_mode(mode: str) -> bool:
    """Return whether a CTF mode keeps selected coarse parents in final ranking."""

    return mode == "coarse_to_fine_40_16_keep_parent"


def is_mixed_global_refine_mode(mode: str) -> bool:
    """Return whether a block mode keeps a global rail plus local refinement."""

    return mode in {
        "mixed_global_refine_40_16",
        "mixed_global_refine_40_16_stride_8",
    }


def coarse_to_fine_spec(mode: str) -> tuple[int, int]:
    """Return ``(coarse_block_size, fine_block_size)`` for a CTF mode."""

    resolved = block_mode_from_name(mode)
    if resolved == "coarse_to_fine_40_16":
        return (40, 16)
    if resolved == "coarse_to_fine_40_16_keep_parent":
        return (40, 16)
    if resolved == "coarse_to_fine_32_16":
        return (32, 16)
    raise ValueError(f"not a coarse-to-fine mode: {mode!r}")


def mixed_global_refine_spec(mode: str) -> tuple[int, int]:
    """Return ``(global_block_size, fine_block_size)`` for mixed-router modes."""

    resolved = block_mode_from_name(mode)
    if resolved == "mixed_global_refine_40_16":
        return (40, 16)
    if resolved == "mixed_global_refine_40_16_stride_8":
        return (40, 16)
    raise ValueError(f"not a mixed global-refine mode: {mode!r}")


def mixed_global_refine_child_stride(mode: str) -> int:
    """Return the local child stride for mixed global-refine modes."""

    resolved = block_mode_from_name(mode)
    if resolved == "mixed_global_refine_40_16":
        return 16
    if resolved == "mixed_global_refine_40_16_stride_8":
        return 8
    raise ValueError(f"not a mixed global-refine mode: {mode!r}")


def generate_child_block_candidates(
    *,
    token_count: int,
    parent_candidates: Sequence[BlockCandidate],
    fine_block_size: int,
    block_mode: str,
    fine_stride: int | None = None,
) -> tuple[BlockCandidate, ...]:
    """Generate deterministic fine child candidates inside parent spans.

    This is used by benchmark-only coarse-to-fine routing experiments. Child
    IDs preserve parent lineage so qualitative inspection can show which coarse
    region produced each final fine block.
    """

    if token_count <= 0:
        raise ValueError("token_count must be > 0")
    if fine_block_size <= 0:
        raise ValueError("fine_block_size must be > 0")
    stride = fine_block_size if fine_stride is None else fine_stride
    if stride <= 0:
        raise ValueError("fine_stride must be > 0 when set")
    if not block_mode.strip():
        raise ValueError("block_mode must be non-empty")

    children: list[BlockCandidate] = []
    sorted_parents = sorted(
        parent_candidates,
        key=lambda candidate: (
            candidate.token_start,
            candidate.token_end,
            candidate.block_id,
        ),
    )
    seen_spans: set[tuple[int, int]] = set()
    for parent in sorted_parents:
        parent_end = min(parent.token_end, token_count)
        if parent.token_start >= parent_end:
            continue
        for token_start in range(parent.token_start, parent_end, stride):
            token_end = min(token_start + fine_block_size, parent_end)
            token_len = token_end - token_start
            if token_len <= 0:
                break
            span_key = (token_start, token_end)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            children.append(
                BlockCandidate(
                    block_id=len(children),
                    candidate_id=_child_candidate_id(
                        parent_candidate_id=parent.candidate_id,
                        fine_block_size=fine_block_size,
                        fine_stride=stride,
                        token_start=token_start,
                        token_len=token_len,
                    ),
                    block_mode=block_mode,
                    block_size=fine_block_size,
                    stride=stride,
                    token_start=token_start,
                    token_len=token_len,
                    parent_block_id=parent.block_id,
                    parent_candidate_id=parent.candidate_id,
                    parent_token_start=parent.token_start,
                    parent_token_len=parent.token_len,
                    candidate_role="child",
                )
            )
            if token_end >= parent_end:
                break
    return tuple(children)


def retain_parent_and_child_candidates(
    *,
    parent_candidates: Sequence[BlockCandidate],
    child_candidates: Sequence[BlockCandidate],
    block_mode: str,
) -> tuple[BlockCandidate, ...]:
    """Return re-indexed parent+child candidates for parent-retention CTF.

    The retained parent spans stay in the final ranking pool alongside their
    fine children. IDs are re-issued deterministically because the final selector
    expects a single block-id namespace for one metadata population.
    """

    if not block_mode.strip():
        raise ValueError("block_mode must be non-empty")
    combined: list[BlockCandidate] = []
    for parent in sorted(
        parent_candidates,
        key=lambda candidate: (
            candidate.token_start,
            candidate.token_end,
            candidate.block_id,
        ),
    ):
        combined.append(
            BlockCandidate(
                block_id=len(combined),
                candidate_id=f"{parent.candidate_id}__parent",
                block_mode=block_mode,
                block_size=parent.block_size,
                stride=parent.stride,
                token_start=parent.token_start,
                token_len=parent.token_len,
                parent_block_id=parent.block_id,
                parent_candidate_id=parent.candidate_id,
                parent_token_start=parent.token_start,
                parent_token_len=parent.token_len,
                candidate_role="parent",
            )
        )
    for child in sorted(
        child_candidates,
        key=lambda candidate: (
            candidate.token_start,
            candidate.token_end,
            candidate.block_size,
            candidate.candidate_id,
        ),
    ):
        combined.append(
            BlockCandidate(
                block_id=len(combined),
                candidate_id=child.candidate_id,
                block_mode=block_mode,
                block_size=child.block_size,
                stride=child.stride,
                token_start=child.token_start,
                token_len=child.token_len,
                parent_block_id=child.parent_block_id,
                parent_candidate_id=child.parent_candidate_id,
                parent_token_start=child.parent_token_start,
                parent_token_len=child.parent_token_len,
                candidate_role="child",
            )
        )
    return tuple(combined)


def _mode_specs(
    mode: BlockModeName,
    *,
    default_block_size: int,
    overlap_stride: int | None,
) -> tuple[tuple[int, int], ...]:
    if mode == "fixed":
        return ((default_block_size, default_block_size),)
    if mode.startswith("fixed_"):
        size = int(mode.removeprefix("fixed_"))
        return ((size, size),)
    if mode == "multiscale_16_32":
        return ((16, 16), (32, 32))
    if mode == "multiscale_16_24_32":
        return ((16, 16), (24, 24), (32, 32))
    if mode == "overlap_16_stride_8":
        stride = 8 if overlap_stride is None else overlap_stride
        if stride <= 0:
            raise ValueError("overlap_stride must be > 0")
        if stride > 16:
            raise ValueError("overlap_stride must be <= 16 for overlap_16_stride_8")
        return ((16, stride),)
    if is_coarse_to_fine_mode(mode):
        # Standalone ingest uses the coarse phase. The benchmark harness builds
        # child candidates explicitly after ranking coarse regions.
        coarse_block_size, _fine_block_size = coarse_to_fine_spec(mode)
        return ((coarse_block_size, coarse_block_size),)
    if is_mixed_global_refine_mode(mode):
        # Standalone ingest uses the global rail. The benchmark harness adds
        # local child candidates after scoring the global fixed-size pass.
        global_block_size, _fine_block_size = mixed_global_refine_spec(mode)
        return ((global_block_size, global_block_size),)
    raise ValueError(f"unsupported block mode: {mode!r}")


def _candidate_id(
    *,
    block_size: int,
    stride: int,
    token_start: int,
    token_len: int,
) -> str:
    token_end = token_start + token_len
    return f"s{block_size}_stride{stride}_t{token_start}_{token_end}"


def _child_candidate_id(
    *,
    parent_candidate_id: str,
    fine_block_size: int,
    fine_stride: int,
    token_start: int,
    token_len: int,
) -> str:
    token_end = token_start + token_len
    return (
        f"{parent_candidate_id}__child_s{fine_block_size}_"
        f"stride{fine_stride}_t{token_start}_{token_end}"
    )
