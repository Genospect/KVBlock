"""Synthetic workload presets for the V1 selector benchmark suite."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from kvblock.benchmark.selector_microbench import PopulationProfile

WorkloadName = Literal[
    "default",
    "low_confidence",
    "rail_dominated",
    "long_context_reasoning_synthetic",
    "code_reference_synthetic",
    "repeated_reference_synthetic",
    "adversarial_semantic_synthetic",
]


@dataclass(frozen=True, slots=True)
class BenchmarkWorkload:
    """Workload descriptor used by benchmark case builders."""

    name: WorkloadName
    population_profile: PopulationProfile
    description: str
    implementation_status: str = "implemented"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly workload record."""

        return asdict(self)


_WORKLOADS: dict[str, BenchmarkWorkload] = {
    "default": BenchmarkWorkload(
        name="default",
        population_profile="default",
        description="Generic synthetic selector workload with query summaries near rotating target blocks.",
    ),
    "low_confidence": BenchmarkWorkload(
        name="low_confidence",
        population_profile="low_confidence",
        description="Clustered summaries that stress confidence margins and fallback behavior.",
    ),
    "rail_dominated": BenchmarkWorkload(
        name="rail_dominated",
        population_profile="rail_dominated",
        description="Synthetic recency-heavy workload that stresses recent and anchor rails.",
    ),
    "long_context_reasoning_synthetic": BenchmarkWorkload(
        name="long_context_reasoning_synthetic",
        population_profile="default",
        description="Placeholder for future long-context reasoning synthetic traces.",
        implementation_status="placeholder_maps_to_default",
    ),
    "code_reference_synthetic": BenchmarkWorkload(
        name="code_reference_synthetic",
        population_profile="default",
        description="Placeholder for future code-reference synthetic traces.",
        implementation_status="placeholder_maps_to_default",
    ),
    "repeated_reference_synthetic": BenchmarkWorkload(
        name="repeated_reference_synthetic",
        population_profile="rail_dominated",
        description="Placeholder for future repeated-reference synthetic traces.",
        implementation_status="placeholder_maps_to_rail_dominated",
    ),
    "adversarial_semantic_synthetic": BenchmarkWorkload(
        name="adversarial_semantic_synthetic",
        population_profile="low_confidence",
        description="Placeholder for future adversarial semantic synthetic traces.",
        implementation_status="placeholder_maps_to_low_confidence",
    ),
}


def resolve_workload(name: str) -> BenchmarkWorkload:
    """Resolve a workload preset name."""

    try:
        return _WORKLOADS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark workload: {name}") from exc


def available_workloads() -> tuple[BenchmarkWorkload, ...]:
    """Return all known workload descriptors in stable order."""

    return tuple(_WORKLOADS[name] for name in _WORKLOADS)


def workload_names() -> tuple[str, ...]:
    """Return all known workload names in stable order."""

    return tuple(_WORKLOADS.keys())
