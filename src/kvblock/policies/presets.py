"""Named policy presets for repeatable experiments."""

from __future__ import annotations

from dataclasses import replace

from kvblock.policies.base import KVBlockPolicy


POLICY_PRESETS: dict[str, KVBlockPolicy] = {
    "quality_guarded_static": KVBlockPolicy(
        name="quality_guarded_static",
        block_size=40,
        stride=8,
        selector="mixed_global_refine",
        representation_source="query_only_last_layer",
        qk_aggregation="block_max",
        shortlist_m=32,
        semantic_k=8,
        halo=2,
        keep_recent_blocks=4,
        keep_anchor_blocks=True,
        fallback_mode="confidence_guarded",
        fallback_margin=0.05,
        max_selected_fraction=0.20,
        metadata={
            "refinement": "dense_qk_token_refine",
            "selection_score": "softmax_mass",
            "mix": "semantic_refined_mix",
        },
    ),
    "efficiency_guarded_static": KVBlockPolicy(
        name="efficiency_guarded_static",
        block_size=32,
        selector="qk_blockmax",
        shortlist_m=24,
        semantic_k=8,
        halo=1,
        keep_recent_blocks=4,
        keep_anchor_blocks=True,
        fallback_mode="confidence_guarded",
        fallback_margin=0.05,
        max_selected_fraction=0.15,
    ),
    "debug_dense": KVBlockPolicy(
        name="debug_dense",
        selector="dense",
        semantic_k=0,
        halo=0,
        keep_recent_blocks=0,
        keep_anchor_blocks=False,
        fallback_mode="dense",
        max_selected_fraction=None,
    ),
    "debug_recent_only": KVBlockPolicy(
        name="debug_recent_only",
        selector="recent_only",
        semantic_k=0,
        halo=0,
        keep_recent_blocks=4,
        keep_anchor_blocks=False,
        fallback_mode="none",
        max_selected_fraction=None,
    ),
    "debug_random_sparse": KVBlockPolicy(
        name="debug_random_sparse",
        selector="random_sparse",
        semantic_k=8,
        halo=0,
        keep_recent_blocks=0,
        keep_anchor_blocks=False,
        fallback_mode="none",
        max_selected_fraction=0.20,
        metadata={"seed": 0},
    ),
}


def list_policy_presets() -> tuple[str, ...]:
    """Return available preset names."""

    return tuple(sorted(POLICY_PRESETS))


def get_policy_preset(name: str) -> KVBlockPolicy:
    """Return a copy of a named policy preset."""

    try:
        preset = POLICY_PRESETS[name]
    except KeyError as exc:
        known = ", ".join(list_policy_presets())
        raise KeyError(f"unknown policy preset {name!r}; known presets: {known}") from exc
    return replace(preset, metadata=dict(preset.metadata))
