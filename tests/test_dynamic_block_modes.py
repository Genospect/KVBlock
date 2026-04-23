from __future__ import annotations

from kvblock.kv.block_modes import (
    block_modes_from_names,
    coarse_to_fine_spec,
    generate_block_candidates,
    generate_child_block_candidates,
    mixed_global_refine_spec,
    retain_parent_and_child_candidates,
)


def test_multiscale_block_candidates_are_stable_and_labeled() -> None:
    first = generate_block_candidates(
        token_count=40,
        mode="multiscale_16_24_32",
        default_block_size=16,
    )
    second = generate_block_candidates(
        token_count=40,
        mode="multiscale_16_24_32",
        default_block_size=16,
    )

    assert [candidate.to_dict() for candidate in first] == [
        candidate.to_dict() for candidate in second
    ]
    assert len({candidate.block_id for candidate in first}) == len(first)
    assert len({candidate.candidate_id for candidate in first}) == len(first)
    assert first[0].candidate_id == "s16_stride16_t0_16"
    assert any(candidate.candidate_id == "s24_stride24_t0_24" for candidate in first)
    assert any(candidate.candidate_id == "s32_stride32_t0_32" for candidate in first)


def test_overlap_block_candidates_use_configured_stride() -> None:
    candidates = generate_block_candidates(
        token_count=34,
        mode="overlap_16_stride_8",
        default_block_size=16,
        overlap_stride=8,
    )

    assert [candidate.token_start for candidate in candidates] == [0, 8, 16, 24]
    assert [candidate.token_len for candidate in candidates] == [16, 16, 16, 10]
    assert candidates[-1].candidate_id == "s16_stride8_t24_34"


def test_block_mode_parser_validates_names() -> None:
    assert block_modes_from_names(("fixed_16", "multiscale_16_32")) == (
        "fixed_16",
        "multiscale_16_32",
    )


def test_coarse_to_fine_mode_generates_stable_coarse_regions() -> None:
    candidates = generate_block_candidates(
        token_count=82,
        mode="coarse_to_fine_40_16",
        default_block_size=16,
    )

    assert coarse_to_fine_spec("coarse_to_fine_40_16") == (40, 16)
    assert [candidate.candidate_id for candidate in candidates] == [
        "s40_stride40_t0_40",
        "s40_stride40_t40_80",
        "s40_stride40_t80_82",
    ]


def test_mixed_global_refine_mode_generates_stable_global_regions() -> None:
    candidates = generate_block_candidates(
        token_count=82,
        mode="mixed_global_refine_40_16",
        default_block_size=16,
    )

    assert mixed_global_refine_spec("mixed_global_refine_40_16") == (40, 16)
    assert [candidate.candidate_id for candidate in candidates] == [
        "s40_stride40_t0_40",
        "s40_stride40_t40_80",
        "s40_stride40_t80_82",
    ]


def test_child_candidates_preserve_parent_lineage() -> None:
    parents = generate_block_candidates(
        token_count=82,
        mode="fixed_40",
        default_block_size=16,
    )[:2]

    children = generate_child_block_candidates(
        token_count=82,
        parent_candidates=parents,
        fine_block_size=16,
        block_mode="coarse_to_fine_40_16",
    )

    assert [child.candidate_id for child in children] == [
        "s40_stride40_t0_40__child_s16_stride16_t0_16",
        "s40_stride40_t0_40__child_s16_stride16_t16_32",
        "s40_stride40_t0_40__child_s16_stride16_t32_40",
        "s40_stride40_t40_80__child_s16_stride16_t40_56",
        "s40_stride40_t40_80__child_s16_stride16_t56_72",
        "s40_stride40_t40_80__child_s16_stride16_t72_80",
    ]
    assert children[0].parent_candidate_id == "s40_stride40_t0_40"
    assert children[0].parent_token_start == 0
    assert children[0].parent_token_len == 40
    assert children[0].candidate_role == "child"


def test_parent_retention_candidates_include_parent_and_children() -> None:
    parents = generate_block_candidates(
        token_count=82,
        mode="fixed_40",
        default_block_size=16,
    )[:1]
    children = generate_child_block_candidates(
        token_count=82,
        parent_candidates=parents,
        fine_block_size=16,
        block_mode="coarse_to_fine_40_16_keep_parent",
    )

    combined = retain_parent_and_child_candidates(
        parent_candidates=parents,
        child_candidates=children,
        block_mode="coarse_to_fine_40_16_keep_parent",
    )

    assert [candidate.block_id for candidate in combined] == list(range(len(combined)))
    assert combined[0].candidate_id == "s40_stride40_t0_40__parent"
    assert combined[0].candidate_role == "parent"
    assert combined[1].candidate_role == "child"
    assert combined[1].parent_candidate_id == "s40_stride40_t0_40"
