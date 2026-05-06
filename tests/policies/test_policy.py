import pytest

from kvblock.policies import KVBlockPolicy, get_policy_preset, list_policy_presets


def test_policy_resolve_fills_stride() -> None:
    policy = KVBlockPolicy(block_size=32)

    assert policy.resolve().stride == 32


def test_policy_preset_is_copy() -> None:
    first = get_policy_preset("quality_guarded_static")
    second = get_policy_preset("quality_guarded_static")
    first.metadata["changed"] = True

    assert "quality_guarded_static" in list_policy_presets()
    assert "changed" not in second.metadata


def test_policy_rejects_invalid_fraction() -> None:
    with pytest.raises(ValueError, match="max_selected_fraction"):
        KVBlockPolicy(max_selected_fraction=1.5)
