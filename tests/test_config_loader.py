import json
from pathlib import Path

from kvblock.config.loader import load_config, load_config_dict


def test_load_config_dict_builds_typed_models() -> None:
    config = load_config_dict(
        {
            "model": {"model_name": "tiny", "payload_precision": "fp8"},
            "selector": {"keep_recent_blocks": 6},
            "benchmark": {"dense_refresh_interval": 16},
            "runtime": {"backend": "mock", "device": "cpu"},
        }
    )

    assert config.model.model_name == "tiny"
    assert config.model.payload_precision == "fp8"
    assert config.selector.keep_recent_blocks == 6
    assert config.benchmark.dense_refresh_interval == 16
    assert config.runtime.backend == "mock"


def test_load_config_json(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "model": {"model_name": "json-model"},
                "runtime": {"backend": "vllm", "device": "cuda"},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.model_name == "json-model"
    assert config.runtime.backend == "vllm"


def test_load_config_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[model]
model_name = "toml-model"

[selector]
final_top_k = 12
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.model_name == "toml-model"
    assert config.selector.final_top_k == 12


def test_load_config_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
model:
  model_name: yaml-model
benchmark:
  output_dir: artifacts/custom
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model.model_name == "yaml-model"
    assert config.benchmark.output_dir == "artifacts/custom"


def test_load_config_rejects_unknown_keys() -> None:
    try:
        load_config_dict({"selector": {"unknown": 1}})
    except ValueError as exc:
        assert "Unknown fields" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown selector field")
