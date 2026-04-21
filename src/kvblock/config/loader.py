"""Minimal config loading helpers for JSON, TOML, and YAML."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Mapping

from kvblock.config.models import KVBlockConfig

try:
    import yaml
except ImportError:  # pragma: no cover - exercised when PyYAML is not installed
    yaml = None


def load_config_dict(data: Mapping[str, Any]) -> KVBlockConfig:
    """Build a typed config object from an in-memory mapping."""

    return KVBlockConfig.from_mapping(data)


def load_config(path: str | Path) -> KVBlockConfig:
    """Load a config file from JSON, TOML, or YAML into typed config models."""

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_mapping(config_path)
    return load_config_dict(raw)


def _load_mapping(path: Path) -> Mapping[str, Any]:
    suffix = path.suffix.lower()

    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))

    if suffix in {".toml", ".tml"}:
        return tomllib.loads(path.read_text(encoding="utf-8"))

    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "YAML config loading requires PyYAML to be installed. "
                "Install the optional 'yaml' extra to enable it."
            )
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return {} if loaded is None else loaded

    raise ValueError(
        "Unsupported config format. Expected one of: .json, .toml, .tml, .yaml, .yml"
    )
