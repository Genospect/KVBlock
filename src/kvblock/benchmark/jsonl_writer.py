"""Minimal JSONL writer helpers for benchmark analysis rows."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_jsonl(path: str | Path, records: Sequence[object]) -> int:
    """Write one JSON object per line and return the number of written rows."""

    output_path = Path(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(_record_to_dict(record)))
            handle.write("\n")
    return len(records)


def _record_to_dict(record: object) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        payload = record.to_dict()
        if not isinstance(payload, dict):
            raise TypeError("record.to_dict() must return a dict")
        return payload
    if is_dataclass(record):
        payload = asdict(record)
        if not isinstance(payload, dict):
            raise TypeError("dataclass payload must serialize to a dict")
        return payload
    if isinstance(record, Mapping):
        return dict(record)
    raise TypeError("records must be mappings, dataclasses, or implement to_dict()")
