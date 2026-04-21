"""Minimal CSV writer helpers for benchmark analysis rows."""

from __future__ import annotations

import csv
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


def write_csv(
    path: str | Path,
    records: Sequence[object],
    *,
    field_names: Sequence[str] | None = None,
) -> int:
    """Write analysis records to CSV and return the number of written rows."""

    output_path = Path(path)
    rows = [_record_to_dict(record) for record in records]
    resolved_fields = tuple(field_names or _infer_field_names(records, rows))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in resolved_fields})
    return len(rows)


def _infer_field_names(
    records: Sequence[object], rows: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    if not records:
        return ()
    first = records[0]
    if hasattr(first, "field_names"):
        fields = first.field_names()
        if isinstance(fields, tuple):
            return fields
    return tuple(rows[0].keys()) if rows else ()


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
