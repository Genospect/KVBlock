"""Report records for quality-vs-efficiency benchmark summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class EfficiencyGuardedReportRow:
    """One benchmark row for a policy/dataset/context bucket."""

    policy: str
    dataset: str
    context_length_bucket: str
    selected_token_fraction: float
    answer_score_dense: float
    answer_score_sparse: float
    quality_delta: float
    fallback_rate: float
    selector_ms: float
    estimated_kv_read_reduction: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize this report row."""

        return asdict(self)
