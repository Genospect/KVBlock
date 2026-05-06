"""Adapters around the existing V1 staged selector pipeline."""

from __future__ import annotations

from typing import Any, Sequence

from kvblock.blocks import BlockLayout
from kvblock.kv.metadata import BlockMetadata
from kvblock.plans import SelectedKVPlan
from kvblock.policies import KVBlockPolicy
from kvblock.selector.pipeline import SelectorPipeline, SelectorPipelineConfig
from kvblock.selector.policies import (
    ConfidencePolicy,
    FallbackPolicy,
    StageAPolicy,
    StageBPolicy,
    StageCPolicy,
)


class MixedGlobalRefineSelector:
    """Package-facing adapter for the existing inspectable selector pipeline."""

    name = "mixed_global_refine"

    def select(
        self,
        query_state: Any,
        kv_metadata: Sequence[BlockMetadata],
        layout: BlockLayout,
        policy: KVBlockPolicy,
    ) -> SelectedKVPlan:
        resolved = policy.resolve()
        pipeline = SelectorPipeline(_pipeline_config_from_policy(resolved))
        result = pipeline.run(
            query_state,
            kv_metadata,
            current_step=int(resolved.metadata.get("current_step", 0)),
            step_id=resolved.metadata.get("step_id"),
            context_tokens=layout.total_tokens,
            anchor_block_ids=resolved.metadata.get("anchor_block_ids", ()),
        )
        final = result.trace.final_selection
        logical_block_ids = list(final.final_selected_block_ids)
        token_ranges = layout.token_ranges_for_blocks(logical_block_ids)
        selected_tokens = sum(end - start for start, end in token_ranges)
        fallback_triggered = result.mode != "sparse"
        return SelectedKVPlan(
            logical_block_ids=logical_block_ids,
            selected_token_ranges=token_ranges,
            recent_block_ids=list(final.recent_block_ids),
            anchor_block_ids=list(final.anchor_block_ids),
            confidence=_confidence_from_trace(result.trace.confidence),
            fallback_triggered=fallback_triggered,
            fallback_reason=(
                result.trace.fallback.reason_code if fallback_triggered else None
            ),
            selector_name=self.name,
            policy_name=policy.name,
            total_blocks=layout.block_count,
            selected_blocks=len(logical_block_ids),
            total_tokens=layout.total_tokens,
            selected_tokens=selected_tokens,
            metadata={"trace": result.trace.to_dict(), "mode": result.mode},
        )


def _pipeline_config_from_policy(policy: KVBlockPolicy) -> SelectorPipelineConfig:
    keep_anchor_blocks = 2 if policy.keep_anchor_blocks else 0
    return SelectorPipelineConfig(
        stage_a=StageAPolicy(
            shortlist_size=policy.shortlist_m,
            long_context_shortlist_size=max(policy.shortlist_m, 48),
        ),
        stage_b=StageBPolicy(),
        stage_c=StageCPolicy(
            keep_recent_blocks=policy.keep_recent_blocks,
            keep_anchor_blocks=keep_anchor_blocks,
            semantic_top_k=max(1, policy.semantic_k),
            semantic_top_k_long_context=max(1, max(policy.semantic_k, 16)),
        ),
        confidence=ConfidencePolicy(margin_threshold=policy.fallback_margin),
        fallback=FallbackPolicy(allow_dense_fallback=policy.fallback_mode != "none"),
    )


def _confidence_from_trace(confidence: Any) -> float:
    if confidence.is_confident:
        return 1.0
    normalized = confidence.normalized_margin
    if normalized is None:
        return max(0.0, min(1.0, confidence.raw_margin))
    return max(0.0, min(1.0, normalized))
