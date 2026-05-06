from types import SimpleNamespace

from kvblock import KVBlockPolicy
from kvblock.blocks import BlockLayout
from kvblock.selectors import MixedGlobalRefineSelectedPlanAdapter


layout = BlockLayout.from_token_count(total_tokens=160, block_size=40)
policy = KVBlockPolicy(name="quality_guarded_static")

# In the real LongBench/output path this object is produced by the existing
# mixed/global/refine selector stack. The adapter preserves those selected ids
# and spans while exposing the new SelectedKVPlan API.
legacy_selector_row = SimpleNamespace(
    selected_ids=(1, 3),
    selected_spans=("40:80", "120:160"),
    total_blocks=layout.block_count,
    total_tokens=layout.total_tokens,
    confidence=0.42,
    fallback_triggered=False,
    selector_name="mixed_global_refine_40_16_stride_8",
)

plan = MixedGlobalRefineSelectedPlanAdapter().select(
    legacy_selector_row,
    None,
    layout,
    policy,
)

print(plan.to_json(indent=2))
