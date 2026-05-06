from kvblock import BlockLayout, SelectedKVPlan


layout = BlockLayout.from_token_count(total_tokens=128, block_size=32)
block_ids = [0, 3]
token_ranges = layout.token_ranges_for_blocks(block_ids)

plan = SelectedKVPlan(
    logical_block_ids=block_ids,
    selected_token_ranges=token_ranges,
    total_blocks=layout.block_count,
    total_tokens=layout.total_tokens,
    selector_name="example",
    policy_name="example_policy",
)

print(plan.to_json(indent=2))
