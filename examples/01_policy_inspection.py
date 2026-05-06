from kvblock.policies import get_policy_preset


policy = get_policy_preset("quality_guarded_static").resolve()
print(policy.to_dict())
