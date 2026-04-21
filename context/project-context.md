# KVBlock project context

This folder contains planning context for KVBlock.

Current project stance:
- build instrumentation first
- V1 should be heuristic and bandwidth-first
- access traces and dense oracle comparison are a major project asset
- FlashInfer is the preferred V1 substrate
- the synthetic needle test should be one of the earliest validation tasks
- confidence and fallback logic are critical to the quality/$-per-token tradeoff

Key philosophy:
You cannot optimize what you have not profiled at the DRAM-byte level.
