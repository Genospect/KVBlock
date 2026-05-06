# vLLM Integration Design Notes

This is a planning stub. Do not start with invasive vLLM integration before the
kernel-level sparse decode path is proven.

## Questions to Resolve

- Where does the active vLLM version store per-request block tables?
- At what point can KVBlock observe logical block ids and physical page ids?
- Where can selected page ids enter the attention backend without breaking
  scheduler assumptions?
- Which hooks can be adapter-based, and which would require upstream changes?
- Which internals are unstable enough to avoid in a package API?

## Intended Adapter Responsibilities

- Observe block tables.
- Map `SelectedKVPlan.logical_block_ids` to physical page ids.
- Pass selected page ids only to a backend that physically skips unselected KV
  reads.
- Emit traces that compare sparse selection against dense-oracle usefulness.
