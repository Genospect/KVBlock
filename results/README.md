# Results Organization

Research outputs are preserved by experiment family rather than kept in one flat directory.

- `baselines/`: current and early selector baseline rows and aggregate JSONL.
- `representation/`: hidden-state, key, and query-source representation sweeps.
- `aggregation/`: query/key aggregation ablations.
- `heads/`: per-head diagnostics, ablations, and static head-weight validation.
- `dynamic_blocks/`: fixed-size, multi-scale, overlap-suppression, and coarse-to-fine block experiments.
- `inspection/`: prompt-level qualitative inspection artifacts.
- `plots/`: generated figures.
- `tmp/`: ignored scratch area for throwaway local runs.

The current best local baseline is documented in `docs/current-best-path.md`.
