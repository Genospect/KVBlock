# Profiling on L4

Use layered timing. `nvidia-smi` is useful for coarse utilization but is not
sufficient for per-kernel proof.

## CUDA Events

Use CUDA events for decode-step buckets:

```python
from kvblock.metrics.latency import time_cuda_callable

result, elapsed_ms = time_cuda_callable(lambda: backend.run_decode(...))
```

Measure at least:

- selector time
- logical-to-physical mapping time
- attention backend time
- total decode-step time

## Torch Profiler

Capture kernel names and high-level operator time:

```bash
python -m torch.profiler scripts/run_sparse_correctness_poc.py --device cuda
```

For real scripts, prefer adding an explicit `torch.profiler.profile` block around
the decode loop so warmup and measurement windows are controlled.

## Nsight Systems

Use Nsight Systems to verify launch timelines and overlap:

```bash
nsys profile -o results/profiles/kvblock_decode python scripts/run_flashinfer_sparse_decode_poc.py
```

## Nsight Compute

Use Nsight Compute for kernel memory behavior:

```bash
ncu --set full -o results/profiles/kvblock_kernel python scripts/run_flashinfer_sparse_decode_poc.py
```

The key proof point is reduced K/V memory traffic for sparse decode kernels, not
just lower selected-token counts in a report.
