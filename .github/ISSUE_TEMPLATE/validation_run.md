---
name: Validation run
about: Record a reproducible memory / quality / throughput comparison
title: "[validation] "
labels: validation
assignees: ""
---

## Hardware

- Chip:
- Unified RAM (GB):
- macOS version:

## Software

- Python / mlx / mlx_lm / VeloxQuant-MLX versions (or commit SHA):

## Model and workload

- Model id:
- Prompt length (tokens) or how the prompt was built:
- `max_tokens`:
- Methods compared (e.g. fp16, RVQ-1bit, VecInfer-1bit):

## Metrics reported

Distinguish these clearly:

| Metric | Value | Notes |
| --- | --- | --- |
| Tokens in cache (`offset`) | | |
| Key accounting ratio (`fp16_key_bytes / compressed_key_bytes`) | | |
| Value accounting (if any) | | |
| MLX peak MB (`mx.get_peak_memory`) | | |
| Throughput (tok/s) | | |
| Output quality notes | | |

## Accounting vs resident memory

State whether claimed compression is **key-byte accounting** only, or measured **resident / packed** storage. Default RVQ and VecInfer paths often dequantize into the parent fp16 cache; counters can show large ratios while process RSS barely moves at short context.

## Artifact path

Path to committed `results.json` (and script that produced it):

```text
figures/validation/<model>/results.json
```

## Script used

```bash
# exact command line
```
