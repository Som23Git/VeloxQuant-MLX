---
id: benchmarking
title: Benchmarking Guide
sidebar_label: Benchmarking
slug: /guides/benchmarking
---

# Benchmarking Guide

This guide explains how to use the `veloxquant benchmark` CLI, interpret results, and reproduce the paper numbers from BENCHMARK_RESULTS.md.

## CLI benchmark tool

The built-in benchmark CLI runs a single quantization configuration against a model and reports memory, latency, and quality metrics:

```bash
python -m veloxquant_mlx benchmark \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --method turboquant_rvq \
    --bits 1 \
    --value-bits 2 \
    --seq-len 4096 \
    --num-runs 10 \
    --output ./results/llama3_rvq_1bit.json
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--model` | Required | mlx_lm model path or HuggingFace ID |
| `--method` | Required | Algorithm name (`turboquant_rvq`, `vecinfer`, etc.) |
| `--bits` | `1` | Key bit rate |
| `--value-bits` | `2` | Value bit rate |
| `--seq-len` | `2048` | Sequence length for benchmark |
| `--num-runs` | `5` | Number of timed runs (first is warmup) |
| `--output` | stdout | Path to save JSON results |
| `--artifacts-dir` | `./artifacts/` | Where to load calibration artifacts |
| `--verbose` | False | Print per-run timing |

### Output format

```json
{
  "model": "mlx-community/Llama-3.2-3B-Instruct-4bit",
  "method": "turboquant_rvq",
  "bits": 1,
  "value_bits": 2,
  "seq_len": 4096,
  "results": {
    "peak_memory_mb": 71.3,
    "fp16_equivalent_mb": 536.0,
    "compression_ratio": 7.5,
    "mean_encode_ms": 3.2,
    "mean_decode_ms": 1.8,
    "mean_cosine_similarity": 0.974,
    "tokens_per_second": 42.1
  }
}
```

## Benchmarking in Python

For more control, run benchmarks programmatically:

```python
import mlx_lm
import time
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.observers.memory import MemoryObserver
from veloxquant_mlx.observers.distortion import DistortionObserver
from veloxquant_mlx.observers.latency import LatencyObserver

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Benchmark multiple configs
configs = [
    KVCacheConfig(method="turboquant_rvq", bits=1, value_bits=2),
    KVCacheConfig(method="turboquant_rvq", bits=2, value_bits=2),
    KVCacheConfig(method="qjl", sketch_dim=64),
]

prompt = "Write a detailed essay on climate change and its global impacts." * 10

for config in configs:
    cache = KVCacheBuilder.build(model, config)
    mem = MemoryObserver()
    dist = DistortionObserver()
    lat = LatencyObserver()
    for obs in [mem, dist, lat]:
        obs.attach(cache)

    # Warmup
    mlx_lm.generate(model, tokenizer, prompt=prompt[:500], max_tokens=128, kv_cache=cache)

    # Timed run
    t0 = time.perf_counter()
    mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=512, kv_cache=cache)
    elapsed = time.perf_counter() - t0

    print(f"\n--- {config.method} {config.bits}-bit ---")
    print(f"Memory      : {mem.report().peak_compressed_mb:.1f} MB "
          f"({mem.report().compression_ratio:.1f}×)")
    print(f"Cosine sim  : {dist.report().mean_cosine_similarity:.4f}")
    print(f"Encode lat  : {lat.report().mean_encode_ms:.2f} ms")
    print(f"Wall time   : {elapsed:.2f} s")
```

## Interpreting results

### Compression ratio

`compression_ratio = fp16_equivalent_mb / peak_compressed_mb`. Higher is better — a ratio of 8× means the compressed cache uses 8× less memory than fp16.

### Cosine similarity

The average cosine similarity between original and quantized keys. A value above `0.95` indicates high fidelity. Below `0.90` may cause noticeable generation quality degradation on some tasks.

### Tokens per second

End-to-end throughput including quantization overhead. With Metal kernels, VeloxQuant-MLX typically achieves throughput within `2–5%` of fp16 baseline at 2-bit compression.

## Reproducing paper numbers

The benchmark results in BENCHMARK_RESULTS.md were produced with:

```bash
# Full 10-model sweep (reproduces Table 1)
python benchmark_scripts/benchmark_vecinfer.py \
    --models mlx-community/Llama-3.1-8B-Instruct-4bit \
             mlx-community/Mistral-7B-Instruct-v0.3-4bit \
             mlx-community/Qwen2.5-7B-Instruct-4bit \
    --seq-lens 1024 4096 16384 \
    --methods turboquant_rvq vecinfer ratequant spectral \
    --output ./figures/paper_table1/

# RateQuant outlier study (reproduces Figure 3)
python benchmark_scripts/run_outlier_ratequant.py \
    --model mlx-community/Llama-3.1-8B-Instruct-4bit \
    --target-bits 1.0 1.5 2.0 2.5 3.0 4.0 \
    --output ./figures/outlier_token_ratequant/
```

:::note
Benchmark scripts assume calibration artifacts are already generated in `./artifacts/`. Run `python -m veloxquant_mlx precompute` first for VecInfer, RateQuant, and SpectralQuant.
:::

## See also

- [Observers guide](/guides/observers)
- [Mixed-precision guide](/guides/mixed-precision)
- [CLI reference — benchmark command](/api/core-api)
