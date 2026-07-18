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

## Worked example: VecInfer

[`benchmark_scripts/benchmark_vecinfer.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_vecinfer.py) compares three VecInfer configurations (2-bit, 1.5-bit, 1-bit key/value quantization) against a vanilla fp16 KV cache on the same three short prompts, measuring generation throughput, peak memory, and key/value compression ratio for each. It's a good first script to run if you want to see real, reproducible numbers rather than the illustrative snippets above.

```bash
PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py \
    --model mlx-community/Llama-3.2-1B-Instruct-4bit
```

The first run performs a short calibration pass (training smooth factors and codebooks on synthetic activations) and caches the result under `~/.cache/veloxquant/vecinfer/<model-id>/`, so subsequent runs against the same model are faster.

### Captured output

Run against `mlx-community/Llama-3.2-1B-Instruct-4bit` on Apple Silicon:

```text
Loading mlx-community/Llama-3.2-1B-Instruct-4bit...
  head_dim=64, n_heads=32

--- fp16-baseline ---
  prompt 0: 81 tok in 0.88s (92.1 tok/s)
  prompt 1: 81 tok in 0.70s (115.8 tok/s)
  prompt 2: 81 tok in 0.70s (115.4 tok/s)

--- vecinfer-2bit ---
  prompt 0: 70 tok in 1.00s (70.2 tok/s)
  prompt 1: 78 tok in 0.93s (83.7 tok/s)
  prompt 2: 79 tok in 0.93s (84.8 tok/s)

--- vecinfer-1.5bit ---
  prompt 0: 81 tok in 6.75s (12.0 tok/s)
  prompt 1: 79 tok in 6.68s (11.8 tok/s)
  prompt 2: 76 tok in 6.69s (11.4 tok/s)

--- vecinfer-1bit ---
  prompt 0: 77 tok in 1.20s (64.0 tok/s)
  prompt 1: 82 tok in 1.13s (72.3 tok/s)
  prompt 2: 52 tok in 1.13s (46.1 tok/s)

Summary: figures/vecinfer/Llama-3.2-1B-Instruct-4bit/vecinfer_summary.png
Results: figures/vecinfer/Llama-3.2-1B-Instruct-4bit/results.json

Final:
  fp16-baseline         106.5 tok/s    710.5 MB  key_x=1.00  avg_bits=16.00
  vecinfer-2bit          79.3 tok/s    713.0 MB  key_x=8.00  avg_bits=2.00
  vecinfer-1.5bit        11.7 tok/s    713.6 MB  key_x=10.67  avg_bits=1.50
  vecinfer-1bit          60.9 tok/s    712.9 MB  key_x=16.00  avg_bits=1.00
```

:::note
In this particular run, `vecinfer-1.5bit` was markedly slower (11.7 tok/s) than the 2-bit and 1-bit configs (79.3 and 60.9 tok/s). This is the actual measurement from this script on this machine, not a typo — the 1.5-bit path takes a different, currently less-optimized code path. Re-run yourself before relying on this number; timings are sensitive to thermal state, background load, and MLX/mlx_lm version.
:::

The script saves a 4-panel summary chart alongside a `results.json` with the same numbers:

![VecInfer benchmark summary for Llama-3.2-1B-Instruct-4bit, showing four bar charts: throughput in tokens per second, peak memory in MB, key cache compression ratio, and effective bit-width, each comparing fp16-baseline against vecinfer-2bit, vecinfer-1.5bit, and vecinfer-1bit configurations](/img/benchmarks/vecinfer/vecinfer_summary.png)

The four panels are, left to right:

- **Throughput** — tokens/second for each config. Lower-bit configs generally trade throughput for memory savings, though the actual ordering depends on which code paths are optimized (see the 1.5-bit note above).
- **Peak memory** — peak MLX/Metal memory in MB during generation. In this run all four configs land close together (~710–714 MB) because the model weights dominate total memory at this scale (1B params); the KV-cache savings become more visible at longer sequence lengths or larger models.
- **Key cache compression** — `fp16_key_bytes / compressed_key_bytes` for the key cache. This tracks the configured bit-width directly (2-bit ≈ 8×, 1.5-bit ≈ 10.67×, 1-bit ≈ 16×).
- **Effective bit-width** — the average bits/element actually assigned by the allocator, which should match the config's target (2.0, 1.5, 1.0) — useful as a sanity check that the allocator is behaving as configured.

### Try it yourself

Swap `--model` for any other `mlx-community/*` checkpoint, and use `--max-tokens` to change generation length (default `80`). Output lands in `figures/vecinfer/<model-stem>/` by default, or pass `--output-dir` to redirect it:

```bash
PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --max-tokens 128 \
    --output-dir ./my-results/vecinfer-3b
```

## Worked example: KIVI

[`benchmark_scripts/benchmark_kivi.py`](https://github.com/rajveer43/VeloxQuant-MLX/blob/master/benchmark_scripts/benchmark_kivi.py) benchmarks several KIVI (arXiv:2402.02750) bit-widths against an fp16 baseline. Unlike VecInfer, KIVI is deterministic — there's no codebook calibration step, so every run is a cold start. KIVI keeps a small fp16 "residual window" of the most recent tokens uncompressed, so the script reports both a key-only compression ratio and a lower full-KV ratio that accounts for that residual.

```bash
PYTHONPATH=. python benchmark_scripts/benchmark_kivi.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit
```

:::note[Read this before trusting the throughput numbers]
This script's own docstring calls out a specific past mistake: an earlier RaBitQ benchmark recorded `fp16_ms: 0`, which silently invalidated every speedup number derived from it. `benchmark_kivi.py` was written so the fp16 baseline is **always timed for real**, specifically to avoid repeating that bug. The docstring also sets an expectation worth stating plainly: on Apple Silicon, KIVI's paper-reported speedup comes from a CUDA kernel that has no Metal port here, so the realistic expectation is a **memory win with a throughput cost**, not a free speedup. In the captured run below, throughput was actually flat across all configs (~16 tok/s) rather than costly — that's a property of this particular 3B model and short prompt, not a general claim; re-run at longer sequence lengths before drawing conclusions.
:::

### Captured output

Run against `mlx-community/Llama-3.2-3B-Instruct-4bit` on Apple M4 (24 GB):

```text
Loading mlx-community/Llama-3.2-3B-Instruct-4bit...
  head_dim=128, n_kv_heads=8, n_layers=28
  prompt_tokens=2239 (residual_length=32)
  hardware={'platform': 'macOS-26.5.2-arm64-arm-64bit', 'machine': 'arm64', 'chip': 'Apple M4', 'ram_gb': 24.0}

--- fp16-baseline ---
  121 tok in 7.54s (16.1 tok/s)  peak=2484MB  key_x=1.00  fullKV_x=1.00

--- KIVI-2bit ---
  121 tok in 7.39s (16.4 tok/s)  peak=2600MB  key_x=5.79  fullKV_x=3.98

--- KIVI-3bit ---
  121 tok in 7.51s (16.1 tok/s)  peak=2600MB  key_x=4.34  fullKV_x=3.24

--- KIVI-4bit ---
  121 tok in 7.57s (16.0 tok/s)  peak=2600MB  key_x=3.47  fullKV_x=2.73

Results: figures/kivi/Llama-3.2-3B-Instruct-4bit/results.json
  fp16-baseline      16.1 tok/s   2483.7 MB  key_x=1.00  fullKV_x=1.00  toks=121
  KIVI-2bit          16.4 tok/s   2600.1 MB  key_x=5.79  fullKV_x=3.98  toks=121
  KIVI-3bit          16.1 tok/s   2600.1 MB  key_x=4.34  fullKV_x=3.24  toks=121
  KIVI-4bit          16.0 tok/s   2600.1 MB  key_x=3.47  fullKV_x=2.73  toks=121
```

:::note[Peak memory went up, not down, in this run]
All three KIVI configs show slightly *higher* peak memory (2600 MB) than the fp16 baseline (2484 MB) here — the opposite of what compression should do. At this small a model (3B) and short a prompt (~2.2K tokens), the KV cache isn't yet the dominant consumer of memory, so KIVI's bookkeeping overhead (quantized storage + the fp16 residual window) outweighs its savings. Compression wins become visible at longer context lengths, where the KV cache — not the model weights — dominates memory. Try `--max-tokens` with a much longer prompt to see this shift.
:::

The script saves a 4-panel summary chart alongside a `results.json` with the same numbers:

![KIVI benchmark summary for Llama-3.2-3B-Instruct-4bit on Apple M4, showing four bar charts: throughput in tokens per second, peak memory in MB, key compression ratio, and full-KV compression ratio including the fp16 residual window, each comparing fp16-baseline against KIVI-2bit, KIVI-3bit, and KIVI-4bit](/img/benchmarks/kivi/kivi_summary.png)

The four panels are, left to right:

- **Throughput** — tokens/second for each config. In this run all four configs are within noise of each other (~16–16.4 tok/s) — see the caveat above about not over-generalizing from a short prompt on a small model.
- **Peak memory** — peak MLX/Metal memory in MB during generation. See the note above on why this went up rather than down at this scale.
- **Key compression** — `fp16_key_bytes / compressed_key_bytes` for the key cache only. Tracks the configured bit-width (2-bit ≈ 5.8×, 3-bit ≈ 4.3×, 4-bit ≈ 3.5× in this run — lower than the theoretical 8×/5.3×/4× because of quantization group overhead at `--group-size 32`).
- **Full-KV compression (incl. fp16 residual)** — the realistic end-to-end ratio once KIVI's fp16 residual window (the most recent `--residual-length` tokens, kept uncompressed for accuracy) is included. Always lower than key-only compression — this is the number to use when estimating actual memory savings.

### Try it yourself

KIVI has more tunable parameters than VecInfer. `--model` and `--max-tokens` work the same way; `--group-size` controls the quantization group size (smaller groups → better fidelity, more overhead) and `--residual-length` controls how many recent tokens stay in fp16:

```bash
PYTHONPATH=. python benchmark_scripts/benchmark_kivi.py \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --max-tokens 256 \
    --group-size 64 \
    --residual-length 64 \
    --output-dir ./my-results/kivi-3b
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

- [Observers guide](../guides/observers)
- [Mixed-precision guide](../guides/mixed-precision)
- [CLI reference — benchmark command](../api/core-api)
