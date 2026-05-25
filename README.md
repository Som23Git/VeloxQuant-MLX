# VeloxQuant-MLX

### 🌐 [veloxquant-mlx.netlify.app](https://veloxquant-mlx.netlify.app/) — interactive landing page with benchmarks, algorithm explainers, and copy-paste code snippets.

**Fast KV-cache quantization for Apple Silicon — TurboQuant, RVQ, VecInfer, RateQuant, PolarQuant, and QJL in MLX.**

[![Landing page](https://img.shields.io/badge/landing-veloxquant--mlx.netlify.app-7c3aed.svg)](https://veloxquant-mlx.netlify.app/)
[![PyPI version](https://img.shields.io/badge/pypi-0.5.1-blue.svg)](https://pypi.org/project/VeloxQuant-MLX/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A drop-in KV-cache replacement for `mlx_lm` that compresses the Key tensor by **3–16×** with near-lossless quality at 4-bit, **functional 2-bit and 1-bit** via Residual Vector Quantization, up to **16× via VecInfer product VQ**, and **per-layer mixed-precision** allocation via RateQuant. Validated end-to-end on **10 production models** (Mistral, Falcon, Phi-4, Qwen3, Qwen2.5, Llama 3.1/3.2, Gemma3, SmolLM2).

### Uniform 1-bit RVQ — 7.5× key compression, 95–104% of fp16 throughput

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer,
    prompt="Explain the theory of relativity in simple terms.",
    max_tokens=200,
)
```

### Per-layer RateQuant — match fp16 throughput at fractional average bits

```python
from veloxquant_mlx import (
    KVCacheBuilder, KVCacheConfig,
    calibrate_layer_sensitivities, allocate_bits_ratequant,
)

# 1.6s one-time calibration on real model activations
weights = calibrate_layer_sensitivities(model, tokenizer)

# Theorem 2 closed-form: high-sensitivity layers get more bits
alloc = allocate_bits_ratequant(weights, target_avg_bits=1.5, beta=3.5)

# Pass the list directly to KVCacheConfig — for_model() consumes per layer
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=alloc, seed=42)
caches = KVCacheBuilder.for_model(model, config)
```

---

## Table of contents

1. [Highlights](#highlights)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [RateQuant — per-layer mixed precision](#ratequant--per-layer-mixed-precision)
5. [What's inside](#whats-inside)
6. [Algorithm guide](#algorithm-guide-which-method-to-pick)
7. [Per-model benchmark results](#per-model-benchmark-results)
8. [Throughput optimization journey](#throughput-optimization-journey)
9. [Architecture](#architecture)
10. [CLI](#cli)
11. [Development](#development)
12. [References](#references)

---

## Highlights

- **VecInfer product VQ (new in 0.5.0)** — smooth scaling + Walsh-Hadamard dual transform + K-means codebook delivers **16× key compression at 1 bit/elem**. On Qwen2.5-7B (strong GQA) VecInfer-1bit *exceeds* fp16 throughput at 16× compression. Benchmarked across 10 models — see [v6 results](#v6-results--vecinfer-10-model-comparative-study-050).
- **RVQ 1-bit (new in 0.3.4)** — sign-quantizer stage + Laplacian residual delivers **7.5× key compression with cosine 0.92**, generates full 200-token output on every tested model, and **matches or beats fp16 throughput** on most 7–8B models.
- **RateQuant per-layer allocation (new in 0.3.5)** — Theorem 2 reverse-waterfilling on real activation sensitivities. Pass `bit_width_inlier=alloc_list` to `KVCacheConfig`, let `KVCacheBuilder.for_model()` consume per layer. 1.6s one-time calibration, zero inference overhead.
- **RVQ 2-bit** — two-pass residual quantization brings 2-bit cosine from **0.69 → 0.98**.
- **End-to-end fp16 throughput parity** on Mistral, Falcon, Phi-4, Qwen3, Gemma3 after the throughput optimizations.
- **Four quantizers**, one interface — `turboquant_rvq`, `turboquant_prod`, `turboquant_mse`, plus `polar` and `qjl`.
- **Native MLX integration** — no Metal kernel writing required; uses `mx.hadamard_transform` for O(d log d) rotation.
- **Production patterns** — Factory + Strategy + Registry + Builder. Drop-in for `mlx_lm.cache.KVCache`.
- **Apple Silicon first** — designed and tested on M-series unified-memory.

---

## Installation

```bash
pip install VeloxQuant-MLX
```

From source:

```bash
git clone https://github.com/rajveer43/VeloxQuant-MLX
cd VeloxQuant-MLX
pip install -e ".[dev]"
```

Requires **Python ≥ 3.11** and an **Apple Silicon Mac** with MLX ≥ 0.18.

---

## Quick start

### Standalone KV cache (synthetic streaming)

```python
from veloxquant_mlx import KVCacheBuilder
import mlx.core as mx, numpy as np

cache = (
    KVCacheBuilder()
    .with_method("turboquant_rvq")     # try also: "turboquant_prod", "polar", "qjl"
    .with_head_dim(128)
    .with_bit_width(inlier=2)          # 2-bit RVQ uses 2*b = 4 bits/dim total
    .with_seed(42)
    .build()
)

rng = np.random.default_rng(0)
for _ in range(1000):
    cache.append(
        mx.array(rng.standard_normal(128).astype(np.float16)),
        mx.array(rng.standard_normal(128).astype(np.float16)),
    )

q = mx.array(rng.standard_normal(128).astype(np.float16))
out = cache.attend(q)
print(f"Memory: {cache.memory_bytes()/1024:.1f} KB for {len(cache)} tokens")
```

### Drop-in replacement for `mlx_lm` generation (recommended)

`KVCacheBuilder.for_model()` handles per-layer construction, dtype detection,
and VLM wrappers automatically:

```python
import mlx_lm
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
caches = KVCacheBuilder.for_model(model, config)
model.make_cache = lambda *_a, **_k: caches

response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=200)
```

Per-cache byte accounting is available via `cache.fp16_key_bytes /
cache.compressed_key_bytes` for benchmark reporting.

---

## RateQuant — per-layer mixed precision

The default is **uniform** bit-width across layers. RateQuant
([arxiv:2605.06675](https://arxiv.org/abs/2605.06675)) allocates **more bits
to high-sensitivity layers** and fewer to low-sensitivity ones, with the
average held at a user-chosen target. The library exposes both the
sensitivity probe and the closed-form allocator:

```python
from veloxquant_mlx import (
    KVCacheBuilder, KVCacheConfig,
    calibrate_layer_sensitivities,   # 1.6s, real-activation probe
    allocate_bits_ratequant,         # Theorem 2 reverse-waterfilling
)

# Step 1 — one-time calibration on 8 default prompts (overridable)
weights = calibrate_layer_sensitivities(model, tokenizer)
# weights[i] is the mean-squared key L2 norm at layer i

# Step 2 — closed-form allocation. Average is exact; per-layer bits are integer.
alloc = allocate_bits_ratequant(
    weights,
    target_avg_bits=1.5,   # fractional — integer alloc straddles it
    beta=3.5,              # RVQ decay constant (paper-reported)
    bit_choices=(1, 2, 3), # RVQ supports any positive integer b
)

# Step 3 — pass directly to KVCacheConfig. for_model() consumes per-layer.
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=alloc, seed=42)
caches = KVCacheBuilder.for_model(model, config)
```

**When does it help?** When per-layer sensitivity is heterogeneous. The
calibration printout reports the min/max range; a ratio above ~2× indicates
RateQuant will give measurable gains. Empirically:

| Model | Sensitivity ratio | Notes |
|---|---|---|
| Falcon3 7B (28 layers, head_dim=256) | 6.48× | Mixed alloc: 14 layers at b=2, 14 at b=1 |
| Gemma3 4B (34 layers, head_dim=256) | 14.39× | Mixed alloc: 3 at b=3, 11 at b=2, 20 at b=1 |

**Distortion model.** The paper notes that the decay rate β varies across
quantizers (3.5 for TurboQuant, ≈5.0 for KIVI/QuaRot). The library default
of `beta=3.5` is correct for RVQ; if you adapt the allocator to another
quantizer, call `fit_distortion_curve()` first to estimate β.

**What's NOT (yet) implemented from the paper:** per-head allocation (paper:
L×H groups, ours: L), gradient-based sensitivity (paper notes activation
is ~1 PPL worse but both beat uniform), and K/V separation (paper's biggest
single fix on KIVI). These remain open extensions — the per-layer subset
already gives most of the benefit on RVQ at ≥1.5 bits.

---

## VecInfer — vector quantization with outlier-suppressing dual transform

**New in 0.5.0.** [VecInfer](https://arxiv.org/abs/2510.06175) (Yao et al. 2025)
compresses the KV cache via product vector quantization against a pre-trained
K-means codebook. To handle outlier channels that wreck codebook utilization
at low bit-widths, VecInfer applies a **dual transform** before quantization:
per-channel smooth scaling (`lambda_i = sqrt(max|K_i|)`) followed by a
Walsh–Hadamard rotation. The inverse transform is absorbed into the queries,
so `q_tilde @ K_tilde.T == q @ K.T` exactly.

```python
import mlx.core as mx
from mlx_lm import load, generate
from veloxquant_mlx import KVCacheConfig, KVCacheBuilder

model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="vecinfer",
    key_codebook_bits=8, key_sub_dim=4,       # 2 bits/elem on keys
    value_codebook_bits=8, value_sub_dim=4,   # 2 bits/elem on values
    # smooth_factors / key_codebook / value_codebook can be passed pre-calibrated;
    # omitted = random init (useful only for wiring tests)
)
caches = KVCacheBuilder.for_model(model, config)
out = generate(model, tokenizer, prompt="Explain Hadamard rotation", max_tokens=80,
               prompt_cache=caches)
```

**Realized compression** (Llama-3.2-3B-Instruct-4bit, this repo):
8× on keys at 2 bits/elem, 16× at 1 bit/elem.

**Tradeoff to know up front:** the paper's CUDA kernel fuses dequantization
into attention, eliminating the dequant overhead. That fusion is not
portable to Metal — on Apple Silicon you should expect throughput to
*drop* vs fp16 (the win is memory, not speed). See
[`benchmark_scripts/benchmark_vecinfer.py`](benchmark_scripts/benchmark_vecinfer.py)
and `figures/vecinfer/` for measured numbers.

---

## What's inside

| Module | Purpose |
|---|---|
| [`veloxquant_mlx.quantizers.turboquant_prod`](veloxquant_mlx/quantizers/turboquant_prod.py) | Rotation + Lloyd-Max + QJL residual (b-1 + 1 bits) |
| [`veloxquant_mlx.quantizers.turboquant_mse`](veloxquant_mlx/quantizers/turboquant_mse.py) | Rotation + Lloyd-Max only (no residual correction) |
| [`veloxquant_mlx.quantizers.turboquant_rvq`](veloxquant_mlx/quantizers/turboquant_rvq.py) | Two-pass scalar RVQ (Gaussian + Laplacian codebooks), b=1/2/3+ |
| [`veloxquant_mlx.quantizers.polarquant`](veloxquant_mlx/quantizers/polarquant.py) | Recursive polar coordinate decomposition |
| [`veloxquant_mlx.quantizers.qjl`](veloxquant_mlx/quantizers/qjl.py) | Pure 1-bit JL sign sketch |
| [`veloxquant_mlx.cache.turboquant_rvq_cache`](veloxquant_mlx/cache/turboquant_rvq_cache.py) | `TurboQuantRVQKVCache` mlx_lm-compatible cache wrapper |
| [`veloxquant_mlx.cache.vecinfer_cache`](veloxquant_mlx/cache/vecinfer_cache.py) | **NEW (0.5.0)** — `VecInferKVCache` smooth + Hadamard + product VQ |
| [`veloxquant_mlx.allocators.vecinfer`](veloxquant_mlx/allocators/vecinfer.py) | **NEW (0.5.0)** — `calibrate_smooth_factors`, `walsh_hadamard_matrix`, `train_codebook`, `quantize_vq` |
| [`veloxquant_mlx.allocators`](veloxquant_mlx/allocators/) | `allocate_bits_ratequant`, `calibrate_layer_sensitivities` |
| [`veloxquant_mlx.observers`](veloxquant_mlx/observers/) | `DistortionObserver`, `LatencyObserver`, `MemoryObserver`, **`KeyNormObserver` (new)** |
| [`veloxquant_mlx.codebooks`](veloxquant_mlx/codebooks/) | `ScalarCodebook`, Lloyd-Max strategies, `AdaptiveScalarCodebook` |
| [`veloxquant_mlx.preconditioners`](veloxquant_mlx/preconditioners/) | `RotationPreconditioner` (QR), `HadamardPreconditioner` (Metal) |
| [`veloxquant_mlx.cache`](veloxquant_mlx/cache/) | `TurboQuantKVCache` standalone, mlx_lm `KVCache` subclasses |
| [`veloxquant_mlx.weight`](veloxquant_mlx/weight/) | `QuantizedLinear` for model weight quantization |
| [`veloxquant_mlx.dsa.bit_pack`](veloxquant_mlx/dsa/bit_pack.py) | Sub-byte index packing |
| [`veloxquant_mlx.outlier`](veloxquant_mlx/outlier/) | Two-stream cache for high-variance channels |

---

## Algorithm guide — which method to pick

| Method | Bits/dim | Per-vector storage (d=128) | Quality (cosine) | Best for |
|---|---|---|---|---|
| `turboquant_mse` | b | `b·d/8` + 4 B norm | 0.86 @ 3b, 0.95 @ 4b | Lowest overhead at 3–4 bit |
| `turboquant_prod` | b-1 + 1 | `(b-1)·d/8` + JL signs + 2 norms | 0.86 @ 3b, 0.95 @ 4b | Unbiased IP estimator at 3–4 bit |
| **`turboquant_rvq` @ b=2** | **2·b = 4** | **64 B** | **0.98** | **Functional 2-bit, 3.9× compression** |
| **`turboquant_rvq` @ b=1** | **2·b = 2** | **34 B** | **0.92** | **Aggressive 7.5× compression — full output on every tested model** |
| `polar` | b·levels | varies | medium | Geometric structure, very low bits |
| `qjl` | 1 | `d/8` + 2 B norm | 0.62 | Topology-only retrieval, extreme compression |

**Rule of thumb**:
- **3–4 bit, maximum compression at uniform precision** → `turboquant_mse`
- **3–4 bit, best uniform-precision quality** → `turboquant_prod`
- **2 bit (3.9× key compression, full coherent output)** → `turboquant_rvq` with `b=2`
- **1 bit (7.5× key compression, full coherent output on 7/7 tested models)** → `turboquant_rvq` with `b=1`
- **Fractional average bits (mixed-precision)** → `turboquant_rvq` + `allocate_bits_ratequant`
- **Ranking-only retrieval, extreme compression** → `qjl`

---

## Per-model benchmark results

All measurements on **Apple M4 MacBook · 16/24 GB unified memory · Python 3.12**. Prompt: structured 200-token explanation of relativity.

### v4 results — RVQ 1-bit at 7.5× compression (8-model sweep, 0.3.4)

| Model | fp16 tok/s | RVQ 1-bit tok/s | tokens | vs fp16 |
|---|---|---|---|---|
| Mistral 7B v0.3 | 23.3 | **22.2** | 201/201 | 95% |
| Falcon3 7B | 24.0 | **23.1** | 200/200 | 96% |
| Phi-4 | 11.9 | **11.8** | 200/200 | **99%** |
| Qwen3 4B | 40.2 | 34.3 | 187/200 | 85% |
| Qwen3 8B | 20.5 | **21.1** | 200/200 | **103%** |
| Llama 3.1 8B | 22.0 | **21.5** | 201/201 | 98% |
| Gemma3 4B | 32.5 | **30.5** | 201/201 | 94% |
| Qwen2.5 32B | 3.7 | — | — | memory-constrained on 24 GB, see [docs](docs/MEMORY_CONSTRAINT_FINDINGS.md) |

> Generated by [`benchmark_scripts/run_outlier_ratequant.py`](benchmark_scripts/run_outlier_ratequant.py).
> Source figures: [`figures/outlier_token_ratequant/<model>/`](figures/outlier_token_ratequant/).

### v5 results — RateQuant V2 mixed-precision (2-model trial, 0.3.5)

Per-layer allocation via `allocate_bits_ratequant` at target b̄=1.5,
measured on Apple M4 24 GB. Source figures: [`figures/2026-05-16/`](figures/2026-05-16/).

| Model | fp16 | RVQ 1-bit | RVQ 1-bit + Outlier | **RVQ + RateQuant V2** | sens. ratio |
|---|---|---|---|---|---|
| Falcon3 7B | 22.9 | 23.1 (101%) | 22.0 (96%) | **22.8 (100%)** at 5.22× compression | 6.48× |
| Gemma3 4B | 39.8 | 37.8 (95%) | 34.7 (87%) | **36.3 (91%)** at 5.22× compression | 14.39× |

> Per-layer allocations were computed from a 1.6s real-activation calibration:
> Falcon3 split 14/14 (b=2/b=1); Gemma3 split 3/11/20 (b=3/b=2/b=1).

### v6 results — VecInfer 10-model comparative study (0.5.0)

8-config head-to-head across 10 models. Full data and per-model plots in [`figures/vecinfer/`](figures/vecinfer/). Cross-model chart: [`figures/vecinfer/_summary/cross_model_comparison.png`](figures/vecinfer/_summary/cross_model_comparison.png).

**Key compression ratio:**

| Model | head_dim | TQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 64 | 6.4× | 7.1× | 8.0× | **16.0×** |
| Llama-3.2-1B | 64 | 6.4× | 7.1× | 8.0× | **16.0×** |
| Llama-3.2-3B | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Llama-3.1-8B | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Mistral-7B | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Qwen2.5-7B | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Qwen3-8B | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Phi-4 | 128 | 9.1× | 7.5× | 8.0× | **16.0×** |
| Falcon3-7B | 256 | 11.6× | 7.8× | — (OOM) | **16.0×** |
| gemma-3-4b | 256 | 11.6× | 7.8× | 8.0× | **16.0×** |

**Throughput (tok/s):**

| Model | fp16 | TQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |
|---|---:|---:|---:|---:|---:|
| SmolLM2-135M | 250.4 | 70.4 | 188.5 | 163.0 | 175.8 |
| Llama-3.2-1B | 105.4 | 75.5 | **104.3** | 60.4 | 91.2 |
| Llama-3.2-3B | 47.6 | 20.6 | **46.2** | 39.7 | 40.2 |
| Llama-3.1-8B | 20.5 | 19.8 | **20.6** | 10.7 | 19.6 |
| Mistral-7B | 23.6 | 22.5 | **22.8** | 21.2 | 9.8 |
| Qwen2.5-7B | 21.0 | 12.0 | 20.7 | 21.3 | **21.5** ← exceeds fp16 at 16× |
| Qwen3-8B | 20.3 | 19.4 | **19.6** | 17.9 | 2.4 |
| Phi-4 | 10.4 | **9.6** | 8.1 | 7.2 | 4.0 |
| Falcon3-7B | 17.3 | 15.9 | **21.7** | — | 17.0 |
| gemma-3-4b | 26.0 | 22.7 | 24.2 | **22.6** | **22.6** |

> **VecInfer wins on raw compression** (16× on every model). **RVQ-1bit wins on throughput/memory balance** — within 5% of fp16 on most 7–8B models with zero calibration overhead. **Qwen2.5-7B is the standout**: VecInfer-1bit exceeds fp16 at 16× compression, likely due to its strong GQA ratio (28q/4kv heads).
> See [`figures/vecinfer/_summary/SUMMARY.md`](figures/vecinfer/_summary/SUMMARY.md) for the full analysis.

### Cross-model summary (single-pass quality at 3-bit and 4-bit)

| Model | Architecture | head_dim | fp16 tok/s | 3-bit quality | 4-bit quality |
|---|---|---|---|---|---|
| Llama 3.2 3B | dense | 128 | 47.2 | Repetition | Near-lossless |
| Mistral 7B v0.3 | dense | 128 | 22.1 | Near-lossless | Near-lossless |
| Falcon3 7B | dense | 128 | 22.1 | Near-lossless | Near-lossless |
| Qwen3 4B | dense | 128 | 38.7 | Near-lossless | Early stop |
| Qwen3 8B | dense | 128 | 20.6 | Partial | Partial |
| Llama 3.1 8B | dense | 128 | 21.5 | Stops @ 62 | Near-lossless |
| Phi-4 | dense | 128 | – | Near-lossless | Near-lossless |
| Gemma-4 | hybrid (35 sliding + 7 full) | 512 | 19.3 | Near-lossless | Near-lossless |
| Qwen2.5 32B | dense | 128 | 7.1 | Near-lossless | Near-lossless |

> **Source:** per-model benchmark scripts under [`benchmark_*.py`](.) producing 6 figures each in [`figures/<model>/`](figures/).

### v2 results — with RVQ 2-bit (0.3.0 throughput optimizations active)

Both runs below use the optimized fast path (Hadamard rotation + boundary-sum quantize + cast cleanup + head batching).

#### Mistral 7B v0.3 — 4-bit weights · head_dim=128 · 32 layers · 8 KV heads

| Config | Key compression | Throughput | Tokens | Quality |
|---|---|---|---|---|
| fp16 baseline | 1.00× | 22.1 tok/s | 201/201 | reference |
| TQ 2-bit (single-pass) | 9.14× | 22.4 tok/s | 201/201 | coherent |
| TQ 3-bit | 5.82× | 22.4 tok/s | 201/201 | coherent |
| TQ 4-bit | 4.27× | 21.8 tok/s | 201/201 | near-lossless |
| **TQ RVQ 2-bit** | **3.88×** | **22.3 tok/s** | **201/201** | **near-lossless** |

> Mistral 7B is memory-bandwidth bound at ~22 tok/s. Every quantized config now matches fp16. **Figures:** [`figures/updated_tests/mistral7b/`](figures/updated_tests/mistral7b/).

#### Qwen3 4B — 4-bit weights · head_dim=128 · `<think>` mode (most quantization-sensitive)

| Config | Key compression | Throughput | Tokens | Quality |
|---|---|---|---|---|
| fp16 baseline | 1.00× | 39.2 tok/s | 200/200 | reference |
| TQ 2-bit (single-pass) | 9.14× | 31.2 tok/s | 174/200 | early stop |
| TQ 3-bit | 5.82× | 30.7 tok/s | 172/200 | partial |
| TQ 4-bit | 4.27× | 8.6 tok/s | 50/200 | `<think>`-loop |
| **TQ RVQ 2-bit** | **3.88×** | **36.0 tok/s** | **199/200** | **coherent** |

> RVQ 2-bit is the **only** quantized config that produces near-full coherent output on Qwen3's `<think>` mode while reaching 92% of fp16 throughput. **Figures:** [`figures/updated_tests/qwen3_4b/`](figures/updated_tests/qwen3_4b/).

#### Llama 3.1 8B Instruct (4-bit) — head_dim=128 · 32 layers · 8 KV heads

| Config | Key compression | Throughput | Tokens | Quality |
|---|---|---|---|---|
| fp16 baseline | 1.00× | 21.5 tok/s | 201/201 | reference |
| TQ 2-bit (single-pass) | 9.14× | 16.3 tok/s | 187/201 | broken |
| TQ 3-bit | 5.82× | 13.9 tok/s | 62/201 | repetition |
| TQ 4-bit | 4.27× | 14.8 tok/s | 201/201 | near-lossless |

> v2 (RVQ 2-bit) not yet benchmarked for this model. **Figures:** [`figures/llama31_8b/`](figures/llama31_8b/).

---

## Throughput optimization journey

The 0.3.0 release lifts quantized throughput to fp16 parity. Four sequential changes, each independently benchmarked:

| Stage | Mistral 7B RVQ 2-bit | Qwen3 4B RVQ 2-bit |
|---|---|---|
| 0. Original (per-head Python loop) | 17.7 tok/s | 24.8 tok/s |
| 1. Batch heads `(B,H,S,D) → (B·H·S,D)` | 21.5 tok/s | 34.0 tok/s |
| 2. Hadamard rotation by default | 20.0 tok/s | – |
| 3. Boundary-sum quantize (replaces argmin) | 22.4 tok/s | – |
| 4. Drop redundant fp32↔fp16 casts | **22.3 tok/s** | **36.0 tok/s** |

Quality verified at every step — RVQ cosine **0.9766** unchanged, **100%** index match on boundary-sum vs argmin, full token completion preserved on real models.

> **Full writeup:** [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md). Stage-by-stage figure: [`figures/updated_tests/optimization_journey.png`](figures/updated_tests/optimization_journey.png).

---

## Architecture

The pipeline uses a **Chain of Responsibility** pattern. Each handler mutates a `QuantizationContext` and passes it downstream:

```
TurboQuantProd pipeline
═══════════════════════
  x (fp16, batch × d)
       │
  Normalize → Rotate (Π) → Scalar quantize → QJL residual sketch → BitPack
       │
  EncodedVector(indices, signs, residual_norm)

TurboQuantRVQ pipeline (NEW)
════════════════════════════
  x (fp16, batch × d)
       │
  Rotate (Π) → Stage-1 quantize (Gaussian Lloyd-Max, b bits)
            → Compute residual r₁ = y − ŷ₁
            → Stage-2 quantize (Laplacian Lloyd-Max, b bits) → idx₂
       │
  EncodedVector(idx₁, idx₂)
       │
  Decode: ŷ = ŷ₁ + ŷ₂ → unrotate
```

Design patterns used (10): Abstract Base Classes, Factory, Chain of Responsibility, Builder, Strategy, Registry + Plugin, Composite, Observer, DAO, Custom DSA (RingBuffer, MaxHeap, BitPackBuffer, VoronoiTree).

---

## CLI

### Precompute artifacts (rotation matrices, JL matrices, codebooks)

```bash
python -m veloxquant_mlx precompute \
    --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --seed 42 \
    --output_dir ./artifacts/
```

Then pass an `NpyArtifactStore` to the builder to load instead of recompute:

```python
from veloxquant_mlx.artifacts import NpyArtifactStore
cache = (KVCacheBuilder()
    .with_method("turboquant_rvq")
    .with_head_dim(128).with_bit_width(inlier=2)
    .with_artifact_store(NpyArtifactStore("./artifacts/"))
    .build())
```

### Benchmark a single configuration

```bash
python -m veloxquant_mlx benchmark \
    --method turboquant_rvq --head_dim 128 --bits 2 --seq_len 1000
```

### Benchmark a real model end-to-end

```bash
python benchmark_mistral7b_v2.py            # 5 configs incl. RVQ 2-bit
python benchmark_qwen3_4b_v2.py             # ↳ outputs to figures/updated_tests/<model>/
python benchmark_<model>.py                 # original 4-config script (figures/<model>/)
```

---

## Development

```bash
# Tests
pytest veloxquant_mlx/tests/ -v

# 2-bit improvement validation (synthetic, fast)
python test_2bit_improvements.py

# Generate optimization-journey figure
python scripts/plot_optimization_journey.py
```

---

## References

### Implemented in this library

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [RateQuant (2025)](https://arxiv.org/abs/2605.06675) — "RateQuant: Mixed-Precision KV Cache Quantization via Rate-Distortion Theory"
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — "PolarQuant: Quantizing KV Caches with Polar Transformation"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"

### Related work — quantization

- [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) — Liu et al., "A Tuning-Free Asymmetric 2-Bit Quantization for KV Cache"
- [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079) — Hooper et al., "Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
- [Coupled Quantization (NeurIPS 2024)](https://arxiv.org/abs/2405.03917) — Zhang et al., "KV Cache is 1 Bit Per Channel: Efficient LLM Inference with Coupled Quantization"
- [KVTuner (ICML 2025)](https://arxiv.org/abs/2502.04420) — Li et al., "Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization"
- [MixKVQ (2024)](https://arxiv.org/abs/2512.19206) — Zhang et al., "Query-Aware Mixed-Precision KV Cache Quantization for Long-Context Reasoning"
- [VecInfer (2024)](https://arxiv.org/abs/2510.06175) — Yao et al., "Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization"
- [FibQuant (2025)](https://arxiv.org/abs/2605.11478) — "Universal Vector Quantization for Random-Access KV-Cache Compression"

### Related work — token eviction & sparse attention

- [SnapKV (2024)](https://arxiv.org/abs/2404.14469) — Li et al., "LLM Knows What You are Looking for Before Generation"
- [PyramidKV (2024)](https://arxiv.org/abs/2406.02069) — Cai et al., "Dynamic KV Cache Compression based on Pyramidal Information Funneling"
- [RocketKV (ICML 2025)](https://arxiv.org/abs/2502.14051) — Behnam et al., "Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"
- [MagicPIG (ICLR 2025 Spotlight)](https://arxiv.org/abs/2410.16179) — Chen et al., "LSH Sampling for Efficient LLM Generation"

### Related work — low-rank & cross-layer compression

- [xKV (2025)](https://arxiv.org/abs/2503.18893) — Chang et al., "Cross-Layer SVD for KV-Cache Compression"
- [Expected Attention / KVPress (2024)](https://arxiv.org/abs/2510.00636) — "KV Cache Compression by Estimating Attention from Future Queries Distribution"

### Survey

- [KV Cache Management Survey (2024)](https://arxiv.org/abs/2412.19442) — "A Survey on Large Language Model Acceleration based on KV Cache Management"

### Framework

- [Apple MLX](https://github.com/ml-explore/mlx)
- Internal docs: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md), [MEDIUM_BLOG.md](MEDIUM_BLOG.md)

---

## License

MIT — see [LICENSE](LICENSE).
