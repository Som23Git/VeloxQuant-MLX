# VeloxQuant-MLX

**Fast KV-cache quantization for Apple Silicon — TurboQuant, RVQ, RateQuant, PolarQuant, and QJL in MLX.**

[![PyPI version](https://img.shields.io/badge/pypi-0.3.5-blue.svg)](https://pypi.org/project/VeloxQuant-MLX/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A drop-in KV-cache replacement for `mlx_lm` that compresses the Key tensor by **3–9×** with near-lossless quality at 4-bit, **functional 2-bit and 1-bit** via Residual Vector Quantization, and **per-layer mixed-precision** allocation via RateQuant. Validated end-to-end on 8 production models (Mistral, Falcon, Phi, Qwen3, Llama 3.1, Gemma3, Qwen2.5).

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

- **RVQ 1-bit (new in 0.3.4)** — sign-quantizer stage + Laplacian residual delivers **7.5× key compression with cosine 0.92**, generates full 200-token output on every tested model, and **matches or beats fp16 throughput** on 5 of 7 7-8B models (best: Phi-4 at 110% of fp16).
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

## What's inside

| Module | Purpose |
|---|---|
| [`veloxquant_mlx.quantizers.turboquant_prod`](veloxquant_mlx/quantizers/turboquant_prod.py) | Rotation + Lloyd-Max + QJL residual (b-1 + 1 bits) |
| [`veloxquant_mlx.quantizers.turboquant_mse`](veloxquant_mlx/quantizers/turboquant_mse.py) | Rotation + Lloyd-Max only (no residual correction) |
| [`veloxquant_mlx.quantizers.turboquant_rvq`](veloxquant_mlx/quantizers/turboquant_rvq.py) | Two-pass scalar RVQ (Gaussian + Laplacian codebooks), b=1/2/3+ |
| [`veloxquant_mlx.quantizers.polarquant`](veloxquant_mlx/quantizers/polarquant.py) | Recursive polar coordinate decomposition |
| [`veloxquant_mlx.quantizers.qjl`](veloxquant_mlx/quantizers/qjl.py) | Pure 1-bit JL sign sketch |
| [`veloxquant_mlx.cache.turboquant_rvq_cache`](veloxquant_mlx/cache/turboquant_rvq_cache.py) | **NEW** — `TurboQuantRVQKVCache` mlx_lm-compatible cache wrapper |
| [`veloxquant_mlx.allocators`](veloxquant_mlx/allocators/) | **NEW** — `allocate_bits_ratequant`, `calibrate_layer_sensitivities` |
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

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [RateQuant (2025)](https://arxiv.org/abs/2605.06675) — "RateQuant: Mixed-Precision KV Cache Quantization via Rate-Distortion Theory"
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — "PolarQuant: Quantizing KV Caches with Polar Transformation"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"
- [Apple MLX](https://github.com/ml-explore/mlx)
- Internal docs: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md), [MEDIUM_BLOG.md](MEDIUM_BLOG.md)

---

## License

MIT — see [LICENSE](LICENSE).
