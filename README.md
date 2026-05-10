# VeloxQuant-MLX

**Fast KV-cache quantization for Apple Silicon — TurboQuant, PolarQuant, RVQ, and QJL in MLX.**

[![PyPI version](https://img.shields.io/badge/pypi-0.3.0-blue.svg)](https://pypi.org/project/VeloxQuant-MLX/)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black.svg)]()
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A drop-in KV-cache replacement for `mlx_lm` that compresses the Key tensor by **3–9×** with near-lossless quality at 4-bit and **functional 2-bit** via Residual Vector Quantization. Validated end-to-end on 8 production models (Llama, Mistral, Falcon, Qwen, Phi, Gemma, DeepSeek-V2-Lite).

```python
from mlx_kv_quant import KVCacheBuilder
import mlx.core as mx, numpy as np

cache = (KVCacheBuilder()
    .with_method("turboquant_rvq")     # new in 0.3.0 — usable 2-bit
    .with_head_dim(128)
    .with_bit_width(inlier=2)
    .build())

rng = np.random.default_rng(0)
for _ in range(1000):
    cache.append(mx.array(rng.standard_normal(128).astype(np.float16)),
                 mx.array(rng.standard_normal(128).astype(np.float16)))

print(f"{cache.memory_bytes()/1024:.1f} KB for {len(cache)} tokens")
```

---

## Table of contents

1. [Highlights](#highlights)
2. [Installation](#installation)
3. [Quick start](#quick-start)
4. [What's inside](#whats-inside)
5. [Algorithm guide](#algorithm-guide-which-method-to-pick)
6. [Per-model benchmark results](#per-model-benchmark-results)
7. [Throughput optimization journey](#throughput-optimization-journey)
8. [Architecture](#architecture)
9. [CLI](#cli)
10. [Development](#development)
11. [References](#references)

---

## Highlights

- **RVQ 2-bit (new)** — two-pass residual quantization brings 2-bit cosine from **0.69 → 0.98**, finally making 2-bit usable for real inference.
- **End-to-end fp16 throughput parity** on Mistral 7B (22.3 vs 22.1 tok/s) and 92% on Qwen3 4B after the 0.3.0 throughput optimizations.
- **Three quantizers**, one interface — `turboquant_prod`, `turboquant_mse`, `turboquant_rvq`, plus `polar` and `qjl`.
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
from mlx_kv_quant import KVCacheBuilder
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

### Drop-in replacement for `mlx_lm` generation

See [`benchmark_core.py`](benchmark_core.py) for the full pattern. Short version:

```python
from benchmark_core import TurboQuantRVQMLXKVCache  # MLX KVCache subclass
import mlx_lm

model, tokenizer = mlx_lm.load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Patch make_cache to inject our compressed cache
def _make_compressed():
    return [TurboQuantRVQMLXKVCache(n_kv_heads=8, head_dim=128, bits=2, seed=i)
            for i in range(model.args.num_hidden_layers)]
model.make_cache = _make_compressed

response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=200)
```

---

## What's inside

| Module | Purpose |
|---|---|
| [`mlx_kv_quant.quantizers.turboquant_prod`](mlx_kv_quant/quantizers/turboquant_prod.py) | Rotation + Lloyd-Max + QJL residual (b-1 + 1 bits) |
| [`mlx_kv_quant.quantizers.turboquant_mse`](mlx_kv_quant/quantizers/turboquant_mse.py) | Rotation + Lloyd-Max only (no residual correction) |
| [`mlx_kv_quant.quantizers.turboquant_rvq`](mlx_kv_quant/quantizers/turboquant_rvq.py) | **NEW** — two-pass scalar RVQ (Gaussian + Laplacian codebooks) |
| [`mlx_kv_quant.quantizers.polarquant`](mlx_kv_quant/quantizers/polarquant.py) | Recursive polar coordinate decomposition |
| [`mlx_kv_quant.quantizers.qjl`](mlx_kv_quant/quantizers/qjl.py) | Pure 1-bit JL sign sketch |
| [`mlx_kv_quant.codebooks`](mlx_kv_quant/codebooks/) | `ScalarCodebook`, Lloyd-Max strategies, **`AdaptiveScalarCodebook`** |
| [`mlx_kv_quant.preconditioners`](mlx_kv_quant/preconditioners/) | `RotationPreconditioner` (QR), `HadamardPreconditioner` (Metal) |
| [`mlx_kv_quant.cache`](mlx_kv_quant/cache/) | `TurboQuantKVCache` standalone, MLX `KVCache` subclasses |
| [`mlx_kv_quant.weight`](mlx_kv_quant/weight/) | `QuantizedLinear` for model weight quantization |
| [`mlx_kv_quant.dsa.bit_pack`](mlx_kv_quant/dsa/bit_pack.py) | Sub-byte index packing |
| [`mlx_kv_quant.outlier`](mlx_kv_quant/outlier/) | Two-stream cache for high-variance channels |

---

## Algorithm guide — which method to pick

| Method | Bits/dim | Per-token storage (d=128) | Quality (cosine) | Best for |
|---|---|---|---|---|
| `turboquant_mse` | b | `b·d/8` + 4 B norm | 0.86 @ 3b, 0.95 @ 4b | Default 3–4 bit, lowest memory overhead |
| `turboquant_prod` | b-1 + 1 | `(b-1)·d/8` + JL signs + 2 norms | 0.86 @ 3b, 0.95 @ 4b | Unbiased IP estimator, slightly higher quality |
| **`turboquant_rvq`** | **2·b** | **`2·b·d/8`** + 2 B norm | **0.98 @ b=2** | **Functional 2-bit** — only method that works at b=2 |
| `polar` | b·levels | varies | medium | Geometric structure, very low bits |
| `qjl` | 1 | `d/8` + 2 B norm | 0.62 @ 1b | Topology-only retrieval, extreme compression |

**Rule of thumb**:
- **3–4 bit, max compression** → `turboquant_mse`
- **3–4 bit, best quality** → `turboquant_prod`
- **2 bit (3.88× key compression with full coherence)** → `turboquant_rvq`
- **1 bit (extreme compression, ranking only)** → `qjl`

---

## Per-model benchmark results

All measurements on **Apple M4 MacBook · 16 GB unified memory · Python 3.12**. Prompt: structured 200-token explanation of relativity. Each model runs fp16 + TurboQuant 2/3/4-bit; v2 runs add RVQ 2-bit.

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
python -m mlx_kv_quant precompute \
    --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --seed 42 \
    --output_dir ./artifacts/
```

Then pass an `NpyArtifactStore` to the builder to load instead of recompute:

```python
from mlx_kv_quant.artifacts import NpyArtifactStore
cache = (KVCacheBuilder()
    .with_method("turboquant_rvq")
    .with_head_dim(128).with_bit_width(inlier=2)
    .with_artifact_store(NpyArtifactStore("./artifacts/"))
    .build())
```

### Benchmark a single configuration

```bash
python -m mlx_kv_quant benchmark \
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
pytest mlx_kv_quant/tests/ -v

# 2-bit improvement validation (synthetic, fast)
python test_2bit_improvements.py

# Generate optimization-journey figure
python scripts/plot_optimization_journey.py
```

---

## References

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — "PolarQuant: Quantizing KV Caches with Polar Transformation"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"
- [Apple MLX](https://github.com/ml-explore/mlx)
- Internal docs: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md), [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md), [MEDIUM_BLOG.md](MEDIUM_BLOG.md)

---

## License

MIT — see [LICENSE](LICENSE).
