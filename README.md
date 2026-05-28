<div align="center">

<!-- Replace with your generated cover image -->
<img src="assets/veloxquant.png" alt="VeloxQuant-MLX" width="860" />

<h1>VeloxQuant-MLX</h1>

<p>
  <strong>Fast KV Cache Quantization for Apple Silicon</strong><br/>
  TurboQuant · RVQ · VecInfer · RateQuant · PolarQuant · QJL — in MLX
</p>

<p>
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/badge/pypi-0.5.1-0078d4?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-0078d4?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon%20M1+-black?style=flat-square&logo=apple&logoColor=white" alt="Platform"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="License"/></a>
  <img src="https://img.shields.io/badge/tests-212%20passing-22c55e?style=flat-square" alt="Tests"/>
</p>

<p>
  <a href="https://veloxquant-mlx.netlify.app/"><img src="https://img.shields.io/badge/landing%20page-veloxquant--mlx.netlify.app-7c3aed?style=flat-square" alt="Landing"/></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.5.1-64748b?style=flat-square" alt="Changelog"/></a>
  <a href="MEDIUM_BLOG_METAL_KERNELS.md"><img src="https://img.shields.io/badge/blog-Metal%20kernels-f97316?style=flat-square" alt="Blog"/></a>
</p>

</div>

---

A KV-cache compression library for `mlx_lm` that compresses the Key tensor up to **16× with near-lossless quality** on Apple M-series chips. Ships six quantization strategies — from zero-calibration 1-bit RVQ to product-VQ with dual outlier suppression — plus hand-written Metal compute kernels that make the hot path **13× faster** and **98% lighter on peak memory** at long context lengths. Plug it in with three lines; `mlx_lm.generate` runs unchanged.

---

## Numbers that matter

| Metric | Value | Notes |
|---|---|---|
| Max key cache compression | **16×** | VecInfer-1bit, head_dim=128 |
| Metal kernel speedup | **13×** | `quantize_vq` at S=2048+ |
| Peak memory reduction | **98%** | 729 MB → 12 MB, Falcon3-7B shape |
| RVQ-1bit compression | **7.5×** | Near-zero throughput cost |
| FP16 throughput retained | **100%** | Qwen2.5-7B at 16× compression |
| Production models validated | **10** | Llama, Mistral, Qwen, Phi, Gemma |

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [RateQuant — per-layer mixed precision](#ratequant--per-layer-mixed-precision)
4. [VecInfer — 16× product VQ](#vecinfer--16-product-vq)
5. [Metal kernels](#metal-kernels--new-in-051)
6. [Benchmark results](#benchmark-results)
7. [Algorithm guide](#algorithm-guide)
8. [What's inside](#whats-inside)
9. [Architecture](#architecture)
10. [CLI](#cli)
11. [Development](#development)
12. [References](#references)

---

## Installation

```bash
pip install VeloxQuant-MLX
```

**Requirements:** Apple Silicon M1+, Python ≥ 3.11, MLX ≥ 0.18, NumPy ≥ 1.26.

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/rajveer43/VeloxQuant-MLX
cd VeloxQuant-MLX
pip install -e ".[dev]"
```

</details>

---

## Quickstart

### RVQ 1-bit — 7.5× compression, no calibration (recommended)

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

### VecInfer 1-bit — 16× compression, Metal kernels auto-detected

```python
import mlx_lm
from veloxquant_mlx import KVCacheConfig, KVCacheFactory
from veloxquant_mlx.allocators.vecinfer import calibrate_smooth_factors, train_codebook

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# One-time offline calibration — save and reuse
smooth   = calibrate_smooth_factors(sample_keys)
codebook = train_codebook(sample_keys_flat, n_centroids=256, sub_dim=8)

config = KVCacheConfig(
    method="vecinfer",
    head_dim=128,
    key_codebook_bits=8,
    key_sub_dim=8,
    smooth_factors=smooth,
    key_codebook=codebook,
    use_metal_kernels=None,   # None=auto-detect, True=require, False=forbid
)
caches = KVCacheFactory.create_for_model(model, config)

response = mlx_lm.generate(model, tokenizer,
    prompt="Write a 5,000-word analysis of the RLHF literature.",
    max_tokens=5000,
    prompt_cache=caches,
)
```

### RateQuant — mixed precision per layer

```python
from veloxquant_mlx import (
    KVCacheBuilder, KVCacheConfig,
    calibrate_layer_sensitivities,
    allocate_bits_ratequant,
)

# Step 1 — 1.6s one-time probe on real activations
weights = calibrate_layer_sensitivities(model, tokenizer)

# Step 2 — closed-form reverse-waterfilling allocation
alloc = allocate_bits_ratequant(weights, target_avg_bits=1.5, beta=3.5)
# e.g. [1, 2, 1, 1, 3, 1, 2, ...]  — one int per layer

# Step 3 — build per-layer caches
config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=alloc, seed=42)
caches = KVCacheBuilder.for_model(model, config)
```

---

## RateQuant — per-layer mixed precision

[RateQuant (arxiv:2605.06675)](https://arxiv.org/abs/2605.06675) allocates more bits to high-sensitivity layers and fewer to low-sensitivity ones via Theorem 2 reverse-waterfilling, with the average held at a user-chosen target. A sensitivity ratio above ~2× indicates measurable gains over uniform allocation.

| Model | Sensitivity ratio | Allocation | Result |
|---|---|---|---|
| Falcon3-7B (28 layers, head_dim=256) | 6.48× | 14 × b=2, 14 × b=1 | **100% fp16** at 5.22× compression |
| Gemma3-4B (34 layers, head_dim=256) | 14.39× | 3 × b=3, 11 × b=2, 20 × b=1 | **91% fp16** at 5.22× compression |

> **What's not yet implemented from the paper:** per-head allocation, gradient-based sensitivity, K/V separation. Per-layer already captures most of the benefit at ≥1.5 bits.

---

## VecInfer — 16× product VQ

[VecInfer (arxiv:2510.06175)](https://arxiv.org/abs/2510.06175) (Yao et al. 2025) applies a **dual transform** to keys before product VQ encoding:

1. **Smooth scaling** — per-channel `λ = √(max|K|)` suppresses outlier magnitudes
2. **Walsh-Hadamard rotation** — spreads energy uniformly across all dims
3. **K-means product VQ** — encode sub-vectors against a calibrated codebook

The inverse transform is absorbed into queries so `q @ K.T` is preserved exactly. At 1 bit/elem a 128-dim key becomes 16 bytes instead of 256 — **16× compression**.

**Standout result:** Qwen2.5-7B VecInfer-1bit *exceeds* fp16 throughput at 16× compression, likely due to its strong GQA ratio (28q/4kv heads).

---

## Metal kernels — new in 0.5.1

The VecInfer `quantize_vq` hot path is now a 30-line Metal Shading Language shader, JIT-compiled by `mx.fast.metal_kernel` on first use. Same Python API — no changes required.

<div align="center">
  <img src="figures/metal/summary.png" alt="Metal kernel benchmark — quantize latency, speedup, and peak memory" width="820"/>
  <br/><sub>Benchmarked on Apple Silicon GPU. Left: quantize latency. Center: speedup factor. Right: peak memory.</sub>
</div>

<br/>

| Metric | Pure MLX | Metal kernel | Delta |
|---|---|---|---|
| Quantize latency (S=8192) | 228 ms | **15.6 ms** | **14.7×** faster |
| Peak memory (Falcon3-7B shape) | 729 MB | **12 MB** | **98%** reduction |
| API change required | — | None | `use_metal_kernels=None` auto-detects |

**Why the memory win:** the `[N, n_centroids, sub_dim]` diff tensor is never materialised — the argmin accumulator lives entirely in thread-local registers.

**Honest caveat:** the kernel pays a ~50–200 µs launch overhead per call. On tiny models (SmolLM2-135M, ~60 launches/token) that overhead can exceed the savings. Built for the regime that needs it: 7B+ models at realistic context lengths.

<details>
<summary>The full 30-line Metal kernel</summary>

```metal
// One thread per sub-vector. Argmin lives in registers — no diff tensor.
uint vec_idx  = thread_position_in_grid.x;
uint N_total  = x_shape[0];
if (vec_idx >= N_total) { return; }

uint n_centroids = codebook_shape[0];
uint sub_dim     = codebook_shape[1];
uint x_base      = vec_idx * sub_dim;

float best_dist = INFINITY;
uint  best_idx  = 0;

for (uint c = 0; c < n_centroids; ++c) {
    uint  cb_base = c * sub_dim;
    float dist    = 0.0f;
    for (uint i = 0; i < sub_dim; ++i) {
        float d = float(x[x_base + i]) - float(codebook[cb_base + i]);
        dist += d * d;
    }
    if (dist < best_dist) { best_dist = dist; best_idx = c; }
}

out[vec_idx] = best_idx;
```

</details>

Read the full writeup: [MEDIUM_BLOG_METAL_KERNELS.md](MEDIUM_BLOG_METAL_KERNELS.md)

---

## Benchmark results

### 10-model comparative study — VecInfer vs RVQ (v0.5.0)

<div align="center">
  <img src="figures/vecinfer/_summary/cross_model_comparison.png" alt="Cross-model comparison — VecInfer vs RVQ-1bit across 10 models" width="820"/>
  <br/><sub>End-to-end <code>mlx_lm.generate</code> · 200-token prompt · 120-token generation · Apple M-series unified memory</sub>
</div>

<br/>

**Compression ratio:**

| Model | RVQ-1bit | VecInfer-1bit |
|---|---|---|
| SmolLM2-135M | 7.1× | **16×** |
| Llama-3.2-1B | 7.1× | **16×** |
| Llama-3.2-3B | 7.5× | **16×** |
| Llama-3.1-8B | 7.5× | **16×** |
| Mistral-7B | 7.5× | **16×** |
| Qwen2.5-7B | 7.5× | **16×** |
| Qwen3-8B | 7.5× | **16×** |
| Phi-4 | 7.5× | **16×** |
| Falcon3-7B | 7.8× | **16×** |
| gemma-3-4b | 7.8× | **16×** |

**Throughput (tok/s):**

| Model | fp16 | RVQ-1bit | VecInfer-1bit |
|---|---|---|---|
| SmolLM2-135M | 250.4 | 188.5 | 175.8 |
| Llama-3.2-1B | 105.4 | **104.3** | 91.2 |
| Llama-3.2-3B | 47.6 | **46.2** | 40.2 |
| Llama-3.1-8B | 20.5 | **20.6** | 19.6 |
| Mistral-7B | 23.6 | **22.8** | 9.8 |
| Qwen2.5-7B | 21.0 | 20.7 | **21.5** ⬆ exceeds fp16 at 16× |
| Qwen3-8B | 20.3 | **19.6** | 2.4 |
| Phi-4 | 10.4 | 8.1 | 4.0 |
| Falcon3-7B | 17.3 | **21.7** | 17.0 |
| gemma-3-4b | 26.0 | 24.2 | **22.6** |

> **RVQ-1bit** is the safe default — within 5% of fp16 on most 7–8B models with zero calibration. **VecInfer-1bit** wins on memory (always 16×) and throughput on strong-GQA models (Qwen2.5, Gemma).

<details>
<summary>Throughput optimisation journey (v0.3.0)</summary>

<div align="center">
  <img src="figures/updated_tests/optimization_journey.png" alt="Throughput optimisation journey" width="700"/>
</div>

Four sequential changes to lift quantized throughput to fp16 parity:

| Stage | Mistral-7B RVQ-2bit | Qwen3-4B RVQ-2bit |
|---|---|---|
| 0. Original (per-head Python loop) | 17.7 tok/s | 24.8 tok/s |
| 1. Batch heads `(B,H,S,D) → (B·H·S,D)` | 21.5 tok/s | 34.0 tok/s |
| 2. Hadamard rotation by default | 20.0 tok/s | — |
| 3. Boundary-sum quantize (replaces argmin) | 22.4 tok/s | — |
| 4. Drop redundant fp32↔fp16 casts | **22.3 tok/s** | **36.0 tok/s** |

Full writeup: [OPTIMIZATION_FINDINGS.md](OPTIMIZATION_FINDINGS.md)

</details>

<details>
<summary>RateQuant V2 mixed-precision results (v0.3.5)</summary>

Per-layer allocation at target b̄=1.5, measured on Apple M4 24 GB.

| Model | fp16 | RVQ-1bit | RVQ + RateQuant V2 | Sens. ratio |
|---|---|---|---|---|
| Falcon3-7B | 22.9 | 23.1 (101%) | **22.8 (100%)** at 5.22× | 6.48× |
| Gemma3-4B | 39.8 | 37.8 (95%) | **36.3 (91%)** at 5.22× | 14.39× |

Source figures: [`figures/2026-05-16/`](figures/2026-05-16/)

</details>

<details>
<summary>RVQ 1-bit 8-model sweep (v0.3.4)</summary>

All on Apple M4 MacBook 16/24 GB. Prompt: 200-token explanation of relativity.

| Model | fp16 tok/s | RVQ-1bit tok/s | vs fp16 |
|---|---|---|---|
| Mistral-7B v0.3 | 23.3 | 22.2 | 95% |
| Falcon3-7B | 24.0 | 23.1 | 96% |
| Phi-4 | 11.9 | 11.8 | **99%** |
| Qwen3-4B | 40.2 | 34.3 | 85% |
| Qwen3-8B | 20.5 | 21.1 | **103%** |
| Llama-3.1-8B | 22.0 | 21.5 | 98% |
| Gemma3-4B | 32.5 | 30.5 | 94% |

Source figures: [`figures/outlier_token_ratequant/`](figures/outlier_token_ratequant/)

</details>

---

## Algorithm guide

| Method | Bits/dim | Compression | Quality (cosine) | Calibration | Best for |
|---|---|---|---|---|---|
| `turboquant_mse` | b | ~9× @ 2b | 0.86 @ 3b | None | Lowest overhead at 3–4 bit |
| `turboquant_prod` | b | ~9× @ 2b | 0.95 @ 4b | None | Unbiased IP estimator at 3–4 bit |
| **`turboquant_rvq` @ b=1** | **2** | **7.5×** | **0.92** | **None** | **Default — full output on all 10 tested models** |
| `turboquant_rvq` @ b=2 | 4 | 3.9× | 0.98 | None | 2-bit with near-lossless quality |
| `turboquant_rvq` + RateQuant | 1.5 avg | 5.2× | ≈0.96 | 1.6s | Heterogeneous layer sensitivity |
| **`vecinfer` @ 1-bit** | **1** | **16×** | model-dependent | Codebook | **Max compression, strong-GQA models** |
| `polar` | b×levels | varies | medium | None | Geometric key distributions |
| `qjl` | 1 | ~16× | 0.62 | None | Ranking-only retrieval, extreme compression |

**Quick decision:**
- No calibration, best default → **`turboquant_rvq` b=1**
- Max compression, Qwen2.5/Gemma → **`vecinfer` 1-bit**
- Heterogeneous layers (sens. ratio >2×) → **RateQuant** on top of RVQ
- 2-bit, near-lossless → **`turboquant_rvq` b=2**

---

## What's inside

| Module | Purpose |
|---|---|
| [`veloxquant_mlx/quantizers/turboquant_rvq`](veloxquant_mlx/quantizers/turboquant_rvq.py) | Two-pass scalar RVQ — Gaussian + Laplacian codebooks, b=1/2/3+ |
| [`veloxquant_mlx/quantizers/turboquant_prod`](veloxquant_mlx/quantizers/turboquant_prod.py) | Rotation + Lloyd-Max + QJL residual (b-1 + 1 bits) |
| [`veloxquant_mlx/quantizers/turboquant_mse`](veloxquant_mlx/quantizers/turboquant_mse.py) | Rotation + Lloyd-Max, no residual correction |
| [`veloxquant_mlx/quantizers/polarquant`](veloxquant_mlx/quantizers/polarquant.py) | Recursive polar coordinate decomposition |
| [`veloxquant_mlx/quantizers/qjl`](veloxquant_mlx/quantizers/qjl.py) | Pure 1-bit Johnson-Lindenstrauss sign sketch |
| [`veloxquant_mlx/cache/vecinfer_cache`](veloxquant_mlx/cache/vecinfer_cache.py) | `VecInferKVCache` — smooth + Hadamard + product VQ |
| [`veloxquant_mlx/cache/turboquant_rvq_cache`](veloxquant_mlx/cache/turboquant_rvq_cache.py) | `TurboQuantRVQKVCache` — mlx_lm-compatible wrapper |
| [`veloxquant_mlx/allocators/vecinfer`](veloxquant_mlx/allocators/vecinfer.py) | `calibrate_smooth_factors`, `train_codebook`, `quantize_vq` |
| [`veloxquant_mlx/allocators`](veloxquant_mlx/allocators/) | `allocate_bits_ratequant`, `calibrate_layer_sensitivities` |
| [`veloxquant_mlx/metal`](veloxquant_mlx/metal/) | Hand-written Metal MSL kernels, JIT via `mx.fast.metal_kernel` |
| [`veloxquant_mlx/preconditioners`](veloxquant_mlx/preconditioners/) | `RotationPreconditioner` (QR), `HadamardPreconditioner` |
| [`veloxquant_mlx/observers`](veloxquant_mlx/observers/) | `DistortionObserver`, `LatencyObserver`, `MemoryObserver`, `KeyNormObserver` |
| [`veloxquant_mlx/codebooks`](veloxquant_mlx/codebooks/) | `ScalarCodebook`, Lloyd-Max strategies, `AdaptiveScalarCodebook` |
| [`veloxquant_mlx/dsa/bit_pack`](veloxquant_mlx/dsa/bit_pack.py) | Sub-byte index packing |
| [`veloxquant_mlx/outlier`](veloxquant_mlx/outlier/) | Two-stream cache for high-variance channels |
| [`veloxquant_mlx/weight`](veloxquant_mlx/weight/) | `QuantizedLinear` for model weight quantization |

---

## Architecture

<details>
<summary>Pipeline diagrams & design patterns</summary>

**TurboQuantRVQ pipeline:**
```
x (fp16, batch × d)
     │
Rotate (Π)
     │
Stage-1 quantize  (Gaussian Lloyd-Max, b bits)  →  idx₁
     │
Compute residual  r₁ = y − ŷ₁
     │
Stage-2 quantize  (Laplacian Lloyd-Max, b bits) →  idx₂
     │
EncodedVector(idx₁, idx₂)
     │
Decode: ŷ = ŷ₁ + ŷ₂  →  unrotate
```

**VecInfer pipeline:**
```
x (fp16, B × H × S × D)
     │
Smooth scale  (λᵢ = √max|Kᵢ|, per channel)
     │
Walsh-Hadamard rotation  O(d log d)
     │
K-means product VQ  (sub-vectors against codebook)
     │
Packed indices  →  16× smaller than fp16 keys
```

**Design patterns used (10):** Abstract Base Classes, Factory, Chain of Responsibility, Builder, Strategy, Registry + Plugin, Composite, Observer, DAO, Custom DSA (RingBuffer, MaxHeap, BitPackBuffer, VoronoiTree).

</details>

---

## CLI

```bash
# Precompute rotation matrices, JL matrices, codebooks
python -m veloxquant_mlx precompute \
    --head_dim 128 --bits 1 2 3 4 --jl_dim 128 --seed 42 \
    --output_dir ./artifacts/

# Synthetic benchmark — single config
python -m veloxquant_mlx benchmark \
    --method turboquant_rvq --head_dim 128 --bits 2 --seq_len 1000

# End-to-end model benchmarks
python benchmark_scripts/benchmark_vecinfer.py   # VecInfer 10-model sweep
python benchmark_scripts/run_outlier_ratequant.py # RateQuant mixed-precision
```

Load precomputed artifacts to skip re-computation at runtime:

```python
from veloxquant_mlx.artifacts import NpyArtifactStore

cache = (KVCacheBuilder()
    .with_method("turboquant_rvq")
    .with_head_dim(128).with_bit_width(inlier=2)
    .with_artifact_store(NpyArtifactStore("./artifacts/"))
    .build())
```

---

## Development

```bash
# Full test suite (212 tests, includes 7 Metal parity tests)
pytest veloxquant_mlx/tests/ -v

# 2-bit improvement validation — fast synthetic run
python test_2bit_improvements.py

# Generate optimization-journey figure
python scripts/plot_optimization_journey.py
```

Contributions welcome — please open an issue first for anything beyond a small bugfix. See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## References

<details>
<summary>Papers implemented in this library</summary>

- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [RateQuant (2025)](https://arxiv.org/abs/2605.06675) — "RateQuant: Mixed-Precision KV Cache Quantization via Rate-Distortion Theory"
- [VecInfer (2024)](https://arxiv.org/abs/2510.06175) — Yao et al., "Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization"
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — "PolarQuant: Quantizing KV Caches with Polar Transformation"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"

</details>

<details>
<summary>Related work</summary>

**Quantization:**
- [KIVI (ICML 2024)](https://arxiv.org/abs/2402.02750) — Liu et al., "A Tuning-Free Asymmetric 2-Bit Quantization for KV Cache"
- [KVQuant (NeurIPS 2024)](https://arxiv.org/abs/2401.18079) — Hooper et al., "Towards 10 Million Context Length LLM Inference with KV Cache Quantization"
- [Coupled Quantization (NeurIPS 2024)](https://arxiv.org/abs/2405.03917) — Zhang et al., "KV Cache is 1 Bit Per Channel"
- [KVTuner (ICML 2025)](https://arxiv.org/abs/2502.04420) — Li et al., "Sensitivity-Aware Layer-Wise Mixed-Precision KV Cache Quantization"
- [MixKVQ (2024)](https://arxiv.org/abs/2512.19206) — Zhang et al., "Query-Aware Mixed-Precision KV Cache Quantization"
- [FibQuant (2025)](https://arxiv.org/abs/2605.11478) — "Universal Vector Quantization for Random-Access KV-Cache Compression"

**Token eviction & sparse attention:**
- [SnapKV (2024)](https://arxiv.org/abs/2404.14469) — Li et al., "LLM Knows What You are Looking for Before Generation"
- [PyramidKV (2024)](https://arxiv.org/abs/2406.02069) — Cai et al., "Dynamic KV Cache Compression based on Pyramidal Information Funneling"
- [RocketKV (ICML 2025)](https://arxiv.org/abs/2502.14051) — Behnam et al., "Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"
- [MagicPIG (ICLR 2025 Spotlight)](https://arxiv.org/abs/2410.16179) — Chen et al., "LSH Sampling for Efficient LLM Generation"

**Low-rank & cross-layer:**
- [xKV (2025)](https://arxiv.org/abs/2503.18893) — Chang et al., "Cross-Layer SVD for KV-Cache Compression"
- [KVPress (2024)](https://arxiv.org/abs/2510.00636) — "KV Cache Compression by Estimating Attention from Future Queries Distribution"

**Survey:**
- [KV Cache Management Survey (2024)](https://arxiv.org/abs/2412.19442) — "A Survey on LLM Acceleration based on KV Cache Management"

**Framework:** [Apple MLX](https://github.com/ml-explore/mlx)

</details>

---

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
  <sub>Built for Apple Silicon · Engineered for speed · MIT License</sub>
  <br/>
  <sub>
    <a href="https://veloxquant-mlx.netlify.app/">Landing page</a> ·
    <a href="https://github.com/rajveer43/VeloxQuant-MLX/issues">Issues</a> ·
    <a href="MEDIUM_BLOG.md">Blog: 10-model study</a> ·
    <a href="MEDIUM_BLOG_METAL_KERNELS.md">Blog: Metal kernels</a>
  </sub>
</div>
