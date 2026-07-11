<div align="center">

<!-- Replace with your generated cover image -->
<img src="assets/veloxquant.png" alt="VeloxQuant-MLX" width="860" />

<h1>VeloxQuant-MLX</h1>

<p>
  <strong>Fast KV Cache Quantization for Apple Silicon</strong><br/>
  TurboQuant · RVQ · VecInfer · RateQuant · PolarQuant · QJL · SpectralQuant · CommVQ · RaBitQ — in MLX
</p>

<p>
  <a href="https://pypi.org/project/VeloxQuant-MLX/"><img src="https://img.shields.io/pypi/v/VeloxQuant-MLX?style=flat-square&logo=pypi&logoColor=white&color=0078d4" alt="PyPI"/></a>
  <a href="https://pepy.tech/project/VeloxQuant-MLX"><img src="https://img.shields.io/pepy/dt/VeloxQuant-MLX?style=flat-square&logo=python&logoColor=white&color=7c3aed&label=downloads" alt="Downloads"/></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.11+-0078d4?style=flat-square&logo=python&logoColor=white" alt="Python"/></a>
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon%20M1+-black?style=flat-square&logo=apple&logoColor=white" alt="Platform"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-22c55e?style=flat-square" alt="License"/></a>
  <img src="https://img.shields.io/badge/tests-817%2F821%20passing-22c55e?style=flat-square" alt="Tests"/>
  <a href="https://doi.org/10.5281/zenodo.20647305"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20647305-1f6feb?style=flat-square" alt="DOI"/></a>
</p>

<p>
  <a href="https://veloxquant-mlx.netlify.app/"><img src="https://img.shields.io/badge/landing%20page-veloxquant--mlx.netlify.app-7c3aed?style=flat-square" alt="Landing"/></a>
  <a href="CHANGELOG.md"><img src="https://img.shields.io/badge/changelog-0.34.0-64748b?style=flat-square" alt="Changelog"/></a>
  <a href="blogs/metal-kernels.md"><img src="https://img.shields.io/badge/blog-Metal%20kernels%20v1-f97316?style=flat-square" alt="Blog"/></a>
  <a href="blogs/turboquant-metal-kernels.md"><img src="https://img.shields.io/badge/blog-TurboQuant%20Metal%20kernels-f97316?style=flat-square" alt="Blog v2"/></a>
  <a href="https://github.com/sponsors/rajveer43"><img src="https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?style=flat-square&logo=githubsponsors&logoColor=white" alt="GitHub Sponsors"/></a>
</p>

</div>

---

A KV-cache compression library for `mlx_lm` that compresses the Key tensor up to **16× with near-lossless quality** on Apple M-series chips. Ships **thirty-seven compression strategies** — from zero-calibration 1-bit RVQ to RaBitQ (1-bit keys + MSE-b4 values) which achieves **6× full KV compression** and fits **6× more context** in the same RAM budget on Falcon3-7B, through twelve token-eviction caches (SnapKV, StreamingLLM, H2O, TOVA, PyramidKV, SqueezeAttention, ChunkKV, L2Norm's intrinsic key-norm scorer — EMNLP 2024 — Q-Filters' query-agnostic projection scorer, Keyformer's Gumbel-regularized heavy-hitter scorer — MLSys 2024 — MorphKV's recent-window correlation retention — ICML 2025 — and KVzip's context-reconstruction reliance scorer — NeurIPS 2025), a cache-**merging** cache (CaM) that folds evicted tokens into survivors instead of dropping them, three cross-layer methods (XQuant's code reuse, MiniCache's SLERP merge, and xKV's joint shared-subspace SVD across a layer group), and a calibration-free universal-codebook VQ (NSNQuant, NeurIPS 2025) that reshapes K/V onto one fixed Gaussian codebook instead of fitting a codebook to the data, plus a sliding-window quantizer (SKVQ, COLM 2024) that regroups head-dim channels by their statistics and clip-searches each quantization group's range — plus a hand-written Metal compute kernel that makes the VecInfer **quantize** hot path **6.9–14.7× faster** (13× at S=2048) and **98% lighter on peak memory** at the OOM-trigger shape. (The companion dequant kernel is at MLX `mx.take` parity — the speedup is on the quantize path.) Plug it in with three lines; `mlx_lm.generate` runs unchanged.

---

## Numbers that matter

| Metric | Value | Notes |
|---|---|---|
| Max key cache compression | **16×** | VecInfer-1bit, head_dim=128 |
| Metal kernel speedup | **13×** | `quantize_vq` at S=2048 (range 6.9–14.7× over S=128–8192) |
| Peak memory reduction | **98%** | 729 MB → 12 MB, Falcon3-7B shape |
| RVQ-1bit compression | **7.5×** | Near-zero throughput cost |
| FP16 throughput retained | **100%** | Qwen2.5-7B at 16× compression |
| SpectralQuant compression | **5.33×** | per-model measured (Qwen2.5-0.5B / Gemma-4-4B), same bit-width |
| SpectralQuant cosine sim | **+3pp** | over TurboQuant on Qwen2.5-0.5B |
| **RaBitQ full KV compression** | **6×** | 1-bit keys + MSE-b4 values, Falcon3-7B |
| **RaBitQ context at 8 GB** | **~103k tokens** (est.) | KV-only linear extrapolation from measured memory rows; vs ~17k fp16 — 6× more context |
| **CommVQ key compression** | **64×** | RoPE-commutative VQ, D=128, n_cb=4 |
| **KIVI-2bit key compression** | **5.8×** | per-channel keys / per-token values; measured on Llama-3.2-3B, Qwen2.5-7B, Mistral-7B |
| **KIVI-2bit full-KV compression** | **~4×** | incl. fp16 residual window (32 tokens); 100–106% of fp16 throughput |
| Production models validated | **12** | Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon |

---

## Table of contents

1. [Installation](#installation)
2. [Quickstart](#quickstart)
3. [Method library](#method-library) — all 34 methods at a glance
4. [Metal kernels](#metal-kernels--new-in-051)
5. [Benchmark results](#benchmark-results)
6. [Algorithm guide](#algorithm-guide) — pick a quantizer by workload
7. [What's inside](#whats-inside)
8. [Architecture](#architecture)
9. [CLI](#cli)
10. [Development](#development)
11. [Blog posts](#blog-posts)
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

## Method library

All 37 methods share the same 3-line integration (`method="<id>"` in `KVCacheConfig`).
Each links to its full page — mechanism, config, evidence, and honest limitations — on
the [documentation site](https://veloxquant-mlx.netlify.app/docs/algorithms/overview).
A few representative methods have runnable examples in [Quickstart](#quickstart) above;
the rest follow the identical pattern.

### Quantization — compress every token

| Method | `method=` | What it does | New in |
|---|---|---|---|
| [TurboQuant RVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rvq) | `turboquant_rvq` | Residual VQ, zero calibration — **the default** (7.5× @ 1-bit) | — |
| [VecInfer](https://veloxquant-mlx.netlify.app/docs/algorithms/vecinfer) | `vecinfer` | Dual-transform product VQ, Metal-accelerated (16×) | 0.4.0 |
| [SpectralQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/spectral) | `spectral` | Rotate keys into eigenbasis — best quality-per-bit | 0.6.0 |
| [RateQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/ratequant) | *(allocator)* | Per-layer mixed precision via reverse-waterfilling | — |
| [RaBitQ](https://veloxquant-mlx.netlify.app/docs/algorithms/rabitq) | `rabitq` | 1-bit keys + MSE-b4 values — **6× full KV** | 0.7.0 |
| [QJL](https://veloxquant-mlx.netlify.app/docs/algorithms/qjl) | `qjl` | 1-bit JL sketch, simplest/fastest to set up | — |
| [PolarQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/polarquant) | `polar` | Polar-coordinate quant for geometric key distributions | — |
| [CommVQ](https://veloxquant-mlx.netlify.app/docs/algorithms/commvq) | `comm_vq` | RoPE-commutative VQ, exact inner product (ICML 2025) | — |
| [KIVI](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi) | `kivi` | Tuning-free asymmetric 2-bit baseline | 0.8.0 |
| [KIVI-Sink](https://veloxquant-mlx.netlify.app/docs/algorithms/kivi-sink) | `kivi_sink` | Sink-protected low-bit quantization | 0.9.0 |
| [SKVQ-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/skvq) | `skvq` | **Channel reordering + clipped dynamic quant** behind a sliding fp16 window + sink filter (COLM 2024) — permutations from first-chunk stats, no calibration | 0.30.0 |
| [SVDq](https://veloxquant-mlx.netlify.app/docs/algorithms/svdq) | `svdq` | Sub-2-bit keys (~1.25 bit) via prefill SVD | 0.10.0 |
| [Kitty](https://veloxquant-mlx.netlify.app/docs/algorithms/kitty) | `kitty` | Adaptive channel precision, zero calibration | 0.11.0 |
| [KVQuant-NUQ](https://veloxquant-mlx.netlify.app/docs/algorithms/kvquant) | `kvquant` | Non-uniform datatype + outlier isolation | 0.14.0 |
| [NSNQuant-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/nsnquant) | `nsnquant` | Calibration-free **universal-codebook** VQ — NSN + Hadamard reshape K/V to one fixed Gaussian codebook (NeurIPS 2025) | 0.28.0 |
| [ZipCache-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/zipcache) | `zipcache` | Per-token mixed bit-width by key-norm saliency | 0.18.0 |
| [GEAR](https://veloxquant-mlx.netlify.app/docs/algorithms/gear) | `gear` | Error-feedback: low-rank + sparse residual correction | 0.17.0 |
| [CacheGen](https://veloxquant-mlx.netlify.app/docs/algorithms/cachegen) | `cachegen` | Entropy-coded cache — storage win on correlated KV | 0.16.0 |

### Low-rank & cross-layer — compress across dimensions or depth

| Method | `method=` | What it does | New in |
|---|---|---|---|
| [PALU](https://veloxquant-mlx.netlify.app/docs/algorithms/palu) | `palu` | True low-rank latent storage of both K and V | 0.15.0 |
| [XQuant](https://veloxquant-mlx.netlify.app/docs/algorithms/xquant) | `xquant` | Cross-layer code reuse — adjacent layers share codes | 0.12.0 |
| [MiniCache](https://veloxquant-mlx.netlify.app/docs/algorithms/minicache) | `minicache` | Cross-layer SLERP merge — deep layer pairs cost one | 0.16.0 |
| [xKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/xkv) | `xkv` | Cross-layer **shared-subspace** SVD — one basis jointly fit across a layer group | 0.27.0 |
| [AdaKV-proxy](https://veloxquant-mlx.netlify.app/docs/algorithms/adakv) | `adakv` | Per-head adaptive bit budget, layered on KIVI | 0.13.0 |

### Token eviction & merging — drop or merge low-value tokens

| Method | `method=` | What it does | New in |
|---|---|---|---|
| [SnapKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/snapkv) | `snapkv` | Prefill observation-window eviction, once at prefill end | 0.19.0 |
| [StreamingLLM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/streaming_llm) | `streaming_llm` | Sink + recency window, constant memory | 0.20.0 |
| [H2O-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/h2o) | `h2o` | Cumulative attention-mass heavy-hitter eviction | 0.21.0 |
| [TOVA-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/tova) | `tova` | Memoryless current-step attention-weight eviction | 0.22.0 |
| [PyramidKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/pyramidkv) | `pyramidkv` | H2O eviction with a per-layer pyramid budget | 0.23.0 |
| [SqueezeAttention-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/squeeze) | `squeeze` | 2D layer×token data-driven budget eviction | 0.24.0 |
| [ChunkKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/chunkkv) | `chunkkv` | Chunk-level eviction (`chunk_size=1` == H2O) | 0.25.0 |
| [CaM-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/cam) | `cam` | Cache **merging** — merge evicted tokens, don't drop (`cam_merge=drop` == H2O) | 0.26.0 |
| [L2Norm-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/knorm) | `knorm` | Intrinsic key-norm eviction — low norm ⇒ important (EMNLP 2024); zero per-step scoring cost, path-independent | 0.29.0 |
| [Q-Filters-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/qfilters) | `qfilters` | Query-agnostic projection eviction — score by projection onto a frozen per-head key-SVD direction (preprint); sign-ambiguous, path-dependent | 0.31.0 |
| [Keyformer-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/keyformer) | `keyformer` | Gumbel-regularized heavy-hitter eviction (MLSys 2024) — H2O's accumulator plus frozen Gumbel noise that rescues late-rising tokens; `keyformer_tau=0` == H2O | 0.32.0 |
| [MorphKV-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/morphkv) | `morphkv` | Recent-window correlation retention (ICML 2025) — ranks stored tokens by a sliding window of recent attention, dropping stale early heavy-hitters; `morphkv_window=1` == TOVA | 0.33.0 |
| [KVzip-adapted](https://veloxquant-mlx.netlify.app/docs/algorithms/kvzip) | `kvzip` | Context-reconstruction reliance eviction (NeurIPS 2025) — keeps the KV pairs the model most relies on to reconstruct its own context (query-agnostic); `kvzip_probe=latest` == TOVA | 0.34.0 |

> Every "-adapted" method is an honest adaptation, not a faithful port — the cache
> wrapper sees per-layer K/V but not the model's true query/attention maps, so
> attention-based signals use a key-as-query proxy. Each method's docs page states its
> specific limitations plainly.


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

Read the full writeup: [blogs/metal-kernels.md](blogs/metal-kernels.md)

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

Picking a **quantizer** by measured bits, compression, and quality — a numbers-first
companion to the [Method library](#method-library) inventory above (which also covers
the eviction and low-rank families).

| Method | Bits/dim | Compression | Quality (cosine) | Calibration | Best for |
|---|---|---|---|---|---|
| `turboquant_mse` | b | ~9× @ 2b | 0.86 @ 3b | None | Lowest overhead at 3–4 bit |
| `turboquant_prod` | b | ~9× @ 2b | 0.95 @ 4b | None | Unbiased IP estimator at 3–4 bit |
| **`turboquant_rvq` @ b=1** | **2** | **7.5×** | **0.92** | **None** | **Default — full output on all 12 tested models** |
| `turboquant_rvq` @ b=2 | 4 | 3.9× | 0.98 | None | 2-bit with near-lossless quality |
| `turboquant_rvq` + RateQuant | 1.5 avg | 5.2× | ≈0.96 | 1.6s | Heterogeneous layer sensitivity |
| **`vecinfer` @ 1-bit** | **1** | **16×** | model-dependent | Codebook | **Max compression, strong-GQA models** |
| **`spectral` @ b=3** | **3** | **5.33×** | **0.91 (Qwen2.5)** | **~5s once** | **Best quality-per-bit, any model** |
| **`comm_vq`** | **1 (uint8 idx)** | **64× keys** | RoPE-exact | EM training | **RoPE-compatible VQ, ICML 2025** |
| **`rabitq` keys + MSE-b4 vals** | **1 + 4** | **6× full KV** | approx | IVF fit | **Max context length, same RAM** |
| `polar` | b×levels | varies | medium | None | Geometric key distributions |
| `qjl` | 1 | ~16× | 0.62 | None | Ranking-only retrieval, extreme compression |

**Quick decision:**
- No calibration, best default → **`turboquant_rvq` b=1**
- Max compression, Qwen2.5/Gemma → **`vecinfer` 1-bit**
- Best quality at moderate compression → **`spectral` b=3** (requires ~5s calibration)
- Heterogeneous layers (sens. ratio >2×) → **RateQuant** on top of RVQ
- 2-bit, near-lossless → **`turboquant_rvq` b=2**
- **Max context length, fixed RAM** → **`rabitq` keys + MSE-b4 values** (6× full KV)
- **RoPE-compatible exact VQ** → **`comm_vq`** (ICML 2025, 64× key compression)

---

## What's inside

| Module | Purpose |
|---|---|
| [`veloxquant_mlx/spectral/spectral_quant`](veloxquant_mlx/spectral/spectral_quant.py) | `SpectralQuantizer` — eigenvector rotation + signal/noise codebooks, b=3 |
| [`veloxquant_mlx/spectral/calibrate`](veloxquant_mlx/spectral/calibrate.py) | `calibrate_spectral_rotation`, `calibrate_from_vectors`, on-disk rotation cache |
| [`veloxquant_mlx/spectral/bit_allocator`](veloxquant_mlx/spectral/bit_allocator.py) | `water_fill_bits` — water-filling bit allocation per eigenvalue |
| [`veloxquant_mlx/spectral/participation_ratio`](veloxquant_mlx/spectral/participation_ratio.py) | `compute_participation_ratio`, `compute_spectral_gap` |
| [`veloxquant_mlx/quantizers/rabitq`](veloxquant_mlx/quantizers/rabitq.py) | `RaBitQQuantizer` — IVF + randomised Hadamard + 1-bit sign packing + Metal Hamming search |
| [`veloxquant_mlx/quantizers/comm_vq`](veloxquant_mlx/quantizers/comm_vq.py) | `CommVQQuantizer` — RoPE-commutative residual VQ, commutativity projection in EM M-step |
| [`veloxquant_mlx/metal/_rabitq`](veloxquant_mlx/metal/_rabitq.py) | `rabitq_hamming_score` — Metal XOR+popcount Hamming distance kernel |
| [`veloxquant_mlx/metal/_comm_vq`](veloxquant_mlx/metal/_comm_vq.py) | `comm_vq_decode_metal` — fused centroid gather + RoPE Metal kernel |
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
# Full test suite (includes Metal parity tests)
pytest veloxquant_mlx/tests/ -v

# 2-bit improvement validation — fast synthetic run
python test_2bit_improvements.py

# Generate optimization-journey figure
python scripts/plot_optimization_journey.py
```

Contributions welcome — please open an issue first for anything beyond a small bugfix. See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## Blog posts

All blog posts live in the [`blogs/`](blogs/) directory and are published at
**https://veloxquant-mlx.netlify.app/docs/blog/**.

| File | Description | Live |
|---|---|---|
| [`blogs/overview.md`](blogs/overview.md) | High-level overview of VeloxQuant-MLX and its goals | [↗](https://veloxquant-mlx.netlify.app/docs/blog/overview) |
| [`blogs/10-model-study.md`](blogs/10-model-study.md) | End-to-end benchmark study across 10 production models | [↗](https://veloxquant-mlx.netlify.app/docs/blog/10-model-study) |
| [`blogs/hands-on.md`](blogs/hands-on.md) | Hands-on tutorial: compressing your first model | [↗](https://veloxquant-mlx.netlify.app/docs/blog/hands-on) |
| [`blogs/kivi.md`](blogs/kivi.md) | Deep dive into the KIVI asymmetric quantization baseline | [↗](https://veloxquant-mlx.netlify.app/docs/blog/kivi) |
| [`blogs/metal-kernels.md`](blogs/metal-kernels.md) | How the Metal compute kernel cuts quantize latency 13× | [↗](https://veloxquant-mlx.netlify.app/docs/blog/metal-kernels) |
| [`blogs/results.md`](blogs/results.md) | Detailed benchmark results and analysis | [↗](https://veloxquant-mlx.netlify.app/docs/blog/results) |
| [`blogs/tensorops-research.md`](blogs/tensorops-research.md) | TensorOps research notes and findings | [↗](https://veloxquant-mlx.netlify.app/docs/blog/tensorops-research) |
| [`blogs/turboquant-metal-kernels.md`](blogs/turboquant-metal-kernels.md) | TurboQuant + Metal kernels: combined writeup | [↗](https://veloxquant-mlx.netlify.app/docs/blog/turboquant-metal-kernels) |

---

## References

<details>
<summary>Papers implemented in this library</summary>

- [RaBitQ (SIGMOD 2024)](https://arxiv.org/abs/2402.02855) — Gao et al., "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound for Approximate Nearest Neighbor Search" — 1-bit randomised Hadamard quantization with formal error guarantees
- [Ascend-RaBitQ (2026)](https://arxiv.org/abs/2605.16007) — He et al., "Ascend-RaBitQ: Heterogeneous NPU-CPU Acceleration of Billion-Scale Similarity Search with 1-bit Quantization" — heterogeneous pipeline inspiration for key+value joint compression
- [CommVQ (ICML 2025)](https://arxiv.org/abs/2506.18879) — Apple ML Research, "CommVQ: Commutative Vector Quantization for KV Cache Compression" — RoPE-commutative additive codebook VQ
- [SpectralQuant (2026)](https://arxiv.org/abs/2506.xxxxx) — "3% Is All You Need: Breaking TurboQuant's Compression Limit via Spectral Structure" — eigenvector PCA rotation + signal/noise codebooks, 5.95× at higher quality than TurboQuant
- [TurboQuant (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al., "Online Vector Quantization with Near-optimal Distortion Rate"
- [RateQuant (2025)](https://arxiv.org/abs/2605.06675) — "RateQuant: Mixed-Precision KV Cache Quantization via Rate-Distortion Theory"
- [VecInfer (2024)](https://arxiv.org/abs/2510.06175) — Yao et al., "Efficient LLM Inference with Low-Bit KV Cache via Outlier-Suppressed Vector Quantization"
- [PolarQuant (AISTATS 2026)](https://arxiv.org/abs/2502.02617) — "PolarQuant: Quantizing KV Caches with Polar Transformation"
- [QJL (2024)](https://arxiv.org/abs/2406.03482) — Zandieh et al., "QJL: 1-Bit Quantized JL Transform for KV Cache Quantization"
- [PALU (ICLR 2025)](https://arxiv.org/abs/2407.21118) — Chang et al., "Palu: Compressing KV-Cache with Low-Rank Projection" — group-head low-rank projection of keys and values (true latent storage)
- [CacheGen (SIGCOMM 2024)](https://arxiv.org/abs/2310.07240) — Liu et al., "CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving" — entropy coding of the quantized KV via token-wise locality
- [MiniCache (NeurIPS 2024)](https://arxiv.org/abs/2405.14366) — Liu et al., "MiniCache: KV Cache Compression in Depth Dimension for Large Language Models" — cross-layer SLERP merge of adjacent layers' KV directions
- [GEAR (arXiv:2403.05527)](https://arxiv.org/abs/2403.05527) — Kang et al., "GEAR: An Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of LLM" — error feedback: low-rank residual + sparse outlier correction over a base quantizer
- [ZipCache (NeurIPS 2024)](https://arxiv.org/abs/2405.14256) — He et al., "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification" — per-token mixed bit-width via saliency-adaptive allocation (adapted: key-norm proxy for the attention-score saliency signal)
- [SnapKV (ICLR 2025)](https://arxiv.org/abs/2404.14469) — Yuan et al., "SnapKV: LLM Knows What You are Looking for Before Generation" — token eviction via prefill observation-window attention scoring (adapted: key-as-query proxy for the prompt query vectors)
- [StreamingLLM (ICLR 2024)](https://arxiv.org/abs/2309.17453) — Xiao et al., "Efficient Streaming Language Models with Attention Sinks" — structural positional eviction: first N sinks + rolling recency window; constant-memory streaming (adapted: no attention mask adjustment, no RoPE position-ID remapping)
- [H2O (ICLR 2024)](https://arxiv.org/abs/2306.14048) — Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" — cumulative attention-mass token eviction with sink protection; budget-bounded constant-memory cache (adapted: key-as-query proxy for attention weights, no RoPE remapping)
- [TOVA (arXiv:2401.06104)](https://arxiv.org/abs/2401.06104) — Oren et al., "Transformers are Multi-State RNNs" — memoryless current-step attention-weight token eviction; budget-bounded constant-memory cache, reactive counterpart to H2O's inertial cumulative scoring (adapted: key-as-query proxy for attention weights, no RoPE remapping)
- [PyramidKV (arXiv:2406.02069)](https://arxiv.org/abs/2406.02069) — Cai et al., "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling" — layer-adaptive KV budget (large early, small deep, fixed average) over H2O-style cumulative attention-mass eviction; funnels memory to where attention spreads (adapted: fixed linear budget schedule instead of prefill-entropy allocation, key-as-query proxy, no RoPE remapping)
- [SqueezeAttention (arXiv:2404.04793)](https://arxiv.org/abs/2404.04793) — Wang et al., "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget" — 2D (layer × token) budget: measures each layer's attention concentration and reallocates a fixed total budget over H2O-style cumulative attention-mass eviction; the data-driven sibling of PyramidKV (adapted: cosine-dispersion concentration proxy instead of observed attention maps, one-shot re-budget at prefill, key-as-query proxy, no RoPE remapping)
- [ChunkKV (arXiv:2502.00299)](https://arxiv.org/abs/2502.00299) — Liu et al., "ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference" — chunk-level eviction: partitions the sequence into contiguous chunks and keeps or drops whole chunks by pooled importance, preserving local coherence that token-level eviction shreds; `chunk_size=1` reduces to H2O (adapted: mean-pooled per-token score proxy for attention-over-chunk importance, no layer-wise index reuse, key-as-query proxy, no RoPE remapping)
- [CaM (ICML 2024)](https://proceedings.mlr.press/v235/zhang24n.html) — Zhang et al., "CaM: Cache Merging for Memory-efficient LLMs Inference" (PMLR 235:58840-58850) — cache merging: instead of dropping the evicted token, merges it into the surviving token it most resembles, mitigating the output perturbation that dropping causes; `cam_merge=drop` reduces to H2O (adapted: cosine-similarity merge weight instead of the paper's attention-prominence weight, single nearest-survivor merge, key-as-query proxy, no RoPE remapping)
- [L2-norm eviction (EMNLP 2024, arXiv:2406.11430)](https://arxiv.org/abs/2406.11430) — Devoto, Zhao, Scardapane & Minervini, "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression" — intrinsic key-norm eviction: in trained decoder LMs a low key L2 norm predicts high attention, so the cache keeps the lowest-norm tokens; code at https://github.com/alessiodevoto/l2compress (adapted: uniform budget across heads, no RoPE position remapping, optional recent-window and inverted-scorer extensions off by default)
- [Q-Filters (preprint, arXiv:2503.02812)](https://arxiv.org/abs/2503.02812) — "Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression" — score cached keys by their projection onto a per-head direction that predicts attention (the paper estimates it from the SVD of query vectors) (adapted: the direction is estimated from the SVD of the first observed **keys**, not queries — a documented deviation that recovers the dominant axis but not its sign, so `qfilters_sign` is a real ablation; kept set is path-dependent; uniform budget across heads; no RoPE remapping; recent-window extension off by default)
- [Keyformer (MLSys 2024, arXiv:2403.09054)](https://arxiv.org/abs/2403.09054) — Adnan, Arunkumar, Jain, Nair, Soloveychik & Kamath, "Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference" — a Gumbel-noise regularizer on the eviction score that stops "late riser" tokens (low early attention, high later) from being greedily pruned before they recover; code at https://github.com/d-matrix-ai/keyformer-llm (adapted: H2O's key-as-query proxy accumulator for the base score; the paper's annealed, redrawn Gumbel schedule replaced by a **frozen deterministic per-position** Gumbel draw seeded by `keyformer_seed`, since a cache has no trustworthy global step — `keyformer_tau=0` collapses onto H2O-adapted bit-for-bit; uniform budget/tau across heads; no RoPE remapping; recent-window extension off by default)
- [MorphKV (ICML 2025, arXiv:2503.00979)](https://arxiv.org/abs/2503.00979) — Ghadia, Kumar, Jain, Nair & Das, "Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in LLMs" — a constant-size retention rule that ranks stored tokens by correlation with the attention pattern of a **sliding window of recent tokens**, eliminating the "early-token bias" of cumulative (H2O) scoring (adapted: key-as-query proxy for the recent-window attention, since a cache never sees the true query; retention recomputed each step from the live keep set and last `morphkv_window` keys, not the paper's exact refresh cadence — `morphkv_window=1` collapses onto TOVA-adapted bit-for-bit, no H2O collapse claimed; uniform budget/window across heads; no RoPE remapping; the paper's accuracy/memory numbers are the paper's on trained models, not reproduced here)
- [KVzip (NeurIPS 2025 Oral, arXiv:2505.23416)](https://arxiv.org/abs/2505.23416) — Kim, Kim, Kwon, Lee, Yun & Song, "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction" ([code](https://github.com/snu-mllab/KVzip)) — a query-agnostic retention rule that scores a KV pair by how much the model relies on it to **reconstruct its own context** and evicts the least-relied-upon (adapted: key-as-reconstruction-probe proxy, since a cache never runs the model to reconstruct text; reconstruction reliance = the max proxy-attention a stored key receives across the probe rows, recomputed each step from the live keep set — `kvzip_probe="latest"` collapses onto TOVA-adapted bit-for-bit, no H2O collapse claimed; no head-level context-independent scoring; uniform budget/probe across heads; no RoPE remapping; the paper's 3–4×-reduction / ~2×-decode / negligible-loss numbers are the paper's on trained models up to 170K tokens, not reproduced here)
- [SKVQ (COLM 2024, arXiv:2405.06219)](https://arxiv.org/abs/2405.06219) — Duanmu, Yuan, Li, Duan, Zhang & Lin, "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models" — channel reordering (group like-range channels so per-token group min/max stays tight) + clipped dynamic quantization (per-group searched clip factor) behind a sliding fp16 window with an attention-sink filter; code at https://github.com/cat538/SKVQ (adapted: offline KMeans/attention-MSE calibration replaced by first-flushed-chunk statistics and per-group reconstruction-MSE clip search; explicit runtime permutation instead of weight fusion; integer bit-widths and fp16 metadata instead of 1.5-bit packing and FP8)
- [NSNQuant (NeurIPS 2025, arXiv:2505.18231)](https://arxiv.org/abs/2505.18231) — Son, Choi & Yoo, "NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache" — calibration-free universal-codebook VQ: a token-wise Normalize / channel-wise Shift / token-wise Normalize transform plus Hadamard rotation aligns K/V token distributions with the standard normal, so a single codebook built offline from synthetic Gaussian samples quantizes any model at 1–2 bits/element (adapted: post-RoPE keys, explicit value Hadamard, spherical-k-means-only codebook without gradient fine-tune, fp16 metadata without double quantization)
- [xKV (arXiv:2503.18893)](https://arxiv.org/abs/2503.18893) — Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang, Li, Ceze, Wu & Abdelfattah, "xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction" — cross-layer shared-subspace compression: jointly factorizes a fixed-size group of layers' stacked key matrices into one shared SVD basis (motivated by CKA showing dominant subspaces align across nearby layers), amortizing the basis storage cost across every member of the group (adapted: fixed contiguous grouping instead of CKA-validated grouping, no "Selective Reconstruction" decode-time optimization, single-bit-width latent quantization, keys only)

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
- [SqueezeAttention (2024)](https://arxiv.org/abs/2404.04793) — Wang et al., "2D Management of KV-Cache in LLM Inference via Layer-wise Optimal Budget"
- [RocketKV (ICML 2025)](https://arxiv.org/abs/2502.14051) — Behnam et al., "Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression"
- [MagicPIG (ICLR 2025 Spotlight)](https://arxiv.org/abs/2410.16179) — Chen et al., "LSH Sampling for Efficient LLM Generation"

**Low-rank & cross-layer:**
- [KVPress (2024)](https://arxiv.org/abs/2510.00636) — "KV Cache Compression by Estimating Attention from Future Queries Distribution"

**Survey:**
- [KV Cache Management Survey (2024)](https://arxiv.org/abs/2412.19442) — "A Survey on LLM Acceleration based on KV Cache Management"

**Framework:** [Apple MLX](https://github.com/ml-explore/mlx)

</details>

---

## Support

VeloxQuant-MLX has passed **15,000+ downloads** on PyPI. It's free, MIT-licensed,
and built nights-and-weekends — if it saves your Mac some memory (or you just
want to see the 38th method land), you can
[**sponsor on GitHub** 💜](https://github.com/sponsors/rajveer43). Stars, issues, and
PRs are equally appreciated.

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
    <a href="blogs/10-model-study.md">Blog: 10-model study</a> ·
    <a href="blogs/metal-kernels.md">Blog: Metal kernels v1</a> ·
    <a href="blogs/turboquant-metal-kernels.md">Blog: TurboQuant Metal kernels</a>
  </sub>
</div>
