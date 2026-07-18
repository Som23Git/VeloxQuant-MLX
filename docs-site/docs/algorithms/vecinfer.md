---
id: vecinfer
title: VecInfer
sidebar_label: VecInfer
slug: /algorithms/vecinfer
---

# VecInfer

VecInfer is VeloxQuant-MLX's highest-throughput algorithm. It combines **product vector quantization** with per-channel smooth scaling and Metal GPU kernels that deliver **13× faster quantization** than naive MLX operations.

:::warning[Apple Silicon required]
VecInfer depends on `vecinfer_quantize_metal` — a Metal kernel that is only available on macOS M-series.
:::

## How it works

1. **Smooth scaling** — Before quantization, each key channel is scaled by `λᵢ = √max|Kᵢ|`. This suppresses outlier channels that would otherwise dominate the codebook, similar to the technique from SmoothQuant.

2. **Walsh-Hadamard transform** — The scaled keys are rotated by a WHT matrix to decorrelate dimensions, making the distribution more uniform across subspaces.

3. **Product VQ (PVQ)** — The head dimension is split into `M` sub-vectors. Each sub-vector is independently quantized by looking up the nearest centroid in a learned sub-codebook. The result is `M` short integer indices per key vector.

4. **Metal-accelerated lookup** — During attention, `vecinfer_encode_decode_metal` and `compute_query_lut` use GPU parallelism to compute query-codebook inner products with a precomputed look-up table, enabling asymmetric distance computation without full dequantization.

## Key properties

| Property | Value |
|---|---|
| Calibration | Codebook training (≈2 min on 128 samples) |
| Key bits | 1–4 (PVQ subspace bits) |
| Value bits | 2–4 |
| Compression ratio | up to 16× |
| Metal kernel speedup | 13× |
| Metal kernels | `vecinfer_quantize_metal`, `vecinfer_encode_decode_metal` |

## Calibration (one-time setup)

VecInfer requires a trained codebook. This is a one-time step — save the artifacts and reuse them across sessions.

```python
import mlx_lm
from veloxquant_mlx.allocators.vecinfer import (
    calibrate_smooth_factors,
    train_codebook,
)
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Collect smooth scaling factors from 64 samples
smooth_factors = calibrate_smooth_factors(model, tokenizer, num_samples=64)

# Train product-VQ codebook on 128 samples
codebook = train_codebook(
    model, tokenizer,
    smooth_factors=smooth_factors,
    num_samples=128,
    num_centroids=256,   # 256 centroids = 8-bit sub-indices
    num_subspaces=8,     # split head_dim into 8 subspaces
)

# Save for reuse
store = NpyArtifactStore("./veloxquant_artifacts/")
store.save("vecinfer_smooth", smooth_factors)
store.save("vecinfer_codebook", codebook)
```

## Inference

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

store = NpyArtifactStore("./veloxquant_artifacts/")
smooth_factors = store.load("vecinfer_smooth")
codebook = store.load("vecinfer_codebook")

config = KVCacheConfig(
    method="vecinfer",
    bits=2,
    codebook=codebook,
    smooth_factors=smooth_factors,
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Summarise the history of calculus in 300 words.",
    max_tokens=400,
    kv_cache=cache,
)
```

## Fused SDPA (maximum throughput)

For maximum throughput, enable the fused scaled dot-product attention kernel. This avoids materialising the dequantized cache in memory:

```python
from veloxquant_mlx.metal.fused_sdpa import patch_mlx_lm_for_fused_sdpa

patch_mlx_lm_for_fused_sdpa(model)  # monkey-patches mlx_lm attention layers

response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1024, kv_cache=cache)
```

:::tip
`patch_mlx_lm_for_fused_sdpa` patches each attention layer once. Call it after loading the model and before the first `generate` call.
:::

## Configuration reference

```python
KVCacheConfig(
    method="vecinfer",
    bits=2,                 # bits per sub-index (determines centroid count)
    value_bits=2,           # value quantization bits
    codebook=codebook,      # trained product codebook (required)
    smooth_factors=smooth_factors,  # per-channel scaling (required)
    num_subspaces=8,        # number of PVQ subspaces. Default: head_dim // 16
    use_fused_sdpa=True,    # use fused Metal SDPA kernel. Default: True
)
```

| Parameter | Type | Required | Description |
|---|---|---|---|
| `codebook` | `ndarray` | Yes | Trained product codebook from `train_codebook()` |
| `smooth_factors` | `ndarray` | Yes | Per-channel smooth scaling from `calibrate_smooth_factors()` |
| `bits` | `int` | No | Sub-index bit width (2=256 centroids, 3=512). Default: 2 |
| `num_subspaces` | `int` | No | Number of PVQ partitions. Default: `head_dim // 16` |
| `use_fused_sdpa` | `bool` | No | Enable fused Metal SDPA. Default: `True` |

## When to use VecInfer

**Use VecInfer when:**
- You have 2 minutes for calibration
- Throughput is the primary objective
- Context lengths are moderate (up to 8k)
- You want the highest compression ratio

**Consider alternatives when:**
- Zero calibration is required → [TurboQuant RVQ](../algorithms/rvq)
- Context exceeds 8k → [SpectralQuant](../algorithms/spectral)
- Per-layer quality targeting is needed → [RateQuant](../algorithms/ratequant)

## Benchmark results

On Llama-3.1-8B at 4096 context, M3 Pro (source: BENCHMARK_RESULTS.md):

| Config | Memory | Compression | Latency vs fp16 |
|---|---|---|---|
| fp16 baseline | 536 MB | 1× | 1.00× |
| VecInfer 2-bit | 33 MB | 16× | 0.98× |
| VecInfer 4-bit | 67 MB | 8× | 0.99× |

## See also

- [Calibration guide](../guides/calibration)
- [Metal kernels guide](../guides/metal-kernels)
- [API — VecInfer allocators](../api/allocators)
- [API — Metal VecInfer](../api/metal-api)
