---
id: spectral
title: SpectralQuant
sidebar_label: SpectralQuant
slug: /algorithms/spectral
---

# SpectralQuant

SpectralQuant uses **eigenvector rotation** to align the key distribution with the quantizer's assumptions. By rotating keys into the PCA basis, it separates high-variance "signal" dimensions from low-variance "noise" dimensions and applies separate codebooks to each group — achieving high fidelity at long context lengths.

:::warning Apple Silicon required
Requires macOS M-series for Metal kernels and efficient MLX SVD.
:::

## How it works

1. **SVD rotation calibration** — `calibrate_spectral_rotation()` collects key activations from calibration sequences and computes the top-k eigenvectors via SVD. These form the rotation matrix `R`.

2. **Participation ratio** — `compute_participation_ratio()` measures how many dimensions concentrate the variance. A high participation ratio (close to `head_dim`) means keys are uniform; a low ratio means energy is concentrated in a few directions.

3. **Signal/noise split** — Dimensions above the spectral gap (high eigenvalue) are "signal" dimensions. Dimensions below are "noise". The bit allocator assigns more bits to signal dimensions via water-filling.

4. **Rotation + quantize** — At inference, each key is rotated by `R`, then signal dimensions are quantized with a high-bit codebook and noise dimensions with a low-bit codebook.

5. **Inverse rotation at decode** — The dequantized vector is rotated back by `Rᵀ` before being used in attention.

## Key properties

| Property | Value |
|---|---|
| Calibration | SVD rotation ≈3 min on 64 samples |
| Key bits | 2–8 (signal dims), 1–2 (noise dims) |
| Value bits | 2–4 |
| Best for | Long context (8k+), high-fidelity inference |
| Compression | 4–8× |

## Calibration

```python
import mlx_lm
from veloxquant_mlx.spectral.calibrate import (
    calibrate_spectral_rotation,
    save_rotations,
)

model, tokenizer = mlx_lm.load("mlx-community/Qwen2.5-7B-Instruct-4bit")

# Calibrate rotation matrices per layer
rotations = calibrate_spectral_rotation(
    model, tokenizer,
    num_samples=64,
    sequence_length=1024,
    device="gpu",
)

# Save for reuse
save_rotations(rotations, "./veloxquant_artifacts/spectral_rotations/")
print(f"Saved rotations for {len(rotations)} layers")
```

## Inspect spectral properties

```python
from veloxquant_mlx.spectral.participation_ratio import (
    compute_participation_ratio,
    compute_spectral_gap,
)

# For layer 0
pr = compute_participation_ratio(rotations[0].eigenvalues)
gap_idx = compute_spectral_gap(rotations[0].eigenvalues)
print(f"Participation ratio: {pr:.2f} / {rotations[0].head_dim}")
print(f"Signal/noise split at dimension: {gap_idx}")
```

## Inference

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.spectral.calibrate import load_cached_rotations

rotations = load_cached_rotations("./veloxquant_artifacts/spectral_rotations/")

config = KVCacheConfig(
    method="spectral",
    rotations=rotations,
    signal_bits=4,    # bits for high-variance dimensions
    noise_bits=1,     # bits for low-variance dimensions
    value_bits=2,
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Write a comprehensive 5000-word essay on the history of mathematics.",
    max_tokens=5000,
    kv_cache=cache,
)
```

## Water-filling bit allocation

SpectralQuant supports fine-grained bit allocation per dimension via the water-filling algorithm:

```python
from veloxquant_mlx.spectral.bit_allocator import water_fill_bits

# Given eigenvalues and target average bits
per_dim_bits = water_fill_bits(
    eigenvalues=rotations[0].eigenvalues,
    target_avg_bits=3.0,
    min_bits=1,
    max_bits=8,
)
print(per_dim_bits)  # e.g. [8, 8, 6, 4, 2, 1, 1, ...]
```

## Configuration reference

| Parameter | Type | Required | Description |
|---|---|---|---|
| `rotations` | `list[SpectralRotation]` | Yes | Per-layer rotation from `calibrate_spectral_rotation()` or `load_cached_rotations()` |
| `signal_bits` | `int` | No | Bits for high-variance dims. Default: `4` |
| `noise_bits` | `int` | No | Bits for low-variance dims. Default: `1` |
| `value_bits` | `int` | No | Value bit rate. Default: `2` |
| `use_water_filling` | `bool` | No | Use per-dim water-filling allocation. Default: `False` |

## When to use SpectralQuant

**Use SpectralQuant when:**
- Context length exceeds 8k tokens
- Perplexity must be minimised (long sequences amplify accumulation errors)
- The model's key distribution is low-rank (check `compute_participation_ratio`)
- You can spend 3 minutes calibrating

**Consider alternatives when:**
- Zero calibration required → [TurboQuant RVQ](../algorithms/rvq)
- Maximum compression is the goal → [VecInfer](../algorithms/vecinfer)
- Best quality per bit across all lengths → [RateQuant](../algorithms/ratequant)

SpectralQuant's signal/noise split is a **binary** cutoff (participation-ratio
derived) with **uniform bits within each half**. [KVTC-adapted](../algorithms/kvtc)
takes the same local-PCA starting point but replaces the binary cutoff with a
**dynamic-programming-optimal** bit-width *per individual component* (not just
two tiers) — including exactly 0 bits for a component — and adds a
zero-calibration entropy-coding stage on top.

## Benchmark results

On Qwen2.5-7B at 16k context, M3 Max (source: BENCHMARK_RESULTS.md):

| Method | Memory | Perplexity delta |
|---|---|---|
| fp16 baseline | 2048 MB | 0.00 |
| RVQ 2-bit | 256 MB | +0.45 |
| SpectralQuant 4/1-bit | 256 MB | +0.12 |

SpectralQuant achieves **3.75× lower perplexity degradation** at the same memory footprint at 16k context.

## See also

- [Calibration guide](../guides/calibration)
- [API — SpectralQuant](../api/spectral-api)
- [Core concepts — KV cache](../getting-started/concepts)
