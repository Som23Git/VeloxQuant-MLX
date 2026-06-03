---
id: mixed-precision
title: Mixed-Precision Guide
sidebar_label: Mixed Precision
slug: /guides/mixed-precision
---

# Mixed-Precision Guide

Mixed-precision quantization assigns different bit rates to different layers (or tokens) based on their sensitivity to quantization noise. This achieves better accuracy than uniform quantization at the same average memory footprint.

## Why mixed precision?

Not all transformer layers are equally sensitive to quantization. In practice:

- **Early layers** tend to form coarse semantic representations — they can tolerate more compression
- **Middle layers** are often the most sensitive — they compute fine-grained features
- **Final layers** project to vocabulary — moderate sensitivity

Uniform 2-bit quantization wastes bits on insensitive layers and starves sensitive ones. Mixed precision solves this with a per-layer bit assignment that minimises total distortion at a fixed average bit rate.

## Method 1 — RateQuant (automatic allocation)

RateQuant is the fully automatic approach. It probes sensitivity and allocates bits via reverse-waterfilling with no manual tuning.

```python
import mlx_lm
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    fit_distortion_curve,
    allocate_bits_ratequant,
)
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Probe sensitivities (~90 sec)
sensitivities = calibrate_layer_sensitivities(model, tokenizer, num_samples=32)
curves = fit_distortion_curve(sensitivities)

# Allocate: target 2.0 avg bits
bit_allocation = allocate_bits_ratequant(curves, target_bits=2.0, min_bits=1, max_bits=4)

config = KVCacheConfig(method="ratequant", bit_allocation=bit_allocation)
cache = KVCacheBuilder.build(model, config)
```

See the [RateQuant algorithm page](/algorithms/ratequant) for the full reference.

## Method 2 — Manual allocation

If you know which layers are sensitive (from profiling or domain knowledge), you can set the bit allocation manually:

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

# Llama-3.1-8B has 32 layers; layers 8-16 are most sensitive empirically
bit_allocation = {f"layer_{i}": 1 for i in range(32)}
for sensitive_layer in [8, 9, 10, 11, 12, 13, 14, 15, 16]:
    bit_allocation[f"layer_{sensitive_layer}"] = 4

config = KVCacheConfig(
    method="ratequant",
    bit_allocation=bit_allocation,
    value_bits=2,
)
cache = KVCacheBuilder.build(model, config)
```

## Outlier token handling

Some tokens generate anomalously large key norms — "outlier tokens" that carry disproportionate attention weight. Quantizing these at low bit rates degrades quality significantly.

VeloxQuant-MLX detects outlier tokens with `KeyNormObserver` and routes them to a higher-bit quantizer:

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

observer = KeyNormObserver(
    outlier_threshold=3.0,   # tokens > 3σ above mean norm are "outliers"
    window_size=256,          # rolling window for norm statistics
)

config = KVCacheConfig(
    method="ratequant",
    bit_allocation=bit_allocation,
    outlier_observer=observer,
    outlier_bits=8,           # outlier tokens stored at 8 bits
)
cache = KVCacheBuilder.build(model, config)
```

After generation, inspect detected outliers:

```python
report = observer.report()
print(f"Outlier tokens detected: {report.outlier_count}")
print(f"Outlier fraction: {report.outlier_fraction:.2%}")
print(f"Mean outlier norm: {report.mean_outlier_norm:.2f}")
```

## Adaptive bit allocation at runtime

`TurboQuantProdAdaptive` adjusts bit allocation dynamically based on observed distortion during generation:

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProdAdaptive
from veloxquant_mlx.observers.distortion import DistortionObserver

observer = DistortionObserver()
quantizer = TurboQuantProdAdaptive(
    base_bits=2,
    max_bits=4,
    distortion_threshold=0.05,  # increase bits if cosine sim drops below 0.95
    observer=observer,
)
```

## Choosing a target bit rate

| Target avg bits | Memory vs fp16 | Typical perplexity delta |
|---|---|---|
| 4.0 | 4× reduction | < 0.01 |
| 2.0 | 8× reduction | ~0.03–0.08 |
| 1.5 | 10× reduction | ~0.07–0.15 |
| 1.0 | 16× reduction | ~0.20–0.35 |

These are approximate; actual numbers depend on model architecture and calibration quality. Source: BENCHMARK_RESULTS.md.

## See also

- [RateQuant algorithm](/algorithms/ratequant)
- [Observers guide](/guides/observers)
- [API — RateQuant allocators](/api/allocators)
- [API — KeyNormObserver](/api/observers-api)
