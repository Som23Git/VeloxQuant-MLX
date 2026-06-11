---
id: polarquant
title: PolarQuant
sidebar_label: PolarQuant
slug: /algorithms/polarquant
---

# PolarQuant

PolarQuant uses **recursive polar coordinate decomposition** to represent keys as angles rather than Cartesian coordinates. This is particularly effective for models where keys form geometric clusters on a sphere — a distribution that standard scalar quantizers handle poorly.

## How it works

1. **Polar decomposition** — Each key vector `k` is decomposed recursively. At each step, the angle `θᵢ = arccos(kᵢ / ‖k[i:]‖)` is computed and quantized as a binary value (above or below the equator).

2. **Recursive encoding** — The remaining vector after projecting out the first angle is processed by the same decomposition recursively until `head_dim` angles are encoded. Each angle is 1 bit.

3. **Geometric reconstruction** — Decoding reconstructs the original direction by composing the angles in reverse order. The norm is stored separately at full precision.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1 per angle (= 1 bit/dim effectively) |
| Value bits | 2–4 |
| Compression | 8× (keys) |
| Best for | Models with spherical key geometry |
| Metal kernel | `turboquant_scalar_quantize` for norm |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Phi-3-mini-4k-instruct-4bit")

config = KVCacheConfig(
    method="polarquant",
    value_bits=2,
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="What are the main differences between Python and Go?",
    max_tokens=400,
    kv_cache=cache,
)
```

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer

quantizer = PolarQuantizer()

keys = mx.random.normal(shape=(1, 32, 256, 64))  # Phi-3 mini head_dim=64

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)

# Cosine similarity (direction is encoded exactly, norm approximately)
from veloxquant_mlx import cosine_similarity
print(f"Cosine sim: {cosine_similarity(keys, decoded):.4f}")
```

## When to use PolarQuant

**Use PolarQuant when:**
- Key vectors are distributed approximately on a hypersphere (unit norm)
- The model uses normalised attention (Phi-3, Gemma-2 style)
- You want 1-bit keys without calibration and without the JL approximation

**Consider [TurboQuant RVQ](../algorithms/rvq) instead when:**
- Keys are not spherically distributed (most Llama/Mistral variants)
- You need both key and value compression at high quality

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `value_bits` | `int` | `2` | Value quantization bits |
| `norm_bits` | `int` | `8` | Bits for key norm (stored separately). Higher = more accurate magnitude |

## See also

- [CommVQ — polar + RoPE compatibility](../algorithms/commvq)
- [TurboQuant RVQ — better quality for non-spherical keys](../algorithms/rvq)
- [API — PolarQuantizer](../api/quantizers)
