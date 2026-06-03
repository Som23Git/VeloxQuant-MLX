---
id: ratequant
title: RateQuant
sidebar_label: RateQuant
slug: /algorithms/ratequant
---

# RateQuant

RateQuant is the **highest-accuracy algorithm** in VeloxQuant-MLX. It uses a 90-second sensitivity probe to learn which layers need more bits and which can be aggressively compressed — then allocates a mixed-precision budget via reverse-waterfilling.

:::warning Apple Silicon required
Calibration and inference use Metal kernels. Requires macOS M-series.
:::

## How it works

1. **Sensitivity probing** — `calibrate_layer_sensitivities()` runs a short forward pass on 32 calibration sequences and perturbs each layer's KV cache with controlled noise. It measures how much each layer's output changes — the "sensitivity" of that layer to quantization.

2. **Rate-distortion curve** — `fit_distortion_curve()` fits a parametric model `D(r) = α · exp(-β · r)` to the sensitivity measurements, where `r` is the bit rate and `D` is distortion.

3. **Reverse-waterfilling** — `allocate_bits_ratequant()` solves a Lagrangian optimisation: given a target average bit rate, assign bits to each layer to minimise total distortion. Sensitive layers get more bits; insensitive layers get fewer.

4. **Per-layer quantizers** — The allocated bit vector is passed to `KVCacheBuilder`, which assigns a different quantizer (e.g., 4-bit RVQ, 2-bit RVQ, 1-bit QJL) per layer.

## Key properties

| Property | Value |
|---|---|
| Calibration | Sensitivity probe ≈90 seconds |
| Key bits | mixed (e.g., 1–4 per layer) |
| Value bits | mixed |
| Target avg bits | user-specified (e.g., 2.0) |
| Quality | Best in class at equal avg bits |
| Metal kernels | Per-layer (inherits from assigned quantizer) |

## Step 1 — Calibrate

```python
import mlx_lm
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    fit_distortion_curve,
    allocate_bits_ratequant,
)
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Probe all layers — takes ~90 seconds on M3 Pro
sensitivities = calibrate_layer_sensitivities(
    model, tokenizer,
    num_samples=32,       # number of calibration sequences
    sequence_length=512,  # tokens per sequence
)

# Fit distortion curve
distortion_curves = fit_distortion_curve(sensitivities)

# Allocate bits with target average of 2.0 bits per dimension
bit_allocation = allocate_bits_ratequant(
    distortion_curves,
    target_bits=2.0,
    min_bits=1,
    max_bits=4,
)

print(bit_allocation)
# {'layer_0': 3, 'layer_1': 2, 'layer_2': 1, ..., 'layer_31': 2}

# Save for reuse
store = NpyArtifactStore("./veloxquant_artifacts/")
store.save("ratequant_allocation", bit_allocation)
```

## Step 2 — Inference

```python
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

store = NpyArtifactStore("./veloxquant_artifacts/")
bit_allocation = store.load("ratequant_allocation")

config = KVCacheConfig(
    method="ratequant",
    bit_allocation=bit_allocation,  # per-layer dict
    value_bits=2,                   # all layers use same value bits
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Write a detailed analysis of the economic impacts of automation.",
    max_tokens=2048,
    kv_cache=cache,
)
```

## Configuration reference

| Parameter | Type | Required | Description |
|---|---|---|---|
| `bit_allocation` | `dict[str, int]` | Yes | Per-layer bit assignment from `allocate_bits_ratequant()` |
| `value_bits` | `int` | No | Uniform value bit rate. Default: `2` |
| `target_bits` | `float` | No | Target average bits (used if `bit_allocation` not provided). Default: `2.0` |
| `min_bits` | `int` | No | Minimum bits per layer. Default: `1` |
| `max_bits` | `int` | No | Maximum bits per layer. Default: `4` |

## Outlier token handling

RateQuant integrates with the outlier detection system. Tokens with anomalously large key norms (detected by `KeyNormObserver`) are automatically routed to a higher-bit quantizer:

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver

observer = KeyNormObserver(outlier_threshold=3.0)  # 3 σ above mean
config = KVCacheConfig(
    method="ratequant",
    bit_allocation=bit_allocation,
    outlier_observer=observer,
    outlier_bits=8,   # use 8 bits for detected outlier tokens
)
```

See the [Mixed-precision guide](/guides/mixed-precision) for a detailed walkthrough.

## When to use RateQuant

**Use RateQuant when:**
- Quality is the primary objective
- You have 90 seconds for calibration
- You want fine-grained control over the accuracy-memory tradeoff
- You are dealing with models that have heterogeneous layer sensitivities

**Consider alternatives when:**
- Zero calibration required → [TurboQuant RVQ](/algorithms/rvq)
- Maximum compression → [VecInfer](/algorithms/vecinfer)

## Benchmark results

On Llama-3.1-8B at 4096 context, M3 Pro (source: BENCHMARK_RESULTS.md):

| Method | Avg bits | Memory | Perplexity delta |
|---|---|---|---|
| fp16 baseline | 16 | 536 MB | 0.00 |
| RVQ uniform 2-bit | 2.0 | 67 MB | +0.08 |
| RateQuant 2.0 avg | 2.0 | 67 MB | +0.03 |
| RateQuant 1.5 avg | 1.5 | 50 MB | +0.07 |

RateQuant at 2.0 average bits achieves **2.7× lower perplexity degradation** than uniform 2-bit RVQ.

## See also

- [Mixed-precision guide](/guides/mixed-precision)
- [Calibration guide](/guides/calibration)
- [API — RateQuant allocators](/api/allocators)
- [Observers guide](/guides/observers)
