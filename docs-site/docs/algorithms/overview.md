---
id: overview
title: Algorithm Overview
sidebar_label: Overview
slug: /algorithms/overview
---

# Algorithm Overview

VeloxQuant-MLX implements eleven KV cache compression algorithms. This page helps you pick the right one for your workload.

:::warning Apple Silicon required
All algorithms use Metal GPU kernels and require macOS on an M-series chip.
:::

## Comparison table

| Algorithm | Key bits | Val bits | Calibration | Compression | Quality | Best for |
|---|---|---|---|---|---|---|
| [TurboQuant RVQ](../algorithms/rvq) | 1–3 | 2–4 | None | 7.5× | ★★★★ | General purpose, zero setup |
| [VecInfer](../algorithms/vecinfer) | 1–4 | 2–4 | Codebook (2 min) | 16× | ★★★★ | Max throughput, Metal-accelerated |
| [RateQuant](../algorithms/ratequant) | mixed | mixed | Sensitivity (90 s) | 6–12× | ★★★★★ | Best accuracy per bit |
| [SpectralQuant](../algorithms/spectral) | 2–8 | 2–4 | SVD rotation (3 min) | 4–8× | ★★★★★ | Long context, high fidelity |
| [RaBitQ](../algorithms/rabitq) | 1 | fp16 | None | 6× total | ★★★ | Key-only extreme compression |
| [QJL](../algorithms/qjl) | 1 | fp16 | None | 8× key only | ★★★ | Simplest, fastest to set up |
| [PolarQuant](../algorithms/polarquant) | 1–2 | 2 | None | 8× | ★★★ | Geometric key distributions |
| [CommVQ](../algorithms/commvq) | 2–4 | fp16 | None | 4–8× | ★★★★ | RoPE-compatible models |
| [SVDq](../algorithms/svdq) | ~1.25 | fp16 | SVD at prefill | 12.8× key | ★★★ | Sub-2-bit keys, long context |
| [Kitty](../algorithms/kitty) | ~2.5 | fp16 | None | 6.4× key | ★★★★ | Adaptive channel precision, zero calibration |

*Compression ratios measured on Llama-3.1-8B at 4096 context. Source: [BENCHMARK_RESULTS.md](https://github.com/rajveer43/veloxquant-mlx/blob/master/BENCHMARK_RESULTS.md).*

## Decision guide

```
Do you want zero calibration?
├── Yes → TurboQuant RVQ (best quality), QJL (simplest), RaBitQ (1-bit keys)
└── No, I can spend 1–3 minutes calibrating →
    ├── Priority: max compression → VecInfer
    ├── Priority: max quality     → RateQuant or SpectralQuant
    └── Long sequences (8k+)     → SpectralQuant

Is RoPE positional encoding compatibility critical?
└── Yes → CommVQ

Do you have geometric/non-Gaussian key distributions?
└── Yes → PolarQuant

Do key channels have highly non-uniform variance?
└── Yes, want adaptive mixed-precision without calibration → Kitty
```

## Method families

### Zero-calibration methods

These work immediately on any model with no setup beyond installation.

- **[TurboQuant RVQ](../algorithms/rvq)** — The recommended default. Uses analytical Gaussian + Laplacian codebooks precomputed from distribution theory. Two residual passes give excellent fidelity at 1 bit per pass.
- **[QJL](../algorithms/qjl)** — Johnson-Lindenstrauss 1-bit sign sketch. Provably preserves inner products in expectation. Extremely simple — great for prototyping.
- **[RaBitQ](../algorithms/rabitq)** — Randomised Hadamard transform + 1-bit sign packing with IVF clustering. Better than QJL for key-only compression.
- **[PolarQuant](../algorithms/polarquant)** — Recursive polar decomposition for models where keys form geometric clusters.
- **[CommVQ](../algorithms/commvq)** — RoPE-commutative residual VQ: quantization that commutes with rotary position embeddings, preserving exact positional information.
- **[Kitty](../algorithms/kitty)** — Dynamic channel-wise mixed-precision: ranks key channels by online variance and allocates 4-bit to high-variance channels, 2-bit to the rest. Zero calibration, 2.5-bit effective key precision.

### Calibration-required methods

These require a one-time calibration step, but deliver significantly better accuracy per bit.

- **[VecInfer](../algorithms/vecinfer)** — Product VQ with Metal-accelerated codebook lookup. Smooth scaling handles outlier dimensions. The fastest method at inference time due to fused SDPA kernels.
- **[RateQuant](../algorithms/ratequant)** — Mixed-precision allocation via reverse-waterfilling. Probes per-layer sensitivity and allocates more bits to layers that contribute most to output quality. Best accuracy per average bit.
- **[SpectralQuant](../algorithms/spectral)** — SVD rotation aligns key dimensions with high-variance directions. Separate signal/noise codebooks. Best for very long contexts (8k+).

## Mixing methods

The `CompositeQuantizer` chains multiple quantizers in sequence:

```python
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
from veloxquant_mlx.quantizers.qjl import QJLQuantizer

# RVQ for first-pass compression + QJL residual sketch
quantizer = CompositeQuantizer([
    TurboQuantRVQ(bits=1),
    QJLQuantizer(sketch_dim=64),
])
```

## Per-model recommendations

| Model | Recommended algorithm | Notes |
|---|---|---|
| Llama 3.1/3.2 (7–8B) | TurboQuant RVQ 1-bit | Gaussian key distribution, zero setup |
| Mistral 7B / Mixtral | VecInfer 2-bit | Sliding window attention benefits from product VQ |
| Qwen 2.5 (7–14B) | SpectralQuant | Long-context optimised, benefits from SVD rotation |
| Phi-3 Mini | RaBitQ + CommVQ | Small head dim, CommVQ preserves RoPE exactly |
| Gemma 2B/7B | TurboQuant RVQ 2-bit | GQA benefits from slightly higher bit rate |
| Falcon 7B | RateQuant | Alibi positional bias; RateQuant adapts per-layer |

## Next steps

- Pick an algorithm and read its detailed page
- [mlx_lm integration guide](../guides/mlx-lm-integration)
- [Calibration guide](../guides/calibration)
