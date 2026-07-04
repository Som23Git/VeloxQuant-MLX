---
id: intro
title: What is VeloxQuant-MLX?
sidebar_label: Introduction
slug: /getting-started/intro
---

# What is VeloxQuant-MLX?

VeloxQuant-MLX is a production-grade **KV cache compression library** for Apple Silicon (M-series Macs). It implements twenty-nine compression algorithms — quantization plus token eviction — that compress the key-value cache used during LLM inference, reducing peak memory by up to **98%** while maintaining near-lossless output quality.

LLMs like Llama, Mistral, and Qwen store past context in a KV cache that grows linearly with sequence length. On a MacBook M3 Pro with 18 GB unified memory, a 7B model at 8k context can consume 14 GB of cache alone — leaving almost no room for anything else. VeloxQuant-MLX compresses that cache on-the-fly with Metal GPU kernels, making long-context inference practical on consumer hardware.

## Why Apple Silicon?

Apple's M-series chips have a unique advantage: **unified memory**. The GPU and CPU share the same memory pool, which means there is no PCIe bandwidth bottleneck between host and device. VeloxQuant-MLX is built specifically around this architecture:

- Metal GPU kernels run quantization/dequantization directly on the Neural Engine and GPU cores
- MLX — Apple's ML framework — provides the tensor primitives; VeloxQuant-MLX sits on top of it
- Quantized KV cache stays in unified memory, accessed by both the attention kernel and the quantizer with zero copies

## Key metrics

| Metric | Value |
|---|---|
| Max key cache compression | 16× (VecInfer 1-bit) |
| Metal kernel speedup | 13× faster quantization |
| Peak memory reduction | up to 98% |
| RVQ-1bit compression | 7.5× with zero calibration |
| RaBitQ full KV | 6× (keys + values) |
| Validated models | 12 (Llama, Mistral, Qwen, Phi, Gemma, Falcon) |
| Test suite | 212+ passing tests |

## Algorithm overview

VeloxQuant-MLX provides twenty-nine algorithms ranging from zero-calibration 1-bit methods to sophisticated mixed-precision allocators, low-rank latent caches, cross-layer schemes, and token-eviction/merging caches:

| Algorithm | Bits | Calibration | Best for |
|---|---|---|---|
| **TurboQuant RVQ** | 1–3+ | None | General purpose, drop-in replacement |
| **VecInfer** | 1–4 | Codebook training | Maximum throughput |
| **RateQuant** | mixed | 90 seconds | Mixed-precision accuracy-memory tradeoffs |
| **SpectralQuant** | 2–8 | SVD rotation | High-accuracy long context |
| **RaBitQ** | 1 | None | Key-only extreme compression |
| **QJL** | 1 | None | Simplest, fastest |
| **PolarQuant** | 1–2 | None | Geometric key distributions |
| **CommVQ** | 2–4 | None | RoPE-compatible residual VQ |
| **KIVI** | 2 | None | Tuning-free asymmetric baseline |
| **KIVI-Sink** | 2 | None | Sink-protected low-bit quantization |
| **SVDq** | ~1.25 | SVD at prefill | Sub-2-bit keys, long context |
| **Kitty** | ~2.5 | None | Adaptive channel precision |
| **AdaKV-proxy** | 2–4 | None | Per-head adaptive bits, layers on KIVI |
| **XQuant** | ~1.0–1.4 | None | First cross-layer reuse (adjacent layers share codes) |
| **KVQuant-NUQ** | 2–4 | None | Non-uniform datatype + outlier isolation |
| **PALU** | ~0.6 | None | First true latent cache — K and V stored low-rank |
| **CacheGen** | 3–4 | None | First entropy-coded cache — storage win on correlated KV |
| **MiniCache** | fp16 (merged) | None | Cross-layer SLERP merge — deep layer pairs cost one |

See [Algorithm Overview](../algorithms/overview) for a full comparison.

## Next steps

- [Install VeloxQuant-MLX](../getting-started/installation)
- [5-minute quickstart](../getting-started/quickstart)
- [Core concepts — KV cache, quantization](../getting-started/concepts)
