---
id: overview
title: Algorithm Overview
sidebar_label: Overview
slug: /algorithms/overview
---

# Algorithm Overview

VeloxQuant-MLX implements thirty-two KV cache compression algorithms. This page helps you pick the right one for your workload.

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
| [KIVI](../algorithms/kivi) | 2 | 2 | None | 4× total | ★★★ | Tuning-free asymmetric baseline |
| [KIVI-Sink](../algorithms/kivi-sink) | 2 | 2 | None | 4× total | ★★★★ | Sink-protected low-bit quantization |
| [SVDq](../algorithms/svdq) | ~1.25 | fp16 | SVD at prefill | 12.8× key | ★★★ | Sub-2-bit keys, long context |
| [Kitty](../algorithms/kitty) | ~2.5 | fp16 | None | 6.4× key | ★★★★ | Adaptive channel precision, zero calibration |
| [AdaKV-proxy](../algorithms/adakv) | adaptive (2–4) | fp16 | None | adaptive | ★★★★ | Per-head adaptive bits, layers on KIVI |
| [XQuant](../algorithms/xquant) | ~1.0–1.4 | yes | None | 11–16× | ★★★★ | First cross-layer reuse — adjacent layers share codes |
| [KVQuant-NUQ](../algorithms/kvquant) | 2–4 (non-uniform) | 2–4 | None | 5–8× | ★★★★★ | Non-uniform datatype + outlier isolation |
| [PALU](../algorithms/palu) | ~0.6 (low-rank) | ~0.6 (low-rank) | None | high (full-KV) | ★★★ | First true latent cache — both K and V stored low-rank |
| [CacheGen](../algorithms/cachegen) | 3–4 (entropy) | 3–4 (entropy) | None | +10–17% over packing | ★★★ | First entropy-coded cache — storage win on correlated KV |
| [MiniCache](../algorithms/minicache) | fp16 (merged) | fp16 (merged) | None | ~2× on merged layers | ★★★ | Cross-layer SLERP merge — pairs of deep layers cost one |
| [GEAR](../algorithms/gear) | 2–4 (+ feedback) | 2–4 (+ feedback) | None | quality at low bits | ★★★ | First error-feedback cache — residual low-rank + sparse outliers |
| [ZipCache-adapted](../algorithms/zipcache) | adaptive (2–4) | adaptive (2–4) | None | adaptive | ★★★★ | Per-token mixed bit-width — salient tokens get hi_bits, rest get lo_bits |
| [SnapKV-adapted](../algorithms/snapkv) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★★ | Token eviction — keeps only a budget of prefill positions by obs-window attention |
| [StreamingLLM-adapted](../algorithms/streaming_llm) | fp16 (kept tokens) | fp16 (kept tokens) | None | constant memory | ★★★★ | Structural eviction — first N sinks + last W recent tokens; constant-memory streaming |
| [ChunkKV-adapted](../algorithms/chunkkv) | fp16 (kept chunks) | fp16 (kept chunks) | None | constant memory | ★★★★ | Chunk-level eviction — keeps whole contiguous chunks by pooled importance; `chunk_size=1` == H2O |
| [CaM-adapted](../algorithms/cam) | fp16 (merged) | fp16 (merged) | None | constant memory | ★★★★ | Cache merging — merges evicted tokens into similar survivors instead of dropping; `cam_merge=drop` == H2O |
| [xKV-adapted](../algorithms/xkv) | uniform-bit (latent) | fp16 | None | 8–20% fewer bytes vs per-layer SVD | ★★★★ | Cross-layer shared-subspace — joint SVD basis amortized across a layer group |
| [NSNQuant-adapted](../algorithms/nsnquant) | 1–2 (VQ) | 1–2 (VQ) | None (by construction) | ~6.4× at 2-bit incl. metadata | ★★★★ | Calibration-free universal-codebook VQ — NSN + Hadamard reshape data to one fixed Gaussian codebook |
| [L2Norm-adapted](../algorithms/knorm) | fp16 (kept tokens) | fp16 (kept tokens) | None | token count | ★★★ | Intrinsic key-norm eviction — low norm ⇒ important (EMNLP 2024 finding); zero per-step scoring cost, path-independent |

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

Are some attention heads far more quant-sensitive than others?
└── Yes, want a fixed average-bit target with per-head allocation → AdaKV-proxy

Are adjacent layers in your model highly correlated?
└── Yes, want the lowest effective bits by reusing codes across layers → XQuant

Are your K/V distributions heavy-tailed / non-uniform?
└── Yes, want best quality per bit without calibration → KVQuant-NUQ

Do a small fraction of your tokens have disproportionate attention weight?
└── Yes, want token-level bit allocation (not fp16 protection) → ZipCache-adapted

Do you need a hard cap on token count (very long context, fixed RAM budget)?
├── Yes, evict by importance score (score-based) → SnapKV-adapted
└── Yes, constant-memory streaming (positional) → StreamingLLM-adapted
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
- **[AdaKV-proxy](../algorithms/adakv)** — Per-head adaptive bit allocation layered on KIVI: ranks heads by online inter-token key-norm variance and solves a per-head bit budget so the average bits/element hits a configured target. Zero calibration; complements Kitty's per-channel axis.
- **[XQuant](../algorithms/xquant)** — Cross-layer reuse: adjacent layers are paired (anchor/reuse), the anchor publishes its quantized codes through a shared coordinator, and reuse layers store only their own scale/zero (+ optional residual). The first method to exploit *inter-layer* redundancy — sub-1.4-bit effective keys on correlated models, zero calibration.
- **[KVQuant-NUQ](../algorithms/kvquant)** — Non-uniform quantization datatype: places `2^bits` signpost levels where the data actually is via online Lloyd-Max fitting, plus dense/sparse outlier isolation that carves the top few extreme elements out to fp16. The first non-uniform-datatype method — strictly lower reconstruction error than uniform at the same bit-width, zero calibration.
- **[PALU](../algorithms/palu)** — True low-rank latent storage: fits one shared projection per head-group from the prefill batch and stores the cache as latent codes `[S, r]` for *both* keys and values, reconstructing fp16 only at attend time. Unlike SVDq (keys-only, reconstructs full fp16), the cache itself stays low-rank, so the storage win is real. Layered with mixed-bit latent quantization for a full-KV effective rate below 1 bit/element. Zero calibration.
- **[CacheGen](../algorithms/cachegen)** — Entropy coding: the first method to compress the *codes* themselves rather than just pick a bit-width. Exploits token-wise locality (adjacent tokens' KV are similar) by delta-coding the quantized codes and entropy-coding the low-entropy residual stream toward its Shannon entropy. Reconstruction is identical to group quant; the win is storage, capped to never exceed fixed-width packing. Zero calibration.
- **[MiniCache](../algorithms/minicache)** — Cross-layer depth merging: adjacent middle-to-deep layers share one SLERP-interpolated direction while each keeps its own per-token magnitude, so a pair of layers costs roughly one. High-divergence token pairs are retained unmerged. A different route to inter-layer redundancy than XQuant (which reuses codes); MiniCache merges the tensors. Zero calibration.
- **[xKV-adapted](../algorithms/xkv)** — Cross-layer shared-subspace compression: a fixed-size contiguous group of layers jointly factorizes its stacked key matrices into *one* shared SVD basis via a fan-in/fan-out coordinator, then each layer stores only its own latent codes in that basis. A third route to inter-layer redundancy alongside XQuant (code reuse) and MiniCache (direction merge) — xKV shares an entire subspace amortized across N layers rather than a pairwise code or direction. Keys only (values fp16). Zero calibration.
- **[NSNQuant-adapted](../algorithms/nsnquant)** — Calibration-free universal-codebook VQ: the first method to *reshape the data to match a fixed code* rather than fit a code to the data. A Normalize-Shift-Normalize transform plus a Hadamard rotation maps K/V tokens onto the standard normal distribution, so one codebook built offline from synthetic Gaussian samples — never from model activations — quantizes any model at 1–2 bits/element. Both keys and values quantized; chunk-flush fp16 residual buffer (KIVI's idiom). Zero calibration by construction.
- **[GEAR](../algorithms/gear)** — Error feedback: the first method to reconstruct what an ultra-low-bit base quantizer threw away, rather than just pick a bit-width. It adds a low-rank approximation of the quantization *residual* plus a sparse correction for the few outlier entries the low-rank term cannot absorb — `X ~= Quant_b(X) + L·R + S`. The residual SVD reuses the same shared helper as SVDq/PALU, but applied to the error rather than the signal. Composes over any base quantizer to recover quality at low bits. Zero calibration.
- **[ZipCache-adapted](../algorithms/zipcache)** — Per-token mixed bit-width: the first method to allocate bit-width *per token* within the quantized space. Uses key L2-norm as a saliency proxy (the same proxy as KIVI-Sink and AdaKV-proxy, but with a different decision): the top `hi_fraction` tokens by norm get `hi_bits`; the rest get `lo_bits`. Both groups remain quantized — not fp16 protection. The effective average rate is `hi_frac×hi_bits + (1-hi_frac)×lo_bits`. Labeled "ZipCache-adapted" because the paper's true signal (attention scores) is not observable at the cache level. Zero calibration.
- **[L2Norm-adapted](../algorithms/knorm)** — Intrinsic-signal eviction: the first scorer read directly off the stored key itself. The EMNLP 2024 finding (Devoto et al.): keys with *low* L2 norm attract *high* attention in trained LMs — so keep the lowest-norm tokens. No attention, no proxy, no per-step scoring cost (norms are computed once at insertion), and the kept set is provably identical whether tokens arrive as one prefill block or one at a time. Note the sign inversion vs ChunkKV's key_norm option (which keeps high-norm). Zero calibration.
- **[SnapKV-adapted](../algorithms/snapkv)** — Token eviction: the repo's first method to drop token positions entirely rather than compressing them. During prefill, the last `snap_obs_window` key rows act as proxy queries; their softmax attention over all prefix positions produces per-token importance scores. Only the top-`snap_budget` positions (plus `snap_n_sink` always-kept sink positions) are retained as fp16. Decode tokens are never evicted. The first method where the paper's actual signal (attention scores) is computable at the cache level — key-as-query proxy is stronger than key-norm-only methods. Zero calibration.
- **[StreamingLLM-adapted](../algorithms/streaming_llm)** — Structural positional eviction: the repo's first constant-memory cache. Keeps only the first `stream_n_sink` token positions (attention sinks, frozen forever) and the most recent `stream_window_size` positions (a rolling FIFO). All other positions are permanently dropped. Both prefill and decode tokens go through the same sink+window logic, so the cache never grows beyond `stream_n_sink + stream_window_size` positions regardless of generation length. Orthogonal to SnapKV-adapted (which evicts by score and grows during decode). Zero calibration.

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
