---
id: changelog
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

---

## v0.13.0 — Latest

### New
- **XQuant** (`method="xquant"`) — cross-layer KV cache reuse, the repo's first method to exploit *inter-layer* redundancy. Adjacent attention layers are grouped into anchor/reuse pairs: the anchor quantizes K/V with KIVI-style group quantization and publishes its integer codes through a shared coordinator; reuse layers borrow those codes and store only their own per-group scale/zero (+ optional low-bit residual), correcting the small cross-layer drift. Drives effective per-element key bits below 1.4 on correlated models (11–16× key bandwidth reduction across a group). Both keys and values compressed; zero calibration. A faithful adaptation of XQuant (arXiv:2510.11236, EMNLP 2025): the paper couples layers in a modified attention forward pass; we coordinate through a shared object so `mlx_lm.generate` stays untouched.
- `XQuantKVCache` — new cache wrapper in `veloxquant_mlx/cache/xquant_cache.py` with anchor/reuse role dispatch
- `XQuantCoordinator` — shared cross-layer code store in `veloxquant_mlx/cache/xquant_coordinator.py`, injected by `KVCacheBuilder.for_model()`
- XQuant utilities in `veloxquant_mlx/quantizers/xquant.py`: `pair_layers()`, `quantize_codes()`, `compute_reuse_params()`, `dequant_with_params()`, `quantize_residual()`, `cross_layer_similarity()`
- New `KVCacheConfig` fields: `xquant_group_size`, `xquant_base_bits`, `xquant_residual_bits`, `xquant_group_quant_size`, `xquant_max_ctx`
- `KVCacheBuilder.for_model()` now builds one shared coordinator and assigns anchor/reuse roles for `method="xquant"` (other methods unchanged)
- 16 new tests in `tests/cache/test_xquant_cache.py`: factory dispatch, `for_model` pairing, coordinator round-trip, anchor/reuse shape (prefill + decode), value reconstruction, residual-0 tolerance, residual lowers MSE, correlated near-self-quant, uncorrelated residual recovery (negative control), byte accounting, effective-bits, decode synchronization, token-budget guard, `group_size=3`, determinism
- `benchmark_scripts/benchmark_xquant.py` — throughput + memory sweep over `group_size ∈ {2,3}`, `residual_bits ∈ {0,1}` vs KIVI-2bit, SVDq-1.25bit, fp16, plus measured cross-layer key similarity

---

## v0.12.0

### New
- **AdaKV-proxy** (`method="adakv"`) — per-head adaptive bit allocation layered on KIVI-style group quantization. Ranks attention heads by online inter-token key-norm variance (an attention-free proxy for head importance), then solves a per-head bit budget so the average bits/element matches a configured target — high-importance heads get more bits, low-importance heads fewer. Zero calibration; values left at fp16. A *proxy* adaptation of Ada-KV (arXiv:2407.11550): true Ada-KV adapts the per-head *eviction* budget from softmax attention weights, which live outside the cache contract; we adapt the per-head *bit* budget instead.
- `AdaKVCache` — new cache wrapper in `veloxquant_mlx/cache/adakv_cache.py`
- AdaKV utilities in `veloxquant_mlx/quantizers/adakv.py`: `compute_head_norm_variance()`, `allocate_head_bits()` (budget allocator with greedy round-trip correction), `quantize_head()`
- New `KVCacheConfig` fields: `adakv_target_avg_bits`, `adakv_lo_bit`, `adakv_mid_bit`, `adakv_hi_bit`, `adakv_group_size`, `adakv_update_interval`
- 14 new tests in `tests/cache/test_adakv_cache.py`: factory dispatch, shape preservation (prefill + decode), values unchanged, high-importance heads get more bits, average bits matches target, equal-importance uniform degradation, lower MSE than lo_bit on the high-importance head, running norm-accumulator correctness, decode accumulation, byte accounting, avg_bits range, single-head trivial allocation, determinism
- `benchmark_scripts/benchmark_adakv.py` — throughput + memory sweep over `target_avg_bits ∈ {2.0, 2.5, 3.0}` vs KIVI-2bit, Kitty-2.5bit, fp16

---

## v0.11.0

### New
- **Kitty** (`method="kitty"`) — dynamic channel-wise mixed-precision key quantization. Ranks key channels by online per-channel variance at every step; top-25% channels get 4-bit, remaining 75% get 2-bit asymmetric group quantization. Achieves ~2.5-bit effective key precision (6.4× bandwidth reduction vs fp16). Zero calibration — no SVD, no codebook training, works on any model immediately. Values left at fp16. Inspired by Kitty (arXiv:2511.18643).
- `KittyKVCache` — new cache wrapper in `veloxquant_mlx/cache/kitty_cache.py`
- Kitty utilities in `veloxquant_mlx/quantizers/kitty.py`: `rank_channels_by_sensitivity()`, `quantize_mixed_channels()`, `compute_running_variance()`
- `veloxquant_mlx/quantizers/_quant_utils.py` — shared `_group_quant_dequant` helper extracted from `svdq.py` (no behavior change; both quantizers import from here)
- New `KVCacheConfig` fields: `kitty_hi_fraction`, `kitty_hi_bit`, `kitty_lo_bit`, `kitty_group_size`
- 14 new tests in `tests/cache/test_kitty_cache.py`: factory dispatch, shape preservation (prefill + decode), values unchanged, channel ranking correctness, hi-channel lower error than lo-channel, MSE vs uniform 2-bit on high-variance data, running variance accumulator, decode accumulation, byte accounting, avg_bits range, hi_fraction boundary cases, determinism
- `benchmark_scripts/benchmark_kitty.py` — throughput + memory sweep vs KIVI-2bit, SVDq-1.25bit, fp16

---

## v0.10.0

### New
- **SVDq** (`method="svdq"`) — sub-2-bit key compression via offline SVD + mixed-precision latent coding. Computes a truncated SVD of the prefill key matrix once, projects all keys into the low-rank latent space, and applies 4-bit / 2-bit mixed quantization ordered by singular value magnitude. Achieves ~1.25-bit effective key precision (12.8× bandwidth reduction vs fp16). Values left at fp16. Inspired by SVDq (arXiv:2502.15304).
- `SVDqKVCache` — new cache wrapper in `veloxquant_mlx/cache/svdq_cache.py`
- SVD utilities in `veloxquant_mlx/quantizers/svdq.py`: `svd_compress_keys()`, `quantize_latents_mixed()`, `reconstruct_keys()`
- New `KVCacheConfig` fields: `svdq_rank`, `svdq_energy_threshold`, `svdq_hi_bit`, `svdq_lo_bit`, `svdq_hi_fraction`, `svdq_group_size`
- 12 new tests in `tests/cache/test_svdq_cache.py`: SVD projection correctness, shape preservation, MSE vs naive 2-bit on low-rank data, decode accumulation, byte accounting, sub-2-bit effective bit-width, energy threshold rank selection, determinism

---

## v0.9.0

### New
- **KIVI-Sink** (`method="kivi_sink"`) — attention sink protection layered on KIVI group quantization. Tokens with anomalously high key L2-norm are kept in fp16 and excluded from quantization-parameter calibration, preventing sink outliers from inflating group scale and degrading neighboring tokens. Inspired by KVSink (Su & Yuan, COLM 2025).
- `SinkProtectedKVCache` — new cache wrapper in `veloxquant_mlx.cache.sink_cache`
- `KVCacheConfig.n_sink_tokens` — new field (default 5). Composes with KIVI's `residual_length`; byte accounting tracks `sink_fp16_bytes` separately with no double-counting. `n_sink_tokens=0` reproduces plain KIVI bit-for-bit.
- 9 new tests in `tests/cache/test_sink_cache.py`: sink detection, fp16 preservation, MSE improvement over plain KIVI, accounting partition, determinism. Full suite: 344/348 passing.

---

## v0.8.0

### New
- **KIVI** (`method="kivi"`) — tuning-free asymmetric 2-bit group quantization (Liu, Yuan et al., ICML 2024). Per-channel keys, per-token values; no codebook training, no rotation.
- `KIVIQuantizer` — registered as `"kivi"` in `QuantizerRegistry`
- `KIVIKVCache` — mlx_lm `update_and_fetch` wrapper with fp16 residual window (`residual_length`) and full byte-accounting
- `KVCacheConfig.kivi_group_size` — new field (default 32)
- Benchmark results on Llama-3.2-3B, Qwen2.5-7B, Mistral-7B (Apple M4): **KIVI-2bit ≈ 5.8× key / ≈ 4× full-KV at 100–106% of fp16 throughput**
- 25 new tests; 334/339 passing

---

## v0.7.0

### New
- **RaBitQ** — randomised Hadamard + 1-bit sign packing with IVF clustering for extreme key compression
- **SpectralQuant** — eigenvector-rotated quantization with signal/noise codebooks and water-filling bit allocation
- **CommVQ** — RoPE-commutative residual VQ for exact positional encoding compatibility
- `SpectralQuantKVCache`, `PolarQuantKVCache` — new cache wrappers
- `calibrate_spectral_rotation()`, `save_rotations()`, `load_cached_rotations()`
- `compute_participation_ratio()`, `compute_spectral_gap()`
- `water_fill_bits()` — per-dimension water-filling allocator
- `rabitq_hamming_score` — Metal XOR+popcount Hamming distance kernel
- `comm_vq_decode_metal` — fused centroid gather + RoPE Metal kernel
- 212+ passing tests

### Changed
- `KVCacheConfig` gains `signal_bits`, `noise_bits`, `rotations` fields for SpectralQuant
- `KVCacheFactory` and `KVCacheBuilder` updated for all new cache types

---

## v0.6.0

### New
- **PolarQuant** — recursive polar coordinate decomposition for spherical key distributions
- `PolarQuantizer`, `PolarQuantKVCache`
- `CommVQQuantizer` — first version (flat codebook, no Metal fusion yet)
- `TurboQuantProdAdaptive` — distortion-driven dynamic bit allocation

### Changed
- `CompositeQuantizer` — supports arbitrary-depth chains; cycle detection via `CyclicPipelineError`

---

## v0.5.1

### New
- **Metal GPU kernels for VecInfer** — hand-written Metal Shading Language shaders replacing pure-MLX hot paths
  - `vecinfer_quantize_metal` — fused nearest-centroid argmin, **13× speedup, 98% peak-memory reduction**
  - `vecinfer_dequant_metal` — bit-exact drop-in for `dequantize_vq`
  - `metal_available()` — capability probe
- `KVCacheConfig.use_metal_kernels` — three-state flag (`None` = auto-detect, `True` = require, `False` = force MLX)
- `VecInferKVCache` now dispatches to Metal kernels when available (zero API change)
- 7 new parity tests in `tests/cache/test_vecinfer_metal_parity.py`

---

## v0.5.0

### New
- **VecInfer** — product VQ with outlier-suppressing dual transform
  - `calibrate_smooth_factors()` — per-channel `λᵢ = √max|Kᵢ|`
  - `walsh_hadamard_matrix()`, `apply_dual_transform_keys/queries()`
  - `train_codebook()`, `quantize_vq()`, `dequantize_vq()`
  - `compute_query_lut()` — fused-score fast path
- `VecInferKVCache` — mlx_lm-compatible cache with `update_and_fetch`
- **Benchmarks**: 8× key compression at 2-bit, 16× at 1-bit on Llama-3.2-1B/3B

### Notes
- Throughput trades slightly vs fp16 (CUDA kernel fusion not available on Metal at this version)

---

## v0.3.6

### Breaking change
- **Package renamed**: `mlx_kv_quant` → `veloxquant_mlx`
- All imports must be updated: `from mlx_kv_quant import ...` → `from veloxquant_mlx import ...`
- No backward-compatibility shim

---

## v0.3.5

### New
- **RateQuant** becomes a first-class feature
  - `allocate_bits_ratequant()` — reverse-waterfilling allocator (arxiv:2605.06675)
  - `calibrate_layer_sensitivities()` — activation-norm sensitivity probe (1.6s)
  - `fit_distortion_curve()` — fits `D(b) = α·β^(-b)` per layer
- `TurboQuantRVQKVCache` — mlx_lm-compatible cache wrapper for RVQ
- `KeyNormObserver`, `KeyNormReport` — per-token key norm tracking
- `KVCacheConfig.bit_width_inlier` accepts `list[int]` for per-layer allocation
- 27 new tests (187 total passing)

### Results (M4 24 GB)

| Model | fp16 PPL | RVQ 1-bit | RateQuant 1.5-bit | Compression |
|---|---|---|---|---|
| Falcon3 7B | 22.9 | 23.1 | 22.8 | 5.22× |
| Gemma3 4B | 39.8 | 37.8 | 36.3 | 5.22× |

---

## v0.3.0

### New
- **QJL** — Johnson-Lindenstrauss 1-bit sign sketch cache
- `QJLQuantizer`, `QJLKVCache`
- `qjl_encode`, `qjl_inner_product` Metal kernels
- `DistortionObserver` — cosine similarity and IP error tracking
- `LatencyObserver` — encode/decode timing profiling
- `MemoryObserver` — peak memory and compression ratio

---

## v0.2.0

### New
- **TurboQuant RVQ** — two-pass residual VQ with Gaussian + Laplacian codebooks
- `TurboQuantRVQ` quantizer with Walsh-Hadamard preprocessing
- `turboquant_scalar_quantize`, `turboquant_hadamard_quantize` Metal kernels
- `turboquant_bit_pack`, `turboquant_bit_unpack` — sub-byte packing
- `KVCacheConfig`, `KVCacheFactory`, `KVCacheBuilder` — unified configuration API
- `NpyArtifactStore`, `MemoryArtifactStore` — artifact persistence
- `QuantizerRegistry` — plugin registration

---

## v0.1.0

### Initial release
- Core abstractions: `Quantizer`, `KVCache`, `Preconditioner`, `Codebook` ABCs
- `TurboQuantMSE` — MSE-optimal rotation + Lloyd-Max scalar quantization
- `ScalarCodebook`, `AdaptiveScalarCodebook`
- `RotationPreconditioner`, `JLSketchPreconditioner`
- `RingBuffer`, `AVLTree`, `BitPackBuffer` data structures
- Basic test suite (48 tests)

---

*Full commit-level history: [GitHub Commits](https://github.com/rajveer43/veloxquant-mlx/commits/master)*
