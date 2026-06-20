---
id: changelog
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

---

## v0.9.0 — Latest

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
