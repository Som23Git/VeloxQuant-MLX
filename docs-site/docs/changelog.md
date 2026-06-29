---
id: changelog
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

---

## v0.17.0 — Latest

### New
- **GEAR** (`method="gear"`) — the repo's first **error-feedback** KV cache. Every other method picks a bit-width or a cache layout and lives with the quantization error; GEAR makes *any* ultra-low-bit base quantizer near-lossless by reconstructing what it threw away, via the three-part decomposition `X ~= Quant_b(X) + L·R + S`: an ultra-low-bit base quant, a **low-rank** approximation of the quantization *residual* `E = X - dequant(Quant_b(X))`, and a **sparse** matrix correcting the top-magnitude outlier entries the low-rank term cannot absorb. Unlike CacheGen (reconstruction identical to group quant), GEAR's reconstruction genuinely **recovers quality** the base bit-width loses. GEAR-adapted (arXiv:2403.05527, Kang et al.): the residual SVD is computed per `update_and_fetch` call (reusing the SVDq/PALU prefill-SVD pattern) and GEAR's fused dequant CUDA kernel is not ported — we reconstruct fp16 then call MLX SDPA, so stored size shrinks but attend-time peak memory does not.
  - `GEARKVCache` (`veloxquant_mlx/cache/gear_cache.py`); primitives in `veloxquant_mlx/quantizers/gear.py`: `quantize_base`, `residual`, `lowrank_error`, `sparse_outliers`, `gear_compress`, `gear_reconstruct`, `gear_bytes`, `base_only_bytes`, `gear_quant_dequant`. The base quant is borrowed from CacheGen and the truncated-SVD helper (`_quant_utils._truncated_svd`) is shared with SVDq/PALU.
  - Config: `gear_bits`, `gear_rank`, `gear_energy_threshold`, `gear_sparse_fraction`, `gear_group_size`, `gear_quantize_values`
  - 10 cache tests + 13 quantizer tests; `benchmark_scripts/benchmark_gear.py` (offline-synthetic, not run)
  - Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates the `gear_*` fields automatically via `dataclasses.replace`.

### Honest scope
- GEAR's **stored** cache (base codes + low-rank factors + sparse triples) shrinks, but the working set *during* attention is the reconstructed fp16 K/V — attend-time peak memory is not reduced. The low-rank factors and sparse triples are overhead, so the rank must be genuinely low relative to the head dim (the GEAR premise); the overhead is reported honestly and never hidden.
- Quality evidence is unit-test level (synthetic low-rank-plus-outlier data); no model-level benchmark run yet.

---

## v0.16.0

### New
- **CacheGen** (`method="cachegen"`) — the repo's first **entropy-coded** KV cache. Every other method packs codes at a fixed bit-width; CacheGen exploits token-wise locality (adjacent tokens' KV are similar) by applying a reversible token-delta transform to the quantized codes and compressing the low-entropy residual stream toward its Shannon entropy. Reconstruction is identical to plain group quant (lossless over the codes); the contribution is the storage accounting. CacheGen-adapted (arXiv:2310.07240, SIGCOMM 2024): rather than ship a serial range codec that would bottleneck MLX decode, the entropy-coded byte size is modelled from the measured symbol entropy and capped at the fixed-width packed size, so savings are never negative (exactly 0% on incompressible iid data, ~10–17% on correlated data).
  - `CacheGenKVCache` (`veloxquant_mlx/cache/cachegen_cache.py`); primitives in `veloxquant_mlx/quantizers/cachegen.py`: `quantize_to_codes`, `dequant_codes`, `token_delta`, `symbol_entropy_bits`, `entropy_coded_bytes`, `fixed_width_bytes`, `cachegen_quant_dequant`
  - Config: `cachegen_bits`, `cachegen_group_size`, `cachegen_use_delta`
  - 12 cache tests + 9 quantizer tests; `benchmark_scripts/benchmark_cachegen.py` (not run)
- **MiniCache** (`method="minicache"`) — cross-layer compression in the **depth dimension**. Adjacent middle-to-deep layers have nearly identical KV directions, so a pair is merged into one shared SLERP-interpolated direction plus each layer's own per-token magnitude (a pair costs ~one layer). High-divergence token pairs are kept unmerged (the retention set). A different route to inter-layer redundancy than [XQuant](algorithms/xquant) — XQuant reuses quantized *codes*, MiniCache merges the *tensors* via spherical interpolation. MiniCache-adapted (arXiv:2405.14366, NeurIPS 2024): faithful to the magnitude/direction SLERP + token retention, integrated via a shared `MiniCacheCoordinator` (the XQuant pattern) rather than a modified attention forward.
  - `MiniCacheKVCache` (`veloxquant_mlx/cache/minicache_cache.py`), `MiniCacheCoordinator` (`veloxquant_mlx/cache/minicache_coordinator.py`); primitives in `veloxquant_mlx/quantizers/minicache.py`: `pair_layers_depth`, `to_mag_dir`, `slerp`, `merge_pair`, `reconstruct_layer`, `merge_similarity`
  - Config: `minicache_start_frac`, `minicache_group_size`, `minicache_retention_threshold`, `minicache_slerp_t`, `minicache_max_ctx`
  - 11 cache tests + 11 quantizer tests; `benchmark_scripts/benchmark_minicache.py` (not run)
  - Requires `KVCacheBuilder.for_model()` for the shared coordinator; a single factory-built cache is a degenerate lossless-passthrough primary.

### Honest scope
- Both are **storage**-compression methods: CacheGen's entropy coding and MiniCache's merge both reduce stored cache size but reconstruct fp16 for SDPA, so neither reduces working-set memory at attend time. On Apple Silicon's bandwidth-bound decode they are lower-leverage than the low-rank (PALU/SVDq) and quantization methods.
- Quality evidence is unit-test level (synthetic data); no model-level benchmark run yet.

---

## v0.15.0

### New
- **PALU** (`method="palu"`) — true low-rank latent storage for **both keys and values**, the repo's first method where the cache itself stays low-rank rather than reconstructing full fp16 for storage. At prefill it partitions heads into `palu_n_head_groups` groups, fits one shared projection per group via group-head SVD (G-LRD), and stores the projected codes `[S, r]` directly; full fp16 K/V is reconstructed only at attend time. Latents are mixed-bit quantized (top-25% of channels by singular value at 4-bit, the rest at 2-bit) for a full-KV effective rate below 1 bit/element on low-rank data. Unlike [SVDq](algorithms/svdq) — keys-only, reconstructs full fp16 and so wins on bandwidth accounting — PALU bypasses the parent fp16 ring buffer entirely (the storage win is real). Zero calibration. A PALU-adapted (arXiv:2407.21118, ICLR 2025) implementation: we fit projections from the prefill batch instead of an offline calibration set, and we do **not** port PALU's fused low-rank-reconstruction attention kernel (we reconstruct then call MLX SDPA), so peak memory during attention is not reduced — only stored cache size.
- `PALUKVCache` — new cache wrapper in `veloxquant_mlx/cache/palu_cache.py` (true latent storage; parent fp16 buffer bypassed, own offset bookkeeping)
- PALU primitives in `veloxquant_mlx/quantizers/palu.py`: `head_group_bounds()`, `group_head_svd()`, `project_to_latent()`, `reconstruct_from_latent()`, `quantize_latent()` (reuses the SVDq mixed-bit latent coder)
- New `KVCacheConfig` fields: `palu_rank`, `palu_energy_threshold`, `palu_n_head_groups`, `palu_hi_bit`, `palu_lo_bit`, `palu_hi_fraction`, `palu_group_size`, `palu_quantize_values`
- 13 tests in `tests/cache/test_palu_cache.py` + 9 in `tests/quantizers/test_palu.py`: factory dispatch, no-bits-leak, group projections stored, shape (prefill + decode), **latent-storage assertion** (buffers hold `[S, r]`, parent `keys is None`), PALU-beats-naive-2bit on both K and V, decode accumulation + offset growth, both-tensors-compressed accounting, low-rank-only values, sub-2-bit effective rate, energy-threshold rank, head-grouping, group SVD subspace recovery, determinism
- `benchmark_scripts/benchmark_palu.py` — throughput + memory sweep vs SVDq, KIVI, fp16, plus an offline full-KV reconstruction-MSE harness (PALU vs naive 2-bit on low-rank K and V)

### Fixed
- `KVCacheBuilder.for_model()` now propagates **all** method-specific config fields (`svdq_*`, `kitty_*`, `kvquant_*`, `palu_*`, …) to each per-layer cache via `dataclasses.replace`. Previously it rebuilt the per-layer config field by field and silently dropped method hyperparameters, so methods built through `for_model` fell back to defaults regardless of what the user passed.

---

## v0.14.0

### New
- **KVQuant-NUQ** (`method="kvquant"`) — non-uniform quantization datatype plus dense/sparse outlier isolation, the repo's first method that places quantization levels by the data distribution rather than uniformly. For each group it fits `2^bits` signpost levels via online 1-D Lloyd-Max (k-means), and carves the top-magnitude `outlier_fraction` of elements out to an fp16 sparse side-channel so a handful of outliers cannot stretch the level range. Keys are quantized per-channel (levels frozen after prefill), values per-token. At equal bit-width this strictly reduces reconstruction error on non-uniform K/V — measured ~73% lower MSE than uniform at 3-bit on Laplacian data. Zero calibration. A faithful adaptation of KVQuant (arXiv:2401.18079, NeurIPS 2024): we implement the two cache-observable pillars (NUQ + dense/sparse) and document the third (pre-RoPE key quantization, which needs a model-forward hook) as out of scope.
- `KVQuantKVCache` — new cache wrapper in `veloxquant_mlx/cache/kvquant_cache.py`
- NUQ utilities in `veloxquant_mlx/quantizers/kvquant.py`: `fit_nuq_levels()` (Lloyd-Max), `quantize_nuq()`, `dequant_nuq()`, `split_dense_sparse()` (outlier isolation), `nuq_quant_dequant()` (drop-in for `_group_quant_dequant`), `nuq_distortion()`
- New `KVCacheConfig` fields: `kvquant_bits`, `kvquant_outlier_fraction`, `kvquant_group_size`, `kvquant_lloyd_iters`, `kvquant_refit_interval`
- 15 new tests in `tests/cache/test_kvquant_cache.py`: factory dispatch, shape (prefill + decode), value reconstruction, NUQ-beats-uniform on non-uniform data, NUQ-not-worse on uniform data, Lloyd-Max monotone convergence, top-k outlier selection, outlier isolation lowers MSE, `outlier_fraction=0` pure-NUQ, level-table determinism, frozen-key-levels decode, byte accounting, effective-bits range, per-channel/per-token axis correctness, determinism
- `benchmark_scripts/benchmark_kvquant.py` — throughput + memory sweep over `bits ∈ {2,3}` and an outlier ablation vs KIVI (uniform), SVDq, fp16, plus offline NUQ-vs-uniform reconstruction MSE

---

## v0.13.0

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
