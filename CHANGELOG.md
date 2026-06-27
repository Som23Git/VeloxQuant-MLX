# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

> Detailed release notes for 0.10.0–0.14.0 (SVDq, Kitty, AdaKV-proxy, XQuant,
> KVQuant-NUQ) live in the docs-site changelog
> (`docs-site/docs/changelog.md`). The entries below cover the latest releases
> and the original 0.9.0 baseline.

## [0.16.0] — 2026-06-26

### Added — CacheGen: entropy-coded KV cache (`method="cachegen"`)

- **`veloxquant_mlx.cache.cachegen_cache.CacheGenKVCache`** — the repo's first
  **entropy-coded** cache. *Inspired by, not a faithful port of,* "CacheGen: KV
  Cache Compression and Streaming for Fast LLM Serving" (Liu et al., **SIGCOMM
  2024**, arXiv:2310.07240). Every other method packs codes at a fixed
  bit-width; CacheGen exploits token-wise locality (adjacent tokens' KV are
  similar) by applying a reversible token-delta transform to the quantized codes
  and compressing the low-entropy residual stream toward its Shannon entropy.
  Reconstruction is identical to plain group quant (lossless over the codes).
- **Adaptation:** rather than ship a serial range codec (which would bottleneck
  MLX's parallel decode), the entropy-coded byte size is modelled from the
  measured symbol entropy of the delta stream and **capped at the fixed-width
  packed size** — a real coder falls back to raw packing when the stream is
  incompressible, so savings are never negative (exactly 0% on iid data, ~10–17%
  on token-correlated data).
- Primitives in `veloxquant_mlx/quantizers/cachegen.py`: `quantize_to_codes`,
  `dequant_codes`, `token_delta`, `symbol_entropy_bits`, `entropy_coded_bytes`,
  `fixed_width_bytes`, `cachegen_quant_dequant`.
- Config: `cachegen_bits`, `cachegen_group_size`, `cachegen_use_delta`.
- **Tests** — `tests/cache/test_cachegen_cache.py` (12) +
  `tests/quantizers/test_cachegen.py` (9): lossless reconstruction vs group
  quant, reversible token-delta, delta-entropy < raw-entropy on correlated data,
  positive savings on correlated / never-negative on iid, entropy primitives
  (0 for constants, 1 bit for 50/50, bounded by log2-alphabet), byte-accounting
  ordering, decode, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_cachegen.py` (offline entropy
  harness + throughput vs KIVI/fp16). **Not yet run.**

### Added — MiniCache: cross-layer depth-dimension merge (`method="minicache"`)

- **`veloxquant_mlx.cache.minicache_cache.MiniCacheKVCache`** +
  **`MiniCacheCoordinator`** — cross-layer compression in the **depth
  dimension**. *Inspired by* "MiniCache: KV Cache Compression in Depth Dimension
  for Large Language Models" (Liu et al., **NeurIPS 2024**, arXiv:2405.14366).
  Adjacent middle-to-deep layers have nearly identical KV directions, so a pair
  is merged into one shared **SLERP**-interpolated direction plus each layer's
  own per-token magnitude (a pair costs ~one layer). High-divergence token pairs
  are kept unmerged (the retention set). A different route to inter-layer
  redundancy than XQuant — XQuant reuses quantized *codes*, MiniCache merges the
  *tensors*.
- **Adaptation:** faithful to the magnitude/direction SLERP + token retention;
  integrated via a shared coordinator (the XQuant pattern) rather than a modified
  attention forward. The primary layer publishes its KV so the later-arriving
  merge layer can perform the merge — both then reconstruct from the shared
  direction.
- Primitives in `veloxquant_mlx/quantizers/minicache.py`: `pair_layers_depth`,
  `to_mag_dir`, `slerp`, `merge_pair`, `reconstruct_layer`, `merge_similarity`.
- Config: `minicache_start_frac`, `minicache_group_size`,
  `minicache_retention_threshold`, `minicache_slerp_t`, `minicache_max_ctx`.
- **Tests** — `tests/cache/test_minicache_cache.py` (11) +
  `tests/quantizers/test_minicache.py` (11): role assignment (early all primary,
  deep has merge), SLERP endpoints/unit-norm/collinear-fallback, similar layers
  merge MSE < 2e-4 with 0% retention, opposite directions 100% retained and
  reconstructed exactly, magnitude preservation, `n_retained+n_merged==total`,
  degenerate lossless passthrough, coordinator `max_ctx` guard, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_minicache.py` (offline merge-
  quality harness + throughput vs XQuant/KIVI/fp16). **Not yet run.**

### Honest scope

- Both are **storage**-compression methods: CacheGen's entropy coding and
  MiniCache's merge reduce stored cache size but reconstruct fp16 for SDPA, so
  neither reduces working-set memory at attend time. On Apple Silicon's
  bandwidth-bound decode they are lower-leverage than the low-rank (PALU/SVDq)
  and quantization methods.
- Quality evidence is unit-test level (synthetic data); no model-level benchmark
  or downstream-task evaluation has been run.

## [0.15.0] — 2026-06-26

### Added — PALU: true low-rank latent storage for keys *and* values (`method="palu"`)

- **`veloxquant_mlx.cache.palu_cache.PALUKVCache`** — the first method in the
  suite where the KV cache *itself* stays low-rank. *Inspired by, not a faithful
  port of,* "PALU: Compressing KV-Cache with Low-Rank Projection" (Chang et al.,
  **ICLR 2025**, arXiv:2407.21118). At prefill it partitions the attention heads
  into `palu_n_head_groups` contiguous groups and fits one shared projection per
  group via group-head SVD (PALU's G-LRD), then stores the projected codes
  `[S, r]` **directly** — full fp16 keys/values are reconstructed only at attend
  time. The latents are mixed-bit quantized (top-25% of channels by singular
  value at 4-bit, the rest at 2-bit, reusing the SVDq latent coder) for a
  full-KV effective rate below 1 bit/element on low-rank data. Unlike SVDq
  (keys-only, reconstructs full fp16 and so wins on byte-accounting/bandwidth),
  PALU bypasses the parent `mlx_lm` fp16 ring buffer entirely and tracks its own
  offset — the stored-cache win is real.
- **`veloxquant_mlx.quantizers.palu`** — pure primitives `head_group_bounds`,
  `group_head_svd`, `project_to_latent`, `reconstruct_from_latent`,
  `quantize_latent`.
- **`KVCacheConfig`** — new fields `palu_rank`, `palu_energy_threshold`
  (default 0.90), `palu_n_head_groups` (default 4), `palu_hi_bit`, `palu_lo_bit`,
  `palu_hi_fraction`, `palu_group_size`, `palu_quantize_values` (default True;
  `False` → low-rank-only with fp16 latents).
- **Tests** — `tests/cache/test_palu_cache.py` (13) + `tests/quantizers/test_palu.py`
  (9): factory dispatch, no-`.bits`-leak, group projections stored,
  prefill/decode shape, the **latent-storage assertion** (buffers hold `[S, r]`,
  parent `keys is None`), PALU-beats-naive-2bit on **both** K and V, decode
  accumulation + offset growth, both-tensors-compressed accounting,
  low-rank-only values, sub-2-bit effective rate, energy-threshold rank,
  head-grouping, group-SVD subspace recovery, determinism.
- **Benchmark** — `benchmark_scripts/benchmark_palu.py` (fp16 / KIVI-2bit /
  SVDq / PALU-LR-only / PALU-LR+mixed / PALU-aggressive) plus an offline
  full-KV reconstruction-MSE harness. **Not yet run** — no throughput or
  compression figures are claimed for this method until its `results.json` is
  committed.

### Fixed

- `KVCacheBuilder.for_model()` now propagates **all** method-specific config
  fields (`svdq_*`, `kitty_*`, `kvquant_*`, `palu_*`, …) to each per-layer cache
  via `dataclasses.replace`. Previously it rebuilt the per-layer config field by
  field and silently dropped method hyperparameters, so any method built through
  `for_model` ran with default hyperparameters regardless of the user's config.

### Honest scope

- PALU's fused low-rank-reconstruction attention kernel is **not** ported — we
  reconstruct fp16 then call MLX SDPA. The storage is low-rank, but the working
  set during attention is briefly the reconstructed fp16 K/V, so peak memory at
  attend time is not reduced — only the stored cache size. Documented as a known
  simplification.
- Quality evidence is unit-test level (synthetic low-rank data); no model-level
  benchmark or downstream-task evaluation has been run.

## [0.9.0] — 2026-06-12

### Added — KVSink-adapted sink protection (`method="kivi_sink"`)

- **`veloxquant_mlx.cache.sink_cache.SinkProtectedKVCache`** — dynamic
  attention-sink protection layered on KIVI group quantization. *Inspired
  by, not a faithful port of,* "KVSink: Understanding and Enhancing the
  Preservation of Attention Sinks in KV Cache Quantization for LLMs"
  (Su & Yuan, **COLM 2025**, arXiv:2508.04257): the paper detects sinks via
  hidden-state outlier channels at a model-specific emergence layer, which
  cache wrappers cannot see; this implementation uses the cache-observable
  proxy of **anomalously high key L2-norm** (mean over KV heads, running
  top-k of absolute positions). Selected tokens are kept fp16 and —
  critically, per the paper — **excluded from quantization-parameter
  calibration** (sink rows are replaced by the nearest non-sink row before
  group min/max is computed; without this, a large-magnitude sink inflates
  its group's scale and ruins every neighbor even though the sink itself is
  restored — our tests reproduce that failure when calibration exclusion is
  omitted).
- **`KVCacheConfig.n_sink_tokens`** — new field (default 5, the paper's k).
  Composes with KIVI's `residual_length` window; byte accounting tracks
  `sink_fp16_bytes` separately from `residual_fp16_bytes` with no double
  counting. `n_sink_tokens=0` reproduces plain KIVI bit-for-bit (tested).
- **Tests** — `tests/cache/test_sink_cache.py` (9 tests): planted-sink
  detection + bit-exact fp16 preservation; sink-protected MSE < plain KIVI
  at equal bit-width; **dynamic selection MSE < Preserve-First-N at equal
  fp16 budget** (the KVSink paper's central claim, reproduced at cache
  level on synthetic planted-sink data); accounting partition; determinism.
  Full suite: **344 passed / 348 collected** (4 pre-existing flaky VecInfer
  parity tests, unrelated).
- **Benchmark script** — `benchmark_scripts/benchmark_sink.py` (fp16 /
  KIVI-2bit / +sink k=5 / +sink k=20, long-prompt protocol). **Not yet
  run** — no throughput or compression figures are claimed for this method
  until its `results.json` is committed.

### Honest scope

- Known v1 limitation: sink selection is **prefill-dominant** — tokens
  quantized in earlier calls are not retroactively restored if they later
  qualify as sinks. Sinks emerge among early tokens in practice, which
  arrive in the prefill block where protection is fully effective.
- Quality evidence is unit-test level (synthetic planted sinks); no
  model-level benchmark or downstream-task evaluation has been run.

## [0.8.0] — 2026-06-10

### Added — KIVI: tuning-free asymmetric group quantization (baseline)

- **`veloxquant_mlx.quantizers.kivi.KIVIQuantizer`** — re-implementation of
  "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (Liu, Yuan
  et al., **ICML 2024**, arXiv:2402.02750). Deterministic asymmetric min/max
  group quantization: **per-channel keys** (group along the token axis) and
  **per-token values** (group along the channel axis). No codebook training,
  no rotation, no RNG. Registered as `"kivi"` in `QuantizerRegistry`.
- **`veloxquant_mlx.cache.kivi_cache.KIVIKVCache`** — mlx_lm
  `update_and_fetch` wrapper. Keeps the most-recent `residual_length` tokens
  in fp16 (KIVI's residual window) and quantizes only tokens that age out.
  Full byte-accounting (`compressed_key_bytes`, `fp16_key_bytes`,
  `residual_fp16_bytes`); never exposes `.bits`. Selectable via
  `KVCacheConfig(method="kivi", bit_width_inlier=2, kivi_group_size=32,
  residual_length=32)`.
- **`KVCacheConfig.kivi_group_size`** — new field (default 32).
- **Benchmarks** — `benchmark_scripts/benchmark_kivi.py` records throughput,
  peak memory, and realized key / full-KV compression with a **real fp16
  baseline timing** and a `hardware` block, under
  `figures/kivi/<model>/results.json`. Measured on Llama-3.2-3B, Qwen2.5-7B,
  Mistral-7B (Apple M4): **KIVI-2bit ≈ 5.8× key / ≈ 4× full-KV at 100–106%
  of fp16 throughput**.
- **Figures** — `scripts/plot_kivi.py` emits four figures (compression vs
  quality, throughput, analytic memory-at-scale, KIVI-vs-VecInfer) +
  `figures/kivi/results_summary.json`, all read from committed JSONs.
- **Tests** — `tests/quantizers/test_kivi.py` and
  `tests/cache/test_kivi_cache.py`: shape/dtype, deterministic seeded
  reconstruction cosine/SNR per bit-width, monotone-quality-in-bits,
  residual-window correctness, byte-accounting, no-`.bits`-leak. **+25 tests
  (334/339 pass; the 5 failures are the pre-existing flaky VecInfer parity
  tests documented in `paper/EVIDENCE_TABLE.md`, unrelated to KIVI).**

### Honest scope

- KIVI's published *speedup* is a CUDA kernel that does not port to Metal; on
  Apple Silicon the win is **memory**, not raw speed.
- Compression only manifests once context exceeds the residual window; at
  short prompts the whole prefill stays fp16 (realized ratio 1.0×).
- Peak runtime memory is **not** reduced (keys dequantize to fp16 before SDPA).
- KIVI-2bit is genuinely lossy on raw keys (synthetic cosine ~0.93); VecInfer
  compresses harder. KIVI's role is the recognized, calibration-free baseline.

## [0.5.1] — 2026-05-25

### Added — Metal compute kernels for VecInfer (Phase 1)

- **`veloxquant_mlx.metal`** — new subpackage with hand-written Metal
  Shading Language shaders that replace pure-MLX hot paths in
  `VecInferKVCache`. JIT-compiled on first use via `mx.fast.metal_kernel`.
  - `vecinfer_quantize_metal` — fused nearest-centroid argmin. Squared
    distance is accumulated in thread-local registers so the kernel never
    materializes the `[chunk, n_centroids, sub_dim]` diff tensor that
    OOMed Falcon3-7B-style configurations on the pure-MLX path.
    **Measured: 6.9–13× speedup, 98% peak-memory reduction at the OOM
    trigger shape (head_dim=256, n_centroids=256, sub_dim=4).**
  - `vecinfer_dequant_metal` — bit-exact drop-in for `dequantize_vq`.
    Ships at MLX `mx.take` parity (no speedup); included as a building
    block for the Phase-2 fused dequant+SDPA kernel.
  - `metal_available()` capability probe.
- **`KVCacheConfig.use_metal_kernels`** — three-state opt-in flag.
  `None` (default) auto-detects, `True` requires Metal, `False` forces
  the pure-MLX path for debugging/parity testing.
- **`VecInferKVCache`** now dispatches to the Metal kernels when
  available — zero public-API change. Existing benchmark scripts pick
  up the speedup automatically.
- **Tests**: `veloxquant_mlx/tests/cache/test_vecinfer_metal_parity.py`
  — 7 new tests covering flag resolution, shape/dtype preservation,
  reconstruction-MSE parity vs pure-MLX, no `.bits` leak, byte-account
  consistency, head_dim=256 sanity. **All 212 tests pass.**
- **Scripts** (`scripts/`):
  - `metal_quantize_proof.py` — correctness + speedup + memory benchmark.
  - `metal_dequant_proof.py` — same for the dequant kernel.
  - `metal_end_to_end_smoke.py` — `mlx_lm.generate` parity smoke test.
  - `metal_falcon3_unblock.py` — Falcon3-7B-shape sanity check.

### Notes

- Phase 2 (fused dequant+SDPA so fp16 keys are never materialized) is
  scoped but not yet implemented.
- The dequant kernel is at-parity with MLX's tuned `mx.take`; the win
  here is the quantize kernel.

## [0.5.0] — 2026-05-23

### Added — VecInfer (vector quantization with outlier-suppressing dual transform)

- **`veloxquant_mlx.allocators.vecinfer`** — algorithmic primitives for
  VecInfer (arxiv:2510.06175, Yao et al. 2025):
  - `calibrate_smooth_factors(keys)` → per-(head, channel) `lambda_i = sqrt(max|K_i|)`.
  - `walsh_hadamard_matrix(d)` → orthonormal rotation; `d` must be power-of-2.
  - `apply_dual_transform_keys / queries` → preserve `q @ K.T` under
    smooth + Hadamard (Eq. 7), with GQA fallback when smooth was
    calibrated on more heads than the cache stores.
  - `train_codebook`, `quantize_vq`, `dequantize_vq` → product VQ with a
    pure-numpy Lloyd's k-means.
  - `compute_query_lut` → optional fused-score fast path.
- **`veloxquant_mlx.cache.vecinfer_cache.VecInferKVCache`** — mlx_lm
  `update_and_fetch` wrapper that quantizes and immediately dequantizes
  keys/values so downstream SDPA sees standard fp16 tensors. Tracks
  `compressed_key_bytes`, `fp16_key_bytes`, `codebook_bytes`,
  `assigned_avg_bits`. Selectable via `KVCacheConfig(method="vecinfer", ...)`.
- **Benchmarks**: 8× key compression at 2-bit, 16× at 1-bit on
  Llama-3.2-1B/3B-Instruct-4bit. Plots and `results.json` under
  `figures/vecinfer/<model>/`. Run:
  `PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py --model <hf-id>`
- **Tradeoff**: throughput drops vs fp16 (the paper's CUDA kernel fusion
  is not portable to Metal). The win on Apple Silicon is memory
  compression, not raw speed.
- 18 new tests (`tests/allocators/test_vecinfer.py`,
  `tests/cache/test_vecinfer_cache.py`).

---

## [0.3.6] — 2026-05-17

### Breaking Change — Package namespace renamed

- **`mlx_kv_quant` → `veloxquant_mlx`**: The Python import namespace now
  matches the PyPI distribution name `VeloxQuant-MLX`. All imports must be
  updated: `from mlx_kv_quant import ...` → `from veloxquant_mlx import ...`.
  No backward-compatibility shim is provided; this is a clean break at pre-1.0.

---

## [0.3.5] — 2026-05-16

### Added — RateQuant becomes a first-class library feature

- **`veloxquant_mlx.allocators.allocate_bits_ratequant`** — RateQuant Theorem 2
  closed-form reverse-waterfilling allocator (arxiv:2605.06675). Given a list
  of per-layer sensitivities and a fractional `target_avg_bits`, returns an
  integer-valued list of bit-widths whose mean exactly matches the target.
  Defaults match the paper's RVQ-fitted β=3.5; configurable per quantizer.
- **`veloxquant_mlx.allocators.calibrate_layer_sensitivities`** — one-pass
  activation-norm probe. Runs 8 default calibration prompts (overridable),
  collects per-token squared key L2 norm via a transparent KV-cache subclass.
  Returns one float per attention layer; ratios above ~2× indicate
  RateQuant will deliver measurable gains.
- **`veloxquant_mlx.allocators.fit_distortion_curve`** — least-squares fit of
  `D(b) = α·β^(-b)` on synthetic unit-norm Gaussian keys. Use this if
  adapting the allocator to a different quantizer family (paper reports
  β≈5.0 for KIVI/QuaRot vs 3.5 for TurboQuant).
- **`KVCacheConfig.bit_width_inlier`** now accepts `int | list[int]`.
  When a list is supplied, `KVCacheBuilder.for_model(model, config)` consumes
  element `i` for layer `i`. Length mismatch raises `QuantizerConfigError`.
  `KVCacheFactory.create()` continues to require an int (the list path
  dispatches through `for_model` to per-layer factory calls).
- **`veloxquant_mlx.cache.turboquant_rvq_cache.TurboQuantRVQKVCache`** —
  library-grade mlx_lm-compatible cache wrapper around `TurboQuantRVQ`.
  Exposes `compressed_key_bytes`, `fp16_key_bytes`, and `assigned_bits`
  (never `bits` — that name collides with mlx_lm's quantized-SDPA dispatch).
- **`veloxquant_mlx.observers.KeyNormObserver`** and `KeyNormReport` —
  event-driven observer that accumulates per-token key L2 norm² and reports
  mean / min / max plus a `heterogeneity_ratio` property (predicts RateQuant
  benefit).
- **`turboquant_rvq` registered** in `KVCacheFactory.create()` — users can
  now configure RVQ via `method="turboquant_rvq"` in `KVCacheConfig` without
  manually constructing the cache class.
- **27 new tests** across `tests/allocators/`, `tests/observers/`, and
  `tests/cache/test_turboquant_rvq_cache.py`. Full suite: 187 passing.

### Changed
- `KVCacheBuilder.with_bit_width(inlier=...)` now accepts a list for
  per-layer RateQuant allocations.
- Top-level package re-exports `allocate_bits_ratequant`,
  `calibrate_layer_sensitivities`, `fit_distortion_curve`,
  `KeyNormObserver`, and `KeyNormReport`.
- `pyproject.toml`: version 0.3.5; added `maintainers`, `Author`, `Changelog`,
  `Documentation` URLs so PyPI displays attribution cleanly.

### Results (RateQuant V2 trial — 2 models on Apple M4 24 GB)

| Model | fp16 | RVQ 1-bit | **RVQ + RateQuant V2** (b̄=1.5) | sensitivity ratio |
|---|---|---|---|---|
| Falcon3 7B | 22.9 | 23.1 | **22.8 (100%)** at 5.22× | 6.48× |
| Gemma3 4B | 39.8 | 37.8 | **36.3 (91%)** at 5.22× | 14.39× |

> Per-layer bit allocations from 1.6s real-activation calibration:
> Falcon3 = 14/14 (b=2/b=1); Gemma3 = 3/11/20 (b=3/b=2/b=1).
> Source figures: [`figures/2026-05-16/`](figures/2026-05-16/).

### Known limitations vs paper
- **Per-head granularity** not implemented (paper: L×H groups, ours: L).
  mlx_lm's cache is per-layer; adding per-head requires splitting the cache
  layout. Estimated gain left on the table: ~30% of the paper's headline
  improvement.
- **Gradient-based sensitivity** not implemented (paper uses gradient,
  notes activation is ~1 PPL worse but both beat uniform). Gradient requires
  backprop through `mlx_lm.generate`, which is not currently practical.
- **K/V separate budgets** not implemented (paper's biggest single fix on
  KIVI). Our cache currently only quantizes keys; values pass through fp16.

## [0.3.4] — 2026-05-15

### Added
- **`OutlierTokenRVQMLXKVCache`** (arxiv:2505.10938, ACL 2025) — RVQ 1-bit
  cache that routes high-L2-norm "sink" tokens through an fp16 side buffer
  at prefill. Vectorized mask-blend implementation (no scatter) keeps decode
  S=1 overhead-free. Catches 0.05–0.09% of tokens on Phi-4, Qwen3, Llama,
  Gemma3 — exactly the sink-token pattern the paper predicts.
- **`RateQuantRVQMLXKVCache`** (arxiv:2605.06675) — per-layer integer bit
  allocation via reverse-waterfilling on a fitted distortion curve
  D(b) = α·β^(-b). Computed once at construction, zero inference overhead.
  Uses `.assigned_bits` (not `.bits`) to avoid triggering mlx_lm's quantized
  SDPA path that expects a different cache layout.
- **`benchmark_scripts/outlier_ratequant_core.py`** — 4-config figure
  pipeline (fp16, RVQ 1-bit, RVQ 1-bit + Outlier, RVQ + RateQuant) with
  a dedicated palette and the same 6-PNG layout as `_generate_figures_v3`.
- **`benchmark_scripts/run_outlier_ratequant.py`** — 8-model × 4-config
  benchmark runner with subprocess isolation. Outputs to
  `figures/outlier_token_ratequant/<model>/`.
- **`docs/MEMORY_CONSTRAINT_FINDINGS.md`** — documents the Qwen2.5-32B
  memory-headroom constraint on 24 GB Apple M4 and the watchdog mechanism
  added to protect the GPU from OOM-driven kernel events.
- **`.github/workflows/copyright-watch.yml`** — weekly GitHub Actions job
  that searches the public code index for distinctive class names
  (TurboQuantRVQMLXKVCache, OutlierTokenRVQMLXKVCache, etc.) and fails
  the workflow on any hit, triggering an email per GitHub notification
  settings.
- **`NOTICE`** — explicit attribution-requirements notice that strengthens
  the MIT license terms for DMCA purposes.

### Results (OTRQ sweep, 7 of 8 models, Apple M4 24 GB)

Outlier-Token RVQ matches or **beats fp16 throughput** on 5 of 7 models at
7.5× compression:

| Model | fp16 | RVQ 1-bit | RVQ 1-bit + Outlier | vs fp16 |
|---|---|---|---|---|
| Mistral 7B | 21.4 | 21.9 | **22.2** | **104%** |
| Phi-4 | 10.3 | 9.1 | **11.3** | **110%** |
| Qwen3 4B | 38.9 | 34.7 (187 tok) | **35.7 (196 tok)** | 92% + better completeness |
| Qwen3 8B | 19.6 | 17.1 | **20.3** | **104%** |
| Gemma3 4B | 35.9 | 34.7 | **36.5** | **102%** |
| Llama 3.1 8B | 18.8 | 17.5 | 17.9 | 95% |
| Falcon3 7B | 23.4 | 22.5 | 21.8 | 93% |

Qwen2.5-32B-Instruct-4bit could not complete any non-fp16 OTRQ config on
24 GB unified memory — see `docs/MEMORY_CONSTRAINT_FINDINGS.md`.

### Engineering note
- **Watchdog for large-model runs**: a memory-pressure poller
  (`/tmp/memory_watchdog.sh`) terminates the benchmark process tree if
  free + inactive memory drops below 1 GB. Validated: the watchdog caught
  the Qwen2.5-32B run at 891 MB free and killed cleanly before MLX could
  fault the Metal heap.

## [0.3.3] — 2026-05-12

### Added
- **RVQ 1-bit quantizer** — `TurboQuantRVQ(b=1)` is now fully supported.
  Stage 1 is a 2-level sign quantizer ({−0.798, +0.798} Gaussian Lloyd-Max);
  stage 2 applies a 2-level Laplacian correction to the sign-quantization error.
  Achieves **cosine 0.917 / SNR +7.6 dB** at d=128 on synthetic data, and
  **201 coherent tokens at 97–98% of fp16 throughput** on Mistral 7B and Qwen3 8B.
  Per-vector storage: `ceil(d / 4) + 2` bytes → **7.5× key compression** at d=128.
  Docstring updated with supported bit-widths (b=1, 2, 3+) and expected quality.
- **`benchmark_scripts/run_full_reports.py`** — model-agnostic 8-model × 6-config
  sweep orchestrator. Spawns one fresh Python subprocess per (model, config) to
  guarantee clean MLX graph state. Outputs `figures/2026-05-12/<model>/` with the
  full 6-figure v3 report. Idempotent: skips completed models/configs unless `--force`.
- **`_generate_figures_v3` + `run_benchmark_v3_from_results`** in `benchmark_core.py`
  — v3 figure pipeline extended to 6 configs (fp16 / TQ 2-3-4-bit / RVQ 2-bit ★ /
  RVQ 1-bit ★). New RVQ-1bit ★ traces appear in all 6 figures. Original v2 functions
  left untouched.
- **`benchmark_scripts/run_text_sweep.py`** — lightweight sweep runner used for
  fp16/RVQ-1/RVQ-2/TQ-4 comparison across models; results go to `figures/updated_tests/text_sweep/`.
- **`benchmark_scripts/diagnose_vlm_key_stats.py`** — VLM key-distribution diagnostic.
  Hooks into each layer's `update_and_fetch` to capture real key tensors, then reports
  per-layer L2 norm (image vs text tokens), post-rotation kurtosis, and RVQ-2bit cosine.
  Saves histograms to `figures/updated_tests/qwen2_vl/key_stats/`.
- **`benchmark_scripts/benchmark_qwen2_vl.py`** rewritten with `--run-config` subprocess
  isolation mode. Fixes the MLX graph-reuse bug that caused 2nd+ configs to produce
  0 tokens in the same process.

### Changed
- **`_read_model_cfg()` in `benchmark_core.py`** — new helper that robustly reads
  `(head_dim, n_kv_heads, n_layers)` from any mlx_lm model, handling:
  - Standard text models (Mistral, Qwen3, Llama, Phi) via `model.args`.
  - VLM-style wrappers where `model.args.text_config` is a plain `dict` (Gemma3, Qwen2-VL).
  - GQA models (Gemma3) where `hidden_size // n_heads` gives the wrong `head_dim` —
    always uses direct `attn.head_dim` from layer inspection instead of derived formula.
- **`TurboQuantMLXKVCache` and `TurboQuantRVQMLXKVCache` `update_and_fetch`** —
  dtype-aware norm handling. Safe-norm threshold and scale factor now use `keys.dtype`
  (bfloat16 for Qwen2-VL-7B-bf16, float16 for most text models) instead of always
  casting to float16. Eliminates a redundant cast and preserves the wider exponent
  range of bfloat16 for large-norm image-patch keys.
- **`test_2bit_improvements.py`** — added RVQ b=1 synthetic check (`Extra TQ-RVQ (b=1 x2)`,
  cosine 0.9165) with assert `cosine > 0.80`.

### Fixed
- **Gemma3 `head_dim` detection** — `_read_model_cfg` previously derived `head_dim`
  as `hidden_size // num_attention_heads = 2560 // 8 = 320`, but Gemma3's actual
  per-head dimension is 256. Now reads `attn.head_dim` directly from the layer.
- **VLM benchmark prompt** — `benchmark_qwen2_vl.py` previously rejected the
  Qwen2-VL chat template (which ends with `<|im_start|>assistant\n`) and fell back
  to raw text, degrading quantized output quality. Now always uses the full chat
  template unconditionally.

### Results (v3 sweep, Apple M4 16GB, figures/2026-05-12/)

Full 6-config benchmark across 8 models (Apple M4 16GB):

| Model | fp16 tok/s | RVQ 1-bit ★ | RVQ 2-bit ★ | TQ 4-bit | RVQ 1-bit compr. | vs fp16 |
|---|---|---|---|---|---|---|
| Mistral 7B v0.3 | 23.3 | **22.2** (201 tok) | 22.5 (201) | 21.4 (201) | 7.53× | **95%** |
| Falcon3 7B | 24.0 | **23.1** (200 tok) | 22.7 (200) | 22.1 (200) | 7.76× | **96%** |
| Phi-4 | 11.9 | **11.8** (200 tok) | 11.7 (200) | 11.4 (200) | 7.53× | **99%** |
| Qwen3 4B | 40.2 | **34.3** (187 tok) | 35.0 (197) | 33.5 (199) | 7.53× | **85%** |
| Qwen3 8B | 20.5 | **21.1** (200 tok) | 20.7 (200) | 19.8 (200) | 7.53× | **103%** |
| Llama 3.1 8B | 22.0 | **21.5** (201 tok) | 20.9 (201) | 20.3 (201) | 7.53× | **98%** |
| Gemma3 4B | 32.5 | **30.5** (201 tok) | 29.2 (201) | 27.7 (201) | 7.76× | **94%** |
| Qwen2.5 32B | 3.7 | **3.9** (200 tok) | 4.2 (200) | 3.9 (200) | 7.53× | **107%** |

Notable: on Qwen3-8B, Phi-4, and Qwen2.5-32B, RVQ configs **match or exceed fp16 throughput** (all memory-bandwidth bound). At 32B scale, RVQ 2-bit achieves 4.2 tok/s vs fp16's 3.7 tok/s (114%) — the KV-cache compression benefit grows with model size. TQ single-pass 2-bit degrades severely on Qwen2.5-32B (5 tokens) and is not suitable for this model; RVQ consistently delivers full outputs across all models and bit-widths.

## [0.3.2] — 2026-05-12

### Added
- VLM support for **Qwen2-VL-7B-Instruct-bf16** via `build_vlm_caches()` and
  `KVCacheBuilder.for_model()`.
- `benchmark_scripts/benchmark_qwen2_vl.py` — VLM benchmark with image+text prompt
  capability (text-only path validated; image path requires mlx-vlm).

## [0.3.1] — 2026-05-10

### Changed
- README restructured with TOC, algorithm picker table, per-model benchmark tables,
  and throughput optimization journey. All emojis removed for plain-text rendering.
- Distribution metadata now reflects the new structure.

## [0.3.0] — 2026-05-10

### Added
- **`TurboQuantRVQ`** — two-pass Residual Vector Quantization quantizer that lifts
  2-bit cosine similarity from 0.69 → **0.98** and SNR from −0.5 dB → **13.2 dB**.
  Stage 1 uses N(0, 1/d) Lloyd-Max; stage 2 fits a Laplacian PDF on the per-coordinate
  residual. Total storage 2·b bits/dim. Registered as `turboquant_rvq` in the registry.
- **`AdaptiveScalarCodebook`** — wrapper that refits codebook centroids from observed
  post-rotation distribution after a calibration phase. Plumbed via
  `TurboQuantProd(use_adaptive_codebook=True)` and `TurboQuantProdAdaptive`.
- **Adaptive JL sketch dimension** — `TurboQuantProd.m_default(d, b)` now returns
  `d` at b ≤ 2 and `min(d, 64)` at b ≥ 3, doubling the QJL correction budget at 2-bit.
- **Optimization journey figure** — [`figures/updated_tests/optimization_journey.png`](figures/updated_tests/optimization_journey.png).
- **`OPTIMIZATION_FINDINGS.md`** — full writeup of bottleneck analysis and four-stage
  speedup attribution.
- **`benchmark_mistral7b_v2.py`** and **`benchmark_qwen3_4b_v2.py`** — 5-config v2
  benchmark scripts that include `TurboQuantRVQMLXKVCache` alongside the existing
  fp16/2/3/4-bit configurations.
- **`test_2bit_improvements.py`** — synthetic validation script with asserts for all
  three 2-bit accuracy improvements.

### Changed
- **Throughput parity with fp16** for quantized configs on memory-bound models:
  Mistral 7B RVQ 2-bit at 22.3 tok/s vs fp16 22.1 tok/s. Qwen3 4B RVQ 2-bit at
  36.0 tok/s vs fp16 39.2 tok/s (92% of fp16). Achieved via four sequential changes:
  1. Single shared quantizer with `(B·H·S, D)` flat batching (eliminates per-head Python loop).
  2. Hadamard rotation by default in benchmark wrappers (`use_hadamard=True`).
  3. Boundary-sum `quantize()` in `ScalarCodebook` (replaces broadcast-argmin).
  4. Dropped redundant fp32 ↔ fp16 casts in `update_and_fetch`.
- `ScalarCodebook.__init__` now sorts centroids and precomputes Voronoi boundaries
  in `self._boundaries_mx`. `quantize()` returns 100% index-match output vs the prior
  argmin path.
- `TurboQuantMLXKVCache` and `TurboQuantRVQMLXKVCache` in `benchmark_core.py` use a
  single shared quantizer instance instead of `n_kv_heads` separate ones.

### Performance
- Mistral 7B RVQ 2-bit: **17.7 → 22.3 tok/s** (+26%).
- Qwen3 4B RVQ 2-bit: **24.8 → 36.0 tok/s** (+45%).
- Boundary-sum quantize verified bitwise-identical to broadcast-argmin (100.00% index match on synthetic test).

### Quality
- RVQ 2-bit synthetic cosine **0.9766** preserved through every optimization step.
- Real-model output completeness preserved at every step:
  - Mistral 7B: 201/201 tokens across all 5 configs.
  - Qwen3 4B `<think>` mode: 199/200 tokens for RVQ 2-bit (vs 50/200 for single-pass 4-bit).

## [0.2.0] — 2025-05-07

### Added
- Published to PyPI as `VeloxQuant-MLX`
- `veloxquant` CLI entry point (alias for `mlx-kv-quant`)
- 2-bit quantization support in benchmark suite (11.6× compression ratio)
- Per-model benchmark scripts: Falcon3-7B, Mistral-7B, Qwen3-4B, Qwen3-8B, Qwen2.5-32B, Gemma-4, Phi-4
- `benchmark_core.py` unified benchmark runner with 6-figure report generation
- Validated across 7 models: near-lossless at 3-bit and 4-bit; 2-bit degrades gracefully

### Changed
- Package distribution name renamed from `mlx-kv-quant` → `VeloxQuant-MLX`
- Status classifier updated from Alpha → Beta

## [0.1.0] — 2025-04-01

### Added
- Initial implementation of TurboQuant KV cache quantization for Apple Silicon MLX
- PolarQuant and QJL algorithms
- Chain-of-Responsibility quantization pipeline
- Lloyd-Max scalar codebooks
- Random orthogonal rotation preconditioner
- Builder pattern (`KVCacheBuilder`) for fluent cache construction
- Observer framework (latency, memory, distortion)
- Precompute CLI for offline codebook generation
- Full test suite
