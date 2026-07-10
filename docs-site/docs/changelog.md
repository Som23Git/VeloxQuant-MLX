---
id: changelog
title: Changelog
sidebar_label: Changelog
slug: /changelog
---

# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

---

## v0.32.0 — Latest

### New
- **Keyformer-adapted** (`method="keyformer"`) — Gumbel-regularized heavy-hitter eviction. Structurally H2O's proxy-attention accumulator plus **Gumbel noise** on the eviction logits, so a "late riser" (a token that reads low early, before the queries that attend to it arrive) is not deterministically pruned before it can recover. Inspired by "Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient Generative Inference" (Adnan et al., **MLSys 2024**, arXiv:2403.09054) — documented as "Keyformer-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `KeyformerKVCache` (`veloxquant_mlx/cache/keyformer_cache.py`); primitives in `veloxquant_mlx/quantizers/keyformer.py`: `keyformer_update` (additive proxy-attention accumulation + `score + tau·gumbel` eviction ranking), `keyformer_get_kv`, byte helpers, and a deterministic per-position Gumbel draw.
  - Config: `keyformer_budget` (512), `keyformer_n_sink` (4), `keyformer_recent` (0, extension), `keyformer_tau` (1.0; **0 = H2O-adapted**), `keyformer_seed` (0).
  - 29 tests (17 quantizer + 12 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_keyformer.py`).

### Honest scope
- **`keyformer_tau=0` collapses onto H2O-adapted, bit-for-bit** — the only thing Keyformer adds over H2O is the Gumbel regularizer, and a test asserts the `tau=0` kept set equals H2O's; the benchmark prints an `h2o` cross-check column.
- **Frozen per-position Gumbel, not the paper's annealed schedule.** The paper redraws Gumbel noise and anneals a temperature across generation; a cache has no trustworthy global step, so we draw one deterministic Gumbel value per token position (seeded by `keyformer_seed` + a per-head running position) and freeze it. Preserves the "don't doom a borderline token on one low reading" intent; not claimed equivalent to the schedule.
- **Key-as-query proxy** (same as H2O/SnapKV-adapted): the incoming key stands in for the unseen query.
- **Mechanism evidence is the survival rate.** Under constructed late-riser geometry, greedy `tau=0` evicts the planted riser 100% of the time while `tau=6` rescues it ~75% of the time; the downstream probe perturbation is a noisier, regime-dependent secondary effect, reported as-is. No RoPE remapping. Uniform budget/tau across heads. No model-level perplexity/throughput benchmark — offline-synthetic survival-rate, output-perturbation and byte-accounting only.

---

## v0.31.0

### New
- **Q-Filters-adapted** (`method="qfilters"`) — query-agnostic projection eviction, the library's fourth eviction scorer class (after attention/proxy, structural, and intrinsic-norm). Each cached key is scored by its projection onto a single frozen per-head direction; over budget, the highest-scoring tokens are kept (sinks and an optional recent window protected). Inspired by "Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression" (arXiv:2503.02812, **preprint**) — documented as "Q-Filters-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `QFiltersKVCache` (`veloxquant_mlx/cache/qfilters_cache.py`); primitives in `veloxquant_mlx/quantizers/qfilters.py`: `estimate_filter_dir` (top singular vector of the observed keys, frozen after `qfilters_calib_tokens`), `qfilters_update`/`qfilters_get_kv`, byte helpers (K+V fp16 plus the float32 filter direction).
  - Config: `qfilters_budget` (512), `qfilters_n_sink` (4), `qfilters_recent` (0, extension), `qfilters_calib_tokens` (128), `qfilters_sign` (1; -1 = inverted ablation).
  - 27 tests (12 quantizer + 15 cache) and a deterministic offline benchmark (`benchmark_scripts/benchmark_qfilters.py`).

### Honest scope
- **The filter is key-SVD-derived, not query-SVD-derived.** The paper estimates the direction offline from a sample of query vectors; a cache-side library never sees queries, so we substitute the SVD of the first observed *keys*. This recovers the dominant *axis* but not which *end* is important — the sign a query would disambiguate. The committed benchmark shows the key-SVD recovering the planted axis (`filter_cosine ≈ 0.97`) while which raw sign arm wins flips row to row, so `qfilters_sign` is a **genuine ablation**. Nothing here is claimed equivalent to the paper's filter.
- **Path-dependent** (unlike L2Norm): prefill-in-one-block and token-by-token decode can freeze different filters and diverge; there is deliberately no prefill/decode bit-for-bit equivalence guarantee.
- Preprint, no venue. No RoPE remapping after eviction. Uniform budget across heads. `qfilters_recent` is an extension, off by default. No model-level perplexity/throughput benchmark — offline-synthetic output-perturbation and byte-accounting only.

---

## v0.30.1

### Fixed
- **PyPI package metadata only — no code changes.** PyPI mirrors such as pepy.tech showed no summary/version/license/author because the published metadata was malformed for downstream consumers: the Summary was a ~700-character method list (now a one-line summary), the License field embedded the entire MIT license text via `license = { file = "LICENSE" }` (now a PEP 639 SPDX expression, `License-Expression: MIT`), and the `Author:` field was empty (now populated alongside `Author-email:`). Wheel/sdist contents are otherwise identical to 0.30.0.

---

## v0.30.0

### New
- **SKVQ-adapted** (`method="skvq"`) — sliding-window quantization with two mechanisms new to the library: **channel reordering** (permute head-dim channels so channels of similar dynamic range share a quantization group — per-head permutations sorted by range, frozen from the first flushed chunk) and **clipped dynamic quantization** (each group's min/max window shrunk by a per-group grid-searched clip factor α, saturating a few extremes to buy finer resolution everywhere else; α=1 is always in the grid so the search never loses under its own metric). Both K and V quantized with per-token channel groups behind a sliding fp16 window (the NSNQuant chunk-flush idiom) with the paper's attention-sink filter (first `skvq_n_sink` tokens stay fp16). Inspired by "SKVQ: Sliding-window Key and Value Cache Quantization for Large Language Models" (Duanmu, Yuan, Li, Duan, Zhang, Lin, COLM 2024, arXiv:2405.06219) — documented as "SKVQ-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `SKVQKVCache` (`veloxquant_mlx/cache/skvq_cache.py`); primitives in `veloxquant_mlx/quantizers/skvq.py`: `channel_permutation`, `invert_permutation`, `apply_permutation`, `clipped_group_quant`/`clipped_group_dequant` (vectorized per-group α search, α folded into the stored lo/scale — nothing extra kept), `skvq_round_trip`, byte helpers.
  - Prefill and token-by-token decode produce **bit-for-bit identical caches** (chunk boundaries, first-chunk permutation statistics, clip search, and sink restore are all functions of the same chunk contents — pinned by test). Deterministic end to end: no RNG anywhere.
  - Config: `skvq_bits_key`/`skvq_bits_value` (default 2/2), `skvq_group_size` (32), `skvq_window` (128), `skvq_n_sink` (5), `skvq_reorder` (True; False = identity ablation), `skvq_clip_search` (True) / `skvq_clip_alpha`, `skvq_max_ctx`. No coordinator — the default `KVCacheBuilder.for_model()` path returns one `SKVQKVCache` per layer.
  - 13 quantizer tests + 18 cache tests; `benchmark_scripts/benchmark_skvq.py` + committed `skvq_benchmark_results.json` — under a heterogeneous-channel regime (2.5-decade smooth scale spread), reordering cuts key MSE a further **16.9%** on top of clip search and collapses per-channel normalized error ~450×; clip search adds **14.0%** on top of reordering; under the homogeneous control reordering buys **−0.3%** (nothing), reported in full. The repo's KIVI reference wins several heterogeneous rows outright (its per-channel key scheme is intrinsically immune to channel heterogeneity) — reported as measured.

### Honest scope
- The paper's offline calibration (KMeans channel clustering on WikiText-2 + attention-output-MSE clip search, permutation fused into projection weights) is replaced by **first-chunk statistics** (sort by dynamic range; per-group reconstruction-MSE grid search) with an explicit runtime permute/inverse-permute — a documented adaptation, not the paper's pipeline.
- No 1.5-bit value packing and no FP8(E4M3) metadata (both CUDA artifacts); integer bit-widths and fp16 metadata, all counted in the byte accounting.
- That real transformer K/V exhibit the heterogeneous-channel regime is the paper's premise (shared with KIVI/KVQuant), not something the offline-synthetic benchmark can validate — the homogeneous control shows reordering buys nothing without it.
- No model-level (perplexity/throughput) benchmark run.

---

## v0.29.0

### New
- **L2Norm-adapted** (`method="knorm"`) — the repo's first **intrinsic-signal** eviction cache: token importance is read directly off the stored key vector's L2 norm, with the counterintuitive sign the paper reports in trained decoder LMs — **low norm ⇒ high future attention** — so the cache keeps the lowest-norm tokens. No attention scores, no key-as-query proxy (the approximation H2O/SnapKV/TOVA need), no structure-only recency rule: the paper's actual signal is fully observable at the cache level, making this the cleanest adaptation in the eviction family. L2Norm-adapted ("A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression", Devoto, Zhao, Scardapane, Minervini, EMNLP 2024, arXiv:2406.11430) — documented as "L2Norm-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `L2NormKVCache` (`veloxquant_mlx/cache/knorm_cache.py`); primitives in `veloxquant_mlx/quantizers/knorm.py`: `KnormState`, `init_knorm_state`, `knorm_update` (vectorized — one protected top-k per incoming block, no per-token softmax-over-cache loop), `knorm_get_kv`, `knorm_fp16_bytes`, `full_knorm_fp16_bytes`.
  - Because the score is intrinsic (computed once at insertion, never updated): eviction is **~100–800× faster than H2O-adapted** at prefill on the committed harness (0.3 ms vs 240 ms at S=1024), and with `knorm_recent=0` the kept set is **path-independent** — prefill-in-one-block and token-by-token decode produce bit-for-bit identical caches (the "keep k best with a heap" invariant, pinned by test at both the primitive and wrapper level). No accumulating-score method has this property.
  - Config: `knorm_budget` (default 512), `knorm_n_sink` (default 4), `knorm_recent` (default 0 — trailing protected window, an extension beyond the paper; enabling it breaks path independence), `knorm_keep` (`"low"` = paper finding | `"high"` = inverted ablation arm). No coordinator — the default `KVCacheBuilder.for_model()` path returns one `L2NormKVCache` per layer.
  - 10 quantizer tests + 14 cache tests, including the bit-for-bit path-independence check and a mechanism test under paper-like geometry; `benchmark_scripts/benchmark_knorm.py` + committed `knorm_benchmark_results.json` — under geometry constructed to exhibit the paper's correlation, keep-low beats random eviction by **+0.17** mean output perturbation and the inverted scorer by **+0.21**; under the isotropic control the advantage **reverses** (keep-low ~0.07 *worse* than random — softmax favors high-norm keys on isotropic Gaussians), reported in full.

### Honest scope
- The low-norm ⇒ high-attention correlation is the paper's **empirical claim about trained models** — the offline-synthetic benchmark validates the machinery under constructed geometry, not the correlation itself, and the isotropic control shows the method can underperform random eviction when that geometry is absent.
- No RoPE position-ID remapping after eviction; uniform budget and n_sink across heads (same as the rest of the eviction family); `knorm_recent` and `knorm_keep="high"` are extensions beyond the paper, both off by default.
- No model-level (perplexity/throughput) benchmark run.

---

## v0.28.0

### New
- **NSNQuant-adapted** (`method="nsnquant"`) — the repo's first **calibration-free distribution-matching VQ**: instead of fitting a codebook to the data (per-sequence k-means, EM) or using a data-independent geometric code (signs, polar grids), NSNQuant **reshapes the data to match a fixed code**. A Normalize-Shift-Normalize transform (token-norm → channel-mean shift → token-norm) plus a Hadamard rotation maps K/V tokens onto the standard normal distribution, so one codebook built offline from synthetic Gaussian samples — never from model activations — quantizes any model at 1–2 bits/element. NSNQuant-adapted ("NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit Vector Quantization of KV Cache", Son, Choi, Yoo, NeurIPS 2025, arXiv:2505.18231) — documented as "NSNQuant-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `NSNQuantKVCache` (`veloxquant_mlx/cache/nsnquant_cache.py`) — single-layer wrapper, no coordinator, with a chunk-flush fp16 residual buffer (KIVI's idiom): every `nsn_residual_length` tokens flush through the pipeline as one self-contained chunk with its own online channel mean; prefill and decode produce identical quantized state by construction. Primitives in `veloxquant_mlx/quantizers/nsnquant.py`: `nsn_transform`, `nsn_inverse`, `build_universal_codebook`, `vq_encode`, `vq_decode`, `hadamard_forward`/`hadamard_inverse` (reusing `mx.hadamard_transform` via the repo's existing Hadamard infrastructure).
  - Config: `nsn_bits` (default 2: uint8 sign mask + uint8 index per 8-dim subvector = 2 bits/element; 1: index only), `nsn_residual_length` (default 64; paper suggests 128 for 1-bit), `nsn_codebook_size` (default 256), `nsn_subvector_dim` (default 8), `nsn_seed` (default 1234), `nsn_max_ctx` (default 8192). Both keys **and** values quantized, mirroring the paper (unlike the keys-only SVDq/xKV precedent).
  - 16 quantizer tests + 19 cache tests, including a mechanism-validation ablation (on channel-biased input the full NSN pipeline must beat the identical Hadamard+VQ without NSN by a pinned margin) and a prefill-vs-decode path-independence check; `benchmark_scripts/benchmark_nsn.py` + committed `nsn_benchmark_results.json` — NSN gains +0.038 (2-bit) / +0.110 (1-bit) reconstruction cosine over the no-NSN ablation at strong channel bias, honestly collapsing to ~+0.001–0.002 on already-centered input; 0.96–0.98 cosine at ~2.5 effective bits/element (metadata included), beating a KIVI-2bit baseline on every row of the sweep.
  - Honest scope: post-RoPE keys (the paper applies NSN pre-RoPE with a custom kernel — the central simplification of this adaptation), explicit value Hadamard (no projection-layer fusion), spherical-k-means-only codebook (no gradient fine-tune), fp16 metadata (~0.5 bits/element overhead vs the paper's double-quantized ~0.23), no fused kernels, no model-level perplexity/throughput benchmark — offline reconstruction-quality and byte-accounting numbers only.

---

## v0.27.0

### New
- **xKV-adapted** (`method="xkv"`) — the repo's **third cross-layer** mechanism, alongside XQuant (code reuse) and MiniCache (SLERP direction merge). A fixed-size contiguous group of layers jointly factorizes its stacked key matrices into **one shared SVD basis** via a fan-in/fan-out coordinator; every group member then stores only its own latent codes in that shared basis, amortizing the basis storage cost across the whole group. xKV-adapted ("xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction", Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang, Li, Ceze, Wu, Abdelfattah, arXiv:2503.18893, preprint) — documented as "xKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `XKVCache` (`veloxquant_mlx/cache/xkv_cache.py`); `XKVCoordinator` (`veloxquant_mlx/cache/xkv_coordinator.py`) — a fan-in-then-fan-out coordinator, distinct from XQuant/MiniCache's single-publisher pattern since the joint SVD needs every group member's keys before any of them can compress; primitives in `veloxquant_mlx/quantizers/xkv.py`: `pair_layers_grouped`, `joint_svd_compress`, `project_into_shared_basis`, `reconstruct_from_shared_basis`, `quantize_latents_uniform`.
  - Config: `xkv_group_size` (default 2), `xkv_rank` (default `None` -> energy-threshold selection), `xkv_energy_threshold` (default 0.95), `xkv_latent_bits` (default 4), `xkv_group_quant_size` (default 32), `xkv_max_ctx` (default 8192). Keys only — values pass through fp16 unchanged, mirroring SVDq's precedent.
  - 9 quantizer tests + 14 cache tests, including a group-of-1 degeneracy check (`joint_svd_compress` on a single matrix matches SVDq's plain single-layer SVD) and a mechanism-validation test (shared structure across synthetic layers reconstructs better than independent per-layer SVD on unrelated noise at matched rank); `benchmark_scripts/benchmark_xkv.py` + committed `xkv_benchmark_results.json` — sweeps group size (2–4) and a synthetic shared-structure knob, showing near-parity reconstruction MSE (within ~1%) and 8–20% fewer bytes than independent per-layer SVD, improving with larger group sizes.
  - Honest scope: fixed contiguous grouping (no CKA-based layer-alignment validation), no "Selective Reconstruction" decode-time optimization, single-bit-width latent quantization (not SVDq-style mixed-bit routing), no model-level perplexity/throughput benchmark — offline reconstruction-quality and byte-accounting numbers only.

---

## v0.26.0

### New
- **CaM-adapted** (`method="cam"`) — the repo's **eighth eviction configuration** and the first on the **merge-vs-drop** axis. Every other eviction method permanently discards the tokens it evicts; CaM instead **merges** each evicted token into the surviving token it most resembles (a cosine-weighted blend of the value rows, and optionally the keys), then removes only the redundant slot — so the information is folded into a neighbour rather than lost. The eviction *choice* is H2O's; only the disposition differs. With `cam_merge="drop"` it reduces **bit-for-bit** to H2O-adapted. CaM-adapted ("CaM: Cache Merging for Memory-efficient LLMs Inference", Zhang et al., ICML 2024, PMLR 235:58840-58850) — documented as "CaM-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `CaMKVCache` (`veloxquant_mlx/cache/cam_cache.py`); primitives in `veloxquant_mlx/quantizers/cam.py`: `most_similar_survivor`, `merge_pair`, `CaMState`, `init_cam_state`, `cam_update`, `cam_get_kv`, `cam_fp16_bytes`, `full_cam_fp16_bytes`.
  - Config: `cam_budget` (default 512), `cam_n_sink` (default 4), `cam_merge` (`"sim_weighted"` | `"mean"` | `"drop"`, default `"sim_weighted"`), `cam_merge_keys` (default False). No coordinator — each layer merges independently; the default `KVCacheBuilder.for_model()` path returns one `CaMKVCache` per layer.
  - 18 quantizer tests + 14 cache tests, including a bit-for-bit `cam_merge="drop"` == H2O equivalence (identical kept keys *and* values vs `H2OKVCache`) at both the primitive and cache level; `benchmark_scripts/benchmark_cam.py` + committed `cam_benchmark_results.json`.

### Honest scope
- Cosine-similarity merge weight rather than the paper's attention-prominence weight (which is ~0 for a just-appended token that overflows before accumulating mass — the common streaming case); single nearest-survivor merge (no multi-target soft assignment / sampling); key-as-query proxy; no RoPE remapping; uniform budget across heads.
- No model-level (perplexity/throughput) benchmark run. The offline harness measures output **perturbation** (cosine distance of the compressed-cache attention output vs the full-cache output over probe queries) against the H2O `drop` baseline; the measured finding is that `sim_weighted` merging reduces perturbation and the gain grows with compression ratio (e.g. 0.955 → 0.708 at `seq=1024, budget=64`, 16×), shrinking to ~0 at low compression where dropping barely hurts. Not an end-to-end task-quality claim.

---

## v0.25.0

### New
- **ChunkKV-adapted** (`method="chunkkv"`) — the repo's **seventh eviction configuration** and the first to evict at **chunk** rather than **token** granularity. The sequence is partitioned into contiguous chunks of `chunk_size` tokens; each chunk is kept or dropped as a whole, ranked by a mean-pooled per-token importance proxy (H2O cumulative attention mass, or key L2 norm). Keeping whole contiguous spans preserves local structure that token-level eviction shreds. When `chunk_size=1` it reduces **bit-for-bit** to H2O-adapted. ChunkKV-adapted (arXiv:2502.00299, Liu et al., 2025) — documented as "ChunkKV-adapted (VeloxQuant-MLX implementation)," not a faithful port.
  - `ChunkKVCache` (`veloxquant_mlx/cache/chunkkv_cache.py`); primitives in `veloxquant_mlx/quantizers/chunkkv.py`: `chunk_partition`, `chunk_scores`, `chunkkv_keep_mask`, `ChunkKVState`, `init_chunkkv_state`, `chunkkv_update`, `chunkkv_trim_to`, `chunkkv_get_kv`, `chunkkv_fp16_bytes`, `full_chunkkv_fp16_bytes`.
  - Config: `chunkkv_budget` (default 512), `chunkkv_chunk_size` (default 8), `chunkkv_n_sink` (default 4), `chunkkv_score` (`"attn_mass"` | `"key_norm"`, default `"attn_mass"`). No coordinator — each layer resolves its own chunks; the default `KVCacheBuilder.for_model()` path returns one `ChunkKVCache` per layer.
  - 19 quantizer tests + 14 cache tests, including a bit-for-bit `chunk_size=1` == H2O equivalence at both the primitive and cache level; `benchmark_scripts/benchmark_chunkkv.py` + committed `chunkkv_benchmark_results.json` (offline-synthetic).

### Honest scope
- Pooled per-token score as a proxy for the paper's attention-over-chunk importance; no layer-wise kept-index reuse (each layer resolves chunks independently).
- Key-as-query proxy for the `attn_mass` scorer (same as H2O-adapted); no RoPE position-ID remapping after eviction; uniform budget across heads within a layer.
- Whole-chunk retention lets heads settle at slightly different counts — the wrapper trims every head to the common minimum so the emitted tensor is rectangular.
- No model-level (perplexity/throughput) benchmark run yet. The committed harness measures compression, kept-token count, and eviction latency on synthetic data; larger chunks cut the pure-Python eviction pass sharply (~12.7× fewer passes at `C=16` vs `C=1` on the `seq=1024, budget=128` shape) while holding compression. ChunkKV's semantic-coherence advantage is a real-attention property and is not claimed from the synthetic harness.

---

## v0.20.0

### New
- **StreamingLLM-adapted** (`method="streaming_llm"`) — the repo's first **constant-memory** cache and first **structural positional eviction** method. Keeps only the first `stream_n_sink` token positions (frozen attention sinks) and the most recent `stream_window_size` positions (rolling FIFO). All other positions are permanently evicted. Both prefill and decode tokens go through the same logic — the cache never exceeds `stream_n_sink + stream_window_size` positions regardless of generation length. StreamingLLM-adapted (arXiv:2309.17453, ICLR 2024, Xiao et al.) — positional eviction (no scoring, no calibration); documented as "StreamingLLM-adapted (VeloxQuant-MLX implementation)."
  - `StreamingLLMKVCache` (`veloxquant_mlx/cache/streaming_llm_cache.py`); primitives in `veloxquant_mlx/quantizers/streaming_llm.py`: `StreamingWindow`, `init_streaming_window`, `stream_update`, `stream_get_kv`, `stream_fp16_bytes`, `full_stream_fp16_bytes`.
  - Config: `stream_n_sink` (default 4), `stream_window_size` (default 512). Single-layer; `KVCacheBuilder.for_model()` propagates all `stream_*` fields via `dataclasses.replace`.
  - 17 quantizer tests + 15 cache tests; `benchmark_scripts/benchmark_streaming_llm.py` (offline-synthetic, not run).

### Honest scope
- No attention mask adjustment: the model attends to all returned K/V positions; only the number of K/V rows is bounded.
- No RoPE position-ID remapping: original token positions are preserved in returned rows.
- Fixed `stream_n_sink` count — not adaptive.
- No model-level benchmark run yet; streaming_ratio and constant-memory property verified on synthetic data (32/32 tests passing).

---

## v0.19.0

### New
- **SnapKV-adapted** (`method="snapkv"`) — the repo's first **token eviction** method and the first where the paper's actual signal (observation-window attention scores) is computable at the cache level without model interception. During prefill, the last `snap_obs_window` key rows act as proxy queries; their softmax attention over all prefix positions scores each token. Only the top-`snap_budget` positions (plus `snap_n_sink` always-kept sink positions) are retained as fp16. Decode tokens are never evicted. SnapKV-adapted (arXiv:2404.14469, ICLR 2025, Yuan et al.) — key-as-query proxy and no max-pool smoothing; documented as "SnapKV-adapted (VeloxQuant-MLX implementation)."
  - `SnapKVKVCache` (`veloxquant_mlx/cache/snapkv_cache.py`); primitives in `veloxquant_mlx/quantizers/snapkv.py`: `obs_window_attention_scores`, `snap_select_indices`, `snapkv_compress`, `snapkv_fp16_bytes`, `full_fp16_bytes`.
  - Config: `snap_budget`, `snap_obs_window`, `snap_n_sink`. Single-layer; `KVCacheBuilder.for_model()` propagates all `snap_*` fields via `dataclasses.replace`.
  - 18 quantizer tests + 12 cache tests; `benchmark_scripts/benchmark_snapkv.py` (offline-synthetic, not run).
  - Single-layer (no coordinator); eviction is per-head, uniform budget.

### Honest scope
- The key-as-query proxy is weaker than true query vectors from the prompt (not observable at `update_and_fetch`). Still stronger than key-norm-only methods (computes the actual attention distribution from K).
- No max-pool smoothing (paper's `kernel_size > 1`).
- Uniform `snap_budget` across all heads.
- No model-level benchmark run yet; eviction ratio and attention-coverage lift verified on synthetic data.

---

## v0.18.0

### New
- **ZipCache-adapted** (`method="zipcache"`) — the repo's first **per-token mixed bit-width** cache. The top `hi_fraction` of tokens by key L2-norm (the saliency proxy) are quantized at `hi_bits`; the rest at `lo_bits`. Both groups remain quantized — not fp16. ZipCache-adapted (arXiv:2405.14256, NeurIPS 2024, He et al.): the paper's true signal is normalized attention scores, which are not observable by a cache wrapper; key L2-norm is the proxy (same as KIVI-Sink and AdaKV-proxy, but here the decision is bit-width routing rather than fp16 protection or head budgeting).
  - `ZipCacheKVCache` (`veloxquant_mlx/cache/zipcache_cache.py`); primitives in `veloxquant_mlx/quantizers/zipcache.py`: `token_key_norms`, `saliency_mask`, `channel_quant`, `channel_dequant`, `zipcache_compress`, `zipcache_reconstruct`, `zipcache_bytes`, `base_only_bytes`, `zipcache_quant_dequant`.
  - Config: `zipcache_hi_bits`, `zipcache_lo_bits`, `zipcache_hi_fraction`, `zipcache_group_size`, `zipcache_quantize_values`.
  - 16 quantizer tests + 11 cache tests; `benchmark_scripts/benchmark_zipcache.py` (offline-synthetic, not run).
  - Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all `zipcache_*` fields via `dataclasses.replace`.

### Honest scope
- The saliency proxy (key L2-norm) is weaker than true attention scores. This is the third use of the key-norm proxy in this repo; each prior use is on a different decision (KIVI-Sink: fp16 protection; AdaKV-proxy: head budget).
- The effective average key rate is `hi_frac×hi_bits + (1-hi_frac)×lo_bits` — between `lo_bits` and `hi_bits`, as expected.
- No model-level benchmark run yet; stored bytes and reconstruction MSE are test-verified on synthetic data.

---

## v0.17.0

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
