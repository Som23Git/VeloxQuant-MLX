# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

> Detailed release notes for 0.10.0–0.14.0 (SVDq, Kitty, AdaKV-proxy, XQuant,
> KVQuant-NUQ) live in the docs-site changelog
> (`docs-site/docs/changelog.md`). The entries below cover the latest releases
> and the original 0.9.0 baseline.

## [0.29.0] — 2026-07-07

### Added — L2Norm: intrinsic key-norm eviction (`method="knorm"`)

- **`veloxquant_mlx.cache.knorm_cache.L2NormKVCache`** — the library's
  **thirty-second configuration** and its first **intrinsic-signal** eviction
  cache. *Inspired by, not a faithful port of,* "A Simple and Effective L2
  Norm-Based Strategy for KV Cache Compression" (Devoto, Zhao, Scardapane &
  Minervini, EMNLP 2024, arXiv:2406.11430). Every eviction method shipped so
  far scores tokens with attention / a key-as-query proxy (SnapKV, H2O,
  TOVA, PyramidKV, SqueezeAttention, ChunkKV, CaM) or pure structure
  (StreamingLLM, sink, sliding-window); L2Norm reads importance **directly
  off the stored key** — the paper's trained-LM finding is that *low* key
  norm predicts *high* attention, so the lowest-norm tokens are kept. Note
  the sign inversion vs ChunkKV's `key_norm` option and ZipCache's saliency
  proxy (which treat high norm as important) — the inversion is the paper's
  empirical content.
- **`veloxquant_mlx.quantizers.knorm`** — `KnormState`, `init_knorm_state`,
  `knorm_update` (vectorized: norms are computed once at insertion and never
  updated, so eviction is a single protected top-k per incoming block — no
  per-token softmax-over-cache loop like H2O), `knorm_get_kv`,
  `knorm_fp16_bytes`, `full_knorm_fp16_bytes`.
- **Two properties fall out of the intrinsic score** (both measured/pinned):
  - **Speed:** 0.3–1.2 ms per prefill block vs H2O-adapted's 37–275 ms on
    identical inputs in the committed harness (~100–800×).
  - **Path independence** (`knorm_recent=0`): the kept set equals the global
    budget-best over all tokens seen regardless of arrival grouping (the
    "keep k best with a heap" invariant) — prefill and token-by-token decode
    produce bit-for-bit identical caches, pinned by test at the primitive
    and wrapper level. No accumulating-score method has this property.
- **Config** — `knorm_budget` (default 512), `knorm_n_sink` (default 4),
  `knorm_recent` (default 0; extension, breaks path independence when on),
  `knorm_keep` (`"low"` paper default | `"high"` inverted ablation).
  Build-time validation (keep mode, sinks+recent < budget). No coordinator.
- **Tests** — 10 quantizer + 14 cache tests (903 total collected).
- **Benchmark** — `benchmark_scripts/benchmark_knorm.py` + committed
  `knorm_benchmark_results.json` (offline-synthetic): under geometry
  constructed to exhibit the paper's correlation, keep-low beats random
  eviction by **+0.17** mean output perturbation and the inverted scorer by
  **+0.21**, and beats H2O-adapted on most rows at matched budget; under
  the isotropic control the advantage **reverses** (keep-low ~0.07 worse
  than random), reported in full. **Explicitly NOT a model-level
  perplexity/throughput benchmark.**

### Honest scope

- The low-norm ⇒ high-attention correlation is the **paper's empirical
  claim about trained models**. Synthetic data cannot validate it — the
  benchmark validates the machinery under constructed geometry and shows
  the method underperforming random eviction when that geometry is absent.
- No RoPE position-ID remapping after eviction; uniform budget/n_sink
  across heads (same as the rest of the eviction family).
- `knorm_recent` and `knorm_keep="high"` are extensions beyond the paper,
  both off by default.
- No model-level benchmark run.

## [0.28.0] — 2026-07-06

### Added — NSNQuant: calibration-free universal-codebook VQ (`method="nsnquant"`)

- **`veloxquant_mlx.cache.nsnquant_cache.NSNQuantKVCache`** — the library's
  **thirty-first configuration** and its first **calibration-free
  distribution-matching VQ**. *Inspired by, not a faithful port of,*
  "NSNQuant: A Double Normalization Approach for Calibration-Free Low-Bit
  Vector Quantization of KV Cache" (Son, Choi & Yoo, NeurIPS 2025,
  arXiv:2505.18231). Every other VQ method in the repo either fits its
  codebook to the data (RVQ's per-sequence k-means, CommVQ) or uses a
  data-independent geometric code (RaBitQ signs, VecInfer binary, PolarQuant
  grids, QJL sketches); NSNQuant inverts the relationship — a
  **Normalize-Shift-Normalize transform + Hadamard rotation reshapes K/V
  tokens onto the standard normal distribution**, so one codebook built
  offline from synthetic Gaussian samples (never model activations)
  quantizes any model at 1–2 bits/element.
- **`veloxquant_mlx.quantizers.nsnquant`** — pure primitives: `nsn_transform`
  / `nsn_inverse` (token-norm → channel-mean shift → token-norm, exact
  restoration `x̂ = s1·(s2·x_nsn + o)`), `build_universal_codebook`
  (deterministic seeded spherical k-means on synthetic standard-normal
  samples; "magnitude" positive-orthant variant for 2-bit + sign mask,
  "signed" variant for 1-bit), `vq_encode`/`vq_decode` (8-dim subvector
  cosine matching, uint8 indices), `hadamard_forward`/`hadamard_inverse`
  (reusing `mx.hadamard_transform` — norm-preserving, so it composes with
  NSN's stored scales).
- **Chunk-flush residual buffer** — KIVI's fp16-residual idiom, upgraded:
  every `nsn_residual_length` tokens age past the quantized frontier as one
  self-contained chunk with its own online channel mean (no frozen
  statistics, no coordinator, chunk *i* forever independent of later
  arrivals). Prefill and decode produce identical quantized state by
  construction — verified bit-for-bit by test. Unlike KIVI's
  incoming-block-only simplification, decode tokens *do* get quantized once
  they age out.
- **Config** — `nsn_bits` (default 2: uint8 sign mask + uint8 codebook index
  per 8-dim subvector = 2 bits/element; 1: index only), `nsn_residual_length`
  (default 64; the paper recommends 128 for 1-bit), `nsn_codebook_size`
  (default 256), `nsn_subvector_dim` (default 8), `nsn_seed` (default 1234),
  `nsn_max_ctx` (default 8192). Both keys **and** values quantized (mirrors
  the paper; unlike the keys-only SVDq/xKV precedent). Build-time validation
  with clear messages (bits ∈ {1,2}, head_dim divisibility, Hadamard
  compatibility).
- **Byte accounting** — payload at exactly `nsn_bits` bits/element plus fp16
  metadata counted honestly (`s1`+`s2` per token, `o` per chunk ≈ 0.5
  bits/element at defaults — the paper double-quantizes these to ~0.23; we
  don't, and say so); `residual_fp16_bytes` reported separately as a
  snapshot so ratios aren't inflated. ~2.5 effective bits/element at 2-bit
  defaults → ~6.4× vs fp16.
- **Tests** — 16 quantizer + 19 cache tests (871 total passing), including a
  mechanism-validation ablation (on channel-biased input the full NSN
  pipeline must beat the identical Hadamard+VQ without NSN by a pinned
  margin) and the prefill-vs-decode path-independence check.
- **Benchmark** — `benchmark_scripts/benchmark_nsn.py` + committed
  `nsn_benchmark_results.json` (offline-synthetic, no model download):
  NSN gains **+0.038 (2-bit) / +0.110 (1-bit)** reconstruction cosine over
  the no-NSN ablation at strong synthetic channel bias, and the gain
  honestly **collapses to ~+0.001–0.002 when the input is already centered**
  (NSN only helps when there is a bias to remove); 0.96–0.98 cosine at ~2.5
  effective bits/element, above a KIVI-2bit baseline (0.66–0.88) on every
  row of the sweep. **Explicitly NOT a model-level perplexity/throughput
  benchmark.**

### Honest scope

- **Post-RoPE keys** — the paper applies NSN to keys *before* RoPE and defers
  RoPE onto the stored mean inside a custom attention kernel; our cache
  wrappers receive post-RoPE keys from `update_and_fetch`, so NSN + Hadamard
  run post-RoPE. This is the central simplification of the adaptation.
- **No value-projection Hadamard fusion** (model surgery) — the value-side
  Hadamard is applied explicitly to cached tensors instead.
- **No gradient fine-tuning of the codebook** — seeded spherical k-means
  only; expect a slightly worse codebook than the paper's.
- **No 4-bit double quantization of metadata** — fp16, counted.
- **No fused kernels** — MLX ops; on Apple Silicon the win is memory, not
  throughput, exactly as with KIVI.
- **No model-level benchmark run** — offline-synthetic reconstruction and
  byte-accounting numbers only.

## [0.27.0] — 2026-07-06

### Added — xKV: cross-layer shared-subspace key compression (`method="xkv"`)

- **`veloxquant_mlx.cache.xkv_cache.XKVCache`** — the library's **thirtieth
  configuration** and the **third cross-layer** mechanism, alongside XQuant
  (code reuse) and MiniCache (SLERP direction merge). *Inspired by, not a
  faithful port of,* "xKV: Cross-Layer KV-Cache Compression via Aligned
  Singular Vector Extraction" (Chang, Lin, Lin, Chiang, Akhauri, Dai, Jiang,
  Li, Ceze, Wu & Abdelfattah, arXiv:2503.18893, preprint). Every other
  cross-layer method either reuses one anchor's codes (XQuant) or merges a
  *pair* of layers' directions (MiniCache); xKV instead **jointly factorizes
  a whole group of layers** into one shared low-rank SVD basis, amortizing
  the basis storage cost across every member of the group.
- **`veloxquant_mlx.cache.xkv_coordinator.XKVCoordinator`** — a
  fan-in-then-fan-out coordinator: every group member publishes its own raw
  prefill keys; once all members of a group have published for the same
  token range, the joint SVD runs once and the resulting shared basis is
  broadcast back to every member (including whichever one triggered the
  computation). This is a different coordination shape than XQuant/MiniCache,
  which have a single publisher and one or more readers.
- **`veloxquant_mlx.quantizers.xkv`** — pure primitives: `pair_layers_grouped`
  (fixed-size contiguous grouping, including a trailing partial group),
  `joint_svd_compress` (stack N layers' centered keys, single truncated SVD),
  `project_into_shared_basis`, `reconstruct_from_shared_basis`,
  `quantize_latents_uniform`.
- **Grouping** — `xkv_group_size` (default 2) chunks attention-bearing layers
  into fixed contiguous groups; layer 0 of each group is the conventional
  "leader" (the only member reporting the amortized `shared_basis_bytes`
  cost, avoiding double-counting when bytes are summed across layers).
- **Config** — `xkv_group_size` (default 2), `xkv_rank` (default `None` →
  energy-threshold selection), `xkv_energy_threshold` (default 0.95),
  `xkv_latent_bits` (default 4 — single-bit-width latent quantization, not
  SVDq-style mixed-bit routing), `xkv_group_quant_size` (default 32),
  `xkv_max_ctx` (default 8192). Keys only — values pass through fp16
  unchanged, mirroring SVDq's existing precedent in this repo.
- **Tests** — 9 quantizer tests + 14 cache tests (all passing), including a
  group-of-1 degeneracy check (`joint_svd_compress` on a single matrix
  matches SVDq's plain single-layer SVD at the same rank) and a
  mechanism-validation test (a shared basis fit jointly across synthetic
  layers with genuinely shared low-rank structure reconstructs better than
  independent per-layer SVD on unrelated noise at matched rank).
- **Benchmark** — `benchmark_scripts/benchmark_xkv.py` + committed
  `xkv_benchmark_results.json` (offline-synthetic). Sweeps group size (2–4)
  and a synthetic shared-structure fraction against an independent-per-layer
  -SVD baseline at matched rank: reconstruction MSE lands within ~1% of
  independent SVD across every configuration tested (near-parity, not a
  quality regression), while byte cost is **8–20% lower** than independent
  SVD, improving with larger group sizes — the amortization win the
  shared-basis mechanism is designed to deliver.

### Honest scope

- Fixed contiguous layer grouping — no CKA-based (Centered Kernel Alignment)
  validation that the grouped layers actually share a subspace, unlike the
  paper's empirical per-architecture grouping.
- No "Selective Reconstruction" — the paper's decode-time latency
  optimization (exactly reconstruct a subset of group layers, derive the
  rest) is not implemented; every layer is fully reconstructed on every
  fetch, like every other wrapper in this repo.
- Single-bit-width latent quantization, not SVDq's importance-ranked
  mixed-bit routing — xKV's distinguishing feature is the shared basis, not a
  novel bit-allocation scheme.
- **No model-level (perplexity/throughput) benchmark run.** The harness
  measures reconstruction-MSE parity and byte-accounting savings against an
  independent-SVD baseline, and an output-perturbation proxy — not end-to-end
  task quality on a real model.
- Docs: new `docs-site/docs/algorithms/xkv.md`, sidebar + overview + intro +
  changelog entries, cross-links from XQuant and MiniCache pages. README/
  landing counts: twenty-nine → thirty strategies; version bump 0.26.0 →
  0.27.0.

## [0.26.0] — 2026-07-04

### Added — CaM: cache merging (merge evicted tokens instead of dropping) (`method="cam"`)

- **`veloxquant_mlx.cache.cam_cache.CaMKVCache`** — the library's **eighth
  eviction configuration** and the first on the **merge-vs-drop** axis. *Inspired
  by, not a faithful port of,* "CaM: Cache Merging for Memory-efficient LLMs
  Inference" (Zhang, Du, Luo, Zhong, Zhang, Liu & Ji, ICML 2024, PMLR
  235:58840-58850). Every other eviction method permanently discards the tokens it
  evicts; CaM instead **merges** each evicted token into the surviving token it
  most resembles (a cosine-weighted blend of the value rows, and optionally the
  keys), then removes only the redundant slot. The eviction *choice* is H2O's;
  only the disposition differs. With `cam_merge="drop"` it reduces **bit-for-bit**
  to H2O-adapted.
- **`veloxquant_mlx.quantizers.cam`** — pure primitives: `most_similar_survivor`
  (nearest retained non-sink key by cosine), `merge_pair` (the weighted blend),
  `CaMState` + `init_cam_state` / `cam_update` / `cam_get_kv` / `cam_fp16_bytes` /
  `full_cam_fp16_bytes`.
- **Merge modes** — `cam_merge="sim_weighted"` (default) blends by
  `w = clip(cos(k_evicted, k_survivor), 0, 1)`; `"mean"` is an unweighted average;
  `"drop"` skips the blend (== H2O). Values are always merged; keys only when
  `cam_merge_keys=True`.
- **Config** — `cam_budget` (default 512), `cam_n_sink` (default 4), `cam_merge`
  (default `"sim_weighted"`), `cam_merge_keys` (default False). No coordinator;
  the default `KVCacheBuilder.for_model()` path returns one `CaMKVCache` per layer.
- **Tests** — 18 quantizer tests + 14 cache tests (all passing), including a
  bit-for-bit `cam_merge="drop"` == H2O equivalence (identical kept keys *and*
  values vs `H2OKVCache`) at both the primitive and cache level.
- **Benchmark** — `benchmark_scripts/benchmark_cam.py` + committed
  `cam_benchmark_results.json` (offline-synthetic, Apple Silicon). Measures output
  **perturbation** (cosine distance of the compressed-cache attention output vs the
  full cache over probe queries) against the H2O `drop` baseline; `sim_weighted`
  merging reduces perturbation and the gain grows with compression ratio
  (0.955 → 0.708 at `seq=1024, budget=64`, 16×), shrinking to ~0 at low compression.

### Honest scope

- Cosine-similarity merge weight rather than the paper's attention-prominence
  weight (which is ~0 for a just-appended token that overflows before it
  accumulates mass — the common streaming case); single nearest-survivor merge (no
  multi-target soft assignment / sampling); key-as-query proxy; no RoPE remapping;
  uniform budget across heads.
- **No model-level (perplexity/throughput) benchmark run.** The harness measures
  the output-perturbation proxy CaM targets, not end-to-end task quality.
- Docs: new `docs-site/docs/algorithms/cam.md`, sidebar + overview + intro +
  changelog entries, cross-links from H2O and ChunkKV. README/landing counts:
  twenty-eight → twenty-nine strategies; version bump 0.25.0 → 0.26.0.

## [0.25.0] — 2026-07-04

### Added — ChunkKV: chunk-level (semantic-block) eviction (`method="chunkkv"`)

- **`veloxquant_mlx.cache.chunkkv_cache.ChunkKVCache`** — the library's **seventh
  eviction configuration** and the first to evict at **chunk** rather than **token**
  granularity. *Inspired by, not a faithful port of,* "ChunkKV: Semantic-Preserving
  KV Cache Compression for Efficient Long-Context LLM Inference" (Liu et al., 2025,
  arXiv:2502.00299). Every other eviction method scores and drops individual tokens;
  ChunkKV partitions the sequence into contiguous chunks of `chunk_size` tokens and
  keeps or drops each chunk *as a whole*, preserving local coherence that token-level
  eviction shreds. When `chunk_size=1` it reduces **bit-for-bit** to H2O-adapted.
- **`veloxquant_mlx.quantizers.chunkkv`** — pure primitives: `chunk_partition`
  (split into sink + body chunks), `chunk_scores` (mean-pool a per-token score into
  per-chunk scores), `chunkkv_keep_mask` (chunk-aligned keep-mask for a budget),
  `ChunkKVState` + `init_chunkkv_state` / `chunkkv_update` / `chunkkv_trim_to` /
  `chunkkv_get_kv` / `chunkkv_fp16_bytes` / `full_chunkkv_fp16_bytes`.
- **Chunk-importance proxy** — `chunkkv_score="attn_mass"` (default) mean-pools H2O's
  cumulative attention mass; `chunkkv_score="key_norm"` mean-pools the key L2 norm
  (calibration-free, coarser). Sinks (`chunkkv_n_sink`) are always kept and never
  grouped into an evictable chunk.
- **Config** — `chunkkv_budget` (default 512), `chunkkv_chunk_size` (default 8),
  `chunkkv_n_sink` (default 4), `chunkkv_score` (`"attn_mass"` | `"key_norm"`).
  No coordinator: each layer resolves its own chunks, so the default
  `KVCacheBuilder.for_model()` path returns one `ChunkKVCache` per layer. Whole-chunk
  retention lets heads settle at slightly different counts, so the wrapper trims every
  head to the common minimum (`chunkkv_trim_to`) to emit a rectangular tensor.
- **Tests** — 19 quantizer tests + 14 cache tests (all passing), including a
  bit-for-bit `chunk_size=1` == H2O equivalence (identical kept keys *and* values vs
  `H2OKVCache`) at both the primitive and cache level. Survivors verified to be whole
  chunks; sinks always preserved; both score modes exercised; deterministic.
- **Benchmark** — `benchmark_scripts/benchmark_chunkkv.py` + committed
  `chunkkv_benchmark_results.json` (offline-synthetic, Apple Silicon). Confirms
  `chunk_size=1` reproduces H2O and that larger chunks cut the pure-Python eviction
  pass sharply (~12.7× fewer/faster passes at `C=16` vs `C=1` on the
  `seq=1024, budget=128` shape) while holding compression.

### Honest scope

- Mean-pooled per-token score as a proxy for the paper's attention-over-chunk
  importance; no layer-wise kept-index reuse (each layer resolves chunks independently).
- Key-as-query proxy for the `attn_mass` scorer (same as H2O-adapted); no RoPE
  position-ID remapping after eviction; uniform budget across heads within a layer.
- **No model-level (perplexity/throughput) benchmark run.** The harness measures
  compression, kept-token count, and eviction latency on synthetic data. ChunkKV's
  semantic-coherence advantage is a real-attention property and is not claimed from
  the synthetic harness.
- Docs: new `docs-site/docs/algorithms/chunkkv.md`, sidebar + overview + changelog
  entries, cross-links from SnapKV and SqueezeAttention. README intro now reads
  "twenty-eight compression strategies" (seven token-eviction caches). Landing page
  updated with a ChunkKV card, picker entry, quickstart tab, and what's-new item.

## [0.24.1] — 2026-07-04

### Changed — documentation & landing page

- **README** — dynamic shields.io PyPI version badge (auto-reads the live release),
  new pepy.tech total-downloads badge, tests updated to 750/756, changelog badge to
  0.24.1, and the intro now reads "twenty-seven compression strategies" (six of them
  token-eviction caches). No code or API changes.
- **Landing page** — "Method Library" redesign: uniform card grid grouped by category
  (Eviction / Quantization / Low-rank / Cross-layer), quiet version metadata, a single
  NEW pill on the three latest methods, and progressive-disclosure `<details>`
  expanders. De-duplicated the install/quickstart sections and added a SqueezeAttention
  quickstart tab. Fixed an invisible footer tagline and stale test/version counts.

## [0.24.0] — 2026-07-03

### Added — SqueezeAttention: 2D layer×token data-driven budget eviction (`method="squeeze"`)

- **`veloxquant_mlx.cache.squeeze_cache.SqueezeAttentionCache`** — the library's
  first **2D (layer × token)** budget eviction method and the first with a
  **data-driven** per-layer budget. *Inspired by, not a faithful port of,*
  "SqueezeAttention: 2D Management of KV-Cache in LLM Inference via Layer-wise
  Optimal Budget" (Wang et al., 2024, arXiv:2404.04793). SqueezeAttention is
  H2O's cumulative-attention-mass eviction with a per-layer budget that is
  *measured*, not assumed: each layer reports its attention **concentration**
  during prefill and a fixed total budget is reallocated toward broad
  (low-concentration) layers and away from concentrated ones. When
  `squeeze_strength=0.0` it reduces exactly to uniform H2O.
- **`concentration_score(keys)`** — an attention-free concentration proxy: mean
  pairwise cosine similarity of a layer's key set. High → keys cluster →
  attention concentrated → the layer needs *less* budget.
- **The allocator — `squeeze_budgets(concentrations, avg_budget, n_sink, strength)`** —
  reallocates a fixed total by inverse-concentration (mean held ≈ `avg_budget`,
  floored at `n_sink + 1`); `strength` interpolates linearly between uniform
  (`0.0`) and the full split (`1.0`).
- **`SqueezeCoordinator`** — the first eviction method with a **runtime
  re-budgeting** step. A single shared coordinator (injected at
  `KVCacheBuilder.for_model()` build time) collects per-layer concentration
  during prefill, computes the schedule **once at the prefill boundary**, and
  publishes each layer's resolved budget; over-budget layers are then trimmed by
  H2O score. Unlike XQuant / MiniCache it exchanges only per-layer scalars and
  runs its allocation exactly once — decode steps use the frozen schedule.
- **Sixth distinct eviction configuration in VeloxQuant-MLX** — completing the
  budget-axis matrix: SnapKV (prefill-only), StreamingLLM (positional), H2O
  (uniform), TOVA (memoryless), PyramidKV (fixed per-layer pyramid),
  SqueezeAttention (data-driven per-layer budget).
- **Registered** as `method="squeeze"` in `KVCacheFactory`; new config fields
  `squeeze_budget` (avg, default 512), `squeeze_n_sink` (4), `squeeze_strength`
  (1.0), `squeeze_resolved_budget` (override, None).
- **28 quantizer + 19 cache tests — all 47 passing.** A synthetic benchmark
  (`benchmark_scripts/benchmark_squeeze.py`) sweeps
  `(n_layers, seq_len, avg_budget, strength)` and was run on Apple Silicon;
  results committed in `squeeze_benchmark_results.json`. Confirms the design:
  `strength=0.0` gives uniform budgets (== H2O); `strength>0` reallocates so the
  broad early layer keeps more than the concentrated deep layer; schedule mean
  ≈ `avg_budget`.

#### Adaptation limitations (documented, not a faithful port)

- Key-as-query proxy for both concentration measurement and within-layer
  eviction (same as H2O-adapted / PyramidKV-adapted).
- Cosine-dispersion proxy for attention entropy (paper reads actual attention
  maps, not visible at cache level).
- One-shot re-budget at the prefill boundary, frozen for decode.
- No RoPE position-ID remapping; uniform budget across heads within a layer.
- Benchmark is synthetic (schedule / kept-token / compression only); no
  model-level perplexity or throughput figure is claimed.

## [0.23.1] — 2026-07-03

### Changed

- **License** — extended the copyright notice to `2025-2026` to reflect ongoing
  active development. No code or API changes; this is a metadata-only release so
  the corrected copyright year is rendered on the PyPI project page.

## [0.23.0] — 2026-07-02

### Added — PyramidKV: layer-adaptive budget attention-mass eviction (`method="pyramidkv"`)

- **`veloxquant_mlx.cache.pyramidkv_cache.PyramidKVCache`** — the library's first
  **layer-adaptive budget** eviction method. *Inspired by, not a faithful port of,*
  "PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling"
  (Cai et al., 2024, arXiv:2406.02069). PyramidKV is H2O's cumulative-attention-mass
  eviction with a **per-layer budget** instead of a single global one: early layers
  (broad attention) get a large budget, deep layers (concentrated attention) get a
  small one, holding the *average* budget fixed so total memory matches a uniform
  baseline. When the pyramid is flat (`pyramid_beta=1.0`) it reduces exactly to
  H2O-adapted.
- **The allocator — `pyramid_budgets(n_layers, avg_budget, n_sink, beta)`** — returns
  the per-layer budget schedule (monotonically decreasing, mean ≈ `avg_budget`,
  floored at `n_sink + 1`). Resolved once at `KVCacheBuilder.for_model()` build time
  and baked into each layer's config as `pyramid_resolved_budget`. **No runtime
  coordinator** is needed (unlike XQuant / MiniCache) — layers never exchange data
  during generation; the only cross-layer signal is each layer's index, consumed at
  build time.
- **Fifth distinct eviction configuration in VeloxQuant-MLX:**
  - SnapKV-adapted — score-based, once at prefill end.
  - StreamingLLM-adapted — positional (recency + sink), constant-memory.
  - H2O-adapted — cumulative attention mass, **uniform** budget, every step.
  - TOVA-adapted — current-step attention weight (memoryless), every step.
  - PyramidKV-adapted — H2O scoring with a **per-layer pyramid** budget.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy (same as H2O-adapted / SnapKV-adapted).
  - Fixed monotone (linear) budget schedule rather than the paper's
    prefill-entropy-derived allocation — funneling shape preserved, exact per-layer
    values not data-driven.
  - No RoPE position-ID remapping after eviction.
  - Uniform budget across heads within a layer (the pyramid is across layers).
- Primitives in `veloxquant_mlx/quantizers/pyramidkv.py`: `pyramid_budgets`,
  `PyramidState`, `init_pyramid_state`, `pyramid_update`, `pyramid_get_kv`,
  `pyramid_fp16_bytes`, `full_pyramid_fp16_bytes`.
- Config: `pyramid_budget` (int, default 512, the average/fallback), `pyramid_n_sink`
  (int, default 4), `pyramid_beta` (float, default 2.0 — pyramid steepness; 1.0 = flat).
  Single-cache `KVCacheFactory.create` (no layer context) falls back to
  `pyramid_budget` and behaves as one uniform-budget H2O layer.
- **Tests** — `tests/quantizers/test_pyramidkv.py` (24 tests) +
  `tests/cache/test_pyramidkv_cache.py` (19 tests): allocator shape/monotonicity/
  mean-preservation/flat==uniform/sink-floor/edge-cases, budget enforcement, sink
  protection, byte accounting, determinism, and `for_model` producing a decreasing
  pyramid of per-layer budgets (early layers keep more tokens than deep layers).
- Offline-synthetic harness in `benchmark_scripts/benchmark_pyramidkv.py` sweeping
  `(n_layers, seq_len, avg_budget, beta)` on synthetic fp16 data — **run on Apple
  Silicon**; results committed in `benchmark_scripts/pyramidkv_benchmark_results.json`
  (24 configs). They confirm the design end-to-end: `beta=1.0` gives a flat schedule
  (== uniform H2O), `beta>1.0` gives strictly decreasing budgets with early layers
  retaining more tokens than deep layers, and schedule mean == `avg_budget`
  everywhere. No model-level perplexity/throughput figures are claimed.

---

## [0.22.0] — 2026-07-01

### Added — TOVA: current-step attention-weight eviction, memoryless (`method="tova"`)

- **`veloxquant_mlx.cache.tova_cache.TOVAKVCache`** — the library's first
  **memoryless** eviction method. *Inspired by, not a faithful port of,*
  "Transformers are Multi-State RNNs" (Oren et al., 2024, arXiv:2401.06104), whose
  TOVA (Token Omission Via Attention) policy keeps a fixed-size cache by dropping,
  at each step, the single token receiving the **lowest attention weight in the
  current step**. On every step (prefill and decode alike), the approximate
  current-step attention distribution over the post-append cache is computed using
  the **new key vector as a proxy query** (true queries are not visible at
  cache-wrapper level — same approximation as SnapKV-adapted and H2O-adapted).
  When the cache exceeds `tova_budget`, the **lowest current-step-weight non-sink
  token** is permanently evicted. The cache is bounded at all times to
  `tova_budget` positions.
- **Fourth distinct eviction axis in VeloxQuant-MLX — and the key contrast with H2O:**
  - SnapKV-adapted — score-based, fires once at prefill end; grows during decode.
  - StreamingLLM-adapted — positional (recency + sinks), constant-memory throughout.
  - H2O-adapted — **cumulative** attention mass (inertial: past heavy hitters resist eviction).
  - TOVA-adapted — **current-step** attention weight (memoryless: a token that stops
    being attended to is evicted even if it dominated earlier). TOVA is the more
    reactive policy; H2O is the more conservative one.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy: approximates the paper's true query attention signal.
  - No RoPE position-ID remapping after eviction.
  - Uniform `tova_budget` and `tova_n_sink` across all heads.
- Primitives in `veloxquant_mlx/quantizers/tova.py`: `TovaState`,
  `init_tova_state`, `tova_update`, `tova_get_kv`, `tova_fp16_bytes`,
  `full_tova_fp16_bytes`. No `scores` field — state carries no cross-step history.
- Config: `tova_budget` (int, default 512), `tova_n_sink` (int, default 4).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all
  `tova_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_tova.py` (19 tests) +
  `tests/cache/test_tova_cache.py` (15 tests): init state, no-scores-field assertion,
  single-token bootstrap, multi-token absorption, budget enforcement (never exceeded
  across 30 decode steps), sink protection (sinks always present after evictions),
  n_sink=0 edge case, memorylessness (no scores carried across steps), current-step
  eviction correctness (a token orthogonal to the current key is dropped over a
  similar one), byte accounting formula, compression_ratio, tokens_seen, factory
  dispatch, for_model propagation, determinism.
- Offline-synthetic harness in `benchmark_scripts/benchmark_tova.py` sweeping
  `(seq_len, budget, n_sink)` on synthetic fp16 data — **run on Apple Silicon**;
  results committed in `benchmark_scripts/tova_benchmark_results.json` (28 configs).
  Measured compression ratio equals `seq_len / budget` exactly across every config
  (e.g. 2048 tokens at budget 64 → 32×). No model-level perplexity/throughput
  figures are claimed.

---

## [0.21.0] — 2026-07-01

### Added — H2O: cumulative attention-mass heavy-hitter oracle eviction (`method="h2o"`)

- **`veloxquant_mlx.cache.h2o_cache.H2OKVCache`** — the library's first
  **continuous-decode cumulative-score eviction** method. *Inspired by, not a
  faithful port of,* "H2O: Heavy-Hitter Oracle for Efficient Generative Inference
  of Large Language Models" (Zhang et al., ICLR 2024, arXiv:2306.14048). On every
  step (prefill and decode alike), each incoming token's approximate attention
  distribution over the existing cache is computed using the **new key vector as a
  proxy query** (true queries are not visible at cache-wrapper level — same
  approximation as SnapKV-adapted). The resulting softmax weights are accumulated
  into a per-token cumulative importance score. When the cache exceeds
  `h2o_budget`, the **lowest-score non-sink token** is permanently evicted.
  The cache is thus bounded at all times to `h2o_budget` positions.
- **Third distinct eviction axis in VeloxQuant-MLX:**
  - SnapKV-adapted — score-based, fires once at prefill end; grows during decode.
  - StreamingLLM-adapted — positional (recency + sinks), constant-memory throughout.
  - H2O-adapted — cumulative attention mass, budget-bounded at every step.
- **Adaptation limitations (documented, not hidden):**
  - Key-as-query proxy: approximates the paper's true query attention signal.
  - No RoPE position-ID remapping after eviction.
  - Uniform `h2o_budget` and `h2o_n_sink` across all heads.
  - Scores accumulate as a running sum of softmax weights; some paper variants
    accumulate unnormalised logits — may diverge at very low budgets.
- Primitives in `veloxquant_mlx/quantizers/h2o.py`: `H2OState`,
  `init_h2o_state`, `h2o_update`, `h2o_get_kv`, `h2o_fp16_bytes`,
  `full_h2o_fp16_bytes`.
- Config: `h2o_budget` (int, default 512), `h2o_n_sink` (int, default 4).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all
  `h2o_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_h2o.py` (18 tests) +
  `tests/cache/test_h2o_cache.py` (15 tests): init state, single-token bootstrap,
  multi-token absorption, budget enforcement (never exceeded across 30 decode steps),
  sink protection (sinks always present after evictions), n_sink=0 edge case,
  score non-negativity, score accumulation across steps, byte accounting formula,
  compression_ratio, tokens_seen, factory dispatch, for_model propagation,
  determinism.
- Offline-synthetic harness in `benchmark_scripts/benchmark_h2o.py` sweeping
  `(seq_len, budget, n_sink)` on synthetic fp16 data. Not yet run on Apple Silicon
  hardware.

---

## [0.20.0] — 2026-07-01

### Added — StreamingLLM: sink + recency-window structural eviction (`method="streaming_llm"`)

- **`veloxquant_mlx.cache.streaming_llm_cache.StreamingLLMKVCache`** — the repo's
  first **constant-memory** cache and first **structural positional eviction** method.
  *Inspired by, not a faithful port of,* "Efficient Streaming Language Models with
  Attention Sinks" (Xiao et al., ICLR 2024, arXiv:2309.17453). Keeps only the first
  `stream_n_sink` token positions (frozen as attention sinks) and the most recent
  `stream_window_size` positions (rolling FIFO). All other positions are permanently
  evicted. Both prefill (`S > 1`) and decode (`S == 1`) tokens are processed
  identically through the same sink+window logic — the cache **never** grows beyond
  `stream_n_sink + stream_window_size` positions regardless of how many tokens are
  generated. The `streaming_ratio` and `tokens_in_window` properties report storage
  accounting.
- **Orthogonal to SnapKV-adapted**: SnapKV evicts by importance score at prefill and
  then grows during decode; StreamingLLM-adapted evicts continuously by position and
  stays constant-memory throughout generation.
- **Adaptation limitations (documented, not hidden):**
  - No attention mask adjustment — the model attends to all returned K/V positions; only
    the number of K/V rows is bounded.
  - No RoPE position-ID remapping — original token positions preserved in returned rows;
    remapping requires model-level patching.
  - Fixed `stream_n_sink` count — not adaptive.
- Primitives in `veloxquant_mlx/quantizers/streaming_llm.py`: `StreamingWindow`,
  `init_streaming_window`, `stream_update`, `stream_get_kv`, `stream_fp16_bytes`,
  `full_stream_fp16_bytes`.
- Config: `stream_n_sink` (int, default 4), `stream_window_size` (int, default 512).
  Single-layer (no coordinator); `KVCacheBuilder.for_model()` propagates all `stream_*`
  fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_streaming_llm.py` (17 tests) +
  `tests/cache/test_streaming_llm_cache.py` (15 tests): init shapes, sink absorption,
  FIFO trimming, constant-memory guarantee (30-step stress), stream_get_kv shape/dtype/
  sink-first ordering, byte accounting, streaming_ratio, large-prefill trim, n_sink=0
  edge, determinism, for_model config propagation. **32/32 passing.**
- Offline-synthetic harness in `benchmark_scripts/benchmark_streaming_llm.py` sweeping
  `(seq_len, window_size)` on synthetic data. Not yet run on Apple Silicon hardware.

---

## [0.19.0] — 2026-07-01

### Added — SnapKV: prefill observation-window token eviction (`method="snapkv"`)

- **`veloxquant_mlx.cache.snapkv_cache.SnapKVKVCache`** — the repo's first
  **token eviction** cache and the first where the paper's actual attention
  signal is computable at cache level without model surgery. *Inspired by, not
  a faithful port of,* "SnapKV: LLM Knows What You are Looking for Before
  Generation" (Yuan et al., ICLR 2025, arXiv:2404.14469). During prefill
  (`S > 1`), the last `snap_obs_window` key rows act as proxy queries; scaled
  dot-product softmax over all `S` prefix key positions gives per-token
  importance scores. The top-`snap_budget` tokens (plus `snap_n_sink`
  always-kept sink positions) are retained as fp16. All evicted positions are
  permanently dropped. Decode tokens (`S == 1`) are always appended — never
  evicted. The `eviction_ratio` and `keep_rate` properties report the storage
  accounting.
- **Adaptation:** the paper uses the final prompt *query* vectors for the
  observation window (not visible to a cache wrapper). We substitute the last
  `snap_obs_window` *key* vectors as proxy queries — stronger than key-norm
  alone (computes the actual attention distribution from K) but still an
  approximation. No max-pool smoothing (paper's `kernel_size > 1`). Uniform
  budget across all heads. Documented as "SnapKV-adapted (key-as-query proxy)"
  throughout; never claimed as a faithful port.
- Primitives in `veloxquant_mlx/quantizers/snapkv.py`: `obs_window_attention_scores`,
  `snap_select_indices`, `snapkv_compress`, `snapkv_fp16_bytes`, `full_fp16_bytes`
  (+ `SnapKVState`).
- Config: `snap_budget` (int, default 512), `snap_obs_window` (int, default 32),
  `snap_n_sink` (int, default 4). Single-layer (no coordinator); `KVCacheBuilder.for_model()`
  propagates all `snap_*` fields via `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_snapkv.py` (18 tests) +
  `tests/cache/test_snapkv_cache.py` (13 tests): obs-window scores shape, dtype,
  value range; `obs_window` clamp; `snap_select_indices` exact count, sorted order,
  sink guarantee, high-score preference; `snapkv_compress` output shape/dtype;
  budget≥S no-eviction edge case; byte accounting; eviction ratio > 1; keep rate
  in range; decode accumulation; decode-only no-eviction; determinism;
  `for_model` propagation.
- **Benchmark** — `benchmark_scripts/benchmark_snapkv.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** key-as-query proxy; no max-pool smoothing; no per-head budget;
  no model-level benchmark yet.

## [0.18.0] — 2026-06-30

### Added — ZipCache: saliency-adaptive per-token mixed-precision (`method="zipcache"`)

- **`veloxquant_mlx.cache.zipcache_cache.ZipCacheKVCache`** — the repo's first
  **per-token mixed bit-width** cache. *Inspired by, not a faithful port of,*
  "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token
  Identification" (He et al., NeurIPS 2024, arXiv:2405.14256). The top
  `zipcache_hi_fraction` of tokens by key L2-norm are quantized at `zipcache_hi_bits`;
  the rest at `zipcache_lo_bits`. Both groups remain quantized — this is not fp16
  protection (KIVI-Sink) nor head budgeting (AdaKV-proxy). Effective average key rate:
  `hi_frac × hi_bits + (1-hi_frac) × lo_bits`.
- **Adaptation:** the paper's true saliency signal is normalized attention scores,
  which are not observable by a cache wrapper. Key L2-norm is the proxy (same signal
  used by KIVI-Sink and AdaKV-proxy, but with a different decision — bit-width routing
  rather than fp16 protection or head budgeting). This is the third use of the key-norm
  proxy; the proxy weakness is documented plainly.
- Primitives in `veloxquant_mlx/quantizers/zipcache.py`: `token_key_norms`,
  `saliency_mask`, `channel_quant`, `channel_dequant`, `zipcache_compress`,
  `zipcache_reconstruct`, `zipcache_bytes`, `base_only_bytes`, `zipcache_quant_dequant`
  (+ `ZipCacheState`).
- Config: `zipcache_hi_bits`, `zipcache_lo_bits`, `zipcache_hi_fraction`,
  `zipcache_group_size`, `zipcache_quantize_values`. Single-layer (no coordinator);
  `KVCacheBuilder.for_model()` propagates all `zipcache_*` fields automatically via
  `dataclasses.replace`.
- **Tests** — `tests/quantizers/test_zipcache.py` (16 tests) +
  `tests/cache/test_zipcache_cache.py` (11 tests): saliency mask selects exact
  top-fraction by key-norm; 4-bit channel quant cosine > 0.995; 2-bit cosine > 0.8;
  compress/reconstruct shape and dtype; `hi_fraction=0` and `=1` edge cases;
  byte ordering `compressed ≤ fp16`, mixed-bit ≥ all-lo-bit baseline; effective avg
  bits in `[lo_bits, hi_bits]`; values-off passthrough; decode accumulation;
  determinism; build via both `create` and `for_model`.
- **Benchmark** — `benchmark_scripts/benchmark_zipcache.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** proxy weakness (key-norm, not true attention scores) is stated in
  all docs; no model-level benchmark run yet.

## [0.17.0] — 2026-06-29

### Added — GEAR: error-feedback KV cache (`method="gear"`)

- **`veloxquant_mlx.cache.gear_cache.GEARKVCache`** — the repo's first
  **error-feedback** cache. *Inspired by, not a faithful port of,* "GEAR: An
  Efficient KV Cache Compression Recipe for Near-Lossless Generative Inference of
  LLM" (Kang et al., arXiv:2403.05527). Every other method picks a bit-width (or
  a cache layout) and lives with the quantization error; GEAR makes *any*
  ultra-low-bit base quantizer near-lossless by reconstructing what it threw away
  via the three-part decomposition `X ≈ Quant_b(X) + L·R + S`: an ultra-low-bit
  base group quant, a **low-rank** approximation of the quantization residual
  `E = X − dequant(Quant_b(X))`, and a **sparse** matrix correcting the
  top-magnitude outlier entries the low-rank term cannot absorb. Unlike CacheGen
  (reconstruction identical to group quant), GEAR's reconstruction genuinely
  **recovers quality** the base bit-width loses.
- **Adaptation:** the residual SVD is computed per `update_and_fetch` call on the
  tensor the cache holds (reusing the SVDq/PALU prefill-SVD pattern), and GEAR's
  fused streaming-dequant CUDA kernel is **not** ported — we reconstruct fp16
  then call MLX SDPA, so the *stored* cache shrinks but attend-time peak memory
  does not. The base quant is borrowed from CacheGen and the truncated-SVD helper
  (`_quant_utils._truncated_svd`) is shared with SVDq/PALU.
- Primitives in `veloxquant_mlx/quantizers/gear.py`: `quantize_base`, `residual`,
  `lowrank_error`, `sparse_outliers`, `gear_compress`, `gear_reconstruct`,
  `gear_bytes`, `base_only_bytes`, `gear_quant_dequant` (+ `GEARState`).
- Config: `gear_bits`, `gear_rank`, `gear_energy_threshold`,
  `gear_sparse_fraction`, `gear_group_size`, `gear_quantize_values`. Single-layer
  (no coordinator); `KVCacheBuilder.for_model()` propagates the `gear_*` fields
  automatically via `dataclasses.replace`.
- **Tests** — `tests/cache/test_gear_cache.py` (10) +
  `tests/quantizers/test_gear.py` (13): GEAR reconstruction MSE strictly below
  base-quant-alone on low-rank+outlier data; low-rank-alone and sparse-alone each
  help; `rank=0, sparse=0` collapses exactly to base group quant; rank-`r`
  residual recovered to `< eps`; sparse selection picks true top-magnitude
  entries; byte-accounting ordering `base_only ≤ compressed ≤ fp16`;
  `error_recovery_ratio` in `(0,1]`; values-off path; decode accumulation;
  determinism; build via both `create` and `for_model`.
- **Benchmark** — `benchmark_scripts/benchmark_gear.py` (offline-synthetic,
  loads no model). **Not yet run** on hardware for committed numbers.
- **Honest scope:** the stored cache shrinks but reconstruction is fp16 for SDPA,
  so attend-time peak memory is not reduced; the low-rank/sparse factors are
  overhead, so the rank must be low relative to the head dim (the GEAR premise) —
  reported honestly, never hidden.

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
