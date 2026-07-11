# Phase 1 — New-Method Survey V3 (post-KIVI-Sink)

Follow-up to `NEW_METHOD_SURVEY_V2.md` (which led to KIVI-Sink, now shipped in
v0.9.0). The repo now has TurboQuant, RVQ, VecInfer (Metal kernels), RaBitQ,
CommVQ, QJL, PolarQuant, RateQuant, SpectralQuant, KIVI, and KIVI-Sink. The
open gaps are: **sub-2-bit quantization beyond RaBitQ**, **low-rank projection
(simpler than PALU)**, and **cross-layer / prefill-decode asymmetric schemes**.
Token eviction (H2O, SnapKV, AdaKV) is still out of scope for the quantization
pipeline, but hybrids that layer quantization on top are newly interesting.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper. Assessments marked ⚠️ (uncertain) where implementation details required
reading the full HTML. No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Metal fit | Effort | Verdict |
|---|---|---|---|---|---|
| **SVDq** | arXiv:2502.15304 (Feb 2025) | Sub-2-bit key-only compression via **offline SVD + per-channel mixed-precision quant** — 1.25-bit equivalent key cache, values left fp16 | ✅ offline SVD in `mx.linalg.svd`; per-token decode projection is O(d²) not O(sd) | Med | **CHOSEN** |
| **XQuant** | arXiv:2510.11236, **EMNLP 2025** (Yang et al.) | **Cross-layer quantization**: odd layers reuse quantized cache from even layers, storing only per-layer scale/zero-point — sub-1.4-bit equivalent | ⚠️ cross-layer coordinator required; breaks single-wrapper pattern | Med-High | strong future option |
| **AdaKV** | arXiv:2407.11550, **NeurIPS 2025** (Feng et al.) | **Head-adaptive eviction budget**: allocates more cache slots to high-entropy heads, improving any eviction method | ❌ requires softmax attention scores — outside cache-wrapper contract | Med | deferred (needs attn scores) |
| **Kitty** | arXiv:2511.18643 (Nov 2025) | 2-bit quantization with **dynamic channel-wise precision boost** — keeps top-k sensitive key channels at higher bits | ⚠️ channel sensitivity ranking visible from K tensor, but reference implementation uses CUDA dequant kernel | Med | strong future option |
| **KVLinC** | arXiv:2510.05373 (Oct 2025) | Hadamard rotation + **trainable linear correction adapters** that compensate quantisation error in keys | ❌ adapters require 2–11 h training on H200; needs full-precision keys at runtime alongside quantised keys | High | skip (training required, not cache-only) |
| **ThinKV** | arXiv:2510.01290, **ICLR 2026 Oral** (Ramachandran et al.) | Thought-adaptive hybrid quantisation+eviction for reasoning models; <5% cache size | ❌ requires attention sparsity patterns (attention scores); also CoT-specific | High | skip (attn scores; niche) |

---

## Chosen: SVDq (sub-2-bit key cache via offline SVD)

### What the paper actually does

SVDq ([arXiv:2502.15304](https://arxiv.org/abs/2502.15304), Feb 2025, no venue
yet — unreviewed preprint) addresses key-cache compression at extreme bit widths.
Its mechanism in two phases:

**Offline / prefill phase (once per sequence):**
1. After prefill, the key-cache matrix **K ∈ ℝ^{s×d}** for each layer is
   factored via truncated SVD: **K ≈ Û · Σ_r · V^H + K̄**, where the rank-r
   approximation captures the dominant signal and **K̄** is the mean key.
2. The latent representation **L = K · V** (the projection into the top-r
   singular directions) is then quantized **per-latent-channel** with
   non-uniform importance-aware bit allocation — higher-magnitude singular
   channels get more bits.
3. The projection matrix **V** and mean **K̄** are stored in fp16; they are
   O(d²) and negligible for long contexts.

**Online / decode phase (per new token):**
1. Each new key **k_new ∈ ℝ^d** is projected: **l_new = k_new · V**, quantized,
   and appended to the latent cache.
2. For attention, the full key matrix is reconstructed: **K = L_q · V^H + K̄**,
   then used normally.

The paper reports an *equivalent* key precision of **1.25 bits** (via combined
channel truncation + mixed-bit quantization) with substantially lower
reconstruction error than per-channel 4-bit quantization in the original space.
Values are left at fp16 (the paper explicitly notes values show weak low-rank
structure). This is complementary to the repo's existing value compression
(TurboQuant, KIVI, RaBitQ all compress values; SVDq is the first method that
focuses exclusively on keys at extreme bit widths).

### The honest adaptation problem

Three adaptation decisions for VeloxQuant-MLX:

**1. SVD timing.** The paper computes SVD over the full prefill key matrix once.
In the repo's `update_and_fetch` contract, we *do* see all prefill keys as a
batch when `keys.shape[0] > 1` (the prefill call). So SVD is triggered on that
batch and the projection matrix V is stored as a layer attribute. Subsequent
decode calls (shape `[1, ...]`) project into the already-stored V. This fits
cleanly — no model-forward interception needed.

**2. Rank selection.** The paper uses a fixed rank r (e.g., r = d/2). We will
expose this as `KVCacheConfig(svdq_rank=r)` with a sensible default of d//4 (a
more aggressive setting appropriate for Apple Silicon's bandwidth constraint).
Rank can also be set by energy threshold (e.g., retain 95% of singular value
energy), which we will implement as an alternative.

**3. Quantisation of latent channels.** The paper's importance-aware bit
allocation assigns 1, 2, 3, or 4 bits per latent channel based on singular
value magnitude. We implement this as a simple lookup: channels in the top-25%
by singular value get 4-bit (re-using our `TurboQuantizer`), the rest get 2-bit
(re-using `KIVIQuantizer`). This makes SVDq a **compositor over existing
quantizers** rather than a new quantizer — consistent with the repo's design
philosophy.

**What we do NOT implement:** the paper's optional sparsity layer (zeroing
low-magnitude latent channels). That axis is already covered by KIVI-Sink and
would complicate the byte-accounting story. We document this as a known
simplification.

### Why this is the right pick

1. **Fills the genuine gap: sub-2-bit key compression.** RaBitQ covers ~1-bit
   *vector quantisation* (full K+V); SVDq achieves 1.25-bit equivalent on keys
   alone via a mathematically different route (linear projection + quantisation
   in latent space). Together they give the repo two sub-2-bit methods on
   different axes — one for keys, one for both tensors.

2. **Cache-only access — no model surgery.** SVD is computed from the K tensor
   the cache already holds at the end of prefill. No hidden-state hooks, no
   attention-score coupling, no RoPE interception. This is a genuine first-class
   citizen of the `update_and_fetch` design.

3. **Apple Silicon bandwidth story is the strongest of any candidate.** The
   decode memory footprint of the key cache drops from 16 bits/element to ~1.25
   bits/element — a 12.8× reduction in bandwidth pressure. On M-series chips
   where decode is entirely bandwidth-bound, this is the highest-leverage
   single-axis optimisation left on the table. The per-token decode cost (matrix
   multiply **O(r·d)** per step, where r ≤ d/4) is a negligible compute
   overhead on the unified-memory GPU.

4. **Composes with existing quantizers.** The SVDq wrapper delegates to
   `TurboQuantizer` and `KIVIQuantizer` for the actual bit-packing, meaning the
   latent-channel quantisation inherits all the existing unit tests and
   correctness guarantees.

5. **Honest uncertainty:** SVDq is a preprint (no venue as of June 2026). We
   will label it clearly in docs and the CHANGELOG as
   "SVDq-adapted (VeloxQuant-MLX implementation)" and report empirical numbers
   from committed `results.json` — not paper claims.

### Why the alternatives were not chosen

- **XQuant** is architecturally interesting (sub-1.4-bit via cross-layer reuse)
  but requires a layer-pair coordinator that breaks the single-wrapper contract.
  Every even-numbered layer must hold quantised cache for its paired odd layer
  to read during dequantisation. This is a significant refactor of
  `KVCacheConfig` and the layer-iteration logic in `mlx_lm`. Deferred.
- **AdaKV** requires softmax attention weights — outside the cache-only contract.
  An approximation using key-norm as a proxy (as we did for KIVI-Sink) would
  give a weaker signal for head-level budget allocation than for sink
  identification. Not a clean port.
- **Kitty** is the closest competitor. Its "dynamic channel-wise precision boost"
  is conceptually similar to SVDq's importance-aware channel quantisation. The
  key difference: Kitty operates in the *original* key space (ranking raw key
  channels by sensitivity), while SVDq operates in the *latent* space (ranking
  by singular value magnitude). SVDq's latent channels are by construction
  ordered by importance, making the mixed-bit allocation trivial and
  deterministic. Kitty is a strong backup if SVDq's reconstruction quality
  underperforms.
- **KVLinC** and **ThinKV**: eliminated on hard constraints (training required;
  attention scores required).

### Planned artifacts (Phases 2–6)

- `veloxquant_mlx/cache/svdq_cache.py` — `SVDqKVCache` wrapper. Stores
  `self._V` (projection matrix, fp16, shape `[d, r]`) and `self._K_mean`
  (mean key, fp16, shape `[d]`) per-layer after prefill. Decode path: project
  new key, quantise latent, append; reconstruct full keys on fetch.
- `veloxquant_mlx/quantizers/svdq_quantizer.py` — `SVDqQuantizer`: thin wrapper
  that routes latent channels to `TurboQuantizer` (top-25% by singular value)
  or `KIVIQuantizer` (remainder). No new quant logic — pure routing.
- Config: `KVCacheConfig(method="svdq", svdq_rank=None, svdq_energy_threshold=0.95)`
  where `None` rank falls back to energy-threshold selection.
- Tests:
  - SVD projection correctness (reconstruction error < ε vs full K)
  - Prefill-only case (no decode calls) — V and K̄ stored correctly
  - Decode accumulation correctness — sequential keys reconstruct to original
  - Byte-accounting: `compressed_key_bytes` reflects mixed-bit packing, not 16-bit
  - Quality comparison vs KIVI-2bit and RaBitQ at equal byte budgets (benchmark)
- `benchmark_scripts/benchmark_svdq.py` — perplexity + throughput vs KIVI/RaBitQ
  at 1.25-bit, 2-bit, 4-bit equivalent key cache size on M2/M3 Pro.
- CHANGELOG entry, EVIDENCE_TABLE row, docs page.

---

## Sources (verified)

- SVDq — https://arxiv.org/abs/2502.15304 (arXiv Feb 2025, unreviewed preprint)
- XQuant — https://arxiv.org/abs/2510.11236 (EMNLP 2025); code https://github.com/brinenick511/XQuant; ACL Anthology https://aclanthology.org/2025.emnlp-main.494/
- AdaKV — https://arxiv.org/abs/2407.11550 (NeurIPS 2025); code https://github.com/FFY0/AdaKV
- Kitty — https://arxiv.org/abs/2511.18643 (arXiv Nov 2025, unreviewed preprint)
- KVLinC — https://arxiv.org/abs/2510.05373 (arXiv Oct 2025, unreviewed preprint)
- ThinKV — https://arxiv.org/abs/2510.01290 (ICLR 2026 Oral); https://openreview.net/forum?id=SFsvqfNUsh
- MagicPIG (considered, excluded — LSH attention approximation, CUDA-optimised CPU offload) — https://arxiv.org/abs/2410.16179
- RocketKV (considered, excluded — requires attention scores for top-k selection) — https://arxiv.org/abs/2502.14051 (ICML 2025)
- AsymKV (considered, excluded — overlaps KIVI asymmetric axis already in repo) — https://arxiv.org/abs/2410.13212
