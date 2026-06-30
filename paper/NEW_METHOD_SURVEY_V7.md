# Phase 1 — New-Method Survey V7 (post-GEAR)

Follow-up to `NEW_METHOD_SURVEY_V6.md` (which led to GEAR, shipped in 0.17.0).
After GEAR, the repo spans every axis identified across all prior surveys:
scalar/group quant (KIVI, KVQuant-NUQ, TurboQuant), vector quant (RVQ, VecInfer,
CommVQ), low-rank *signal* (SVDq keys, PALU K+V), cross-layer (XQuant code-reuse,
MiniCache SLERP merge), entropy coding (CacheGen), error-feedback (GEAR), and
attention-proxy adaptive schemes (KIVI-Sink, AdaKV-proxy, Kitty).

The one remaining candidate that has never been hard-rejected — only repeatedly
deferred for proxy weakness — is **ZipCache**. Its core deferred reason (requires
attention scores) maps to an existing repo pattern (key-norm proxy), and its
allocation mechanism (per-token mixed bit-width) is genuinely distinct from all
three existing key-norm proxy methods (KIVI-Sink: fp16 protection; AdaKV: head
budget; Kitty: channel-wise mixed bits). This survey formally picks ZipCache.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper. No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **ZipCache** | arXiv:2405.14256, **NeurIPS 2024** (He et al.) | **Per-token mixed bit-width**: saliency proxy (key-norm) routes high-norm tokens to hi_bits, low-norm tokens to lo_bits — within the quantized space (not fp16 protection). Distinct from KIVI-Sink (fp16), AdaKV (head budget), and Kitty (channel selection) | ✅ key-norm proxy is cache-observable; per-token bit dispatch is pure group-quant math | Low-Med | **CHOSEN** |
| **KVLinC** | arXiv:2510.05373 (re-confirmed) | Hadamard rotation + trainable adapters | ❌ training required | High | hard reject (re-confirmed V3–V6) |
| **ThinKV** | arXiv:2510.01290, ICLR 2026 Oral (re-confirmed) | CoT-specific hybrid quant+eviction | ❌ attention scores; CoT-specific | High | hard reject (re-confirmed V3–V6) |

---

## Chosen: ZipCache-adapted (saliency-adaptive per-token mixed precision)

### What the paper actually does

ZipCache ([arXiv:2405.14256](https://arxiv.org/abs/2405.14256), NeurIPS 2024,
He et al.) achieves aggressive KV compression by:

1. **Saliency scoring:** the normalized attention score per token ranks tokens
   by importance. High-attention (salient) tokens get full bits; low-attention
   tokens get ultra-low bits.
2. **Channel-separable quantization:** rather than uniform group quant across
   all channels, ZipCache scales each channel independently for the per-token
   bit assignment. This makes the mixed-bit boundary the only differentiator
   needed per token.
3. **Dynamic per-token bit allocation:** the allocation adapts to the input
   distribution — not a fixed pattern like Kitty's channel ranking.

The paper reports that even a small hi-bit fraction (20%) on the most salient
tokens recovers quality comparable to uniform higher-bit quantization.

### The honest adaptation problem

ZipCache's saliency signal is the **normalized attention score** — not visible
to a cache wrapper. This is the same constraint that caused its deferral across
V1–V6.

**Adaptation:** key L2-norm is a reliable proxy for token importance (the same
signal used by KIVI-Sink for sink detection). High-norm tokens attract
disproportionate attention — the mechanism behind attention-sink formation. Our
implementation:

> Identify the top-`zipcache_hi_fraction` tokens by key L2-norm; quantize them
> at `zipcache_hi_bits`. Quantize the remaining tokens at `zipcache_lo_bits`.
> Both groups remain quantized (not fp16) — this is the key distinction vs
> KIVI-Sink.

This is labeled **ZipCache-adapted (key-norm saliency proxy)** throughout.

### Why this is different from existing key-norm methods

All three existing key-norm methods use the same proxy signal for different ends:

| Method | Signal | Decision | Outcome |
|---|---|---|---|
| KIVI-Sink | key-norm | top-k positions → fp16 protection | Binary: quantized vs not |
| AdaKV-proxy | mean key-norm per head | allocate more tokens to high-entropy heads | Head-level budget |
| ZipCache-adapted | key-norm per token | hi vs lo bits, both quantized | Per-token bit-width |

ZipCache-adapted is the only method that allocates *within the quantized space*
— the hi-bit tokens are still group-quantized, just at a higher bit-width.
Tokens below the saliency threshold get very aggressive compression (2-bit),
while the top fraction retains quality (4-bit). The effective average bit-rate
is `hi_frac * hi_bits + (1 - hi_frac) * lo_bits` — explicitly between lo and hi.

### What we do NOT implement

- True attention-score saliency (requires model-forward interception).
- ZipCache's "channel-separable" quantization in the original sense: in the
  paper this means each *channel* has an independent scale per token (not
  grouped). We use standard per-group min/max (as in KIVI) because it reuses
  existing tested infrastructure. The per-token saliency routing is what we
  implement faithfully.
- Dynamic bit reallocation per decode step (we compute saliency once on the
  incoming block, not retroactively).

### Why this is the right pick

1. **Last deferred candidate — never hard-rejected.** Every other deferred
   method (KVLinC, ThinKV) has a hard constraint (training; attention scores +
   CoT scope). ZipCache's deferred reason is proxy weakness — a constraint the
   repo has already worked around twice with honest labeling.
2. **Mechanistically distinct from existing key-norm methods.** Per-token
   mixed-bit-within-quantized is not fp16 protection (KIVI-Sink), not head
   budgeting (AdaKV), not channel selection (Kitty). This is its own slot.
3. **Zero training, zero model interception, single-layer.** Pure post-K/V
   tensor math in `update_and_fetch`. No coordinator needed.
4. **Honest uncertainty:** third use of the key-norm proxy for the "importance"
   signal. Documents clearly that the proxy is weaker than true attention scores.
   Numbers come from committed `results.json`, not paper claims.

### Planned artifacts (Phases 2–6)

- `veloxquant_mlx/quantizers/zipcache.py` — `token_key_norms`,
  `saliency_mask`, `channel_quant`, `channel_dequant`,
  `zipcache_compress`, `zipcache_reconstruct`, `zipcache_bytes`,
  `base_only_bytes`, `zipcache_quant_dequant` (+ `ZipCacheState`).
- `veloxquant_mlx/cache/zipcache_cache.py` — `ZipCacheKVCache` (single-layer,
  per-token mixed-bit keys, full byte accounting).
- Config: `KVCacheConfig(method="zipcache", zipcache_hi_bits, zipcache_lo_bits,
  zipcache_hi_fraction, zipcache_group_size, zipcache_quantize_values)`.
- Tests:
  - `tests/quantizers/test_zipcache.py` (≥12 tests): saliency mask correctness,
    channel quant round-trips at 4-bit and 2-bit, compress/reconstruct shapes,
    uniform-bits edge cases, byte ordering, values-off path, determinism.
  - `tests/cache/test_zipcache_cache.py` (≥10 tests): factory dispatch, shape,
    decode accumulation, byte ordering, hi/lo fraction edge cases, mask, for_model.
- `benchmark_scripts/benchmark_zipcache.py` — offline reconstruction-MSE vs
  KIVI-2bit and uniform lo-bit baselines. No model loading; writes `results.json`.
  Marked "Not yet run" until executed on hardware.
- `paper/EVIDENCE_TABLE.md` rows, docs page + sidebar + overview + landing card,
  `docs-site/docs/changelog.md`, root `CHANGELOG.md`, `README.md`, `pyproject.toml`.

---

## Sources (verified)

- ZipCache — https://arxiv.org/abs/2405.14256 (NeurIPS 2024, He et al.)
- KVLinC — https://arxiv.org/abs/2510.05373 (hard reject re-confirmed from V3–V6)
- ThinKV — https://arxiv.org/abs/2510.01290 (ICLR 2026 Oral; hard reject re-confirmed from V3–V6)
