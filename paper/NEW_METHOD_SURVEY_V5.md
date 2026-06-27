# Phase 1 — New-Method Survey V5 (post-PALU)

Follow-up to `NEW_METHOD_SURVEY_V4.md` (which led to PALU, now shipped in
0.15.0). With PALU the repo covers low-rank on both tensors; every candidate
from surveys V2–V4 is shipped or hard-rejected. This survey picks **two**
methods from the two remaining axes the repo does not touch: **entropy coding**
of the codes (CacheGen) and **cross-layer tensor merging** (MiniCache).

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper (WebFetch on the arXiv abstract). No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **CacheGen** | arXiv:2310.07240, **SIGCOMM 2024** (Liu et al.) | **Entropy coding** of the quantized codes — the repo packs at fixed bit-width but never entropy-codes. Token-wise locality → delta-code → compress toward Shannon entropy | ✅ pure post-quant byte coding on the K/V the cache holds | Low-Med | **CHOSEN** |
| **MiniCache** | arXiv:2405.14366, **NeurIPS 2024** (Liu et al.) | **Cross-layer tensor merging** — merges adjacent layers' KV into one shared SLERP direction + per-layer magnitude. Distinct from XQuant (code reuse); MiniCache merges the tensors | ✅ SLERP on the K/V the cache holds; reuses the XQuant coordinator pattern | Med | **CHOSEN** |
| **ZipCache** | arXiv:2405.14256, **NeurIPS 2024** | Saliency-aware per-token mixed precision | ❌ true signal is attention scores; cache only sees K/V (already proxied twice via AdaKV / KIVI-Sink) | Med | deferred (weak proxy, third use of same signal) |
| **KVLinC / ThinKV / KVmix** | (from V3/V4) | — | ❌ training / attention scores / RateQuant overlap | — | skip (re-confirmed) |

---

## Chosen A: CacheGen (entropy-coded KV via token locality)

### What the paper actually does

CacheGen ([arXiv:2310.07240](https://arxiv.org/abs/2310.07240), SIGCOMM 2024)
is a serving system that compresses and streams the KV cache. Its compression
core uses a **custom tensor encoder** that leverages the KV cache's
distributional properties:

1. **Token-wise locality** — adjacent tokens' KV are similar, so the *delta*
   between consecutive tokens' quantized values is concentrated near zero.
2. **Layer-wise sensitivity** — deeper layers tolerate coarser quantization.
3. **Arithmetic coding** — the low-entropy delta stream is encoded into a
   compact bitstream with negligible decoding overhead (in their C++/GPU codec).

### The honest adaptation problem

CacheGen's value is the *entropy coder*. A faithful port would ship a serial
range/arithmetic codec — but a per-step serial codec bottlenecks MLX's parallel
decode and adds no quality. So:

> We keep the exact reconstruction (identical to group quant) and **model** the
> entropy-coded byte size from the **measured Shannon entropy** of the
> token-delta code stream, capped at the fixed-width packed size (a real coder
> falls back to raw packing on incompressible data).

This captures the storage win honestly: ~10–17% on token-correlated data,
exactly 0% (never negative) on iid. Documented as "CacheGen-adapted".

### Why it is the right pick

- **Fills a genuinely new axis:** entropy coding. Every other method picks a
  bit-width; none compress the *codes themselves*. Orthogonal to all of them.
- **Cache-only, zero training, deterministic.** Pure post-quant byte math.
- **Honest about leverage:** it is a *storage* win (codes dequant to fp16 for
  SDPA), so on Apple Silicon's bandwidth-bound decode it is lower-leverage than
  PALU/SVDq — stated plainly, not oversold.

## Chosen B: MiniCache (cross-layer depth merge via SLERP)

### What the paper actually does

MiniCache ([arXiv:2405.14366](https://arxiv.org/abs/2405.14366), NeurIPS 2024)
compresses across **network depth**. Observation: KV states are highly similar
between adjacent layers in the middle-to-deep portion of LLMs. Mechanism:

1. Disentangle each KV state into **magnitude** (L2 norm) and **direction**
   (unit vector).
2. **SLERP** the directions of an adjacent layer pair into one shared direction
   (interpolate direction, preserve each layer's length).
3. Store the shared direction once + each layer's magnitude scalars.
4. A **token retention** strategy keeps highly distinct state pairs unmerged.

Reports up to 5.02× with 4-bit quantization on LLaMA-2-7B.

### The honest adaptation problem

Like XQuant, MiniCache's true integration point is the attention forward pass.
We reuse the repo's solved pattern: a **shared coordinator**. The primary layer
publishes its KV; the later-arriving merge layer fetches it, performs the SLERP
merge, and both reconstruct from the shared direction. We implement the
magnitude/direction SLERP and the token-retention set faithfully; we do **not**
additionally low-bit quantize (the paper combines with 4-bit quant — left to
composition). Documented as "MiniCache-adapted".

### Why it is the right pick

- **Completes the cross-layer story.** PALU compresses *within* a layer; XQuant
  *reuses codes across* layers; MiniCache *merges tensors across* layers — a
  third, mathematically distinct cross-layer mechanism (SLERP vs code-sharing).
- **The hard infrastructure already exists.** The XQuant `*Coordinator` +
  `pair_layers` machinery is exactly what MiniCache needs — far lower effort
  than XQuant itself was.
- **Cache-only, zero training, deterministic.** SLERP is pure tensor math; the
  retention set is a cosine threshold.

### Why ZipCache was not chosen

ZipCache's per-token saliency signal is the **normalized attention score** — not
visible to a cache wrapper. The repo has already proxied attention importance via
key-norm twice (AdaKV-proxy, KIVI-Sink); a third method on the same weak proxy
adds little. Deferred.

---

## Planned artifacts (both methods, Phases 2–6)

**CacheGen**
- `veloxquant_mlx/quantizers/cachegen.py` — `quantize_to_codes`, `dequant_codes`,
  `token_delta`, `symbol_entropy_bits`, `entropy_coded_bytes`, `fixed_width_bytes`,
  `cachegen_quant_dequant`.
- `veloxquant_mlx/cache/cachegen_cache.py` — `CacheGenKVCache` (single-layer).
- Config: `cachegen_bits`, `cachegen_group_size`, `cachegen_use_delta`.

**MiniCache**
- `veloxquant_mlx/quantizers/minicache.py` — `pair_layers_depth`, `to_mag_dir`,
  `slerp`, `merge_pair`, `reconstruct_layer`, `merge_similarity`.
- `veloxquant_mlx/cache/minicache_cache.py` + `minicache_coordinator.py`.
- Config: `minicache_start_frac`, `minicache_group_size`,
  `minicache_retention_threshold`, `minicache_slerp_t`, `minicache_max_ctx`.
- Builder: `_build_minicache` (shared coordinator + role assignment).

**Both:** tests (cache + quantizer), benchmark with offline harness (no
model-level numbers until `results.json` committed), docs page + sidebar +
overview, CHANGELOG (root + docs-site), EVIDENCE_TABLE rows, landing cards.

---

## Sources (verified)

- CacheGen — https://arxiv.org/abs/2310.07240 (SIGCOMM 2024)
- MiniCache — https://arxiv.org/abs/2405.14366 (NeurIPS 2024)
- ZipCache — https://arxiv.org/abs/2405.14256 (NeurIPS 2024, deferred — attention-score saliency)
