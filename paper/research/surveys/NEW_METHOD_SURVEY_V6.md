# Phase 1 — New-Method Survey V6 (post-CacheGen/MiniCache)

Follow-up to `NEW_METHOD_SURVEY_V5.md` (which led to CacheGen + MiniCache, both
shipped in 0.16.0). With those two the repo now spans every axis the prior
surveys identified: scalar/group quant (KIVI, KVQuant-NUQ, TurboQuant), vector
quant (RVQ, VecInfer, CommVQ), low-rank *cache* (SVDq keys, PALU K+V),
cross-layer (XQuant code-reuse, MiniCache SLERP merge), entropy coding
(CacheGen), and attention-proxy adaptive schemes (KIVI-Sink, AdaKV-proxy,
Kitty). The one axis no method touches is **error-feedback** — reconstructing
what an ultra-low-bit quantizer threw away. This survey picks the canonical
method on that axis: **GEAR**.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper (WebFetch on the arXiv abstract). No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **GEAR** | arXiv:2403.05527 (Kang et al.) | **Error-feedback**: quantize at ultra-low bits, then add back a **low-rank SVD of the quantization residual** + a **sparse outlier matrix**. Every other method picks a bit-width; none reconstruct the quantizer's *error*. Orthogonal — composes over any base quantizer | ✅ pure post-quant residual math on the K/V the cache holds | Med | **CHOSEN** |
| **ZipCache** | arXiv:2405.14256, NeurIPS 2024 | Saliency-aware per-token mixed precision | ❌ true signal is the normalized attention score; cache only sees K/V (already proxied twice via AdaKV / KIVI-Sink) | Med | deferred (weak proxy, third use of same signal — re-confirmed from V5) |
| **KVLinC** | arXiv:2510.05373 | Hadamard rotation + trainable linear correction adapters | ❌ adapters need 2–11 h training; full-precision keys at runtime | High | skip (training — re-confirmed from V3/V4) |
| **ThinKV** | arXiv:2510.01290, ICLR 2026 Oral | Thought-adaptive hybrid quant+eviction for reasoning | ❌ needs attention sparsity patterns (attention scores); CoT-specific | High | skip (attn scores — re-confirmed from V3/V4) |

---

## Chosen: GEAR (error-feedback — residual low-rank + sparse outliers)

### What the paper actually does

GEAR ([arXiv:2403.05527](https://arxiv.org/abs/2403.05527), Kang et al.) makes
*any* low-bit KV quantizer near-lossless by reconstructing its error. For a KV
matrix `X`, it stores the three-part decomposition

    X  ~=  Quant_b(X)  +  L . R  +  S

where:

1. **`Quant_b(X)`** — most entries quantized to ultra-low precision (the base).
2. **`L . R`** — a **low-rank** matrix approximating the quantization residual
   `E = X - dequant(Quant_b(X))`. The residual of a coherent KV matrix is itself
   low-rank, so a small rank captures most of the lost signal cheaply.
3. **`S`** — a **sparse** matrix correcting the handful of outlier entries whose
   residual the low-rank term cannot absorb (the top-rho by magnitude).

The paper reports near-lossless 4-bit KV compression with up to 2.38x
throughput and up to 2.29x peak-memory reduction.

### The honest adaptation problem

Three decisions for the VeloxQuant-MLX port, each mirroring a pattern the repo
has already solved:

**1. SVD timing.** GEAR fits the residual low-rank offline / per-buffer. As with
SVDq and PALU, the `update_and_fetch` contract gives us the full prefill batch
when `S > 1`, so the residual SVD is computed on that batch and the projection
`R` is stored. Decode tokens (`S == 1`) project their residual into the stored
`R` and append a sparse correction. No model-forward interception.

**2. No fused dequant kernel.** GEAR's reference implementation streams and
fuses dequant into attention with a custom CUDA kernel. We do **not** port it —
we reconstruct fp16 then call MLX SDPA. Consequence: the *stored* cache is
base-codes + `L,R` factors + sparse triples, but the working set *during*
attention is the reconstructed fp16 K/V. Stored size shrinks; attend-time peak
memory does not. Documented as a known simplification.

**3. Base quantizer is borrowed, not reinvented.** The base `Quant_b` is the
asymmetric min/max group quant already shared across the repo
(`_quant_utils._group_quant_dequant`, exposed with codes by
`cachegen.quantize_to_codes`). GEAR is therefore a **compositor over an existing
quantizer plus an error-feedback layer** — consistent with the repo's design
philosophy (SVDq/PALU compose the same way over the mixed-bit latent coder).

**What we do NOT implement:** GEAR's streaming-buffer / fused-kernel serving
path, and its optional per-token sparsity scheduling. The sparse term is a fixed
top-`gear_sparse_fraction` of the post-low-rank residual by magnitude.

### Why this is the right pick

1. **Fills the one genuinely missing axis: error-feedback.** SVDq/PALU take the
   SVD of the *signal*; GEAR takes the SVD of the quantization *error* and adds
   it back. Mathematically distinct, and orthogonal — it can sit on top of any
   base bit-width to recover quality the bit-width alone would lose.
2. **Cache-only access — no model surgery.** The residual is computed from the
   K/V the cache already holds at prefill. No hidden-state hooks, no
   attention-score coupling, no RoPE interception. A first-class
   `update_and_fetch` citizen, single-layer (no coordinator needed, unlike
   XQuant/MiniCache).
3. **Composes with existing quantizers and the SVD machinery.** The base quant
   is borrowed from CacheGen; the truncated-SVD helper is shared with SVDq/PALU.
   The error-feedback layer inherits their tests and correctness guarantees.
4. **Honest leverage statement.** Like CacheGen, the win is partly a *storage*
   story (codes + factors dequant to fp16 for SDPA), but unlike CacheGen it also
   improves *quality at fixed bits* — the error-recovery ratio is measured and
   reported, not asserted.
5. **Honest uncertainty.** Labeled "GEAR-adapted (VeloxQuant-MLX
   implementation)"; numbers come from committed `results.json`, not paper
   claims.

### Why the alternatives were not chosen

- **ZipCache** — its per-token saliency signal is the normalized attention
  score, not visible to a cache wrapper. The repo has already proxied attention
  importance via key-norm twice (AdaKV-proxy, KIVI-Sink); a third method on the
  same weak proxy adds little. Deferred (re-confirmed from V5).
- **KVLinC / ThinKV** — hard-rejected in V3/V4 (training required; attention
  scores required). Re-confirmed, nothing changed.

### Planned artifacts (Phases 2–6)

- `veloxquant_mlx/quantizers/gear.py` — `quantize_base`, `residual`,
  `lowrank_error`, `sparse_outliers`, `gear_compress`, `gear_reconstruct`,
  `gear_bytes`, `gear_quant_dequant` (+ `GEARState`). Base quant borrowed from
  `cachegen`; residual SVD via the shared `_truncated_svd` in `_quant_utils`.
- `veloxquant_mlx/cache/gear_cache.py` — `GEARKVCache` (single-layer, reconstruct
  -to-fp16 + byte accounting; no `.bits` attribute; no coordinator).
- Config: `KVCacheConfig(method="gear", gear_bits, gear_rank,
  gear_energy_threshold, gear_sparse_fraction, gear_group_size,
  gear_quantize_values)`. The single-layer path means `for_model` propagates the
  fields automatically via `dataclasses.replace` — no builder branch needed.
- Tests:
  - GEAR reconstruction MSE **strictly below** base-quant-alone on low-rank +
    outlier synthetic data (the core claim).
  - Pure-low-rank (`sparse_fraction=0`) and pure-sparse (`rank=0`) degrade
    gracefully; residual SVD recovers a known rank-r error to < eps; sparse
    selection picks the true top-frac outliers.
  - Byte-accounting ordering `base_only <= compressed <= fp16`; determinism;
    decode accumulation; build via both `create` and `for_model`.
- `benchmark_scripts/benchmark_gear.py` — offline reconstruction-MSE +
  throughput vs base-quant/KIVI/fp16. **No model loading**; writes `results.json`;
  marked "Not yet run" until executed on hardware.
- CHANGELOG (root + docs-site), EVIDENCE_TABLE rows, docs page + sidebar +
  overview, landing card.

---

## Sources (verified)

- GEAR — https://arxiv.org/abs/2403.05527 (Kang, Zhang, Kundu, Jeong, Liu,
  Krishna, Zhao); code https://github.com/opengear-project/GEAR
- ZipCache — https://arxiv.org/abs/2405.14256 (NeurIPS 2024, deferred —
  attention-score saliency, re-confirmed from V5)
- KVLinC — https://arxiv.org/abs/2510.05373 (re-confirmed reject from V3/V4)
- ThinKV — https://arxiv.org/abs/2510.01290 (ICLR 2026 Oral; re-confirmed reject
  from V3/V4)
