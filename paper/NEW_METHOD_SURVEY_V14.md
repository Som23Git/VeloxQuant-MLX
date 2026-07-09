# Phase 1 — New-Method Survey V14 (post-SKVQ)

Follow-up to `NEW_METHOD_SURVEY_V13.md` (SKVQ-adapted sliding-window channel
reordering + clipped dynamic quantization, shipped in 0.30.0; metadata patch
0.30.1). The repo now spans **33** strategies. V13 evaluated six candidates,
chose SKVQ, and left five deferred/rejected. This survey does **not**
re-open the field — it commits to the one remaining V13 candidate with a
genuine *mechanism* gap (Q-Filters) and confronts, rather than sidesteps,
the observability blocker that made V13 defer it.

**Evidence discipline:** Q-Filters is verified against its arXiv abstract
(arXiv:2503.02812). It is a **preprint** — no venue is claimed. The official
code repository must be re-verified before it is cited in README/docs (do
not assert a URL from memory; confirm it live during implementation).

---

## Candidate table (carried from V13's non-chosen rows)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **Q-Filters** | arXiv:2503.02812 (**preprint**) | **Query-agnostic projection scoring**: a single per-head direction onto which a key's projection predicts the attention it will receive — a *fourth* scorer class the repo lacks (not attention/proxy, not structural, not intrinsic-norm) | ⚠️ the paper estimates the direction from **query** vectors offline; the cache never sees queries — accepted here as an **explicit, documented deviation** (key-SVD substitute), not pretended away | Medium | **CHOSEN** |
| Keyformer | arXiv:2403.09054 (MLSys 2024) | Gumbel-noise-regularized accumulating eviction score | ✅ (proxy-attention) | Low | Reject-for-now: would be the repo's 8th proxy-attention scorer; real venue but near-zero mechanism gap. Strong *fallback* if the Q-Filters key-SVD substitute proves too weak (see below) |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance retention | ✅ | Low | Reject: overlaps H2O-adapted; venue covered |
| MorphKV | arXiv:2503.00979 (preprint; ICML'25 submission unconfirmed) | Constant-size correlation-aware retention | ✅ | Medium | Defer: venue unverified; proxy-attention family again |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ duplicates ZipCache-adapted's mixed-precision axis | Low | Reject: duplicate axis |
| NestedKV | arXiv:2605.26678 (preprint) | Multi-time-scale cosine-anomaly importance | ✅ mechanically | High | Defer a fourth time — fresh four-part preprint, pieces individually unvalidated |

---

## Chosen: Q-Filters (adapted)

### What the paper actually does

"Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression"
([arXiv:2503.02812](https://arxiv.org/abs/2503.02812), preprint) observes
that for a trained attention head the (Query, Key) joint distribution is
**anisotropic**: there exists a single per-head direction — estimated once,
**offline, from the SVD of a sample of query vectors** — such that a key's
projection onto that direction predicts the average attention that key will
receive. Ranking cached keys by this projection therefore approximates
attention-based importance **without computing any attention** and
**without a query at eviction time** ("query-agnostic"). The direction is
computed offline per head and reused for all inputs.

### Why this is the right pick now

1. **A scorer class the repo genuinely lacks.** Every eviction method shipped
   scores tokens by attention/proxy (SnapKV, H2O, TOVA, PyramidKV,
   SqueezeAttention, ChunkKV, CaM), by structure (StreamingLLM, sink,
   sliding-window), or by intrinsic key norm (L2Norm). Q-Filters adds a
   **fourth**: a learned/estimated **projection direction**. Of all V13
   leftovers it is the only one that is not "yet another proxy-attention
   scorer" (Keyformer, Scissorhands, MorphKV) or a duplicate axis (MiKV).
2. **Infrastructure is already proven.** It is a single-layer, no-coordinator,
   fp16, score-and-evict cache — structurally the **KNorm pair**
   (`quantizers/knorm.py`, `cache/knorm_cache.py`) with the scalar norm
   replaced by a projection onto a frozen direction. Risk is concentrated in
   one new function (`estimate_filter_dir`).

### The honesty crux (this is the whole adaptation)

**The paper's filter is derived from query vectors, offline. A cache-side
library never sees query vectors — only the K/V passed to
`update_and_fetch`.** The paper's exact estimator is therefore impossible
here. The adaptation:

- Estimate the per-head direction from the **top right-singular vector of the
  observed key block** (equivalently the top eigenvector of the key
  covariance), computed from the **first `calib_tokens` observed** and then
  frozen. This is a *different estimator of the same head-geometry direction*
  — key-SVD instead of query-SVD.
- This is a **genuine deviation**, not a shortcut, and every surface (module
  docstring, docs page, CHANGELOG "Honest scope", EVIDENCE_TABLE) must state
  it in exactly those terms: *"the paper derives the filter from
  query-distribution SVD offline; we derive it from the first observed key
  chunk's SVD — validated here only under constructed geometry, never claimed
  equivalent to the paper's."*
- The benchmark reports a **filter-cosine** field (how well the key-SVD
  direction recovers a planted axis) as the honest measure of whether the
  substitute stands in acceptably, plus the mandatory **isotropic control**
  where the method shows no advantage.

**Consequence for expectations:** if `filter-cosine` comes back low under
paper-like geometry, the substitute does *not* stand in for the paper's
query-derived filter, and the method should ship with that stated plainly —
or Keyformer (the fallback above) taken instead. This is surfaced, not hidden.

### Path-dependence (honest contrast with KNorm)

KNorm's intrinsic score gives a path-independent kept set (prefill == decode,
bit-for-bit). Q-Filters does **not**: the filter is estimated from whichever
chunk crosses `calib_tokens` first, so prefill-in-one-block and
token-by-token decode can freeze *different* directions and diverge. We do
**not** claim or test bit-for-bit prefill/decode equivalence — only the
weaker true property (given the same frozen filter, scoring/eviction is
order-invariant). Documented as the explicit contrast with KNorm.

### What we do NOT implement

- Query-derived filter estimation (the paper's actual mechanism) — replaced
  by the key-SVD substitute above.
- Any offline calibration corpus; per-head filters are estimated online from
  observed traffic and frozen.
- No RoPE position-ID remapping after eviction (same as every eviction method
  in this repo).
- Uniform budget / n_sink across heads.
- No model-level perplexity/throughput benchmark (offline-synthetic only, as
  for every method — the win measured is memory via byte accounting).

### Planned artifacts (Phases 1–9)

See `paper/IMPLEMENTATION_PROMPT_QFILTERS.md`:
`veloxquant_mlx/quantizers/qfilters.py` (QFiltersState, estimate_filter_dir,
qfilters_update / get_kv / bytes helpers), `veloxquant_mlx/cache/
qfilters_cache.py` (single-layer wrapper modeled on `knorm_cache.py`),
`KVCacheConfig(method="qfilters", qfilters_budget, qfilters_n_sink,
qfilters_recent, qfilters_calib_tokens, qfilters_sign)`, tests (~28 incl. the
planted-direction recovery, given-same-filter order invariance, and the
paper-like-geometry mechanism test — but NOT a prefill/decode equivalence
test), `benchmark_scripts/benchmark_qfilters.py` + committed results JSON
(sign ±1 × KNorm/H2O/random references, two channel regimes, filter-cosine
field), docs page, CHANGELOG 0.31.0, README 33→34, EVIDENCE_TABLE rows,
landing page (34 algorithms, preprint provenance — no fabricated venue),
version bump 0.30.1 → 0.31.0.

---

## Sources (verified this survey)

- Q-Filters — https://arxiv.org/abs/2503.02812 (preprint). Official code
  URL to be re-verified live before citing anywhere.
- Keyformer — https://arxiv.org/abs/2403.09054 (MLSys 2024)
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- MorphKV — https://arxiv.org/abs/2503.00979 (preprint)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried V11–V14)
