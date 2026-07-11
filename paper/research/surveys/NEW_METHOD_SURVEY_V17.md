# Phase 1 — New-Method Survey V17 (post-MorphKV)

Follow-up to `NEW_METHOD_SURVEY_V16.md` (MorphKV-adapted recent-window
correlation retention, shipped in 0.33.0). The repo now spans **37** strategies.
V16 named **NestedKV** as the "best next pick" **but explicitly gated it on a
verified venue** — the same discipline that correctly deferred MorphKV across
V13–V15. NestedKV (arXiv:2605.26678) is still a bare preprint. This survey
therefore picks **KVzip**, which both satisfies the standing verified-venue rule
**and** adds a mechanism axis orthogonal to every eviction scorer in the repo.

**Evidence discipline / venue verified live this survey:**

> "KVzip: Query-Agnostic KV Cache Compression with Context Reconstruction,"
> Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song
> — **NeurIPS 2025 (Oral)**, arXiv:2505.23416, official code
> `github.com/snu-mllab/KVzip`. Accepted, not merely submitted.

Re-check the arXiv abstract + the official repo live before writing it into
README/docs/EVIDENCE_TABLE.

---

## Candidate table (carried from V16, re-scored)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **KVzip** | arXiv:2505.23416 (**NeurIPS 2025 Oral**, confirmed) | **Context-reconstruction reliance** — score a KV pair by how much the model relies on it to *reconstruct its own context*, query-agnostic (a one-time importance profile reused across all future queries). Isolable: `probe="latest"` == latest-token (TOVA-adapted) exactly | ✅ structurally the H2O/TOVA/MorphKV family; the reconstruction-reliance signal is the new axis | Medium | **CHOSEN** |
| NestedKV | arXiv:2605.26678 (preprint) | Multi-time-scale (stable/episodic/current) cosine-anomaly importance + training-free outer router | ✅ mechanically | High | **Best next pick — once a venue exists.** Still a bare preprint; deferred, exactly as MorphKV was |
| Anchor Direction Projection | NeurIPS 2025 | Eviction by projecting keys onto an anchor direction | ⚠️ overlaps Q-Filters / L2Norm projection axis | Low | Reject: axis already covered |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance retention | ⚠️ overlaps H2O-adapted | Low | Reject: duplicate axis |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ duplicates ZipCache-adapted | Low | Reject: duplicate axis |

**Next fallback named explicitly:** **NestedKV** (multi-time-scale importance)
remains the best remaining mechanism gap and becomes the pick the moment it earns
a verified venue. Until then we hold the line, as we did — correctly — for
MorphKV across three surveys.

---

## Chosen: KVzip (adapted)

### What the paper actually does

KVzip quantifies the importance of a cached KV pair by using the **underlying
LLM to reconstruct the original context from the cached KV pairs**, then evicts
the pairs with the lowest reconstruction importance. Crucially the profile is
**query-agnostic**: it is computed once (a few forward passes) against a
reconstruction objective, not against any specific downstream query, so the same
compressed cache serves diverse future queries. The paper reports 3–4× cache-size
reduction and ~2× FlashAttention decode-latency reduction with negligible loss on
LLaMA3.1 / Qwen2.5 / Gemma3 up to 170K-token contexts — those are the paper's
numbers, on trained models, not reproduced here.

### Why this is the right pick now

1. **Verified venue — the standing rule is honored.** NeurIPS 2025 Oral.
   NestedKV, V16's named pick, is still an unverified preprint; picking it would
   break exactly the discipline that (correctly) deferred MorphKV. KVzip lets us
   ship a peer-reviewed method **and** keep the rule.
2. **A genuinely new, isolable axis.** The repo now has nine proxy-attention
   eviction scorers (SnapKV / H2O / TOVA / PyramidKV / SqueezeAttention /
   ChunkKV / CaM / Keyformer / MorphKV). **Every one ranks a token by the
   attention it receives** — cumulative (H2O), latest-query (TOVA/SnapKV), or
   recent-window (MorphKV). **None** ranks by *context-reconstruction reliance*.
   That reconstruction axis is KVzip's contribution and is cleanly ablatable:
   `kvzip_probe="latest"` collapses onto the latest-token (TOVA-adapted) ranking
   bit-for-bit — pinned by a test, the analogue of MorphKV's `window=1`==TOVA and
   Keyformer's `tau=0`==H2O.
3. **Moderate risk, maximal reuse.** Structurally the H2O/TOVA/MorphKV family;
   reuses the single-layer, no-coordinator, fp16, lazy-per-head scaffolding built
   for MorphKV/Keyformer. The only new function is the reconstruction-importance
   scorer.

### The honesty crux (this is the whole adaptation)

**The paper runs the real model to reconstruct text and profiles the attention
each cached key receives under that reconstruction. A cache-side library never
runs the model and never sees the true query.** So the adaptation:

- **Key-as-reconstruction-probe proxy.** The cached KEYS themselves stand in for
  the reconstruction queries — importance ≈ the (max) proxy-attention each stored
  key receives from the reconstruction-probe rows. The same key-as-query
  substitution H2O / TOVA / MorphKV-adapted already make, applied to a
  reconstruction probe instead of a live query. Documented, not the paper's math.
- **Query-agnostic, recomputed — not accumulated.** No cumulative score array is
  stored. Each step, reconstruction importance is recomputed from the live keep
  set against the probe. Query-agnostic in the paper's sense (the probe is not a
  downstream query), and constant, not a growing accumulator.
- **Only the `probe="latest"` reduction is pinned exactly.** With `probe="latest"`
  the reconstruction probe is the single most-recent key, so importance is just
  that key's attention over the keep set — the latest-token (TOVA-adapted)
  ranking. We assert this bit-for-bit. We do **not** claim any H2O collapse.

**Consequence for expectations:** the clean, defensible observable is the
**reconstruction-critical retention rate** under a constructed geometry where the
reconstruction-important region differs from the highest-cumulative-attention
region — a cumulative (H2O-style) keep set retains the wrong (stale) tokens while
KVzip-adapted retains the reconstruction-critical ones. The benchmark
deliberately makes the reconstruction signal weak and per-token noisy so that the
full-context probe materially beats the single-latest-token (`probe="latest"`)
reference — otherwise the probe axis would be vacuous. The downstream
attention-output **perturbation** is a noisier, regime-dependent secondary effect
reported as-is, with a null "flat" control (no reconstruction shift) where KVzip
shows no advantage.

### Path-dependence (honest contrast)

Like H2O/TOVA/MorphKV and unlike L2Norm/KNorm, the kept set is path-dependent (it
depends on the reconstruction probe at each step). We do not claim or test
prefill/decode bit-for-bit equivalence — only that budget is respected on both
paths and that `probe="latest"` reproduces the latest-token ranking exactly.

### What we do NOT implement

- The paper's real context-reconstruction forward passes — replaced by the
  key-as-reconstruction-probe proxy.
- Head-level context-independent scoring / DuoAttention-style head compression.
- No RoPE position-ID remapping after eviction.
- Uniform budget / n_sink / probe across heads.
- No model-level perplexity/throughput/accuracy benchmark; the paper's headline
  numbers (3–4× reduction, ~2× decode, negligible loss up to 170K on
  LLaMA3.1/Qwen2.5/Gemma3) are the paper's, on trained models (offline-synthetic
  only — reconstruction-critical retention rate, output-perturbation, byte
  accounting).

### Delivered artifacts (Phases 1–13)

See `paper/IMPLEMENTATION_PROMPT_KVZIP.md`:
`veloxquant_mlx/quantizers/kvzip.py` (KVzipState, init_kvzip_state,
kvzip_update / get_kv / bytes helpers, `_reconstruction_importance`),
`veloxquant_mlx/cache/kvzip_cache.py` (single-layer wrapper modeled on
`morphkv_cache.py`), `KVCacheConfig(method="kvzip", kvzip_budget, kvzip_n_sink,
kvzip_probe)`, ~32 tests (incl. the `probe="latest"`==TOVA-adapted collapse,
determinism, budget invariants, and the reconstruction-geometry retention
mechanism — but NOT a prefill/decode equivalence test),
`benchmark_scripts/benchmark_kvzip.py` + committed results JSON
(probe ∈ {latest, context} × H2O cumulative cross-check × random,
reconstruction_shift + flat geometries, reconstruction-critical retention field),
docs page, CHANGELOG 0.34.0, README 36→37, EVIDENCE_TABLE rows continuing after
MorphKV, landing page (37 algorithms, NeurIPS 2025 provenance), version bump
0.33.0 → 0.34.0. Plus the funding-link fix (Buy Me a Coffee → GitHub Sponsors)
and the JOSS paper refresh.

---

## Sources (verified this survey)

- KVzip — https://arxiv.org/abs/2505.23416 (**NeurIPS 2025 Oral**, confirmed
  accepted); official code https://github.com/snu-mllab/KVzip. Re-verify the
  abstract + repo live before citing.
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried V11–V17) —
  best next pick once a verified venue exists.
- Anchor Direction Projection — NeurIPS 2025 (axis overlaps Q-Filters/L2Norm).
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
