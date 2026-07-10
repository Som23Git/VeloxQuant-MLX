# Phase 1 — New-Method Survey V15 (post-Q-Filters)

Follow-up to `NEW_METHOD_SURVEY_V14.md` (Q-Filters-adapted query-agnostic
projection eviction, shipped in 0.31.0). The repo now spans **35** strategies.
V14 chose Q-Filters and named **Keyformer** as its explicit fallback. This
survey commits to Keyformer — the strongest remaining candidate with a
*verified venue* — and confronts, rather than sidesteps, the fact that its
"mechanism gap" over H2O is a single regularizer.

**Evidence discipline:** Keyformer is verified against its arXiv abstract
(arXiv:2403.09054) and is a **peer-reviewed MLSys 2024** paper (unlike the
recent run of preprints). The official code repository
(https://github.com/d-matrix-ai/keyformer-llm) should be re-verified live
before it is cited in README/docs.

---

## Candidate table (carried from V14's non-chosen rows)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **Keyformer** | arXiv:2403.09054 (**MLSys 2024**) | **Gumbel-noise regularizer** on the accumulated eviction score — perturbs the keep/evict decision so a "late riser" (low early attention, high later) is not greedily pruned before it recovers. Isolable: `tau=0` == H2O exactly | ✅ structurally the H2O pair; the noise is the only new ingredient | Low | **CHOSEN** |
| MorphKV | arXiv:2503.00979 (preprint; ICML'25 submission unconfirmed) | Constant-size correlation-aware retention | ✅ | Medium | Defer: venue unverified; mechanism (correlation-of-recent-tokens) is a real gap and the best *next* pick if a verified venue is not required |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance retention | ✅ | Low | Reject: overlaps H2O-adapted; venue covered |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ duplicates ZipCache-adapted's mixed-precision axis | Low | Reject: duplicate axis |
| NestedKV | arXiv:2605.26678 (preprint) | Multi-time-scale cosine-anomaly importance | ✅ mechanically | High | Defer a fifth time — fresh multi-part preprint, pieces individually unvalidated |

---

## Chosen: Keyformer (adapted)

### What the paper actually does

"Keyformer: KV Cache Reduction through Key Tokens Selection for Efficient
Generative Inference" (Adnan et al., MLSys 2024,
[arXiv:2403.09054](https://arxiv.org/abs/2403.09054)) observes that naively
evicting by an accumulated attention score is **unstable**: a token that scores
low early — before the tokens that will attend to it have arrived — is evicted
and can never recover, even if it would have become a heavy hitter. Keyformer
regularizes the eviction decision with **Gumbel noise** on the score logits, a
temperature-controlled perturbation annealed toward 0 over generation, so that
borderline tokens are not deterministically pruned on a single low reading.

### Why this is the right pick now

1. **Verified venue.** After a run of preprints (Q-Filters, SKVQ is COLM but
   before that several preprints), Keyformer is a peer-reviewed MLSys 2024
   paper — it strengthens the Sources/EVIDENCE_TABLE venue coverage.
2. **A small but genuine and isolable mechanism gap.** The repo has eight
   proxy-attention scorers, but none injects stochastic regularization into the
   retention decision. Keyformer's whole contribution is that one term, and it
   is cleanly ablatable: `keyformer_tau=0` collapses onto H2O-adapted
   bit-for-bit. That makes the "does the mechanism actually help?" question
   directly testable — the honest ideal.
3. **Lowest risk.** Structurally the H2O pair; the only new function is the
   Gumbel draw. Reuses the KNorm/Q-Filters single-layer, no-coordinator, fp16
   scaffolding built twice already.

### The honesty crux (this is the whole adaptation)

**The paper redraws Gumbel noise and anneals a temperature across the full
generation. A cache-side library processes blocks with no global step counter
it can trust.** So the adaptation:

- Draw **one deterministic Gumbel(0,1) value per token position**, seeded from a
  fixed base seed + the head's running position, and **freeze** it. `tau`
  scales that frozen noise; it is added to the eviction score only for the
  keep/evict decision (the stored cumulative mass stays clean).
- This preserves the mechanism's *intent* — a borderline token is not doomed by
  one low reading — while staying reproducible and order-diagnosable. It is
  **not** the paper's annealing schedule and must never be claimed to be.
- The base score is H2O-adapted's **key-as-query proxy** attention mass (a cache
  never sees the true query), the same substitution H2O/SnapKV-adapted already
  make.

**Consequence for expectations:** the clean, defensible observable is the
**late-riser survival rate** — greedy `tau=0` evicts a planted late-riser 100%
of the time; the Gumbel term rescues it a large fraction of the time. The
downstream attention-output **perturbation** is a noisier, regime-dependent
secondary effect that does *not* uniformly improve, and is reported as-is
rather than cherry-picked.

### Path-dependence (honest contrast)

Like H2O and unlike L2Norm/KNorm, the kept set is path-dependent (accumulating
score). We do not claim or test prefill/decode bit-for-bit equivalence — only
that budget is respected on both paths and that `tau=0` reproduces H2O exactly.

### What we do NOT implement

- The paper's **annealed, redrawn** Gumbel schedule — replaced by the frozen
  per-position draw above (the crux).
- The model's real attention logits — replaced by the key-as-query proxy.
- No RoPE position-ID remapping after eviction.
- Uniform budget / n_sink / tau across heads.
- No model-level perplexity/throughput benchmark (offline-synthetic only —
  survival-rate, output-perturbation, byte accounting).

### Delivered artifacts (Phases 1–9)

See `paper/IMPLEMENTATION_PROMPT_KEYFORMER.md`:
`veloxquant_mlx/quantizers/keyformer.py` (KeyformerState,
init_keyformer_state, keyformer_update / get_kv / bytes helpers, `_gumbel_at`),
`veloxquant_mlx/cache/keyformer_cache.py` (single-layer wrapper modeled on
`h2o_cache.py`), `KVCacheConfig(method="keyformer", keyformer_budget,
keyformer_n_sink, keyformer_recent, keyformer_tau, keyformer_seed)`, 29 tests
(incl. the `tau=0`==H2O collapse, seed-invariance, Gumbel determinism, and the
late-riser survival mechanism — but NOT a prefill/decode equivalence test),
`benchmark_scripts/benchmark_keyformer.py` + committed results JSON
(tau ∈ {0,2,6} × H2O cross-check × random, late_riser + stable geometries,
survival-rate field), docs page, CHANGELOG 0.32.0, README 34→35,
EVIDENCE_TABLE rows 139–148, landing page (35 algorithms, MLSys 2024
provenance), version bump 0.31.0 → 0.32.0.

---

## Sources (verified this survey)

- Keyformer — https://arxiv.org/abs/2403.09054 (MLSys 2024). Official code
  https://github.com/d-matrix-ai/keyformer-llm — re-verify live before citing.
- MorphKV — https://arxiv.org/abs/2503.00979 (preprint) — best next pick
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried V11–V15)
