# Phase 1 — New-Method Survey V16 (post-Keyformer)

Follow-up to `NEW_METHOD_SURVEY_V15.md` (Keyformer-adapted Gumbel-regularized
heavy-hitter eviction, shipped in 0.32.0). The repo now spans **36** strategies.
V13, V14, and V15 all named **MorphKV** as the explicit "best next pick" and
deferred it for **one reason only — its venue was unverified**. This survey
commits to MorphKV because that sole deferral reason is now resolved.

**Evidence discipline / the deferral reason is gone:** MorphKV was carried as
"arXiv:2503.00979 (preprint; ICML'25 submission unconfirmed)" through V13–V15.
Re-verified live this survey against the arXiv abstract:

> "Dialogue Without Limits: Constant-Sized KV Caches for Extended Responses in
> LLMs," Ravi Ghadia, Avinash Kumar, Gaurav Jain, Prashant Nair, Poulami Das —
> **Proceedings of the 42nd International Conference on Machine Learning (ICML
> 2025)**, Vancouver. **Accepted, not merely submitted.**

That is a peer-reviewed venue. Re-check the arXiv abstract + any official code
repo live again before writing it into README/docs/EVIDENCE_TABLE.

---

## Candidate table (carried from V15's non-chosen rows)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **MorphKV** | arXiv:2503.00979 (**ICML 2025**, now confirmed) | **Recent-window correlation retention** — keep the tokens whose keys correlate with the attention pattern of a *sliding window* of recent tokens, eliminating the cumulative-scoring "early-token bias." Isolable: `window=1` == latest-token (TOVA-adapted) exactly | ✅ structurally the H2O/TOVA family; the recent-*window* correlation is the new axis | Medium | **CHOSEN** |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance retention | ✅ | Low | Reject: overlaps H2O-adapted; venue already covered |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ duplicates ZipCache-adapted's mixed-precision axis | Low | Reject: duplicate axis |
| NestedKV | arXiv:2605.26678 (preprint) | Multi-time-scale cosine-anomaly importance | ✅ mechanically | High | **Best next pick** — but only once a verified venue exists; still a fresh multi-part preprint with pieces individually unvalidated |

**Next fallback named explicitly:** if a verified venue is not required for the
37th method, **NestedKV** (multi-time-scale importance) is the best remaining
mechanism gap. Otherwise wait for its venue, as we did — correctly — for MorphKV.

---

## Chosen: MorphKV (adapted)

### What the paper actually does

MorphKV maintains a **constant-size** KV cache. Instead of ranking a stored
token by an accumulated attention score (H2O — which suffers "early-token bias,"
where tokens that were heavy hitters early dominate the keep set and crowd out
what the model is *currently* attending to), it ranks stored tokens by their
**correlation with the attention pattern of a sliding window of the most recent
tokens**. Retention therefore tracks what the recent context reads; stale-but-
early tokens are dropped. The paper reports 52.9% memory savings and 18.2%
higher accuracy over prior methods **on trained models** — those are the
paper's numbers, not reproduced here.

### Why this is the right pick now

1. **The sole deferral reason is resolved.** MorphKV was the named best next
   pick in V13, V14, and V15, deferred every time because its venue was
   "submission unconfirmed." It is now confirmed **ICML 2025** — peer-reviewed,
   strengthening the Sources/EVIDENCE_TABLE venue coverage exactly as Keyformer
   (MLSys 2024) did.
2. **A genuine, isolable mechanism gap.** The repo has eight proxy-attention
   scorers, but each ranks a stored token against **either** cumulative history
   (H2O) **or** a single most-recent query (TOVA/SnapKV). **None** ranks by
   correlation with a *window* of recent tokens. That recent-window axis is
   MorphKV's contribution and is cleanly ablatable: `morphkv_window=1` collapses
   onto the latest-token (TOVA-adapted) ranking bit-for-bit — pinned by a test,
   the analogue of Keyformer's `tau=0`==H2O collapse.
3. **Moderate risk, maximal reuse.** Structurally the H2O/TOVA family; reuses
   the single-layer, no-coordinator, fp16, lazy-per-head scaffolding built for
   Keyformer/Q-Filters/KNorm. The only new function is the recent-window
   relevance ranking.

### The honesty crux (this is the whole adaptation)

**The paper uses the model's real attention patterns and a specific refresh
cadence. A cache-side library never sees the true query and processes blocks.**
So the adaptation:

- **Key-as-query proxy.** The incoming KEY stands in for the unseen query when
  estimating the attention each stored key receives — the same substitution
  H2O/TOVA/SnapKV/Keyformer-adapted already make. Documented, not the paper's
  math.
- **Constant-size, recomputed — not accumulated.** We store no cumulative score
  array. Each step, retention is recomputed from the live keep set and a window
  of the last `window` key rows (the trailing recent tokens, themselves
  protected). This preserves the mechanism's intent — track what the recent
  context reads — while staying deterministic and order-diagnosable.
- **Only the `window=1` reduction is pinned exactly.** With `window=1` the
  recent-relevance is just the newest key's attention over the keep set — the
  latest-token (TOVA-adapted) ranking. We assert this bit-for-bit. We do **not**
  claim any H2O collapse: MorphKV recomputes from the live window, it never
  becomes H2O's cumulative-forever rule, so no such equivalence is asserted.

**Consequence for expectations:** the clean, defensible observable is the
**recent-relevant retention rate** under a constructed topic shift — cumulative
H2O scoring retains ~0% of the recent-relevant region (fully captured by stale
early heavy hitters), while MorphKV's recent-window ranking re-targets the cache
toward the region the recent context attends to. The benchmark deliberately
makes the recent signal weak and per-token noisy so that a *wider window*
materially beats the single-latest-token (`window=1`) reference — otherwise the
window axis would be vacuous. The downstream attention-output **perturbation** is
a noisier, regime-dependent secondary effect reported as-is, with a null
"stable" control (no topic shift) where MorphKV shows no advantage.

### Path-dependence (honest contrast)

Like H2O/TOVA and unlike L2Norm/KNorm, the kept set is path-dependent (it
depends on the recent window at each step). We do not claim or test
prefill/decode bit-for-bit equivalence — only that budget is respected on both
paths and that `window=1` reproduces the latest-token ranking exactly.

### What we do NOT implement

- The paper's real attention logits — replaced by the key-as-query proxy.
- The paper's exact refresh cadence / per-head adaptive budgets.
- No RoPE position-ID remapping after eviction.
- Uniform budget / n_sink / window across heads.
- No model-level perplexity/throughput/accuracy benchmark; the paper's headline
  numbers are the paper's, on trained models (offline-synthetic only —
  recent-relevant retention rate, output-perturbation, byte accounting).

### Delivered artifacts (Phases 1–9)

See `paper/IMPLEMENTATION_PROMPT_MORPHKV.md`:
`veloxquant_mlx/quantizers/morphkv.py` (MorphKVState, init_morphkv_state,
morphkv_update / get_kv / bytes helpers, `_recent_relevance`),
`veloxquant_mlx/cache/morphkv_cache.py` (single-layer wrapper modeled on
`keyformer_cache.py`), `KVCacheConfig(method="morphkv", morphkv_budget,
morphkv_n_sink, morphkv_window)`, 32 tests (incl. the `window=1`==TOVA-adapted
collapse, determinism, budget invariants, and the topic-shift retention
mechanism — but NOT a prefill/decode equivalence test),
`benchmark_scripts/benchmark_morphkv.py` + committed results JSON
(window ∈ {1,8,32} × H2O cumulative cross-check × random, topic_shift + stable
geometries, recent-relevant retention field), docs page, CHANGELOG 0.33.0,
README 35→36, EVIDENCE_TABLE rows continuing after Keyformer, landing page
(36 algorithms, ICML 2025 provenance), version bump 0.32.0 → 0.33.0.

---

## Sources (verified this survey)

- MorphKV — https://arxiv.org/abs/2503.00979 (**ICML 2025**, confirmed
  accepted). Re-verify the abstract + official code live before citing.
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried V11–V16) —
  best next pick once a verified venue exists.
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
