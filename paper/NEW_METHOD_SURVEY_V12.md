# Phase 1 — New-Method Survey V12 (post-NSNQuant)

Follow-up to `NEW_METHOD_SURVEY_V11.md` (NSNQuant-adapted calibration-free
universal-codebook VQ, shipped in 0.28.0). The repo now spans 31 strategies;
see V11's opening paragraph for the full taxonomy. This survey is short by
design: V11 already identified and verified the next two candidates and
explicitly deferred them — the work here is choosing between them, not
re-surveying the field.

**Evidence discipline:** both candidates below were already verified in V11
(fetched abstract + author list against the claim). No new IDs introduced.

---

## Candidate table (carried forward from V11's "defer" verdicts)

| Method | Paper (verified in V11) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **L2-norm eviction** ("KnormPress") | arXiv:2406.11430 (**EMNLP 2024**; Devoto, Zhao, Scardapane, Minervini; code https://github.com/alessiodevoto/l2compress) | **Intrinsic-signal eviction**: token importance read directly off the key vector's L2 norm — *low* norm ⇒ high future attention. The repo's first eviction scorer that needs **no attention scores and no proxy attention** | ✅ the paper's actual signal is fully observable at the cache level — a *stronger* fit than SnapKV/H2O, whose true signal (real query vectors) must be proxied | Low | **CHOSEN** |
| **NestedKV** | arXiv:2605.26678 (May 2026, preprint; Chen, Liu, Gao, Fan, Wang, Chu, Lin, Hu) | Multi-time-scale cosine-anomaly importance from the key stream; hierarchical anchors; surprise-gated routing; per-head budgets | ✅ mechanically (key-only, training-free) | High | defer again (still a ~2-month-old preprint with a four-part mechanism whose pieces are individually unvalidated; L2-norm covers the same "importance without attention" gap at a fraction of the risk, with a peer-reviewed venue) |

---

## Chosen: L2-norm eviction (Devoto et al., EMNLP 2024)

### What the paper actually does

"A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
([arXiv:2406.11430](https://arxiv.org/abs/2406.11430), Alessio Devoto, Yu
Zhao, Simone Scardapane, Pasquale Minervini — **EMNLP 2024**) reports a
consistent empirical correlation in trained decoder LMs: **a low L2 norm of a
key embedding usually leads to a high attention score during decoding**. The
influence of a KV pair is therefore largely determined by the key embedding
itself, *before it is ever queried*. Their compression strategy follows
directly: rank cached tokens by key L2 norm and keep the lowest-norm ones,
evicting the highest-norm ones — no attention scores, no proxies, no
training, no calibration.

### Why this is the right pick

1. **Fills the repo's last eviction gap — the signal axis.** Every eviction
   method shipped so far scores tokens with either attention (or a
   key-as-query proxy: SnapKV, H2O, TOVA, PyramidKV, SqueezeAttention,
   ChunkKV's `attn_mass` mode, CaM) or pure structure (StreamingLLM, sink,
   sliding-window). L2-norm is a third scorer class: **intrinsic** — read off
   the stored key itself. (ChunkKV's `key_norm` scoring option and
   ZipCache's saliency proxy both treat *high* norm as important; the
   paper's finding is the **inversion** — low norm attracts attention — so
   this is not a duplicate, and the sign difference is exactly the
   counterintuitive empirical content of the paper.)
2. **The honest-adaptation story is unusually clean.** SnapKV/H2O needed a
   key-as-query proxy because their true signal isn't cache-observable.
   L2-norm's true signal *is* the cache content. Fewer disclaimers than any
   eviction method shipped to date.
3. **Intrinsic scores buy real implementation wins**: the score never
   updates, so (a) eviction vectorizes as one top-k per incoming block — no
   per-token softmax-over-cache loop like H2O; (b) the kept set is
   **path-independent** (prefill vs token-by-token decode provably yield the
   same kept set at the same budget — the "keep k smallest with a max-heap"
   invariant), which becomes a bit-for-bit test no accumulating-score method
   can offer.
4. **Peer-reviewed at a venue new to the project.** EMNLP 2024 adds a fifth
   venue family to the landing page's provenance strip.
5. **Low effort, low risk** — mirrors `h2o.py`/`h2o_cache.py` structurally
   with a simpler scorer. V11 deferred it only because it was "too small to
   headline" while NSNQuant was on the table; it is the natural next release.

### The honest adaptation problem

**1. The correlation is an empirical property of trained models — not
reproducible on isotropic synthetic data.** On random Gaussian keys there is
no low-norm ⇒ high-attention geometry, so an offline-synthetic benchmark
cannot validate the paper's core claim. We handle this the same way CaM's
benchmark handled attention-prominence: construct synthetic geometry that
*exhibits* the paper's reported correlation (low-norm keys aligned with the
query distribution, high-norm keys anti-aligned) and verify the machinery
preserves attention output under that geometry — keep-low ≫ keep-high and
random eviction — while **also reporting the isotropic case where keep-low ≈
random** (no fabricated advantage). The claim "low norm predicts attention in
trained LMs" remains attributed to the paper, never to our benchmark.

**2. Sink positions.** The paper observes the first tokens (attention sinks)
matter despite their norms; standard practice across this repo's eviction
family is an `n_sink` protection window. We keep `knorm_n_sink` (default 4,
same as H2O/CaM/ChunkKV).

**3. Just-generated tokens.** Because the norm is intrinsic, a brand-new
high-norm token can be evicted the moment it arrives. That is the paper's
position taken seriously (importance is predictable before querying), so the
default follows it; an optional `knorm_recent` window (default 0 = off,
faithful) can protect the most recent tokens for callers who want
StreamingLLM-style recency insurance. Documented as an extension, not paper
behavior. (Note: `knorm_recent > 0` breaks the path-independence property;
the equivalence test pins `knorm_recent=0`.)

**4. Keep direction as an exposed knob.** `knorm_keep="low"` (paper default)
vs `"high"` — the inverted scorer is exactly the ablation arm the benchmark
needs, so it ships as a config value rather than benchmark-only code.

### What we do NOT implement

- Per-layer/per-task compression-rate tuning from the paper's evaluation
  sweeps — one uniform `knorm_budget`, overridable per layer via the
  existing `KVCacheBuilder.for_model` config mechanics.
- No RoPE position-ID remapping after eviction (same as every eviction
  method in this repo).
- Uniform budget across heads (same as H2O/TOVA/CaM).

### Planned artifacts (Phases 2–6)

See `paper/IMPLEMENTATION_PROMPT_KNORM.md`:
`veloxquant_mlx/quantizers/knorm.py` (KnormState, vectorized
`knorm_update`, get/bytes helpers), `veloxquant_mlx/cache/knorm_cache.py`
(single-layer wrapper modeled on `h2o_cache.py`),
`KVCacheConfig(method="knorm", knorm_budget, knorm_n_sink, knorm_recent,
knorm_keep)`, tests (~26 incl. the path-independence bit-for-bit check and
the geometry mechanism test), `benchmark_scripts/benchmark_knorm.py` +
committed results JSON (keep-low vs keep-high vs random under both the
paper-like geometry and the isotropic control), docs page, CHANGELOG 0.29.0,
README 31→32, EVIDENCE_TABLE rows, landing page (32 algorithms, EMNLP 2024
provenance item), version bump 0.28.0 → 0.29.0.

---

## Sources (verified in V11)

- L2-norm KV compression — https://arxiv.org/abs/2406.11430 (EMNLP 2024;
  Devoto, Zhao, Scardapane, Minervini; code
  https://github.com/alessiodevoto/l2compress)
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint, May 2026)
