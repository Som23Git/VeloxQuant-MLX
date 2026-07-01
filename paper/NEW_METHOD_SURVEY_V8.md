# Phase 1 — New-Method Survey V8 (post-ZipCache)

Follow-up to `NEW_METHOD_SURVEY_V7.md` (which led to ZipCache, shipped in 0.18.0).
After ZipCache, the repo spans: scalar/group quant (KIVI, KVQuant-NUQ, TurboQuant),
vector quant (RVQ, VecInfer, CommVQ, RaBitQ, QJL), low-rank (SVDq, PALU),
cross-layer (XQuant, MiniCache), entropy coding (CacheGen), error-feedback (GEAR),
per-token saliency routing (ZipCache-adapted), and attention-proxy adaptive schemes
(KIVI-Sink, AdaKV-proxy, Kitty). The one uncovered axis: **token eviction** — dropping
tokens entirely rather than compressing them.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real paper.
No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **SnapKV** | arXiv:2404.14469, ICLR 2025 (Yuan et al.) | **Token eviction**: prefill observation-window attention scores each token; only the top-`budget` tokens are kept. First method on the eviction axis. First where the actual attention signal (not a key-norm proxy) is computable at cache level — the obs-window K self-attention uses only the K matrix visible at `update_and_fetch`. | ✅ prefill K visible; obs-window self-attention computable from K alone | Low-Med | **CHOSEN** |
| **H2O** | arXiv:2306.14048 | Heavy hitter + recency eviction — requires per-decode-step accumulated attention scores | ❌ decode attention not observable by cache wrapper | Med | hard reject (needs decode attn) |
| **ThinKV** | arXiv:2510.01290, ICLR 2026 Oral | Thought-adaptive hybrid quant+eviction for reasoning | ❌ CoT-specific; needs attention sparsity patterns at decode | High | hard reject (re-confirmed V3–V7) |

---

## Chosen: SnapKV-adapted (prefill observation-window token eviction)

### What the paper actually does

SnapKV (arXiv:2404.14469, ICLR 2025, Yuan et al.) identifies which KV pairs are most
important during prefill by computing an "observation window" attention pass — the last
`obs_window` query tokens attend to the full prefix — and pools the resulting attention
scores to rank all prefix tokens. Only the top-`budget` tokens (by pooled attention
score) plus a fixed number of initial "sink" tokens are retained in the KV cache.
Decode-step tokens are always appended. The paper reports 5–20× prefill KV memory
reduction with negligible quality loss on generation tasks.

### The honest adaptation problem

**Adaptation 1 — Key-as-query proxy.** SnapKV uses the last `obs_window` *query*
vectors from the prompt — visible during prefill in the original forward pass but not
in a cache wrapper (only K and V are visible at `update_and_fetch`). We substitute the
last `obs_window` *key* vectors as proxy queries. Key and query spaces are correlated
(both projected from the same residual stream), making this a stronger proxy than
key-norm-only methods (KIVI-Sink, AdaKV-proxy, ZipCache-adapted), but still an
approximation. Documented as "SnapKV-adapted (key-as-query proxy)" throughout.

**Adaptation 2 — No max-pool smoothing.** The paper applies a 1-D max-pool of width
`kernel_size` to the pooled attention vector before ranking, smoothing single-token
spikes. We use mean-pooling only (simpler; no sliding-window kernel needed).

**Adaptation 3 — Eviction, not quantization.** The kept tokens are stored in fp16
(no further quantization). This is a pure eviction method — composable with any
quantizer by applying a quantizer cache to the kept-token subset.

### What we do NOT implement
- Max-pool smoothing (`kernel_size > 1`).
- Dynamic per-head budget adjustment.
- Adaptive observation window sizing.
- Stacking with a quantizer (composable by the user; left to `KVCacheConfig` composition).

### Why this is the right pick
1. **Last uncovered axis.** Every other proposed method re-parameterises an existing
   axis. Token eviction is structurally new — the cache shrinks in token count, not
   in bits per token.
2. **Strongest proxy in the repo.** Key-as-query computes the actual attention
   distribution from K; all four prior key-norm proxy methods (KIVI-Sink, AdaKV-proxy,
   Kitty, ZipCache-adapted) use only L2-norm, a cruder signal.
3. **Zero calibration, no model surgery, single-layer.** Identical constraint profile
   to all successful methods in this repo.
4. **Honest scope.** Key-as-query proxy and no max-pool smoothing are stated plainly
   in all docs. No throughput or perplexity numbers claimed until a hardware benchmark
   JSON is committed.

---

## Sources (verified)

- SnapKV — https://arxiv.org/abs/2404.14469 (ICLR 2025, Yuan et al.)
- H2O — https://arxiv.org/abs/2306.14048 (hard reject — decode attn not observable)
- ThinKV — https://arxiv.org/abs/2510.01290 (hard reject re-confirmed from V3–V7)
