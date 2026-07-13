# Phase 1 — New-Method Survey V19 (post-KVTC)

Follow-up to `NEW_METHOD_SURVEY_V18.md` (KVTC-adapted local-PCA + DP-optimal
bit allocation + entropy coding, shipped in 0.35.0). The repo now spans **37**
strategies. V18 carried **NestedKV** forward as the named next pick,
explicitly gated on it acquiring a verified venue — the same discipline that
correctly deferred MorphKV across V13–V15 and NestedKV itself across
V16–V18. NestedKV (arXiv:2605.26678) was re-checked live this survey and is
**still a bare preprint** (single version, submitted 2026-05-26, no
journal-ref, no OpenReview listing under any tracked venue). It stays
deferred. This survey picks **CurDKV (Value-Guided KV Compression via
Approximated CUR Decomposition)**, which satisfies the standing
verified-venue rule and adds a value-aware token-selection mechanism axis the
repo does not have.

**Evidence discipline / venue verified live this survey:**

> "Value-Guided KV Compression for LLMs via Approximated CUR Decomposition"
> (Sengupta, Chaudhary, Chakraborty) — **NeurIPS 2025 (poster, confirmed)**,
> arXiv:2509.15038, official listing at
> `neurips.cc/virtual/2025/poster/116352`. Confirmed accepted via the
> official NeurIPS virtual site, independent of the arXiv page (which
> carries no venue metadata — normal for NeurIPS, which does not require an
> arXiv comments-field update).

Re-check the arXiv abstract + the neurips.cc poster page live before writing
it into README/docs/EVIDENCE_TABLE.

---

## Candidate table (carried from V18, re-scored, new candidates added)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **CurDKV** | arXiv:2509.15038 (**NeurIPS 2025**, confirmed poster) | **Value-aware token selection** — leverage scores from an approximated CUR decomposition of the attention-output subspace (`softmax(QK^T)V`), jointly weighting each token's key **and** value contribution, instead of scoring by key/attention-mass alone | ✅ same key-as-query proxy already documented for H2O/SnapKV; no retraining, no session state | Medium | **CHOSEN** |
| NestedKV | arXiv:2605.26678 (preprint; re-checked live, still no venue) | Multi-time-scale (stable/episodic/current) cosine-anomaly importance + training-free outer router | ✅ mechanically | High | **Still deferred — no verified venue.** Re-check next survey. |
| KeyDiff | arXiv:2504.15364 (**NeurIPS 2025**, confirmed via arXiv comments field) | Pure key-vector geometric-diversity/dissimilarity eviction — no attention scores needed at all | ⚠️ overlaps L2Norm/KNorm-adapted (already in repo) — both score purely from key geometry; KeyDiff's pairwise-dissimilarity criterion is a narrower distinction from KNorm's norm-magnitude criterion than a full new axis | Low–Medium | Reject: axis too close to KNorm-adapted |
| MixKVQ | arXiv:2512.19206 (**ACL 2026**, confirmed via ACL Anthology) | "Query-aware" mixed-precision channel selection | ❌ requires the **live query vector** at inference, not just a key-as-query proxy — the mechanism is defined in terms of relevance to the actual query, which the cache wrapper cannot approximate honestly the way attention-mass scoring does | — | Reject: breaks cache-only scope |
| GVote / Adaptive KV-Cache Compression | arXiv:2509.03136 (header claims ICLR 2026; **no OpenReview/iclr.cc poster page found** — unverified independently) | Monte-Carlo future-query-demand sampling to set an adaptive eviction budget | ⚠️ overlaps AdaKV's adaptive-budget axis; also needs simulated future-query access | — | Reject: venue unverified + axis overlap + scope risk |

**Next fallback named explicitly:** **NestedKV** (multi-time-scale importance)
remains the best remaining mechanism gap and becomes the pick the moment it
earns a verified venue. Until then we hold the line, as we did — correctly —
for MorphKV across three surveys and for NestedKV itself across V16–V19.

---

## Chosen: CurDKV (adapted)

### What the paper actually does

CurDKV reframes KV token selection around the **attention output**, not just
the attention *scores*. Standard heavy-hitter eviction (H2O, SnapKV, TOVA,
Keyformer, MorphKV, KVzip — all already in the repo) scores tokens purely by
how much attention-mass they receive (or, for KVzip, how much the model
"reconstructs" from them) — none of them weight a token by how much its
**value vector actually contributes to the output** once retained. CurDKV
computes an **approximated CUR matrix decomposition** of the
attention-output matrix `softmax(QK^T)V` and derives **leverage scores**
from it — a statistical importance measure over *rows* (tokens) that
captures joint key-relevance **and** value-magnitude/direction contribution
in one score, then keeps the top tokens by that combined leverage score
under a fixed budget. The paper reports up to 9.6% higher accuracy than
prior SOTA eviction baselines at matched compression, and up to 40% latency
reduction under aggressive budgets — those are the paper's numbers, on
trained models, not reproduced here.

### Why this is the right pick now

1. **Verified venue — the standing rule is honored.** NeurIPS 2025, confirmed
   poster via the official virtual site. NestedKV, the carried-forward pick,
   is still an unverified preprint as of this survey; picking it would break
   exactly the discipline that (correctly) deferred MorphKV. CurDKV lets us
   ship a peer-reviewed method **and** keep the rule.
2. **A genuinely new, isolable axis.** The repo has thirteen token-eviction
   methods, and every one of them scores tokens by some function of the
   **key** side alone (attention-mass accumulation, positional recency,
   key-norm, frozen key-SVD projection, Gumbel-regularized attention,
   recent-window correlation, reconstruction reliance). **None of them fold
   the value vector's own contribution into the retention score.** CurDKV's
   leverage-score criterion is derived from the joint `(K, V)` structure via
   CUR decomposition — the first eviction method in the repo where a
   token's *value*, not just its key/attention profile, materially changes
   whether it survives. This is cleanly separable from every existing
   eviction mechanism.
3. **Moderate risk, high reuse.** Structurally the same shape as H2O/SnapKV:
   a per-head budget, a scoring pass, a keep/evict decision at or over
   budget. The one new function is the CUR/leverage-score estimator (small,
   well-specified: approximate row-leverage scores of a matrix via a
   randomized/sketched CUR decomposition over the locally-available K/V
   block — no attention-weight ground truth required beyond the existing
   key-as-query proxy already used for H2O/SnapKV).

### The honesty crux (this is the whole adaptation)

**The paper computes leverage scores from the true `softmax(QK^T)V`
attention-output matrix, using the model's real query vectors across a
prefill/decode trace, and validates end-to-end on trained models.** A
cache-side library has no access to the true query vector at any step
(same limitation already documented for H2O/SnapKV/Keyformer/MorphKV/KVzip)
and no ability to materialize the true attention-output matrix. So the
adaptation:

- **Key-as-query proxy, not the true query vector.** Exactly the same
  documented approximation as H2O/SnapKV: the incoming key vector stands in
  for the true query to build an approximate attention-weight row, since
  the cache wrapper never sees the real query.
- **Approximated CUR decomposition over the locally observable K/V block,
  not the paper's full-sequence attention-output matrix computed with real
  queries.** We build the CUR approximation (and its leverage scores) from
  the proxy-attention-weighted `(K, V)` block available at the cache layer,
  not the paper's ground-truth `softmax(QK^T)V`.
- **Not the paper's exact CUR sketching algorithm** unless a simple,
  well-specified randomized column/row-sampling CUR approximation
  (leverage-score-proportional sampling, e.g. via a small number of power
  iterations for the dominant singular directions) is sufficient to
  reproduce the paper's *qualitative* claim (joint key+value leverage beats
  key-only attention-mass scoring at a matched budget) — document precisely
  which simplification is used.
- **Uniform budget across heads**, matching the repo's existing eviction
  convention (H2O, SnapKV, etc.) — not the paper's potential per-head
  tuning, if any.
- **Clean mechanism observable**: at a **matched token budget**, compare
  retention quality (e.g. reconstruction of a held-out later-step's true
  attention-weighted output, or simple output-approximation error) of
  CurDKV's leverage-score eviction against H2O's cumulative-attention-mass
  eviction, on a planted geometry where a token has **high key-similarity
  but low value-magnitude/orthogonal-value-direction contribution** (H2O
  should keep it since it only looks at key/attention mass; CurDKV should
  correctly deprioritize it since its value contribution is negligible).
  This is the isolable, testable difference between the two mechanisms.
- Nothing here is validated on a trained model — offline-synthetic only.

---

## What we do NOT implement (state plainly, carried into the implementation prompt)

- The paper's exact CUR sketching/leverage-score estimation algorithm,
  unless independently reproduced from the paper with enough precision to
  cite exactly — default to a standard, well-documented randomized
  leverage-score CUR approximation instead, cited generically (Mahoney &
  Drineas-style CUR/leverage-score sampling), not claimed as the paper's own
  implementation detail.
- The paper's ground-truth `softmax(QK^T)V` computed with real query
  vectors (key-as-query proxy instead, same as H2O/SnapKV).
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers (up to 9.6% higher accuracy, up to 40% latency reduction)
  are the paper's — not reproduced here.
