# New-method survey V20 — no method selected this cycle

**Outcome: paused.** Two full search cycles ran against the standing bar
(verified peer-reviewed venue, genuinely new mechanism axis, cache-only /
zero-shot / training-free fit) and neither produced a candidate that clears
all three. Rather than force a weak pick, method #39 is deferred to a future
cycle. This note exists so a future session doesn't re-spend effort on the
same dead ends.

## Hard constraint clarified this cycle

Every one of the 38 methods already in the repo works **zero-shot on any
`mlx_lm` model**, with **no shipped pretrained weights** and **no offline
training**. A brief calibration pass computed live from the model's own
weights/activations at load time (PALU's group-head SVD, RateQuant's
sensitivity pass, VecInfer's codebook fit — seconds, no ground-truth labels
needed) is fine. A trained neural network checkpoint, or anything needing
supervised labels from generation traces, is not — it would require shipping
and versioning per-model-family artifacts, breaking the "three lines of code,
any model" value proposition. This was not previously written down explicitly
as a hard bar; it is now, and future surveys should check it first.

## Cycle 1 — rejected: KVP (Learning to Evict from Key-Value Cache)

- arXiv:2602.10238, Apple. Venue independently confirmed three ways: arXiv
  Comments field ("Accepted to ICML 2026"), `icml.cc/virtual/2026/poster/66783`,
  and `machinelearning.apple.com/research/evict`.
- Mechanism: per-layer-head RL agents learn a reward-guided eviction policy
  from generation traces, frozen and reused at deployment. Genuinely new axis
  (first *learned* scoring function vs. every closed-form heuristic already in
  the repo).
- **Rejected**: requires offline RL training per model/head-config and shipping
  those trained weights. Fails the zero-shot/training-free bar outright.

## Cycle 2 — rejected candidates

- **KeyDiff** (arXiv:2504.15364, NeurIPS 2025 poster, confirmed via
  neurips.cc/virtual/2025/poster/115521) — key-only geometric similarity
  scorer. Rejected: overlaps the existing key-only heavy-hitter axis
  (H2O/SnapKV/PyramidKV/MorphKV); no value signal, a regression relative to
  CurDKV's value-awareness.
- **AttentionPredictor** (arXiv:2502.04077, NeurIPS 2025 poster, confirmed via
  arXiv Comments field "NeurIPS 2025" + neurips.cc/virtual/2025/poster/118927).
  Mechanism: lightweight shared convolutional model predicts *future* attention
  scores from spatiotemporal history; frozen and reused across prompts.
  Considered for a scoped-down "fit cheaply at load time" version — **checked
  directly against the official repo
  (github.com/MIRALab-USTC/LLM-AttentionPredictor)**: it ships pretrained
  checkpoints per specific model (LongChat-7B, LLaMA-3.1-8B), no public
  training script, and the predictor needs supervised ground-truth
  future-attention labels from real generation traces — not derivable from
  static weights the way PALU/RateQuant calibration is. **No honest
  scoped-down version exists.** Rejected on the training-free bar, same as KVP.
- **CAKE** (arXiv:2503.12491, ICLR 2025, confirmed via
  proceedings.iclr.cc/paper_files/paper/2025/hash/dfae940651f3e690a12e19c874edad7c-Abstract-Conference.html)
  — layer-adaptive budget allocator using spatial+temporal attention dynamics.
  Rejected: same axis as PyramidKV/SqueezeAttention already in the repo.
- **ManifoldKV** (arXiv:2602.08343, ICML 2026 regular, confirmed via OpenReview
  API `venue` field) — training-free, but its own abstract frames it as a
  KeyDiff ablation (cosine→Euclidean distance-to-centroid). Rejected: same
  axis as KeyDiff, already rejected above.
- **KQ-SVD** (arXiv:2512.05916, claimed AISTATS 2026 poster) — closed-form
  joint SVD of the bilinear Q·Kᵀ attention operator (distinct from PALU's
  independent per-tensor K/V low-rank projection — would have cleared the
  novelty bar). **Venue claim did not survive independent re-verification**:
  the arXiv abstract page has no Comments/journal-ref field; a direct
  OpenReview search returned no matching record; the AISTATS 2026
  accepted-papers page 404s. A sub-agent's claimed OpenReview API record
  (`"venue":"AISTATS 2026 Poster"`) did not reproduce under direct fetch.
  **Rejected on venue — unconfirmed, treat as a bare preprint until a live
  re-check finds real corroboration.**
- **RocketKV** (arXiv:2502.14051, ICML 2025 poster, confirmed) — training-free,
  but built from SnapKV++ plus top-k sparse attention; not a new axis.
- **LSH-E / HashEvict** (arXiv:2412.16187) — NeurIPS 2024 **workshop** only
  (ML & Compression workshop), not main track. Rejected on venue tier.
- **Expected Attention** (arXiv:2510.00636), **Nexus Sampling**
  (arXiv:2606.23961), **MomentKV** (arXiv:2606.01563), **CapKV**
  (arXiv:2604.25975) — all still bare preprints as of live re-check, no
  Comments/journal-ref field, no venue page match. Worth re-checking in a
  future cycle:
  - MomentKV's moment-statistic compensation for evicted tokens (retaining a
    running statistical summary of dropped tokens rather than discarding them
    outright) would be a genuinely new axis if it clears review.
  - Expected Attention's closed-form Gaussian expected-attention estimate is
    also distinct from everything in the roster, if it clears review.
- **NestedKV** (arXiv:2605.26678) — re-checked again, still no venue.

## Recommendation for next cycle

Re-run the survey in a few months. Prioritize re-checking MomentKV and
Expected Attention's venue status first (mechanistically new and training-free
already, just unreviewed), before searching for entirely new candidates. Any
future venue claim must be corroborated by directly fetching the arXiv
Comments/journal-ref field *and* the venue's own page — a sub-agent's
paraphrase of an API response is not sufficient on its own, per the KQ-SVD
false-positive this cycle.
