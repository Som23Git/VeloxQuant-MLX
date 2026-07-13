# Phase 1 — New-Method Survey V18 (post-KVzip)

Follow-up to `NEW_METHOD_SURVEY_V17.md` (KVzip-adapted context-reconstruction
reliance eviction, shipped in 0.34.0). The repo now spans **37** strategies.
V17 carried **NestedKV** forward as the named next pick, explicitly gated on it
acquiring a verified venue — the same discipline that correctly deferred
MorphKV across V13–V15. NestedKV (arXiv:2605.26678) was re-checked live this
survey and is **still a bare preprint** (submitted 2026-05-26, no venue). It
stays deferred. This survey picks **KVTC (KV Cache Transform Coding)**, which
both satisfies the standing verified-venue rule **and** adds a bit-allocation
mechanism axis the repo does not have.

**Evidence discipline / venue verified live this survey:**

> "KV Cache Transform Coding for Compact Storage in LLM Inference" (NVIDIA) —
> **ICLR 2026 (accepted, poster)**, arXiv:2511.01815,
> OpenReview poster `iclr.cc/virtual/2026/poster/10008708`. Accepted, not
> merely submitted.

Re-check the arXiv abstract + OpenReview page live before writing it into
README/docs/EVIDENCE_TABLE.

---

## Candidate table (carried from V17, re-scored, new candidates added)

| Method | Paper (verified) | What it adds (not in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **KVTC** | arXiv:2511.01815 (**ICLR 2026**, confirmed accepted) | **DP-optimal per-principal-component bit allocation** under a hard total-bit budget (can zero trailing components entirely) + **entropy coding** on top of quantized codes | ✅ structurally the Palu/SVDq/SpectralQuant low-rank family; the DP-allocation + entropy-coding stage is the new axis | Medium | **CHOSEN** |
| NestedKV | arXiv:2605.26678 (preprint; re-checked live, still no venue) | Multi-time-scale (stable/episodic/current) cosine-anomaly importance + training-free outer router | ✅ mechanically | High | **Still deferred — no verified venue.** Re-check next survey. |
| STAR-KV | ICML 2026 Spotlight (confirmed) | Differentiable soft-thresholding for per-head/per-block **rank** selection + mixed-precision on latent channels | ⚠️ overlaps Palu (group-head low-rank) / SVDq (mixed-bit latents) — same two axes (adaptive rank + mixed-precision latents), just a differentiable threshold instead of a fixed energy cutoff | Medium | Reject: axis already covered twice |
| EpiCache | ICML 2026 (confirmed) | Episodic KV management for long-term multi-session conversation under a fixed budget | ⚠️ session/conversation-boundary framing doesn't map onto a single-sequence cache-side library (no session concept in `_MLXKVCache`) | High (needs new abstraction) | Reject: doesn't fit the cache-only scope |
| Anchor Direction Projection | NeurIPS 2025 | Eviction by projecting keys onto an anchor direction | ⚠️ overlaps Q-Filters / L2Norm projection axis | Low | Reject: axis already covered |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance retention | ⚠️ overlaps H2O-adapted | Low | Reject: duplicate axis |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ duplicates ZipCache-adapted | Low | Reject: duplicate axis |

**Next fallback named explicitly:** **NestedKV** (multi-time-scale importance)
remains the best remaining mechanism gap and becomes the pick the moment it
earns a verified venue. Until then we hold the line, as we did — correctly —
for MorphKV across three surveys and for NestedKV itself across V16–V18.

---

## Chosen: KVTC (adapted)

### What the paper actually does

KVTC is a **transform-coding pipeline** applied to cached keys/values:
(1) a **PCA basis** is fit once on a calibration set and reused at inference
to linearly decorrelate features (rotate into principal components); (2) a
**dynamic-programming bit allocator** finds the total-bit-budget-constrained
allocation of bits *per principal component* that minimizes expected
reconstruction error — trailing low-variance components are frequently
assigned **zero bits** (dropped entirely) rather than quantized at a uniform
floor; (3) the quantized component codes are **entropy-coded** (the paper
reports this final stage recovers additional bits beyond what fixed-width
quantization alone achieves). The paper reports up to 20× (up to 40× in
some regimes) KV cache compression at <1pp accuracy loss across LLaMA 3,
Mistral NeMo, and R1-distilled Qwen2.5 (1.5B–70B), evaluated on AIME25,
GSM8K, LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, and RULER — those
are the paper's numbers, on trained models, not reproduced here.

### Why this is the right pick now

1. **Verified venue — the standing rule is honored.** ICLR 2026, accepted
   (poster). NestedKV, the carried-forward pick, is still an unverified
   preprint as of this survey; picking it would break exactly the discipline
   that (correctly) deferred MorphKV. KVTC lets us ship a peer-reviewed
   method **and** keep the rule.
2. **A genuinely new, isolable axis.** The repo has three low-rank / spectral
   methods already — Palu (group-head SVD, latents mixed-bit quantized with a
   fixed top-25%/75% split), SVDq (keys-only SVD, same fixed split), and
   SpectralQuant (eigendecomposition into a signal/noise **binary** cutoff via
   participation ratio, uniform bits within each half, plus a JL-sketch
   residual). **None of them compute a DP-optimal allocation across
   individual components under a hard bit budget**, and **none apply entropy
   coding**. `ratequant`'s waterfilling allocator is closed-form and
   per-*layer*, not per-*component*, and never zeroes an allocation outright.
   KVTC's contribution — optimal bit allocation via DP (component-granularity,
   can zero trailing components) plus entropy coding as a final lossless
   stage — is cleanly separable from the PCA rotation itself, which the repo
   already has building blocks for (`spectral/participation_ratio.py`,
   `svdq.py`'s SVD helpers).
3. **Moderate risk, high reuse.** Structurally the Palu/SVDq family: fit a
   basis, project, quantize latents. The two new functions are the DP
   bit-allocator (small, well-specified: given per-component variances and a
   total-bit budget, minimize Σ distortion(bits_i) subject to Σ bits_i ≤
   budget) and a lightweight entropy-coding pass on the quantized codes.

### The honesty crux (this is the whole adaptation)

**The paper calibrates its PCA basis and its entropy coder on real model
activations across many prompts and validates end-to-end on trained models
across eight benchmarks. A cache-side library has no calibration corpus and
no access to a trained model's true activation distribution.** So the
adaptation:

- **Per-sequence PCA, not a pre-calibrated global basis.** The paper fits one
  basis offline on a calibration dataset and reuses it for all future caches.
  We do not have a calibration corpus wired into `KVCacheBuilder.for_model`,
  so the basis is fit **online from the sequence's own prefill keys** (the
  same "fit locally, no calibration set" pattern SVDq already uses) —
  documented as a **local PCA proxy** for the paper's pre-calibrated global
  basis, not the paper's calibration pipeline.
- **DP allocator over per-component variance, not the paper's rate-distortion
  model fit on real model statistics.** The allocator is implemented exactly
  (given a per-component variance/eigenvalue vector and a total-bit budget,
  DP over discrete bit choices per component to minimize a closed-form
  Gaussian-quantization distortion proxy `α·β^(-b)` per component — reusing
  the repo's existing distortion-curve machinery from `ratequant.py` rather
  than inventing a new one). The DP is real and exact; what's a proxy is the
  *distortion model it optimizes*, which is analytic (Gaussian MSE bound), not
  fit on real LLM activation statistics as the paper does.
- **Entropy coding is a real, measured, lossless stage** — a simple
  order-0 range/Huffman-style coder over the quantized component codes,
  applied and measured for its actual achieved bits-per-code, not assumed at
  the theoretical entropy bound. Report the **realized** post-entropy-coding
  size, not the Shannon-entropy lower bound.
- **Values, not just keys.** Unlike SVDq (keys-only), mirror Palu and apply
  the same PCA + DP-allocate + entropy-code pipeline to values independently,
  since KVTC's paper compresses both.

**Consequence for expectations:** the clean, defensible observable is
**bits-per-component actually spent vs. a fixed-uniform-bit baseline at the
same total budget**, and the **reconstruction MSE / cosine similarity at
matched total bytes** against SVDq's fixed top-25%/75% split and against
Palu's fixed group-rank split — showing the DP allocator reaches a lower
distortion at the same byte budget because it can zero low-variance
components entirely rather than paying a uniform per-component floor. Report
the entropy-coding stage's **realized additional compression** (measured
bits/code vs. the pre-entropy-coding fixed-width bits/code) as a secondary,
honestly-scoped observable — it will be modest on synthetic Gaussian-like
data and should not be oversold.

### Path-dependence (honest contrast)

Unlike H2O/TOVA/MorphKV/KVzip (eviction, path-dependent keep-sets), KVTC is
in the Palu/SVDq/SpectralQuant family: the PCA basis is fit once (at prefill)
and every subsequent token is projected through the same fixed basis and
DP-derived per-component bit allocation. **Not path-dependent** in the
eviction sense — state this contrast explicitly, and pin it with a
determinism test (same input → same basis, same allocation, same codes).

### What we do NOT implement

- The paper's pre-calibrated global PCA basis fit across a calibration
  corpus — replaced by a per-sequence local-PCA proxy (same limitation SVDq
  already documents).
- The paper's rate-distortion model fit on real model activation statistics
  — replaced by the repo's existing analytic Gaussian distortion-curve proxy
  (`fit_distortion_curve` / `α·β^(-b)`), reused rather than re-derived.
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers (up to 20×, up to 40× in some regimes, <1pp accuracy loss
  on LLaMA 3 / Mistral NeMo / R1-Qwen2.5 1.5B–70B across AIME25, GSM8K,
  LiveCodeBench, LongBench, MATH-500, MMLU, Qasper, RULER) are the paper's,
  on trained models (offline-synthetic only here — bits-per-component
  accounting, matched-budget distortion comparison, entropy-coding realized
  gain).
- Arithmetic-coding-grade entropy coder with adaptive context modeling — a
  simple order-0 coder is enough to make the "entropy coding recovers
  additional bits" claim honestly, without over-engineering a codec.

### Delivered artifacts (planned — see implementation prompt)

See `paper/research/implementation_prompts/IMPLEMENTATION_PROMPT_KVTC.md`.

---

## Sources (verified this survey)

- KVTC — https://arxiv.org/abs/2511.01815 (**ICLR 2026**, confirmed accepted,
  poster `iclr.cc/virtual/2026/poster/10008708`); MarkTechPost coverage
  confirms NVIDIA authorship and the 20×/40× headline. Re-verify the abstract
  + OpenReview listing live before citing.
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried V11–V18,
  re-checked live this survey, still no venue) — best next pick once a
  verified venue exists.
- STAR-KV — ICML 2026 Spotlight, Dnotitia/UC San Diego
  (`icml.cc/virtual/2026/poster/61958`) — axis overlaps Palu/SVDq (adaptive
  rank + mixed-precision latents).
- EpiCache — ICML 2026 (`icml.cc/virtual/2026/poster/65405`) — episodic
  multi-session framing doesn't fit the single-sequence cache-only scope.
- Anchor Direction Projection — NeurIPS 2025 (axis overlaps Q-Filters/L2Norm).
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
