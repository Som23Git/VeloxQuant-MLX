# Phase 1 — New-Method Survey V11 (post-xKV)

Follow-up to `NEW_METHOD_SURVEY_V3.md`–`V10.md` (most recently xKV-adapted
cross-layer shared-subspace SVD, shipped in 0.27.0). The repo now spans 30
strategies across: scalar/group quant (KIVI, KVQuant-NUQ, TurboQuant,
RateQuant), vector quant (RVQ, VecInfer, CommVQ, RaBitQ, QJL, PolarQuant,
SpectralQuant), low-rank projection (SVDq, PALU), cross-layer mechanisms
(XQuant rematerialization, MiniCache SLERP merge, xKV shared-subspace joint
SVD), entropy coding (CacheGen), error-feedback (GEAR), saliency-adaptive
mixed precision (ZipCache), attention-proxy adaptive schemes (KIVI-Sink,
AdaKV-proxy, Kitty), score-based prefill eviction (SnapKV), structural
positional eviction (StreamingLLM, sink, sliding-window, TOVA), layer-adaptive
budgets (PyramidKV), 2D layer×token budgets (SqueezeAttention), chunk-level
eviction (ChunkKV), and merge-vs-drop eviction (CaM).

The open gap: **calibration-free distribution matching for vector
quantization.** Every VQ method in the repo either fits its codebook to the
data at hand (RVQ's per-sequence k-means, CommVQ) or uses a data-independent
*geometric* code (RaBitQ sign codes, VecInfer binary, PolarQuant polar grids,
QJL sketches). None of them **reshapes the K/V distribution itself into a
known target distribution so that one fixed, model-independent codebook works
everywhere**. That is a structurally different compression axis: instead of
adapting the code to the data, adapt the data to the code. Its failure mode
(imperfect Gaussianization → codebook mismatch) and its byte-accounting story
(per-chunk normalization statistics amortized over the chunk) are both
distinct from anything shipped. Notably, the repo already owns the key
building block — `HadamardPreconditioner`
(`veloxquant_mlx/preconditioners/rotation.py`) — currently used only by the
geometric-VQ family; this candidate promotes it to a distribution-shaping
role.

**Evidence discipline:** every arXiv ID below was verified to resolve to a
real paper (fetched abstract/HTML + author list, cross-checked against the
claim) before inclusion. No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **NSNQuant** | arXiv:2505.18231 (**NeurIPS 2025**) | **Calibration-free distribution-matching VQ**: token-wise Normalize → channel-wise Shift → token-wise Normalize + Hadamard maps K/V onto ~standard-normal vectors, so a *single universal codebook built offline from synthetic Gaussian samples* quantizes any model at 1–2 bits | ✅ pure K/V tensor transforms; no attention scores, no training, no calibration data — the codebook is built from `randn` samples, not model activations | Med | **CHOSEN** |
| **RotateKV** | arXiv:2501.16383 (IJCAI 2025) | Outlier-aware adaptive rotations (FWHT + channel reordering + pre-RoPE grouped-head rotation) for 2-bit KV quant | ⚠️ two of its three pillars break at our cache position: pre-RoPE grouped-head rotation is impossible (`update_and_fetch` receives post-RoPE keys) and channel reordering is derived from offline outlier profiling (calibration) | Med | reject (mechanism gutted by our post-RoPE, calibration-free constraints) |
| **NestedKV** | arXiv:2605.26678 (May 2026, preprint) | Multi-time-scale cosine-anomaly token importance from the key stream (no attention), hierarchical anchors, surprise-gated routing, per-head budgets | ✅ mechanically (key-only signal, training-free) | High | defer (strong future candidate; very new preprint, complex 4-part mechanism — poor effort/verification ratio *this* release) |
| **LogQuant** | arXiv:2503.19950 (preprint) | Log-distributed position-based precision tiers across the context window | ⚠️ positional precision tiers substantially overlap the StreamingLLM/PyramidKV/SqueezeAttention budget axes already shipped; attention-dependence of its filtering step is ambiguous on first read | Low-Med | reject (weak differentiation + unresolved attention-dependence) |
| **L2-norm eviction ("KnormPress")** | arXiv:2406.11430 (**EMNLP 2024**) | Key-L2-norm saliency: low key norm ⇒ high future attention — an eviction signal computed from keys alone, no attention and no proxy | ✅ trivially | Low | defer (real and clean, but a one-scorer eviction variant is too small to headline a release; natural bundle-in for a future eviction-focused version) |

---

## Chosen: NSNQuant (calibration-free distribution-matching VQ)

### What the paper actually does

NSNQuant ([arXiv:2505.18231](https://arxiv.org/abs/2505.18231), Donghyun Son,
Euntae Choi, Sungjoo Yoo — **NeurIPS 2025**) observes that vector quantization
of the KV cache fails under distribution shift when its codebook is fit on a
calibration dataset. Instead of calibrating the codebook, it *standardizes the
data*:

1. **Normalize (token-wise):** scale each token vector to norm √d (d =
   head_dim), suppressing outlier tokens. Store the scale `s1`.
2. **Shift (channel-wise):** compute the channel-wise mean `o` online from the
   current sequence's tokens and subtract it, zero-centering the distribution.
   Store `o`.
3. **Normalize again (token-wise):** rescale to norm √d (store `s2`). The
   second normalization slightly perturbs the zero mean; the paper shows the
   effect is negligible.
4. **Hadamard transform:** rotates the NSN output so channels decorrelate and
   the empirical distribution closely matches an isotropic standard normal.
5. **Vector quantization with a universal codebook:** 8-dimensional subvectors
   are matched (by cosine distance) against a single codebook that was built
   *offline on synthetic `randn` samples* — k-means initialization plus a
   cosine-objective fine-tune — and reused unchanged for every model, layer,
   and dataset. **NSNQuant-2b** stores an 8-bit sign mask + an 8-bit codebook
   index per subvector (≈2 bits/element); **NSNQuant-1b** stores the index
   only (≈1 bit/element).
6. **Restoration:** `v̂ = s1 · (s2 · v_nsn + o)` after inverse Hadamard;
   metadata (`s1`, `o`, `s2`) is itself double-quantized to ~0.23 extra
   bits/element on average.
7. **Decode:** new tokens accumulate in a small full-precision residual buffer
   (default 64 tokens; 128 recommended for 1-bit) and are flushed through
   NSN+VQ chunk-wise when full.

The paper reports consistent wins over prior 1–2-bit KV quantization and up to
3× throughput gain on their hardware — not applicable claims for us to
inherit, since our target is Apple Silicon/MLX, not their CUDA kernels.

### The honest adaptation problem

**1. Pre-RoPE keys and the custom RoPE kernel are unavailable.** The paper
applies NSN to keys *before* RoPE and defers RoPE onto the stored mean via a
custom attention kernel. Our wrappers sit behind `update_and_fetch` and
receive **post-RoPE keys** — the same constraint that gutted RotateKV's
grouped-head rotation. We adapt by running the full NSN + Hadamard pipeline
post-RoPE on keys, exactly where the cache sees them. This weakens the
Gaussianization slightly (RoPE mixes channel pairs position-dependently) and
is documented plainly as the central simplification of this adaptation.

**2. Value-side Hadamard "fused into the projection layers" is model
surgery.** We do not touch model weights. Adaptation: apply the Hadamard
explicitly to cached values on quantize and its inverse on fetch — extra
FWHT compute per flush/fetch, honest and auditable.

**3. Codebook fine-tuning uses gradients; we have no training loop.** The
paper k-means-initializes on `randn` samples then fine-tunes with gradient
descent on a cosine objective. We keep the property that matters —
**model/data independence** — by building the codebook with deterministic
seeded spherical k-means on synthetic standard-normal 8-dim samples (numpy,
offline at first use, cached as a module-level constant keyed by
`(codebook_size, subvector_dim, seed)`), and we **skip the gradient
fine-tune**. Expect a slightly worse codebook than the paper's; documented,
not hidden. This preserves calibration-freeness: no model activations are
ever used to fit anything.

**4. Metadata double-quantization deferred.** The paper 4-bit-quantizes
`s1`/`o` to reach ~0.23 overhead bits. We store `s1`, `s2` (per token) and `o`
(per chunk) in fp16 and **count them in `*_bytes` honestly** — slightly higher
overhead, simpler audit. 4-bit metadata is future work.

**5. Residual buffer reuses the repo's own pattern.** KIVI already keeps the
most recent `residual_length` tokens fp16; NSNQuant's chunk-flush residual
buffer is the same idiom (buffer fills to `nsn_residual_length`, flushes
through NSN+VQ as one chunk with its own `o`). Per-chunk `o` (recomputed at
each flush, never frozen) follows the paper's online-statistics story and
needs no coordinator — this is a **single-layer method**, the simplest wrapper
shape in the repo.

### What we do NOT implement

- Pre-RoPE key handling + RoPE-aware attention kernel (post-RoPE adaptation
  only).
- Value-projection Hadamard fusion (explicit transform instead).
- Gradient fine-tuning of the codebook (seeded spherical k-means only).
- 4-bit double quantization of NSN metadata (fp16, honestly byte-accounted).
- The paper's fused CUDA/Triton kernels (MLX ops; a Metal kernel is possible
  future work alongside the existing `metal/` kernels).

### Why this is the right pick

1. **Fills a genuine axis gap.** Adapt-the-data-to-the-code is the inverse of
   every VQ method shipped; one universal codebook shared across all models is
   a story no existing method tells.
2. **Strong provenance.** NeurIPS 2025 — joins the landing page's
   peer-reviewed venue strip (which currently tops out at ICLR 2026), not the
   arXiv-preprint bucket.
3. **Calibration-free by construction** — the repo's hardest constraint is the
   paper's headline feature, a rare exact alignment.
4. **High infrastructure reuse, low risk.** `HadamardPreconditioner` and
   `make_hadamard_diagonal` already exist; the residual-buffer idiom is
   KIVI's; the wrapper is single-layer (no coordinator).
5. **Verified, real, matches claims** (title/authors/venue checked against
   arXiv + OpenReview/NeurIPS listings — see Sources).

### Why the alternatives were not chosen

- **RotateKV** — after removing pre-RoPE grouped-head rotation (impossible at
  our hook point) and calibrated channel reordering (violates
  calibration-free), what remains is plain FWHT + quantization, which the repo
  effectively has via `HadamardPreconditioner` + scalar quant. A port would be
  RotateKV in name only.
- **NestedKV** — passes the hard constraints and is the most interesting
  eviction candidate seen since CaM, but it is a very fresh preprint with a
  four-part mechanism (hierarchical anchors + anomaly scoring + head-adaptive
  mixing + surprise gating) whose pieces are individually unvalidated.
  Deferred, explicitly flagged for V12 reconsideration once it has been out
  longer.
- **LogQuant** — positional precision tiers are largely spanned by
  StreamingLLM/PyramidKV/SqueezeAttention already; first-read ambiguity about
  whether its filtering needs attention statistics is disqualifying under the
  evidence discipline until resolved.
- **L2-norm eviction** — genuinely attention-free and peer-reviewed (EMNLP
  2024), but it is one scoring rule, not a release-sized mechanism. Best
  shipped later as a bundled eviction scorer, possibly alongside NestedKV.

### Planned artifacts (Phases 2–6)

See `paper/IMPLEMENTATION_PROMPT_NSNQUANT.md` for the full execution
checklist: `veloxquant_mlx/quantizers/nsnquant.py` (NSN transform/inverse,
universal-codebook builder, subvector VQ encode/decode),
`veloxquant_mlx/cache/nsnquant_cache.py` (single-layer wrapper, chunk-flush
residual buffer), `KVCacheConfig(method="nsnquant", ...)` + builder wiring,
tests (~25), `benchmark_scripts/benchmark_nsn.py` + committed results JSON,
`docs-site/docs/algorithms/nsnquant.md`, CHANGELOG 0.28.0, README 30→31,
EVIDENCE_TABLE row, landing page (hero pill/badge, what's-new, Method Library
card, code tab, stat refresh), version bump 0.27.0 → 0.28.0.

---

## Sources (verified)

- NSNQuant — https://arxiv.org/abs/2505.18231 (NeurIPS 2025; Son, Choi, Yoo;
  OpenReview id boNYskaXnO; NeurIPS 2025 poster listing confirmed)
- RotateKV — https://arxiv.org/abs/2501.16383 (IJCAI 2025;
  https://www.ijcai.org/proceedings/2025/690)
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint, May 2026; Chen, Liu,
  Gao, Fan, Wang, Chu, Lin, Hu)
- LogQuant — https://arxiv.org/abs/2503.19950 (preprint; Chen, Jiang, Zhang,
  He, Luo, Lu, Chen)
- L2-norm KV compression — https://arxiv.org/abs/2406.11430 (EMNLP 2024;
  Devoto, Zhao, Scardapane, Minervini; code
  https://github.com/alessiodevoto/l2compress)
