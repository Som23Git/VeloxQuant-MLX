# Full Autonomous Implementation Prompt — NSNQuant-adapted (v0.28.0)

**Purpose of this document:** a self-contained, execute-without-supervision
spec for shipping the next VeloxQuant-MLX method end-to-end: survey (done,
see `paper/NEW_METHOD_SURVEY_V11.md`) → quantizer primitives → cache wrapper →
config/builder wiring → tests → benchmark → docs → changelog → README →
landing page → version bump → release/publish commands. Written so an agent
picking this up cold, with no other context, can complete every step and
leave the repo in the same finished state as every prior release (0.19.0
through 0.27.0).

**Do not deviate from the "adapted, not faithful port" discipline** that
governs this entire repo: label everything "NSNQuant-adapted (VeloxQuant-MLX
implementation)", report only numbers from a committed results JSON you
generate yourself, never repeat the paper's headline numbers as if they were
measured here, and document every simplification plainly (a "What we do NOT
implement" section, exactly like every other algorithm doc in this repo).

Read `paper/NEW_METHOD_SURVEY_V11.md` first — it contains the full rationale,
the exact mechanism, and the adaptation decisions already made. This document
is the *execution checklist* for that survey's "Planned artifacts" section.
Do not re-derive the design; the survey already settled it. If something in
this checklist conflicts with the survey, the survey wins — fix this document,
don't silently diverge.

---

## 0. Ground rules (apply to every phase)

- Work in small, buildable commits, one per phase below, each passing tests
  before moving to the next phase — mirrors every prior release's git history
  (survey → primitives+tests → cache+tests → bench → docs/release).
- Every new Python file gets a module docstring citing arXiv:2505.18231
  (Son, Choi, Yoo — NeurIPS 2025), stating "adapted, not faithful port", and
  listing what's NOT implemented.
- No new third-party dependencies. Codebook construction uses numpy only
  (deterministic, seeded); runtime transforms use MLX ops.
- **Reuse existing infrastructure — do not reimplement:**
  - Hadamard: `veloxquant_mlx/math/rotation.py::make_hadamard_diagonal`,
    `is_hadamard_compatible`, and
    `veloxquant_mlx/preconditioners/rotation.py::HadamardPreconditioner`
    (has `apply` / `apply_inverse`). NSNQuant uses the **plain** (sign-free)
    Hadamard per the paper's practical choice; if `HadamardPreconditioner`
    hard-requires the random-sign diagonal, pass an all-ones diagonal rather
    than forking the class.
  - Residual-window idiom: study `kivi_cache.py` / `kivi.py`
    (`residual_length` handling) before writing the buffer logic.
  - Uniform-quant helpers in `quantizers/_quant_utils.py` if scale/zero-point
    packing is needed.
- Never invent benchmark numbers. Build the offline-synthetic harness (same
  pattern as `benchmark_scripts/benchmark_xkv.py`) and commit whatever
  results JSON it actually produces when run. If MLX/Metal is unavailable in
  the execution sandbox, state that explicitly in the CHANGELOG entry
  ("NOT YET RUN on hardware") rather than fabricating numbers.
- Run `pytest veloxquant_mlx/tests/ -x -q` after each phase; do not proceed
  past a red test suite. Suite is 844 tests before this work.
- Naming: files `nsnquant.py` (quantizer primitives), `nsnquant_cache.py`,
  `test_nsnquant.py`, `test_nsnquant_cache.py`, `benchmark_nsn.py`,
  `nsnquant.md`, `nsn_benchmark_results.json`. Config prefix `nsn_`,
  method string `"nsnquant"`.

---

## Phase 1 — Quantizer primitives

**File:** `veloxquant_mlx/quantizers/nsnquant.py`

Module docstring: cite arXiv:2505.18231 (Donghyun Son, Euntae Choi, Sungjoo
Yoo — NeurIPS 2025, code linked from OpenReview id boNYskaXnO), state the
adaptation scope (post-RoPE keys, explicit value Hadamard, k-means-only
codebook, fp16 metadata — see survey §"The honest adaptation problem").

Pure functions (no cache state), operating on `(..., T, d)` MLX arrays:

1. `nsn_transform(x) -> (x_nsn, s1, o, s2)`
   - `s1 = norm(x, axis=-1, keepdims=True) / sqrt(d)`; `x1 = x / s1`
     (guard zero-norm tokens with an epsilon, like `_quant_utils` does).
   - `o = mean(x1, axis=-2, keepdims=True)` — channel mean **computed from
     this chunk only** (online statistics; never global, never calibrated).
   - `x2 = x1 - o`; `s2 = norm(x2, axis=-1, keepdims=True) / sqrt(d)`;
     `x_nsn = x2 / s2`.
   - Returned metadata dtypes: fp16 (`s1`, `s2` per token; `o` per chunk).
2. `nsn_inverse(x_nsn, s1, o, s2) -> x̂` — exact restoration
   `x̂ = s1 * (s2 * x_nsn + o)`. Round-trip with unquantized `x_nsn` must be
   exact to fp16 tolerance (test below).
3. `build_universal_codebook(codebook_size=256, subvector_dim=8, seed=1234,
   n_samples=1_000_000, iters=25) -> np.ndarray`
   - Seeded `np.random.default_rng(seed)` standard-normal samples,
     **spherical k-means** (normalize samples and centroids to unit norm each
     iteration; assignment by max cosine). Deterministic: same args → bitwise
     identical output.
   - For the 2-bit variant the codebook is built over **absolute-value
     orthant** vectors: take `|samples|` before k-means so the codebook lives
     in the positive orthant and an 8-bit sign mask restores orientation.
     Build both variants: `kind="signed"` (1-bit: k-means on raw samples) and
     `kind="magnitude"` (2-bit: k-means on `|samples|`).
   - Cache the result in a module-level dict keyed by
     `(codebook_size, subvector_dim, seed, kind)` so it is built once per
     process. Do NOT persist to disk and do NOT commit a binary artifact —
     determinism makes the artifact reproducible.
4. `vq_encode(x_nsn, codebook, bits) -> encoded`
   - Reshape `(..., T, d)` → `(..., T, d/8, 8)` subvectors.
   - `bits=2`: store `signs` (uint8 bitmask per subvector, 8 sign bits) +
     `idx` (uint8 index into the magnitude codebook, matched by cosine
     against `|subvector|`).
   - `bits=1`: store `idx` only (uint8 index into the signed codebook).
   - Requires `d % 8 == 0` — raise `ValueError` otherwise (all supported
     head_dims 64/128 pass).
5. `vq_decode(encoded, codebook, bits) -> x_nsn̂` — lookup (+ sign restore
   for 2-bit), reshape back, then **renormalize each token to norm √d**
   (codebook entries are unit-subvector approximations; renormalization is
   the paper's scale-adjustment analog and is why `s2` stays valid).
6. `hadamard_forward(x)` / `hadamard_inverse(x)` — thin wrappers over
   `HadamardPreconditioner` for head_dim-sized last axes; assert
   `is_hadamard_compatible(d)`.

Pipeline order (both K and V, both post-RoPE as received by the cache):
`nsn_transform` → `hadamard_forward` → `vq_encode`; fetch reverses it. (The
Hadamard is norm-preserving, so applying it after NSN keeps `s1/o/s2`
semantics intact — note this in a comment where the order is fixed.)

**Tests:** `veloxquant_mlx/tests/quantizers/test_nsnquant.py` (≈11 tests)

- NSN round-trip exactness (no VQ): `nsn_inverse(nsn_transform(x)) ≈ x`
  to fp16 rtol.
- Post-NSN statistics: per-token norm ≈ √d; channel means ≈ 0 (before the
  second normalize's small perturbation — assert loose tolerance, cite the
  paper's "negligible deviation" note in the test docstring).
- Hadamard forward/inverse round-trip on `(B,H,T,64)` and `(B,H,T,128)`.
- Codebook determinism: two calls, same seed → identical arrays; different
  seed → different.
- Codebook shape/norm: `(256, 8)`, unit-norm rows; magnitude variant is
  non-negative.
- 2-bit encode/decode round-trip cosine similarity on Gaussian input above a
  fixed floor (calibrate the floor empirically once, then pin it — same
  practice as xKV's reconstruction tests).
- 1-bit round-trip cosine floor (lower than 2-bit's).
- 2-bit beats 1-bit on identical input (mean cosine strictly greater).
- Full pipeline (NSN + Hadamard + VQ + inverse) reconstruction error on
  synthetic *non*-Gaussian input (e.g. heavy-tailed / outlier-channel input,
  the case NSN exists for) is materially better than the same VQ without NSN
  — this is the mechanism-validation test, analogous to xKV's
  shared-structure-helps test.
- `d % 8 != 0` raises `ValueError`.
- Odd shapes: `T=1` (single decode token) works through the full pipeline.

---

## Phase 2 — Cache wrapper

**File:** `veloxquant_mlx/cache/nsnquant_cache.py`

`NSNQuantCache` — **single-layer wrapper, no coordinator** (simplest wrapper
shape in the repo; model on `kivi_cache.py`, not on the xKV/MiniCache
coordinator families).

Constructor: `(head_dim, bits=2, residual_length=64, codebook_size=256,
subvector_dim=8, seed=1234, max_ctx=8192)`.

Behavior contract:

- `update_and_fetch(keys, values)` receives post-RoPE keys. Both K **and** V
  go through NSN + Hadamard + VQ (unlike SVDq/xKV which are keys-only —
  mirror the paper here; it treats both).
- **Prefill** (T > residual_length): quantize the first
  `T - (T % residual_length)` tokens... no — simpler and matching KIVI's
  idiom: quantize `floor(T / residual_length) * residual_length` tokens as
  chunks of `residual_length` (each chunk gets its own `o`; `s1`/`s2` per
  token), keep the remainder in the fp16 residual buffer.
- **Decode**: append to the residual buffer; when it reaches
  `residual_length`, flush it through the pipeline as one chunk and reset.
  No cross-step state other than the stored chunks — no frozen statistics,
  each chunk is self-contained (survey adaptation decision #5).
- **Fetch** returns dequantized chunks concatenated with the fp16 residual —
  same contract as every quantizing wrapper in the repo.
- Byte accounting properties (match the repo's existing `*_bytes` naming
  conventions — read two existing wrappers and copy the exact property names):
  - quantized payload: `T_q * d * bits / 8` per tensor (2-bit: signs+idx =
    2 bits/element; 1-bit: idx = 1 bit/element),
  - metadata: fp16 `s1`+`s2` per token (`2 * 2 * T_q` bytes) + fp16 `o` per
    chunk (`2 * d * n_chunks`), **counted, not waved away** (survey decision
    #4),
  - residual buffer at fp16.
- `max_ctx` guard consistent with other wrappers (raise beyond it).
- Non-attention layers / incompatible shapes fall back to the fallback cache
  exactly like `_build_xquant`/`_build_kivi` do.

**Tests:** `veloxquant_mlx/tests/cache/test_nsnquant_cache.py` (≈14 tests)

- Prefill-then-fetch reconstruction cosine floor (2-bit and 1-bit).
- Chunking arithmetic: T = k·residual_length ± 1 edge cases; residual buffer
  holds exactly `T mod residual_length` tokens.
- Decode accumulation: token-by-token decode across ≥2 flush boundaries;
  fetch length always equals total tokens pushed.
- Flush determinism: same tokens pushed prefill-style vs decode-style yield
  identical quantized state (chunk boundaries identical by construction).
- Per-chunk independence: statistics of chunk i unaffected by later chunks
  (quantize chunk 1, record; push more; re-fetch — chunk 1 bytes unchanged).
- Byte accounting: measured `*_bytes` matches the closed-form above;
  2-bit ≈ 2× the payload of 1-bit; compression ratio vs fp16 crosses the
  expected threshold at T ≫ residual_length.
- Values quantized too (V reconstruction floor, not just K).
- `max_ctx` guard raises.
- Fallback path for non-attention layers unaffected.
- Determinism: same input twice → identical fetched output.
- `for_model`-level wiring smoke test (see Phase 3) with a toy multi-layer
  model config: every attention layer gets an `NSNQuantCache`.

---

## Phase 3 — Config + builder wiring

**File:** `veloxquant_mlx/cache/base.py`

- Add `"nsnquant"` to the `method` Literal (line ~39, after `"xkv"`).
- New config fields, grouped with a `# --- NSNQuant configuration ---`
  comment block exactly like GEAR's/xKV's blocks:
  - `nsn_bits: int = 2`            # 2 = signs+index, 1 = index only
  - `nsn_residual_length: int = 64`  # fp16 buffer; paper recommends 128 for 1-bit
  - `nsn_codebook_size: int = 256`
  - `nsn_subvector_dim: int = 8`
  - `nsn_seed: int = 1234`
  - `nsn_max_ctx: int = 8192`
- Factory dispatch: `elif config.method == "nsnquant": return
  KVCacheBuilder._build_nsnquant(layers, args, config, _FallbackCache)`.
- `_build_nsnquant` — model on `_build_kivi` (single-layer, no coordinator):
  one `NSNQuantCache` per attention-bearing layer, fallback cache elsewhere.
- Export from `veloxquant_mlx/cache/__init__.py` and (if the repo re-exports
  quantizer primitives) `veloxquant_mlx/quantizers/__init__.py` — copy
  whatever export pattern xKV used in commit 7424562.

Validation: `nsn_bits ∈ {1, 2}`, `head_dim % nsn_subvector_dim == 0`,
`is_hadamard_compatible(head_dim)` — raise `ValueError` at build time with a
clear message, not at first update.

---

## Phase 4 — Benchmark

**File:** `benchmark_scripts/benchmark_nsn.py` (+ committed
`benchmark_scripts/nsn_benchmark_results.json`)

Offline-synthetic harness, no model download — same pattern as
`benchmark_xkv.py`:

- Synthetic K/V with realistic structure (reuse the outlier-channel /
  heavy-tailed generators from prior benchmarks if they exist as shared
  helpers; otherwise inline the same recipe `benchmark_xkv.py` used).
- Report, at matched bytes/token:
  - NSNQuant-2b and -1b reconstruction cosine/MSE **with and without the NSN
    step** (ablation — the mechanism's whole claim),
  - versus KIVI (bit_width matched) and one geometric-VQ baseline already in
    the repo (RaBitQ or VecInfer) at their nearest byte-matched configs,
  - bytes/token breakdown: payload vs metadata vs residual, so the fp16
    metadata overhead (survey decision #4) is visible, not hidden.
- Print + JSON-dump. Commit the JSON the run actually produces. **Explicitly
  NOT a model-level perplexity/throughput benchmark** — say so in the file
  docstring and the CHANGELOG, exactly like xKV's benchmark did.

---

## Phase 5 — Docs site

- `docs-site/docs/algorithms/nsnquant.md` — follow `xkv.md`'s structure
  exactly: what it is, the NSN+Hadamard+universal-codebook mechanism (with
  the restoration formula), config table (all `nsn_*` fields), usage snippet,
  "Adaptation notes / What we do NOT implement" (survey §adaptation, all five
  points), benchmark table from the committed JSON, citation block
  (arXiv:2505.18231, NeurIPS 2025).
- `docs-site/sidebars.ts`: add `'algorithms/nsnquant'` in the quantization
  group (near `kivi`/`polarquant`, NOT in cross-layer).
- `docs-site/docs/algorithms/overview.md`: add row; bump any "30 methods"
  phrasing to 31.
- Cross-link: from `kivi.md` (residual-buffer sibling) and one VQ page
  (`rabitq.md` or `vecinfer.md`) with a one-line "differs because it adapts
  the data to a fixed codebook, not the codebook to the data".
- Docs changelog page: 0.28.0 entry.

---

## Phase 6 — CHANGELOG, README, EVIDENCE_TABLE, version bump

- `CHANGELOG.md` `[0.28.0]`: feature summary + **"Honest scope"** section
  (post-RoPE adaptation, no value-projection fusion, k-means-only codebook —
  no gradient fine-tune, fp16 metadata — no double quantization, no fused
  kernels, offline-synthetic benchmark only).
- `README.md`:
  - summary paragraph + method table row (31st method, quantization family),
  - method count 30 → 31 everywhere it appears (grep for `30`
    with context before replacing — do not blind-replace),
  - `EVIDENCE_TABLE.md` row (what's validated: reconstruction + byte
    accounting on synthetic; what's NOT: model-level quality),
  - Sources: NSNQuant citation (Son, Choi, Yoo, NeurIPS 2025,
    arXiv:2505.18231).
- `pyproject.toml`: `version = "0.28.0"`.

---

## Phase 7 — Landing page

All zero-build static files; edit `landing/`, deploy copies via the existing
`cp -r landing/* dist/` flow.

- `landing/index.html`:
  - hero pill `30 algorithms` → `31 algorithms` (line ~60) and
    `See all 30 algorithms` → 31 (line ~66); add NSNQuant to the expanded
    list inside that `<details>`,
  - `<meta name="description">`: append NSNQuant-adapted + "New in 0.28.0"
    sentence (replace the 0.27.0 "New in" clause, keep the method roll-call
    updated),
  - What's-new list (line ~205): new `0.28.0` `<li>` **above** the 0.27.0
    entry, same markup shape (`wn-ver` / `wn-body`),
  - Method Library: picker card + full algo card in the **quantization**
    group (copy an existing VQ card's markup; card-headline stat =
    `universal codebook · calibration-free · N/N tests` with the real final
    test count; card-meta `v0.28.0 · NeurIPS 2025`),
  - code tab: config snippet `method="nsnquant"`,
  - **fix the stale test-count line ~2004** `817/821 tests passing` → the
    real post-implementation collected/passing numbers (844 + new tests;
    run pytest and use actual figures),
  - provenance strip (line ~103): add `NeurIPS 2025 <em>NSNQuant</em>` to the
    existing NeurIPS group → `KVQuant · MiniCache · ZipCache` stays 2024;
    add a new `NeurIPS 2025` item after `ICML 2025` keeping chronological
    order.
- `landing/assets/main.js` line ~60: hero badge text →
  `"v0.28.0 — NSNQuant-adapted calibration-free universal-codebook VQ shipped"`.
- No CSS changes expected (cards and strip already wrap); if a new stat card
  is added instead of edited, reuse `.stat-card` markup as-is.

---

## Phase 8 — Final verification

- `pytest veloxquant_mlx/tests/ -q` — full suite green (expect ≈869: 844 + ~25
  new). Record the exact number; it feeds the landing page + release notes.
- `python -m build` (sdist+wheel) succeeds; `twine check dist/*` passes.
- Docs site builds (`npm run build` in `docs-site/`) with the new page.
- Grep sweep before finishing: `grep -rn "30 algorithms\|0\.27\.0" landing/
  README.md docs-site/docs/ | grep -v CHANGELOG` — nothing stale left except
  historical changelog entries.
- Re-run `benchmark_scripts/benchmark_nsn.py`; confirm committed JSON matches
  the code that ships.

---

## Phase 9 — Release commands (for the maintainer to run — NOT the agent)

Per repo convention the maintainer reviews and runs git/publish commands
themselves. Provide these in the final report, filled in with real numbers:

```bash
# 1. Stage everything shipped in this release
git add veloxquant_mlx/quantizers/nsnquant.py \
        veloxquant_mlx/cache/nsnquant_cache.py \
        veloxquant_mlx/cache/base.py \
        veloxquant_mlx/cache/__init__.py \
        veloxquant_mlx/quantizers/__init__.py \
        veloxquant_mlx/tests/quantizers/test_nsnquant.py \
        veloxquant_mlx/tests/cache/test_nsnquant_cache.py \
        benchmark_scripts/benchmark_nsn.py \
        benchmark_scripts/nsn_benchmark_results.json \
        docs-site/docs/algorithms/nsnquant.md docs-site/sidebars.ts \
        docs-site/docs/algorithms/overview.md \
        README.md CHANGELOG.md pyproject.toml \
        paper/NEW_METHOD_SURVEY_V11.md paper/IMPLEMENTATION_PROMPT_NSNQUANT.md \
        paper/EVIDENCE_TABLE.md \
        landing/index.html landing/assets/main.js

# 2. Commit (fill in final test count and benchmark deltas from the run)
git commit -m "feat(nsnquant): NSNQuant-adapted calibration-free universal-codebook VQ — v0.28.0

<body: follow the 7424562 xKV commit-message structure — code / tests /
benchmark / docs / honest-scope sections, real numbers only>

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

# 3. Tag + GitHub release
git tag -a v0.28.0 -m "v0.28.0 — NSNQuant-adapted calibration-free universal-codebook VQ"
git push origin master --follow-tags
gh release create v0.28.0 --title "v0.28.0 — NSNQuant-adapted" \
  --notes "<release notes: what shipped, honest scope, benchmark summary
from the committed JSON, full test count — draft these from the CHANGELOG
entry>"

# 4. PyPI
rm -rf dist/ && python -m build
twine check dist/*
twine upload dist/*   # uses ~/.pypirc / TWINE_* credentials
```

---

## Appendix — file manifest (new files this release)

- `veloxquant_mlx/quantizers/nsnquant.py`
- `veloxquant_mlx/cache/nsnquant_cache.py`
- `veloxquant_mlx/tests/quantizers/test_nsnquant.py`
- `veloxquant_mlx/tests/cache/test_nsnquant_cache.py`
- `benchmark_scripts/benchmark_nsn.py`
- `benchmark_scripts/nsn_benchmark_results.json`
- `docs-site/docs/algorithms/nsnquant.md`
- `paper/NEW_METHOD_SURVEY_V11.md` (already written)
- `paper/IMPLEMENTATION_PROMPT_NSNQUANT.md` (this file)

## Appendix — files modified this release

- `veloxquant_mlx/cache/base.py` (method literal, config fields, factory,
  `_build_nsnquant`)
- `veloxquant_mlx/cache/__init__.py`, `veloxquant_mlx/quantizers/__init__.py`
- `docs-site/sidebars.ts`, `docs-site/docs/algorithms/overview.md`,
  docs changelog, `kivi.md` + one VQ page (cross-links)
- `README.md` (summary, table, count, Sources), `paper/EVIDENCE_TABLE.md`
- `CHANGELOG.md`, `pyproject.toml` (0.27.0 → 0.28.0)
- `landing/index.html` (hero pill ×2, meta description, what's-new,
  provenance strip, Method Library cards, code tab, stale test-count fix),
  `landing/assets/main.js` (hero badge)
