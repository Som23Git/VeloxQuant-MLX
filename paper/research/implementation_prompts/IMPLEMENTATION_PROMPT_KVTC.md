# Implementation Prompt — KVTC-adapted (v0.35.0)

Execute-cold spec for the **38th** method. This single prompt covers the whole
release: core logic, tests, benchmark, docs, README + hero-pill/count bumps, a
new landing card, and the **git tag + GitHub release + PyPI publish** layer.
Do all of it.

> **Venue confirmed (re-verify live before citing):** "KV Cache Transform
> Coding for Compact Storage in LLM Inference" (NVIDIA) — **ICLR 2026**
> (accepted, poster), arXiv:2511.01815, OpenReview poster
> `iclr.cc/virtual/2026/poster/10008708`. Re-check the arXiv abstract + the
> OpenReview listing live before writing it into README/docs/EVIDENCE_TABLE.

Chosen over the survey's carried-forward pick **NestedKV** because NestedKV
(arXiv:2605.26678) is still a bare preprint with **no verified venue** as of
this survey — the same condition that correctly deferred MorphKV across
V13–V15 and NestedKV itself across V16–V18. KVTC has a peer-reviewed venue
(ICLR 2026) **and** a genuinely new mechanism axis (DP-optimal per-component
bit allocation + entropy coding) not covered by STAR-KV (rejected this
survey — overlaps Palu/SVDq's adaptive-rank + mixed-precision axes).

Model it on the **Palu / SVDq pair** already shipped
(`quantizers/palu.py`+`cache/palu_cache.py`,
`quantizers/svdq.py`+`cache/svdq_cache.py`) — local (per-sequence) SVD/PCA
fit at prefill, latent projection, mixed-bit quantization of latent
channels. Ship it as **"KVTC-adapted (VeloxQuant-MLX implementation)," ICLR
2026 (arXiv:2511.01815) — NOT a faithful port.**

---

## The mechanism gap (why this is not a duplicate)

The repo has three low-rank / spectral methods:

- **Palu** — group-head SVD on **both K and V**; latents mixed-bit quantized
  with a **fixed top-25%/75% split** by singular-value magnitude.
- **SVDq** — keys-only SVD (values stay fp16); same **fixed top-25%/75%**
  mixed-bit split.
- **SpectralQuant** — eigendecomposition into a **binary** signal/noise split
  via participation ratio (`d_eff`), **uniform bits within each half**
  (`b_signal = b_noise = 3` in the paper's primary config), plus a JL-sketch
  residual on the signal half's quantization error.

**All three use a fixed, hand-chosen split** (25/75, or a PR-derived binary
cutoff) — none compute a **provably optimal** allocation for a given total-bit
budget, and none can assign **zero bits** to an individual low-variance
component while giving another component more than the "high" tier gets.
`ratequant`'s waterfilling allocator (`allocators/ratequant.py`,
`spectral/bit_allocator.py::water_fill_bits`) is the closest existing idea —
but it is a **closed-form, continuous, per-layer** proportional allocator
(Theorem 2 reverse waterfilling), not a **discrete, per-component,
DP-optimal** allocator, and it never zeroes an allocation outright.

KVTC's axis: given a vector of per-component variances (from the local PCA)
and a **total bit budget**, use **dynamic programming** to choose an integer
bit-width per component (including **0**, i.e. drop the component entirely)
that minimizes total expected distortion subject to the budget — then
**entropy-code** the resulting quantized codes for a further lossless size
reduction. Neither the DP-optimal discrete allocation nor the entropy-coding
stage exists anywhere in the repo.

### The isolable reduction (must pin exactly)

Design the DP allocator so that at **uniform per-component variance** (a
degenerate all-components-equal input), the DP-optimal allocation reduces to
**uniform bits per component** (each surviving component gets
`floor(budget / n_components)`, remainder distributed to the first
components) — i.e. it collapses to the same allocation a naive uniform
splitter would produce when there is no variance signal to exploit. Pin this
with a dedicated test (the analogue of SVDq's fixed-split baseline, MorphKV's
`window=1`==TOVA, and KVzip's `probe="latest"`==TOVA collapses). Do **NOT**
claim any other collapse (e.g. to SVDq's fixed 25/75 split) — that split is
a *different, non-optimal* allocation and the DP allocator should provably
beat it whenever variance is non-uniform; assert that inequality instead
(see Phase 5).

---

## Non-negotiable honesty constraints (repeat on EVERY surface)

- **Local (per-sequence) PCA, not the paper's pre-calibrated global basis.**
  The paper fits one PCA basis offline on a calibration corpus and reuses it
  for all future caches at inference. This library has no calibration
  pipeline wired into `KVCacheBuilder.for_model`, so the basis is fit
  **online from the sequence's own prefill keys/values** — the same
  "fit-locally, no calibration set" limitation SVDq already documents. State
  this explicitly wherever the paper's calibration is mentioned.
- **DP allocator optimizes an analytic distortion proxy, not a
  real-activation-fit rate-distortion model.** The DP is exact and real
  (correctly finds the budget-constrained minimum of the proxy); what's a
  proxy is the *objective it minimizes* — reuse the repo's existing
  `fit_distortion_curve` / `α·β^(-b)` Gaussian-quantization distortion curve
  from `allocators/ratequant.py` rather than inventing or claiming a
  model-fit rate-distortion curve.
- **Entropy coding is real and measured, not assumed at the Shannon bound.**
  Implement an actual order-0 entropy coder (Huffman or range coding is
  fine) over the quantized component codes and report the **realized**
  post-entropy-coding byte count, not the theoretical entropy lower bound.
  State plainly that this is a simple order-0 coder, not the paper's (if
  more sophisticated) scheme.
- **Both K and V**, mirroring Palu, not keys-only like SVDq — the paper
  compresses both.
- **Not path-dependent** (unlike the eviction family H2O/TOVA/MorphKV/KVzip):
  the PCA basis and DP allocation are fixed once at prefill and reused for
  every subsequent token. State this contrast explicitly and pin it with a
  determinism test.
- **Not the paper's full algorithm.** We implement the *local-PCA + DP-optimal
  per-component bit allocation + order-0 entropy coding* pipeline, NOT the
  paper's pre-calibrated global basis, NOT its real-activation rate-distortion
  model, NOT any trained-model evaluation. The paper's "up to 20× (up to 40×
  in some regimes) compression at <1pp accuracy loss on LLaMA 3 / Mistral NeMo
  / R1-Qwen2.5 1.5B–70B across AIME25/GSM8K/LiveCodeBench/LongBench/MATH-500/
  MMLU/Qasper/RULER" numbers are the PAPER's on trained models — **never
  quote them as ours.**
- **Clean mechanism observable** = at a **matched total byte budget**, compare
  reconstruction distortion (MSE / cosine similarity) of the DP allocator
  against a fixed-uniform-bits baseline and against SVDq's fixed top-25%/75%
  split, on a planted non-uniform-variance geometry — the DP allocator should
  win because it can zero low-variance components instead of paying a uniform
  floor. Report entropy-coding's realized additional compression as a
  secondary, honestly-scoped result — modest on synthetic data, do not oversell.
- Nothing here is validated on a trained model — offline-synthetic only.

---

## Phase 1 — Survey (already written)

`paper/research/surveys/NEW_METHOD_SURVEY_V18.md` is written. Confirm it
still matches the implementation before shipping (mechanism gap, honesty
crux, uniform-variance-collapse reduction, matched-budget-distortion
observable). Re-verify sources live before citing in README/docs.

## Phase 2 — `veloxquant_mlx/allocators/kvtc_dp.py`

New small module — the DP bit allocator, reusable by the quantizer.

- `dp_allocate_bits(variances: np.ndarray, total_bit_budget: int, bit_choices: tuple[int,...] = (0,1,2,3,4,6,8)) -> np.ndarray`
  — for each component `i` with variance `v_i`, distortion at `b` bits is
  the reused analytic proxy `distortion(v_i, b) = v_i * BETA ** (-b)` for
  `b > 0` and `v_i` (full variance retained as error) for `b == 0`. DP over
  components × cumulative-budget-so-far, `O(n_components * total_bit_budget *
  len(bit_choices))`, minimizing `sum(distortion(v_i, b_i))` subject to
  `sum(b_i) <= total_bit_budget`. Return the integer bit-width per
  component, `shape [n_components]`, values from `bit_choices`, **may be 0**.
  Import `BETA`/`fit_distortion_curve` default from `allocators/ratequant.py`
  rather than redefining it — one canonical distortion curve.
- **Uniform-variance collapse:** when all `variances` are equal, must reduce
  to `floor(total_bit_budget / n_components)` per component (remainder to the
  first `total_bit_budget % n_components` components, in index order) —
  make this a property of the DP formulation, not a special-cased branch, if
  possible; if a branch is simpler and clearer, add it but assert the
  equivalence with a test either way.
- Guards: `total_bit_budget >= 0`, `variances` non-negative, `len(variances)
  >= 1`; if `total_bit_budget == 0` all components get `0` bits.

## Phase 3 — `veloxquant_mlx/quantizers/kvtc.py`

Mirror the shape and docstring discipline of `quantizers/svdq.py` /
`quantizers/palu.py`.

- `kvtc_compress(tensor: mx.array, total_bit_budget: int) -> KVTCArtifact` —
  local PCA (reuse `svd_compress_keys`-style local SVD, or factor a shared
  helper — read `svdq.py` first and reuse rather than duplicate): center,
  SVD, project to latents `[S, r]` where `r = min(S, D)` (no fixed-energy
  truncation — the DP allocator itself decides how many components survive
  by assigning some `0` bits), get per-component variance from singular
  values, call `dp_allocate_bits`, quantize each component at its assigned
  bit-width (`0`-bit components are dropped from storage entirely — do not
  store a zero-filled placeholder), then **entropy-code** the quantized
  codes for the surviving components (order-0 coder — see Phase 4).
- `kvtc_decompress(artifact) -> mx.array` — entropy-decode, dequantize each
  surviving component, zero-fill dropped components, un-project
  (`latents @ V.T + mean`), return `[S, D]` fp16/fp32.
- `KVTCArtifact` dataclass: `V` (projection `[D, r]`), `mean` (`[D]`),
  `bit_allocation` (`[r]`, may contain zeros), `entropy_coded_bytes` (per
  surviving component, or one combined blob — document which), `n_survived`.
- Byte-accounting helpers: `kvtc_fp16_bytes` (V + mean + realized
  entropy-coded payload, NOT the pre-entropy-coding fixed-width size), and a
  `kvtc_pre_entropy_bytes` companion so the benchmark can report entropy
  coding's realized delta.
- `__all__` exports.

**Uniform-variance collapse test target:** `dp_allocate_bits` on a synthetic
uniform-variance vector reduces exactly to the uniform-per-component split
described above — pin it here or in Phase 5, not both redundantly.

## Phase 4 — `veloxquant_mlx/quantizers/_entropy_coding.py`

Small, self-contained order-0 entropy coder — do not pull in an external
dependency.

- `entropy_encode(codes: np.ndarray) -> tuple[bytes, dict]` — build a
  frequency table over the observed integer codes, Huffman-code them (stdlib
  `heapq`, no external deps), return the encoded bitstream + the code table
  (the table must be counted in the byte accounting — do not hide its cost).
- `entropy_decode(payload: bytes, table: dict, n: int) -> np.ndarray` —
  exact inverse; round-trip must be lossless (pin with a test).
- Document plainly: **order-0, static per-call table** — not adaptive, not
  the paper's (possibly more sophisticated) scheme. Table overhead is
  real and included in `kvtc_fp16_bytes`.

## Phase 5 — `veloxquant_mlx/cache/kvtc_cache.py`

`KVTCKVCache(_MLXKVCache)` modeled on `PALUKVCache` / `SVDqKVCache`:

- Consume `kvtc_bit_budget` (e.g. total bits per token across all
  components — pick a sane default, document the units clearly, e.g. "total
  bits per token for K, separately for V").
- Fit local PCA once at prefill (first `update_and_fetch` call with `S > 1`),
  store `V`/`mean`/`bit_allocation` as layer state; every subsequent decode
  token is projected through the **same fixed basis and allocation** (not
  path-dependent — state this).
- Applies to **both K and V** independently (mirror Palu, not SVDq's
  keys-only scope).
- No `.bits` in the eviction-cache sense (this isn't an eviction cache).
  Byte props: `kvtc_bytes`, `full_seq_bytes`, `compression_ratio`,
  `pre_entropy_bytes`, `entropy_coding_gain` (`pre_entropy_bytes /
  kvtc_bytes`).
- Validate at construction (delegate to the quantizer's guards).
- Determinism: same input sequence → same basis, same allocation, same
  codes, same decompressed output — pin with a test (the "not path-dependent"
  contrast with the eviction family).

## Phase 6 — `veloxquant_mlx/cache/base.py`

Add `"kvtc"` to the method `Literal`; config block (`kvtc_bit_budget`, a
documented default, e.g. `4` avg-bits-per-component equivalent scaled by
typical `head_dim`); import `KVTCKVCache`; factory branch (`elif
config.method == "kvtc"`, no coordinator); extend the unknown-method error
string with `"kvtc"`. **Read each region before editing.**

## Phase 7 — Tests (~30, match Palu/SVDq's count/discipline)

`tests/allocators/test_kvtc_dp.py` (~10):
- guards (`total_bit_budget >= 0`, non-negative variances, budget==0 → all
  zero).
- **uniform-variance collapse** — assert exact equality with
  `floor(budget/n) [+1 remainder to first components]`.
- budget respected exactly (`sum(bits) <= total_bit_budget`, and as close as
  the discrete `bit_choices` allow — assert no feasible reallocation lowers
  total distortion further, i.e. local-optimality, or assert against a
  brute-force reference on small `n`).
- monotonicity: strictly higher-variance component never gets fewer bits
  than a lower-variance one at the optimum (or document/pin the one
  legitimate tie-breaking exception).
- can assign exactly `0` to a component (planted near-zero-variance
  component with a tight budget).
- determinism (no RNG in the DP itself).

`tests/quantizers/test_entropy_coding.py` (~5):
- round-trip losslessness on random integer code arrays (multiple
  alphabets/sizes).
- table overhead counted in the returned byte accounting.
- degenerate single-symbol input doesn't crash.

`tests/quantizers/test_kvtc.py` (~10):
- init guards.
- **local PCA reduces to identity-ish behavior at full budget** (enough bits
  that no distortion is introduced beyond float rounding) — reconstruction
  cosine similarity ≈ 1.0.
- **DP allocator beats SVDq's fixed top-25%/75% split at a matched total
  byte budget** on a planted non-uniform-variance geometry — assert lower
  MSE / higher cosine similarity than the fixed-split baseline, over several
  seeds (a rate, not one lucky run).
- byte accounting (`kvtc_fp16_bytes` includes entropy-coded payload + table
  + V + mean; `kvtc_pre_entropy_bytes` excludes entropy gain).
- determinism (same input → same everything).
- values compressed too (not keys-only).

`tests/cache/test_kvtc_cache.py` (~5):
- factory dispatch to `KVTCKVCache`.
- construction guards; config propagation via `for_model`.
- basis/allocation fixed after prefill, reused unchanged across decode steps
  (not path-dependent) — pin explicitly.
- byte props; compression_ratio > 1 at a reasonable budget.

## Phase 8 — Benchmark

`benchmark_scripts/benchmark_kvtc.py` + committed `kvtc_benchmark_results.json`:
- SEQ_LENS + BIT_BUDGETS grid (match Palu/SVDq's scale).
- GEOMETRIES = `["skewed_variance", "flat"]` (`skewed_variance`: a planted
  geometry where a few components carry most variance — the case the DP
  allocator should win on; `flat`: null control / near-uniform variance,
  where the DP allocator should match, not dramatically beat, the fixed-split
  baseline).
- Arms: KVTC (DP-allocated), a **fixed-uniform-bits baseline**, and **SVDq's
  fixed top-25%/75% split** at the same matched total byte budget.
- Primary field: reconstruction MSE / cosine similarity at matched budget;
  secondary: entropy-coding realized gain (`pre_entropy_bytes /
  kvtc_bytes`), reported plainly, not oversold. Deterministic in ALL
  non-timing fields (only `_ms` may vary). Offline-synthetic; loads no
  model. Verify determinism by diffing two runs (non-`_ms` fields identical).

## Phase 9 — Docs

- `docs-site/docs/algorithms/kvtc.md` — full page: honesty crux (local PCA
  vs. paper's calibrated global basis; analytic distortion proxy vs.
  paper's fitted rate-distortion model; order-0 entropy coder), the
  uniform-variance-collapse reduction, the matched-budget-distortion
  observable, adaptation limitations, the paper's numbers labeled as the
  paper's.
- `docs-site/sidebars.ts` — add `'algorithms/kvtc'` after `kvzip`.
- `docs-site/docs/algorithms/overview.md` — thirty-seven→thirty-eight + table
  row + bullet. **Read before editing.**
- `docs-site/docs/changelog.md` — v0.35.0 (Latest); move v0.34.0 down.
- Cross-link from `palu.md` and `svdq.md` (fixed-split contrast) and
  `spectral.md` (binary-cutoff contrast).

## Phase 10 — README / CHANGELOG / EVIDENCE_TABLE / pyproject

- README:
  - changelog badge `0.34.0`→`0.35.0`.
  - "**thirty-seven** compression strategies"→"**thirty-eight**".
  - low-rank/spectral family count bump in whatever parenthetical list
    enumerates Palu/SVDq/SpectralQuant — extend it with "…and KVTC's
    DP-optimal per-component bit allocation + entropy coding — ICLR 2026".
  - "All **37** methods"→"All **38**".
  - method-table row after the KVzip row: KVTC-adapted / `kvtc` / "Local PCA
    + DP-optimal per-component bit allocation + entropy coding (ICLR 2026) —
    beats fixed-split mixed-precision at matched byte budget on skewed
    variance" / `0.35.0`.
  - Sources entry (ICLR 2026, arXiv:2511.01815, OpenReview poster, if
    verified live).
  - Sweep for stale `37`/`0.34.0`.
- `CHANGELOG.md` `[0.35.0] — <today>` with **Honest scope** (local PCA, not
  pre-calibrated global basis; analytic distortion proxy; order-0 entropy
  coder; paper numbers not ours); move `[0.34.0]` down.
- `paper/research/EVIDENCE_TABLE.md` — next contiguous rows.
- `pyproject.toml` — version `0.34.0`→`0.35.0`; description "...to
  KVzip"→"...to KVTC", 37→38; **preserve PEP 639 metadata** (`license="MIT"`
  + `license-files=["LICENSE"]`, one-line description, name-only author,
  `requires=["setuptools>=77","wheel"]`).

## Phase 11 — Landing page (new card + counts)

`landing/index.html` + `assets/main.js`. **Read each region before editing.**
- `<meta name="description">`: append KVTC-adapted to the roll-call; change
  "New in 0.34.0…"→"New in 0.35.0: KVTC-adapted local-PCA + DP-optimal bit
  allocation + entropy coding (ICLR 2026) — beats fixed-split mixed-precision
  at matched byte budget."
- hero pill "37 algorithms"→"38 algorithms"; "See all 37"→"See all 38".
- whats-new: add a `0.35.0` `<li>` at the top with the honest crux; keep the
  0.34.0 li below.
- Low-rank/spectral `cat-count` bump (find the right category — Palu/SVDq/
  SpectralQuant's group).
- Roll-call in the "See all" details: +KVTC-adapted.
- New algo card `#algo-kvtc` (clone the `#algo-svdq` or `#algo-palu` card
  block: `card-meta` → `v0.35.0 · ICLR 2026`, `data-tags` includes the
  low-rank/quantization tags, full-desc with the honest crux).
- Picker card + code tab button + panel `#tab-kvtc` (clone the svdq/palu ones).
- **Move the NEW pill off the KVzip card onto `#algo-kvtc`.**
- `assets/main.js` `initBadgeTyping` text → `"v0.35.0 — KVTC-adapted local
  PCA + DP-optimal bit allocation + entropy coding shipped"`.

## Phase 12 — Verify

- Full pytest (expect +~30 new; the ~6 known vecinfer Metal fp16 flakes stay
  flaky, do NOT chase). Confirm zero non-vecinfer failures and all new KVTC
  tests pass.
- `python -m build` + `python -m twine check dist/*` at `0.35.0`; inspect the
  wheel PKG-INFO (Metadata-Version 2.4, Version 0.35.0, License-Expression
  MIT, Author: Rajveer Rathod, **38**-method Summary, `kvtc.py`/
  `kvtc_cache.py`/`kvtc_dp.py`/`_entropy_coding.py` present in the wheel).
- Docs `npm run build` SUCCESS.
- Benchmark determinism (non-`_ms` fields stable across two runs).
- Grep stale-ref sweep (38 consistent across
  README/overview/pyproject/landing/JOSS; no lingering `37`/`0.34.0` in
  places that should now say `38`/`0.35.0`).
- End-to-end factory smoke test: `KVCacheConfig(method="kvtc", …)` →
  `KVCacheBuilder.for_model` → a `compression_ratio > 1` at a reasonable
  budget, on both K and V.

---

## Phase 13 — Release layer (provide as CHAT TEXT ONLY — never execute)

**Standing rule:** the user reviews and runs all git/publish commands
themselves. After implementation, give the v0.35.0 release sequence as **chat
text** for them to run — do NOT execute any `git add`/`commit`/`tag`/`push`/
`gh release`/`gh repo edit`/`twine` yourself. Mirror the exact format used for
the v0.34.0 KVzip release:

1. **Branch:** `git checkout -b release/v0.35.0`
2. **Stage only KVTC paths** (explicit `git add` list — do NOT stage the
   unrelated working-tree noise: `dist_preview/`, `blog_drafts/`,
   `dist_pypi/`, `.claude/`). Include:
   `veloxquant_mlx/allocators/kvtc_dp.py`,
   `veloxquant_mlx/quantizers/kvtc.py`,
   `veloxquant_mlx/quantizers/_entropy_coding.py`,
   `veloxquant_mlx/cache/kvtc_cache.py`, `veloxquant_mlx/cache/base.py`, all
   new test files, `benchmark_scripts/benchmark_kvtc.py` +
   `kvtc_benchmark_results.json`, `docs-site/docs/algorithms/kvtc.md`,
   `docs-site/sidebars.ts`, `overview.md`, `palu.md`, `svdq.md`,
   `spectral.md`, `changelog.md`, `README.md`, `CHANGELOG.md`,
   `paper/research/EVIDENCE_TABLE.md`,
   `paper/research/surveys/NEW_METHOD_SURVEY_V18.md`,
   `paper/research/implementation_prompts/IMPLEMENTATION_PROMPT_KVTC.md`,
   `pyproject.toml`, `landing/index.html`, `landing/assets/main.js`.
3. **Commit:** `git commit -F-` heredoc, **NO Co-Authored-By line**, subject
   `feat(kvtc): KVTC-adapted local-PCA + DP-optimal bit allocation +
   entropy coding — v0.35.0`, body covering the mechanism, the honest scope
   (local PCA not pre-calibrated; analytic distortion proxy; order-0 entropy
   coder; paper numbers not ours).
4. **Tag:** `git tag -a v0.35.0 -m "..."`.
5. **Push:** `git push -u origin release/v0.35.0` then `git push origin v0.35.0`.
6. **Build + check:** `rm -rf dist build *.egg-info && python -m build &&
   python -m twine check dist/*`.
7. **PyPI:** `python -m twine upload dist/veloxquant_mlx-0.35.0*`.
8. **GitHub release:** `gh release create v0.35.0 --repo
   rajveer43/VeloxQuant-MLX --title "..." --notes "$(cat <<'EOF' … EOF)"` with
   KVTC release notes (mechanism table, honest scope, usage snippet).
   Escape code fences in the heredoc as `` \`\`\` ``.

## What we do NOT implement (state plainly)

- The paper's pre-calibrated global PCA basis fit across a calibration
  corpus (local per-sequence PCA instead — the same limitation SVDq already
  documents).
- The paper's rate-distortion model fit on real model activation statistics
  (the repo's existing analytic Gaussian distortion-curve proxy instead).
- A sophisticated adaptive/context-modeled entropy coder (a simple order-0
  Huffman/range coder instead).
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers (up to 20×, up to 40× in some regimes, <1pp accuracy loss
  on LLaMA 3 / Mistral NeMo / R1-Qwen2.5 1.5B–70B across eight benchmarks)
  are the paper's — not reproduced here.
