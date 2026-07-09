# Full Autonomous Implementation Prompt — Q-Filters-adapted (v0.31.0)

**Purpose of this document:** a self-contained, execute-without-supervision
spec for shipping VeloxQuant-MLX's 34th method end-to-end: survey (done, see
`paper/NEW_METHOD_SURVEY_V14.md`) → eviction primitives → cache wrapper →
config/builder wiring → tests → benchmark → docs → changelog → README →
landing page → version bump → release/publish commands. Written so an agent
picking this up cold can complete every step and leave the repo in the same
finished state as every prior release (0.19.0 through 0.30.1).

**Do not deviate from the "adapted, not faithful port" discipline**: label
everything **"Q-Filters-adapted (VeloxQuant-MLX implementation)"**, report
only numbers from a committed results JSON you generate yourself, never
repeat the paper's headline numbers as if measured here, and document every
simplification plainly.

**One extra rule specific to this method — the honesty crux.** The paper's
filter direction is estimated **offline from query vectors**. A cache-side
library *never sees query vectors* — only the K/V passed to
`update_and_fetch`. Our adaptation estimates the filter direction from the
**observed key stream** instead (top right-singular vector of the first
observed chunk). This is a **genuine departure from the paper's mechanism,
not a shortcut.** Every doc, the module docstring, the CHANGELOG "Honest
scope", and the EVIDENCE_TABLE must state it in these terms: *"the paper
derives the filter from query-distribution SVD offline; we derive it from the
first observed key chunk's SVD — a different estimator of the same
head-geometry direction, validated here only under constructed geometry,
never claimed equivalent to the paper's."* The benchmark must include the
**isotropic control** where the method shows *no* advantage, and a
**filter-cosine** field measuring whether the key-derived substitute actually
recovers the intended direction.

Read `paper/NEW_METHOD_SURVEY_V14.md` first. If this checklist conflicts with
the survey, the survey wins.

---

## 0. Ground rules (apply to every phase)

- One buildable commit per phase, tests green before proceeding
  (`pytest veloxquant_mlx/tests/ -x -q`; suite is **934 tests before this
  work, 928 passing** — the 6 failures are pre-existing machine-dependent
  vecinfer Metal fp16-tolerance flakes (4× `test_vecinfer_fused_sdpa`, 2×
  `test_vecinfer_metal_parity`), also failing on clean master; do not chase
  them).
- Every new Python file gets a module docstring citing **arXiv:2503.02812
  (Q-Filters, preprint)**, stating "adapted, not faithful port", and listing
  what's NOT implemented.
- No new dependencies. **Model structurally on the KNorm pair**
  (`quantizers/knorm.py`, `cache/knorm_cache.py`) — the closest existing
  method: a single-layer, no-coordinator, fp16, score-and-evict cache with a
  frozen-at-insertion scalar score. Same State-dataclass / init / update /
  get_kv / bytes API shape, same per-(B,H) state list in the wrapper, same
  byte-accounting property names (`*_kept_bytes`, `full_seq_bytes`,
  `compression_ratio`, `tokens_seen`, `tokens_kept`).
- **No per-token Python loop** — each head's whole `[S, D]` block goes through
  one `qfilters_update` call (scores are a single matmul against the frozen
  direction).
- Naming: `qfilters.py`, `qfilters_cache.py`, `test_qfilters.py`,
  `test_qfilters_cache.py`, `benchmark_qfilters.py`, `qfilters.md`,
  `qfilters_benchmark_results.json`. Config prefix `qfilters_`, method string
  `"qfilters"`, display name **Q-Filters-adapted**.

---

## Phase 1 — Eviction primitives

**File:** `veloxquant_mlx/quantizers/qfilters.py`

Mirror `knorm.py`'s public API, with a **filter direction** estimated once
(from the first observed chunk) and then frozen:

1. `QFiltersState` dataclass — `keys [n_kept, D] fp16 | None`,
   `values | None`, `scores [n_kept] float32 | None` (projection score of each
   kept key onto the frozen filter, computed once at insertion, never
   updated), `filter_dir [D] float32 | None` (frozen query-agnostic
   direction; `None` until estimated), `n_sink`, `budget`, `recent`,
   `calib_tokens` (min tokens observed before the filter is estimated;
   default 128), `sign` (`+1` paper-faithful | `-1` inverted ablation arm).
2. `init_qfilters_state(n_sink, budget, head_dim, recent=0, calib_tokens=128,
   sign=1)`. Guards at init: `n_sink + recent >= budget` → `ValueError`;
   `sign not in {1, -1}` → `ValueError`. If `calib_tokens > budget`, warn
   (the cache can transiently exceed budget before the filter is frozen).
3. `estimate_filter_dir(keys [N, D]) -> [D]` — **the honest core.** Compute
   the top right-singular vector of the mean-centered observed key block
   (equivalently the top eigenvector of the `[D, D]` key covariance; use `mx`
   SVD/eigh — no new deps). Sign-normalize deterministically (force the
   largest-magnitude component positive) so the direction is reproducible.
   Docstring must state: *paper estimates this from query-distribution SVD
   offline; we estimate it from observed keys — a documented deviation.*
4. `qfilters_update(state, new_keys [S, D], new_values [S, D]) ->
   QFiltersState` — vectorized:
   - concatenate incoming K/V onto state;
   - **if `filter_dir is None` and total observed ≥ `calib_tokens`**: estimate
     and freeze `filter_dir` from all keys observed so far, then compute
     `scores = sign · (keys · filter_dir)` for every stored token. Before the
     filter is estimated, keep everything (passthrough) — no eviction until
     the direction exists;
   - for incoming tokens once the filter is frozen: `new_scores = sign ·
     (new_keys · filter_dir)`, concatenate onto `scores`;
   - if `n_total > budget`: protect sinks (first `min(n_sink, n_total)`) and
     the trailing `recent` positions (set their score to `+inf`), keep the
     `budget` **highest-scoring** positions **in original temporal order**
     (sort kept indices ascending before gathering). *(Paper: high projection
     ⇒ high predicted attention ⇒ keep. `sign=-1` inverts for the control.)*
5. `qfilters_get_kv(state)` — identical contract to `knorm_get_kv` (zero-row
   `[0, 1]` placeholders before first update).
6. `qfilters_fp16_bytes(state)` / `full_qfilters_fp16_bytes(tokens_seen,
   head_dim)` — identical formulas to the KNorm versions, **plus** the frozen
   `filter_dir`: `D · 4` bytes (float32) per head, added once the filter
   exists. Byte accounting counts it in full (same discipline as SKVQ's
   permutation tables).

**Two invariants to document in the module docstring (and test):**
- **Frozen-filter determinism:** once `filter_dir` is estimated, a token's
  stored score never changes across subsequent updates.
- **Path-DEPENDENCE, stated honestly:** unlike KNorm, the kept set is **not**
  path-independent — the filter is estimated from whichever chunk crosses
  `calib_tokens` first, so prefill-in-one-block and token-by-token decode can
  estimate *different* directions and diverge. **Do not claim or test
  bit-for-bit prefill/decode equivalence.** Test the weaker true property:
  *given the same frozen `filter_dir`, scoring and eviction are
  order-invariant.* This is the honest contrast with KNorm — call it out.

**Tests:** `veloxquant_mlx/tests/quantizers/test_qfilters.py` (~14)

- Under budget: passthrough, all kept in order.
- Pre-calibration passthrough: fewer than `calib_tokens` seen ⇒ nothing
  evicted even above budget (filter not yet frozen).
- `estimate_filter_dir` recovers a planted dominant direction: keys with
  variance concentrated on one axis ⇒ `cosine(estimated, planted) ≈ 1`.
- Over budget after calibration: exactly `budget` kept; kept set = the
  `budget` highest-projection positions (manual check vs numpy against the
  *frozen* direction); original order preserved.
- Sinks + `recent` protected even with low projection scores;
  `n_sink+recent>=budget` raises; `sign` validation raises.
- `sign=-1` selects the complement ranking.
- Frozen-filter determinism: a stored score never changes once the filter is
  set.
- Given-same-filter order invariance: inject a fixed `filter_dir`, feed the
  same tokens in two orderings, assert identical kept set.
- Byte accounting matches closed form incl. the `filter_dir` float32 term.
- Zero-row placeholder before first update.

---

## Phase 2 — Cache wrapper

**File:** `veloxquant_mlx/cache/qfilters_cache.py`

`QFiltersKVCache(_MLXKVCache)` — copy `L2NormKVCache`'s shape exactly: lazy
per-head state init on first `update_and_fetch`, per-(B,H) `QFiltersState`,
fp16 in/out, **no `.bits` attribute**, byte-accounting properties named
`qfilters_kept_bytes`, `full_seq_bytes`, `compression_ratio`, `tokens_seen`,
`tokens_kept`. Config fields consumed: `qfilters_budget` (default 512),
`qfilters_n_sink` (default 4), `qfilters_recent` (default 0),
`qfilters_calib_tokens` (default 128), `qfilters_sign` (default 1). Validate
`qfilters_sign ∈ {1,-1}` and the sink/recent-vs-budget guard in `__init__`
(delegate to `init_qfilters_state`). No per-token loop.

**Tests:** `veloxquant_mlx/tests/cache/test_qfilters_cache.py` (~14)

- Factory dispatch; shape/dtype preservation; no-`.bits` leak.
- Budget enforcement after a long prefill; passthrough below budget.
- Pre-calibration: short inputs (< `calib_tokens`) pass through untouched.
- Sinks retained across heavy eviction.
- Decode accumulation: token-by-token pushes keep `n_kept` capped, monotone
  `tokens_seen`.
- **Non-equivalence handled gracefully** (not a bug): prefill-vs-decode may
  differ; assert both stay within budget and both freeze a valid unit-norm
  filter — NOT bit-for-bit equal. Docstring cites the path-dependence note.
- `qfilters_sign=-1` produces a different kept set than `+1` on the same
  input.
- **Mechanism test under paper-like geometry:** construct keys where
  high-projection rows align with a probe-query cluster and low-projection
  rows anti-align; attention output using the `sign=+1` cache must be closer
  to the full-cache output than `sign=-1` on the same probes. Validates the
  machinery *given* the paper's reported QK geometry — cite the paper in the
  test docstring; the geometry is the paper's claim, not ours.
- Compression-ratio math; determinism (fixed seed ⇒ identical frozen filter);
  `for_model` wiring with a toy model incl. a non-attention fallback layer.

---

## Phase 3 — Config + builder wiring

**File:** `veloxquant_mlx/cache/base.py`

- Add `"qfilters"` to the `method` Literal (after `"skvq"`).
- Config block after the SKVQ block:
  `# --- Q-Filters-adapted configuration (query-agnostic projection eviction) ---`
  - `qfilters_budget: int = 512`         # max tokens kept (incl. sinks)
  - `qfilters_n_sink: int = 4`           # leading positions never evicted
  - `qfilters_recent: int = 0`           # trailing protected window (extension)
  - `qfilters_calib_tokens: int = 128`   # tokens observed before the filter freezes
  - `qfilters_sign: int = 1`             # +1 = paper direction; -1 = inverted ablation
- Factory branch (comment: no coordinator; projection scorer; default
  `for_model` path). Extend the unknown-method error string with `qfilters`.
- No `__init__.py` exports (KNorm/xKV/NSNQuant precedent).

---

## Phase 4 — Benchmark

**File:** `benchmark_scripts/benchmark_qfilters.py` (+ committed
`benchmark_scripts/qfilters_benchmark_results.json`)

Offline-synthetic, model-free. Two regimes per sweep row:

1. **Paper-like geometry** (`geometry="paper_like"`): keys with an
   anisotropic dominant axis; "important" tokens have high projection onto it
   and align with the probe-query cluster, the rest anti-align — the QK
   anisotropy the paper exploits, constructed explicitly (say so in the
   docstring).
2. **Isotropic control** (`geometry="isotropic"`): plain Gaussian keys — no
   dominant direction carries importance.

Arms at matched budget: `sign=+1` (the method), `sign=-1` (inverted), random
eviction (seeded), **KNorm-adapted** and **H2O-adapted** references (the
repo's intrinsic and accumulating-score baselines). Metric: output
perturbation (`1 − cosine` of probe-query attention output vs the full
uncompressed cache) — same metric family as CaM/ChunkKV/xKV/KNorm. Also
report compression ratio, wall time, and **filter-cosine** (how well the
key-SVD direction recovered the planted axis — the honest measure of whether
the key-derived estimator stands in for the paper's query-derived one).

**Honest expectations to print in the summary:** `sign=+1` should win clearly
under paper-like geometry and show ~no advantage over random under isotropic
— report both directions; never present isotropic rows as a win. State
plainly that the key-derived filter is a documented deviation from the
paper's query-derived filter, and that a low filter-cosine would mean the
substitute is not standing in well.

---

## Phase 5 — Docs site

- `docs-site/docs/algorithms/qfilters.md` — follow `knorm.md`/`skvq.md`
  structure: what it is, the **query-agnostic projection** mechanism, the
  honesty crux (query-SVD → key-SVD deviation) up top and unmissable, the
  scorer-class table (attention/proxy vs structural vs intrinsic-norm vs
  **projection** — Q-Filters adds the fourth class), usage snippet, config
  table, "Adaptation notes / What we do NOT implement" (survey §What we do
  NOT implement, all points), the **path-dependence** contrast with KNorm,
  benchmark table from committed JSON with the isotropic + filter-cosine
  caveats, citation block (arXiv:2503.02812, **preprint** — no venue).
- `docs-site/sidebars.ts`: add `'algorithms/qfilters'` in the eviction family
  (after `'algorithms/knorm'`).
- `docs-site/docs/algorithms/overview.md`: comparison-table row + eviction
  bullet; bump "thirty-three" → "thirty-four".
- Cross-links: from `knorm.md` (same score-and-evict machinery, different
  scorer class + the path-dependence contrast) and `h2o.md` — one line each.
- Docs changelog: 0.31.0 entry above 0.30.1 (move the "— Latest" marker).

---

## Phase 6 — CHANGELOG, README, EVIDENCE_TABLE, version bump

- `CHANGELOG.md` `[0.31.0]`: feature summary + **Honest scope**: filter is
  key-SVD-derived not query-SVD-derived (the crux); preprint, no venue; no
  RoPE remapping; uniform budget; kept set is path-dependent (unlike KNorm);
  `qfilters_recent` extension off by default; no model-level benchmark.
- `README.md`: badge `changelog-0.30.1` → `0.31.0`; summary paragraph
  "thirty-three compression strategies" → "thirty-four", and the
  "eight token-eviction caches" clause → **nine** (SnapKV, StreamingLLM, H2O,
  TOVA, PyramidKV, SqueezeAttention, ChunkKV, L2Norm + Q-Filters — verify the
  current phrasing at README.md line ~35 first and update the number and the
  parenthetical roll-call honestly); TOC "all 33 methods" → 34; "All 33
  methods" (line ~173) → 34; eviction method-table row (`qfilters`, 0.31.0);
  Sources citation (arXiv:2503.02812, preprint — **verify the official code
  URL live before adding it**, do not invent a URL).
- `paper/EVIDENCE_TABLE.md`: rows following the KNorm rows' format,
  **including** one "query-derived filter — NOT implemented; key-derived
  substitute used" row, one "end-to-end perplexity — none claimed / NOT RUN"
  row, and one "anisotropy/attention-prediction claim attributed to paper,
  not validated synthetically" row.
- `pyproject.toml`: `0.30.1` → `0.31.0`. **Do not regress the 0.30.1 PEP 639
  metadata fixes** — keep `license = "MIT"` + `license-files = ["LICENSE"]`,
  the one-line `description`, `requires = ["setuptools>=77", "wheel"]`, and
  the name-only author entry intact.

---

## Phase 7 — Landing page

- `landing/index.html`:
  - hero pill `33 algorithms` → `34`; `See all 33` → `34`; append
    `· Q-Filters` to the details roll-call,
  - meta description: `+ Q-Filters-adapted`; replace the "New in 0.30.x"
    clause with a 0.31.0 Q-Filters sentence,
  - what's-new: 0.31.0 `<li>` above the 0.30.x entry,
  - filter counts: `All (N)` +1; Token Eviction `cat-count` +1,
  - picker card (eviction color family) + full algo card in the **Token
    Eviction** group (copy the KNorm/CaM card markup; headline stat e.g.
    `query-agnostic projection · no attention, no proxy · NN/NN tests` with
    real final counts; card-meta `v0.31.0 · preprint`),
  - provenance strip: Q-Filters is a **preprint** — add it to the existing
    arXiv-only/preprint group or omit it from the venue-grouped strip; **do
    not fabricate a venue** (same call made for RVQ/Kitty/AdaKV in commit
    dc1c371),
  - code tab button + panel after the SKVQ panel,
  - update the requirements-list test-count line with real post-run numbers.
- `landing/assets/main.js`: hero badge → `"v0.31.0 — Q-Filters-adapted
  query-agnostic projection eviction shipped"`.

---

## Phase 8 — Final verification

- Full pytest (record exact collected/passed; expect ≈934 + ~28 new ≈ 962,
  minus the 6 known vecinfer flakes).
- `rm -rf dist/ && python3 -m build --outdir <scratch>` + `twine check` pass
  at 0.31.0; **inspect PKG-INFO** to confirm the 0.30.1 metadata fixes
  survived (`Metadata-Version: 2.4`, one-line `Summary:`, `License-Expression:
  MIT`, `Author: Rajveer Rathod`).
- `npm run build` in docs-site succeeds.
- Re-run `benchmark_qfilters.py`; all quality fields (everything but
  wall-time) identical across runs.
- Grep sweep: `grep -rn "33 algorithms\|thirty-three\|0\.30\.1" landing/
  README.md docs-site/docs/ | grep -v -i changelog` — nothing stale except
  historical changelog entries and the 0.30.x "New in" metadata.

---

## Phase 9 — Release commands (maintainer runs; provide as chat text only)

**Never execute git add/commit/tag/push/gh release/twine via tooling — the
maintainer reviews and runs every one.** Provide, as chat text: scoped
`git add` of this release's files only (the working tree carries unrelated
uncommitted changes — the Buy Me a Coffee batch and others — so do NOT
`git add -A`), a commit following the SKVQ/KNorm message structure (code /
tests / benchmark / docs / honest-scope, real numbers only, trailer
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`), `git tag -a
v0.31.0`, `git push origin master --follow-tags`, `gh release create v0.31.0`
(notes from the CHANGELOG entry), then `rm -rf dist/ && python3 -m build &&
twine check dist/* && twine upload dist/*`.

---

## Appendix — new files this release

- `veloxquant_mlx/quantizers/qfilters.py`
- `veloxquant_mlx/cache/qfilters_cache.py`
- `veloxquant_mlx/tests/quantizers/test_qfilters.py`
- `veloxquant_mlx/tests/cache/test_qfilters_cache.py`
- `benchmark_scripts/benchmark_qfilters.py`
- `benchmark_scripts/qfilters_benchmark_results.json`
- `docs-site/docs/algorithms/qfilters.md`
- `paper/NEW_METHOD_SURVEY_V14.md` (already written)
- `paper/IMPLEMENTATION_PROMPT_QFILTERS.md` (this file)

## Appendix — files modified this release

- `veloxquant_mlx/cache/base.py`
- `docs-site/sidebars.ts`, `docs-site/docs/algorithms/overview.md`,
  `docs-site/docs/changelog.md`, `docs-site/docs/algorithms/knorm.md`,
  `docs-site/docs/algorithms/h2o.md`
- `README.md`, `CHANGELOG.md`, `pyproject.toml`, `paper/EVIDENCE_TABLE.md`
- `landing/index.html`, `landing/assets/main.js`
