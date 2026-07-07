# Full Autonomous Implementation Prompt — L2Norm-adapted (v0.29.0)

**Purpose of this document:** a self-contained, execute-without-supervision
spec for shipping the next VeloxQuant-MLX method end-to-end: survey (done,
see `paper/NEW_METHOD_SURVEY_V12.md`) → eviction primitives → cache wrapper →
config/builder wiring → tests → benchmark → docs → changelog → README →
landing page → version bump → release/publish commands. Written so an agent
picking this up cold can complete every step and leave the repo in the same
finished state as every prior release (0.19.0 through 0.28.0).

**Do not deviate from the "adapted, not faithful port" discipline**: label
everything "L2Norm-adapted (VeloxQuant-MLX implementation)", report only
numbers from a committed results JSON you generate yourself, never repeat the
paper's headline numbers as if measured here, and document every
simplification plainly. **One extra rule specific to this method:** the
paper's core claim (low key norm ⇒ high attention in trained LMs) is an
empirical property of trained models that synthetic data cannot validate —
every doc must attribute that correlation to the paper, and the benchmark
must include the isotropic control where the method shows *no* advantage.

Read `paper/NEW_METHOD_SURVEY_V12.md` first. If this checklist conflicts
with the survey, the survey wins.

---

## 0. Ground rules (apply to every phase)

- One buildable commit per phase, tests green before proceeding
  (`pytest veloxquant_mlx/tests/ -x -q`; suite is 879 tests before this work,
  874 passing — the 5 failures are pre-existing machine-dependent vecinfer
  Metal fp16-tolerance flakes, also failing on clean master; do not chase
  them).
- Every new Python file gets a module docstring citing arXiv:2406.11430
  (Devoto, Zhao, Scardapane, Minervini — EMNLP 2024), stating "adapted, not
  faithful port", and listing what's NOT implemented.
- No new dependencies. **Model structurally on the H2O pair**
  (`quantizers/h2o.py`, `cache/h2o_cache.py`) — same State-dataclass /
  init / update / get_kv / bytes API shape, same per-head state list in the
  wrapper, same byte-accounting property names (`*_kept_bytes`,
  `full_seq_bytes`, `compression_ratio`, `tokens_seen`, `tokens_kept`).
- **Exploit the intrinsic-score advantage** — do NOT copy H2O's per-token
  Python loop. Norms never update, so `knorm_update` takes the whole `[S, D]`
  block at once: compute S norms, concatenate, and if over budget do a single
  protected top-k selection. This is both faster and what makes the
  path-independence property (below) hold exactly.
- Naming: `knorm.py`, `knorm_cache.py`, `test_knorm.py`,
  `test_knorm_cache.py`, `benchmark_knorm.py`, `knorm.md`,
  `knorm_benchmark_results.json`. Config prefix `knorm_`, method string
  `"knorm"`, display name **L2Norm-adapted**.

---

## Phase 1 — Eviction primitives

**File:** `veloxquant_mlx/quantizers/knorm.py`

Mirror `h2o.py`'s public API:

1. `KnormState` dataclass — `keys [n_kept, D] fp16 | None`,
   `values | None`, `norms [n_kept] float32 | None` (L2 norm of each kept
   key row, computed once at insertion, never updated), `n_sink`, `budget`,
   `recent` (protected trailing window, 0 = off), `keep` (`"low"` paper
   default | `"high"` inverted ablation arm).
2. `init_knorm_state(n_sink, budget, head_dim, recent=0, keep="low")`.
3. `knorm_update(state, new_keys [S, D], new_values [S, D]) -> KnormState` —
   vectorized:
   - compute `new_norms = ||new_keys||₂` per row (float32),
   - concatenate keys/values/norms onto state,
   - if `n_total > budget`: build a protected score vector — sinks (first
     `min(n_sink, n_total)` positions) and the trailing `recent` positions
     get `-inf` when `keep="low"` (`+inf` when `"high"`), then keep the
     `budget` lowest-norm (resp. highest-norm) positions **in original
     temporal order** (sort the kept indices ascending before gathering —
     eviction methods in this repo preserve order).
   - Guard: if `n_sink + recent >= budget`, raise `ValueError` at state
     construction time (init), not mid-update.
4. `knorm_get_kv(state)` — identical contract to `h2o_get_kv` (zero-row
   `[0, 1]` placeholders before first update).
5. `knorm_fp16_bytes(state)` / `full_knorm_fp16_bytes(tokens_seen, head_dim)`
   — identical formulas to the H2O versions.

**The path-independence invariant (document in the module docstring):**
with `recent=0`, evicting the current worst-scoring non-sink token whenever
over budget is the classic "keep k best with a heap" algorithm — the final
kept set equals the global budget-best over all tokens seen, **regardless of
arrival grouping**. Prefill-in-one-block and token-by-token decode produce
bit-for-bit identical kept sets. No accumulating-score method (H2O, TOVA)
has this property; it is the method's distinguishing testable invariant.
(`recent > 0` breaks it, because the protected window moves.)

**Tests:** `veloxquant_mlx/tests/quantizers/test_knorm.py` (~12)

- Under budget: passthrough, all tokens kept in order, norms match manual
  `np.linalg.norm` computation.
- Over budget: exactly `budget` kept; kept set = the budget lowest-norm
  positions (manual check vs numpy argsort); original order preserved.
- Sinks protected even when they have the highest norms.
- `recent` window protected when set; `n_sink + recent >= budget` raises.
- `keep="high"` selects exactly the complement ranking (highest-norm kept).
- Norm immutability: a token's stored norm never changes across updates.
- Path independence (`recent=0`): one-shot block vs token-by-token arrival
  yield bit-for-bit identical kept keys AND values.
- Byte accounting matches the closed form.
- Zero-row placeholder before first update.

---

## Phase 2 — Cache wrapper

**File:** `veloxquant_mlx/cache/knorm_cache.py`

`L2NormKVCache(_MLXKVCache)` — copy `H2OKVCache`'s shape exactly: lazy
per-head state init on first `update_and_fetch`, per-(B,H) `KnormState`,
fp16 in/out, **no `.bits` attribute**, byte-accounting properties with the
same names H2O uses (`knorm_kept_bytes`, `full_seq_bytes`,
`compression_ratio`, `tokens_seen`, `tokens_kept`). Config fields consumed:
`knorm_budget` (default 512), `knorm_n_sink` (default 4), `knorm_recent`
(default 0), `knorm_keep` (default `"low"`). Validate `knorm_keep ∈
{"low","high"}` and the sink/recent-vs-budget guard in `__init__`.

Unlike H2O there is no per-token loop anywhere — each head's whole `[S, D]`
block goes through one `knorm_update` call.

**Tests:** `veloxquant_mlx/tests/cache/test_knorm_cache.py` (~14)

- Factory dispatch; shape/dtype preservation; no-`.bits` leak.
- Budget enforcement: `n_kept <= budget` after a long prefill; passthrough
  below budget (bit-for-bit).
- Sinks retained across heavy eviction.
- Decode accumulation: token-by-token pushes keep `n_kept` capped and
  monotone tokens_seen.
- **Prefill-vs-decode bit-for-bit equivalence** at `knorm_recent=0` (the
  invariant test — compare returned K AND V exactly).
- `knorm_keep="high"` produces a different kept set than `"low"` on the
  same input (and both respect the budget).
- Mechanism test under paper-like geometry: construct keys where low-norm
  rows are aligned with a probe-query cluster and high-norm rows are
  anti-aligned; attention output using the keep-low cache must be closer to
  the full-cache output than keep-high on the same probes. (This validates
  the machinery *given* the paper's reported geometry — the geometry itself
  is the paper's claim, cite it in the test docstring.)
- Compression-ratio property math; determinism; `for_model` wiring with a
  toy model incl. a non-attention fallback layer.

---

## Phase 3 — Config + builder wiring

**File:** `veloxquant_mlx/cache/base.py`

- Add `"knorm"` to the `method` Literal (after `"nsnquant"`).
- Config block after the NSNQuant block:
  `# --- L2Norm-adapted configuration (intrinsic key-norm eviction) ---`
  - `knorm_budget: int = 512`   # max tokens kept (incl. sinks)
  - `knorm_n_sink: int = 4`     # leading positions never evicted
  - `knorm_recent: int = 0`     # trailing protected window (0 = paper-faithful)
  - `knorm_keep: str = "low"`   # "low" = paper finding; "high" = inverted ablation
- Factory branch (comment: no coordinator; intrinsic scores; default
  `for_model` path). Extend the unknown-method error string with `knorm`.
- No `__init__.py` exports (xKV/NSNQuant precedent).

---

## Phase 4 — Benchmark

**File:** `benchmark_scripts/benchmark_knorm.py` (+ committed
`benchmark_scripts/knorm_benchmark_results.json`)

Offline-synthetic, model-free. Two data regimes per sweep row:

1. **Paper-like geometry** (`geometry="paper_like"`): keys where a random
   "important" fraction has low norm + alignment with the probe-query
   cluster, the rest high norm + anti-alignment — the correlation the paper
   reports in trained LMs, constructed explicitly (say so in the docstring).
2. **Isotropic control** (`geometry="isotropic"`): plain Gaussian keys —
   the regime where key norm carries no importance signal.

Arms at matched budget: `keep="low"` (the method), `keep="high"` (inverted),
random eviction (seeded), H2O-adapted (the repo's accumulating-score
reference). Metric: output perturbation (1 − cosine of probe-query attention
output vs the full uncompressed cache) — the same metric family as the
CaM/ChunkKV/xKV benchmarks. Also report compression ratio and wall time.

**Honest expectations to print in the summary:** keep-low should win clearly
under paper-like geometry and show ~no advantage over random under the
isotropic control — report both directions; never present the isotropic rows
as a win.

---

## Phase 5 — Docs site

- `docs-site/docs/algorithms/knorm.md` — follow `nsnquant.md`/`xkv.md`
  structure: what it is, the intrinsic-signal mechanism + the inversion
  (low norm ⇒ important), scorer-class table (attention/proxy vs structural
  vs **intrinsic**), usage snippet, config table, "Adaptation notes / What we
  do NOT implement" (survey §adaptation, all four points), the
  path-independence invariant, benchmark table from committed JSON with the
  isotropic caveat, citation block (EMNLP 2024).
- `docs-site/sidebars.ts`: add `'algorithms/knorm'` after
  `'algorithms/cam'` (eviction family), before `'algorithms/xkv'`.
- `docs-site/docs/algorithms/overview.md`: comparison-table row +
  eviction-methods bullet; bump "thirty-one" → "thirty-two".
- Cross-links: from `h2o.md` (same machinery, different scorer class) and
  `chunkkv.md` (its `key_norm` scoring option uses the *opposite* sign —
  worth one sentence) — one line each.
- Docs changelog: 0.29.0 entry above 0.28.0 (move the "— Latest" marker).

---

## Phase 6 — CHANGELOG, README, EVIDENCE_TABLE, version bump

- `CHANGELOG.md` `[0.29.0]`: feature summary + Honest scope (correlation is
  the paper's empirical claim about trained models — synthetic benchmark
  only validates the machinery under constructed geometry; no RoPE
  remapping; uniform budget; `knorm_recent` is an extension, off by
  default; no model-level benchmark).
- `README.md`: badge `changelog-0.28.0` → `0.29.0`; summary paragraph
  ("thirty-one" → "thirty-two", extend the eviction clause: "eight
  token-eviction caches" — count carefully: SnapKV, StreamingLLM, H2O, TOVA,
  PyramidKV, SqueezeAttention, ChunkKV + L2Norm = verify the current
  phrasing first and update the number honestly); TOC + "All 31 methods"
  → 32; eviction method-table row (`knorm`, 0.29.0); Sources citation
  (EMNLP 2024, Devoto et al., with the l2compress code link).
- `paper/EVIDENCE_TABLE.md`: rows 113+ following the NSNQuant rows' format
  (incl. one "end-to-end perplexity — none claimed / NOT RUN" row and one
  "correlation claim attributed to paper, not validated synthetically" row).
- `pyproject.toml`: `0.28.0` → `0.29.0`.

---

## Phase 7 — Landing page

- `landing/index.html`:
  - hero pill `31 algorithms` → `32`; `See all 31` → `32`; append
    `· L2Norm` to the details list,
  - meta description: roll-call `+ L2Norm-adapted`; replace the
    "New in 0.28.0" clause with a 0.29.0 L2Norm sentence,
  - what's-new: 0.29.0 `<li>` above 0.28.0,
  - filter count `All (29)` → `All (30)`; Token Eviction `cat-count` 8 → 9,
  - picker card (eviction color family) + full algo card in the **Token
    Eviction** group (copy the CaM/ChunkKV card markup; headline stat:
    `intrinsic signal · no attention, no proxy · 26/26 tests` with real
    final counts; card-meta `v0.29.0 · EMNLP 2024`),
  - provenance strip: add `EMNLP 2024 <em>L2-norm eviction</em>` between
    the SIGCOMM 2024 and NeurIPS 2024 items (chronological-ish order),
  - code tab button + panel after the NSNQuant panel,
  - update the requirements-list test-count line with real post-run numbers.
- `landing/assets/main.js`: hero badge →
  `"v0.29.0 — L2Norm-adapted intrinsic key-norm eviction shipped"`.

---

## Phase 8 — Final verification

- Full pytest (record exact collected/passed; expect ≈905: 879 + ~26 new).
- `python3 -m build --outdir <scratch>` + `twine check` pass at 0.29.0.
- `npm run build` in docs-site succeeds.
- Re-run `benchmark_knorm.py`; quality fields (all but wall-time) must be
  identical across runs.
- Grep sweep: `grep -rn "31 algorithms\|thirty-one\|0\.28\.0" landing/
  README.md docs-site/docs/ | grep -v -i changelog` — nothing stale except
  historical changelog entries and NSNQuant's own "New in 0.28.0" metadata.

---

## Phase 9 — Release commands (maintainer runs; provide as chat text)

Same structure as v0.28.0: scoped `git add` (this release's files only —
the working tree has unrelated changes), commit following the 4a48279
NSNQuant message structure (code / tests / benchmark / docs / honest-scope,
real numbers only, `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`),
`git tag -a v0.29.0` + `git push origin master --follow-tags`,
`gh release create v0.29.0` with notes drafted from the CHANGELOG entry,
then `rm -rf dist/ && python3 -m build && twine check dist/* && twine upload
dist/*`.

---

## Appendix — file manifest (new files this release)

- `veloxquant_mlx/quantizers/knorm.py`
- `veloxquant_mlx/cache/knorm_cache.py`
- `veloxquant_mlx/tests/quantizers/test_knorm.py`
- `veloxquant_mlx/tests/cache/test_knorm_cache.py`
- `benchmark_scripts/benchmark_knorm.py`
- `benchmark_scripts/knorm_benchmark_results.json`
- `docs-site/docs/algorithms/knorm.md`
- `paper/NEW_METHOD_SURVEY_V12.md` (already written)
- `paper/IMPLEMENTATION_PROMPT_KNORM.md` (this file)

## Appendix — files modified this release

- `veloxquant_mlx/cache/base.py`
- `docs-site/sidebars.ts`, `docs-site/docs/algorithms/overview.md`,
  `docs-site/docs/changelog.md`, `docs-site/docs/algorithms/h2o.md`,
  `docs-site/docs/algorithms/chunkkv.md`
- `README.md`, `CHANGELOG.md`, `pyproject.toml`, `paper/EVIDENCE_TABLE.md`
- `landing/index.html`, `landing/assets/main.js`
