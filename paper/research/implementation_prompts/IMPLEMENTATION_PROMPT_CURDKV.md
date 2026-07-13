# Implementation Prompt — CurDKV-adapted (v0.36.0)

Execute-cold spec for the **38th** method. This single prompt covers the whole
release: core logic, tests, benchmark, docs, README + hero-pill/count bumps, a
new landing card, and the **git tag + GitHub release + PyPI publish** layer.
Do all of it.

> **Venue confirmed (re-verify live before citing):** "Value-Guided KV
> Compression for LLMs via Approximated CUR Decomposition" (Sengupta,
> Chaudhary, Chakraborty) — **NeurIPS 2025** (confirmed poster),
> arXiv:2509.15038, official listing `neurips.cc/virtual/2025/poster/116352`.
> Re-check the arXiv abstract + the neurips.cc poster page live before
> writing it into README/docs/EVIDENCE_TABLE.

Chosen over the survey's carried-forward pick **NestedKV** because NestedKV
(arXiv:2605.26678) is still a bare preprint with **no verified venue** as of
this survey — the same condition that correctly deferred MorphKV across
V13–V15 and NestedKV itself across V16–V19. CurDKV has a peer-reviewed venue
(NeurIPS 2025) **and** a genuinely new mechanism axis (value-aware
leverage-score eviction via approximated CUR decomposition) not covered by
KeyDiff (rejected this survey — overlaps KNorm-adapted's key-geometry axis)
or MixKVQ (rejected — needs a live query vector, breaks cache-only scope).

Model it on the **H2O / SnapKV pair** already shipped
(`quantizers/h2o.py`+`cache/h2o_cache.py`,
`quantizers/snapkv.py`+`cache/snapkv_cache.py`) — fixed per-head token
budget, key-as-query proxy for attention weights, score-then-evict loop. Ship
it as **"CurDKV-adapted (VeloxQuant-MLX implementation)," NeurIPS 2025
(arXiv:2509.15038) — NOT a faithful port.**

---

## The mechanism gap (why this is not a duplicate)

The repo has thirteen token-eviction / merging methods:

- **SnapKV** — prefill observation-window attention scoring, fires once.
- **StreamingLLM** — pure positional (sink + recency), no scoring at all.
- **H2O** — cumulative softmax attention-mass, fires every step.
- **TOVA** — memoryless current-step attention weight only.
- **PyramidKV** — H2O scoring with a per-layer pyramid budget.
- **SqueezeAttention** — 2D layer×token data-driven budget.
- **ChunkKV** — chunk-level H2O-style eviction.
- **CaM** — merges (rather than drops) evicted tokens.
- **L2Norm/KNorm** — pure key-vector norm magnitude, no attention proxy.
- **Q-Filters** — frozen per-head key-SVD projection direction.
- **Keyformer** — Gumbel-regularized heavy-hitter with a temperature knob.
- **MorphKV** — recent-window correlation retention.
- **KVzip** — context-reconstruction reliance scoring.

**Every one of these thirteen methods scores a token using only its *key*
side** (attention-mass over keys, key norm, key-SVD projection, key-based
reconstruction) — **none of them incorporate the value vector's own
contribution** into the retention decision. A token whose key looks
"important" by any of the above criteria but whose value vector is
near-zero or orthogonal to the accumulated output direction is
indistinguishable, under every existing method, from a token whose value
actually matters.

CurDKV's axis: build an **approximated CUR decomposition** of the (proxy)
attention-output block and derive **leverage scores** from it — a joint
`(K, V)` importance measure — then evict by that combined score instead of
a key-only one. This is a new, isolable eviction criterion: **value-aware
retention**, not just key-aware retention.

### The isolable reduction / observable contrast (must pin exactly)

Construct a planted synthetic geometry with two token classes:
1. **High key-similarity, high value-magnitude** — a token whose key
   strongly resembles the query-proxy direction (would score high under
   H2O/attention-mass) *and* whose value vector is large/output-relevant.
2. **High key-similarity, near-zero/orthogonal value contribution** — a
   token whose key looks identical in key-only similarity terms, but whose
   value vector is deliberately planted as near-zero or orthogonal to the
   dominant output direction.

Under this geometry:
- **H2O-style key-only scoring must treat classes 1 and 2 identically**
  (same key similarity ⇒ same attention-mass accumulation ⇒ same
  eviction priority) — pin this as the baseline's blind spot.
- **CurDKV's leverage-score eviction must correctly rank class-1 tokens
  above class-2 tokens** — the value-aware signal breaks the tie that H2O
  cannot break. Pin this as CurDKV's differentiating behavior with a
  dedicated test (the analogue of SVDq's fixed-split baseline, MorphKV's
  `window=1`==TOVA, KVzip's `probe="latest"`==TOVA, and KVTC's
  uniform-variance collapse).

Do **NOT** claim CurDKV strictly dominates H2O on all geometries — only
that it makes value-aware distinctions H2O structurally cannot. On a
geometry where key-similarity and value-magnitude are perfectly correlated,
the two methods should produce near-identical retention (state this as an
explicit non-claim, mirroring KVTC's "flat" control geometry).

---

## Non-negotiable honesty constraints (repeat on EVERY surface)

- **Key-as-query proxy, not the true query vector.** Exactly the same
  documented limitation as H2O/SnapKV/Keyformer/MorphKV/KVzip: the cache
  wrapper never sees the real query, so the incoming key vector stands in
  for it when building the proxy attention-weighted block that feeds the
  CUR approximation.
- **Approximated CUR decomposition over the locally observable proxy
  attention-output block, not the paper's ground-truth `softmax(QK^T)V`
  computed from real queries across the full sequence.** State this
  explicitly wherever leverage scores are mentioned.
- **A standard randomized leverage-score-proportional CUR/column-sampling
  approximation** (cite generically — Mahoney & Drineas-style CUR
  sampling — NOT claimed as a reproduction of the paper's specific sketching
  algorithm, which is not being independently re-derived here). Implement it
  with plain `numpy`/`mlx` linear algebra (SVD-based leverage-score
  estimation is an acceptable, well-specified stand-in for full CUR
  sampling) — no new external dependency.
- **Uniform budget across heads**, matching the repo's existing eviction
  convention (H2O, SnapKV, etc.), not any per-head tuning the paper may use.
- **Not the paper's full algorithm.** We implement *key-as-query proxy +
  approximated leverage-score-based joint (K,V) eviction*, NOT the paper's
  ground-truth attention-output CUR decomposition, NOT any trained-model
  evaluation. The paper's "up to 9.6% higher accuracy than SOTA baselines,
  up to 40% latency reduction" numbers are the PAPER's on trained models —
  **never quote them as ours.**
- **Clean mechanism observable** = the two-class planted geometry above
  (key-similar/value-relevant vs. key-similar/value-irrelevant), comparing
  CurDKV's retention choices against H2O's at a matched token budget.
  Report the class-2-token eviction rate difference plainly; do not oversell
  beyond what the synthetic test actually shows.
- Nothing here is validated on a trained model — offline-synthetic only.

---

## Phase 1 — Survey (already written)

`paper/research/surveys/NEW_METHOD_SURVEY_V19.md` is written. Confirm it
still matches the implementation before shipping (mechanism gap, honesty
crux, the two-class planted-geometry observable). Re-verify sources live
before citing in README/docs.

## Phase 2 — `veloxquant_mlx/quantizers/curdkv.py`

Mirror the shape and docstring discipline of `quantizers/h2o.py`.

- `CurDKVState` dataclass: `keys`, `values` (`[n_kept, D]` fp16 each, or
  `None` before first update), `leverage_scores` (`[n_kept]` float32
  cumulative leverage-score estimate, or `None`), `n_sink`, `budget` — same
  shape as `H2OState`.
- `init_curdkv_state(n_sink: int, budget: int, head_dim: int) -> CurDKVState`
  — mirrors `init_h2o_state`.
- `_leverage_scores(query_proxy: mx.array, keys: mx.array, values: mx.array) -> mx.array`
  — internal helper. Steps:
  1. Compute proxy attention weights exactly as H2O does
     (`softmax((keys @ query_proxy) / sqrt(D))`) — reuse the identical
     scale/softmax formulation, do not reinvent it.
  2. Form the proxy attention-output contribution per token:
     `weighted_values = attn[:, None] * values` (`[n, D]`) — each row is
     that token's weighted contribution to the (proxy) output.
  3. Estimate row-leverage scores of `weighted_values` via its dominant
     left-singular vectors: compute a small-rank SVD (`k = min(n, D, rank_cap)`
     where `rank_cap` is a config knob, default e.g. 16) of
     `weighted_values`, and set each row's leverage score to the squared
     row-norm of its projection onto the top-`k` left singular vectors,
     normalized to sum to 1 (standard leverage-score definition:
     `l_i = sum_j (U[i,j])^2` for the top-`k` `U`). This is the
     "approximated CUR" leverage-score estimate — document precisely that
     this is an SVD-based leverage-score stand-in, not a full CUR
     column/row sampling routine.
  4. Return `[n]` leverage scores.
- `curdkv_update(state, new_keys, new_values, rank_cap: int = 16) -> CurDKVState`
  — mirrors `h2o_update`'s per-token loop and eviction structure exactly
  (bootstrap on first token, accumulate scores, append new token, evict
  lowest-score non-sink token if over budget) but accumulates
  **leverage scores** (via `_leverage_scores`) into `state.leverage_scores`
  instead of H2O's raw cumulative softmax weights. Reuse H2O's sink
  protection pattern (`+inf` on protected scores before `argmin`) verbatim.
- `curdkv_get_kv`, `curdkv_fp16_bytes`, `full_curdkv_fp16_bytes` — identical
  in spirit to the H2O equivalents (byte-accounting only, no new logic).
- `__all__` exports.

**Planted-geometry differentiation test target:** on the two-class synthetic
geometry (Phase 7), `curdkv_update` must retain class-1 tokens over class-2
tokens at a tight budget where `h2o_update` (same query-proxy, same budget)
cannot tell the classes apart and evicts near-uniformly across both.

## Phase 3 — `veloxquant_mlx/cache/curdkv_cache.py`

`CurDKVKVCache(_MLXKVCache)` modeled on `H2OKVCache` line-for-line
(per-head state list, lazy init on first `update_and_fetch`, byte
accounting, no coordinator):

- Consume `curdkv_budget` (int, default 512 — match H2O's default),
  `curdkv_n_sink` (int, default 4 — match H2O's default), `curdkv_rank_cap`
  (int, default 16 — the SVD rank cap for leverage-score estimation).
- Same `update_and_fetch(keys, values)` shape/signature as `H2OKVCache`:
  `[B, H, S, D]` in, `[B, H, n_kept, D]` fp16 out, `n_kept <= curdkv_budget`.
- Byte props: `curdkv_kept_bytes`, `full_seq_bytes`, `compression_ratio`,
  `tokens_seen`, `tokens_kept` — identical names/semantics to `H2OKVCache`'s
  (`h2o_kept_bytes` → `curdkv_kept_bytes`, rest unchanged).
- No `.bits` attribute (eviction cache, not a quantizer).

## Phase 4 — `veloxquant_mlx/cache/base.py`

Add `"curdkv"` to the method `Literal` (after `"kvtc"`); config block
(`curdkv_budget: int = 512`, `curdkv_n_sink: int = 4`, `curdkv_rank_cap: int
= 16`, placed after the KVTC config block); import `CurDKVKVCache`; factory
branch (`elif config.method == "curdkv"`, no coordinator, mirror the `"h2o"`
branch); extend the unknown-method error string with `"curdkv"`. **Read
each region before editing.**

## Phase 5 — Tests (~20, match H2O's count/discipline)

`tests/quantizers/test_curdkv.py` (~12):
- init/guards (budget >= 1, n_sink >= 0, rank_cap >= 1).
- bootstrap on first token (no eviction, matches H2O's bootstrap path).
- never exceeds `budget` tokens after any number of updates.
- sink protection: first `n_sink` tokens never evicted regardless of
  leverage score (plant a deliberately-low-value-contribution token in a
  sink slot, confirm it survives).
- **planted two-class geometry differentiation** (the core new-mechanism
  test): class-1 (key-similar, value-relevant) tokens survive preferentially
  over class-2 (key-similar, value-irrelevant/orthogonal) tokens at a tight
  budget, run over several seeds (a rate, not one lucky run).
- **H2O blind-spot contrast**: on the same planted geometry, confirm
  `h2o_update` (imported from `quantizers/h2o.py`) evicts class-1 and
  class-2 tokens near-uniformly (i.e., demonstrate the baseline actually has
  the blind spot CurDKV fixes — don't just assert CurDKV is good in
  isolation).
- leverage scores are non-negative and finite (no NaN/inf from the SVD
  step on degenerate all-zero-value input — guard this explicitly).
- determinism (no RNG in the leverage-score SVD path itself; same input →
  same eviction sequence).
- byte accounting (`curdkv_fp16_bytes`, `full_curdkv_fp16_bytes`).
- values matter, not just keys (a test where two tokens have IDENTICAL keys
  but different values produces different leverage scores — direct proof
  the mechanism is value-aware).

`tests/cache/test_curdkv_cache.py` (~8):
- factory dispatch to `CurDKVKVCache`.
- construction guards; config propagation via `for_model`
  (`curdkv_budget`/`curdkv_n_sink`/`curdkv_rank_cap`).
- `update_and_fetch` never exceeds `curdkv_budget` positions.
- byte props; `compression_ratio > 1` at a reasonable budget.
- multi-head/multi-batch shape correctness (`[B, H, n_kept, D]`).
- prefill (S > 1) and decode (S == 1) both go through the same eviction
  loop (mirror H2O's test — no prefill-only special case).

## Phase 6 — Benchmark

`benchmark_scripts/benchmark_curdkv.py` + committed
`curdkv_benchmark_results.json`:
- SEQ_LENS + BUDGETS grid (match H2O/SnapKV's scale).
- GEOMETRIES = `["planted_value_divergence", "correlated"]`
  (`planted_value_divergence`: the two-class key-similar/value-divergent
  geometry — the case CurDKV should win on; `correlated`: null control
  where key-similarity and value-magnitude are correlated, where CurDKV
  and H2O should retain near-identical token sets — state this as an
  explicit non-claim, mirroring KVTC's "flat" control).
- Arms: CurDKV (leverage-score eviction) vs. H2O (cumulative attention-mass
  eviction) at the same matched token budget.
- Primary field: class-2 (value-irrelevant) token retention rate at matched
  budget — CurDKV should retain fewer of them than H2O on
  `planted_value_divergence`, and near-equal counts on `correlated`.
  Deterministic in ALL non-timing fields (only `_ms` may vary).
  Offline-synthetic; loads no model. Verify determinism by diffing two runs.

## Phase 7 — Docs

- `docs-site/docs/algorithms/curdkv.md` — full page: honesty crux
  (key-as-query proxy; SVD-based leverage-score stand-in vs. paper's full
  CUR sampling; ground-truth-attention-output vs. proxy-attention-output),
  the two-class planted-geometry observable, adaptation limitations, the
  paper's numbers labeled as the paper's.
- `docs-site/sidebars.ts` — add `'algorithms/curdkv'` after `kvtc`.
- `docs-site/docs/algorithms/overview.md` — thirty-seven→thirty-eight +
  table row + bullet. **Read before editing.**
- `docs-site/docs/changelog.md` — v0.36.0 (Latest); move v0.35.0 down.
- Cross-link from `h2o.md` (key-only vs. value-aware contrast) and, if it
  exists, any KNorm/Q-Filters page (key-geometry-only contrast).

## Phase 8 — README / CHANGELOG / EVIDENCE_TABLE / pyproject

- README:
  - changelog badge `0.35.0`→`0.36.0`.
  - "**thirty-seven** compression strategies"→"**thirty-eight**" (sweep
    ALL instances — headline, bullet, TOC, integration line — this repo has
    previously miscounted this exact field; grep for every occurrence of
    "37"/"thirty-seven" before editing and confirm each one is intentionally
    bumped, not just the first match).
  - "All **37** methods"→"All **38**".
  - token-eviction family bullet/parenthetical: extend with "…and CurDKV's
    value-aware leverage-score eviction — NeurIPS 2025".
  - method-table row in the token-eviction section after the KVzip row:
    CurDKV-adapted / `curdkv` / "Value-aware leverage-score eviction via
    approximated CUR decomposition (NeurIPS 2025) — evicts key-similar but
    value-irrelevant tokens that key-only eviction (H2O) cannot
    distinguish" / `0.36.0`.
  - Sources entry (NeurIPS 2025, arXiv:2509.15038, neurips.cc poster link,
    if verified live).
  - Sweep for stale `37`/`0.35.0`.
- `CHANGELOG.md` `[0.36.0] — <today>` with **Honest scope** (key-as-query
  proxy; SVD-based leverage-score stand-in, not full CUR sampling;
  ground-truth vs. proxy attention-output; paper numbers not ours); move
  `[0.35.0]` down.
- `paper/research/EVIDENCE_TABLE.md` — next contiguous rows.
- `pyproject.toml` — version `0.35.0`→`0.36.0`; description "...to
  KVTC"→"...to CurDKV", 37→38; **preserve PEP 639 metadata** (`license="MIT"`
  + `license-files=["LICENSE"]`, one-line description, name-only author,
  `requires=["setuptools>=77","wheel"]`).

## Phase 9 — Landing page (new card + counts)

`landing/index.html` + `assets/main.js`. **Read each region before editing.**
- `<meta name="description">`: append CurDKV-adapted to the roll-call;
  change "New in 0.35.0…"→"New in 0.36.0: CurDKV-adapted value-aware
  leverage-score eviction (NeurIPS 2025) — distinguishes key-similar but
  value-irrelevant tokens that key-only eviction cannot."
- hero pill "37 algorithms"→"38 algorithms"; "See all 37"→"See all 38".
- whats-new: add a `0.36.0` `<li>` at the top with the honest crux; keep the
  0.35.0 li below.
- Token-eviction `cat-count` bump (find the H2O/SnapKV/TOVA/etc. category).
- Roll-call in the "See all" details: +CurDKV-adapted.
- New algo card `#algo-curdkv` (clone the `#algo-h2o` card block:
  `card-meta` → `v0.36.0 · NeurIPS 2025`, `data-tags` includes the
  token-eviction tags, full-desc with the honest crux).
- Picker card + code tab button + panel `#tab-curdkv` (clone the h2o one).
- **Move the NEW pill off the KVTC card onto `#algo-curdkv`.**
- `assets/main.js` `initBadgeTyping` text → `"v0.36.0 — CurDKV-adapted
  value-aware leverage-score eviction shipped"`.

## Phase 10 — Verify

- Full pytest (expect +~20 new; the ~6 known vecinfer Metal fp16 flakes stay
  flaky, do NOT chase). Confirm zero non-vecinfer failures and all new
  CurDKV tests pass.
- `python -m build` + `python -m twine check dist/*` at `0.36.0`; inspect
  the wheel PKG-INFO (Metadata-Version 2.4, Version 0.36.0,
  License-Expression MIT, Author: Rajveer Rathod, **38**-method Summary,
  `curdkv.py`/`curdkv_cache.py` present in the wheel).
- Docs `npm run build` SUCCESS.
- Benchmark determinism (non-`_ms` fields stable across two runs).
- Grep stale-ref sweep (38 consistent across
  README/overview/pyproject/landing/JOSS; no lingering `37`/`0.35.0` in
  places that should now say `38`/`0.36.0`).
- End-to-end factory smoke test: `KVCacheConfig(method="curdkv", …)` →
  `KVCacheBuilder.for_model` → a `compression_ratio > 1` at a reasonable
  budget, and the planted-geometry differentiation holds through the full
  factory path (not just the bare quantizer function).

---

## Phase 11 — Release layer (provide as CHAT TEXT ONLY — never execute)

**Standing rule:** the user reviews and runs all git/publish commands
themselves. After implementation, give the v0.36.0 release sequence as
**chat text** for them to run — do NOT execute any
`git add`/`commit`/`tag`/`push`/`gh release`/`gh repo edit`/`twine` yourself.
Mirror the exact format used for the v0.35.0 KVTC release:

1. **Branch:** `git checkout -b release/v0.36.0`
2. **Stage only CurDKV paths** (explicit `git add` list — do NOT stage the
   unrelated working-tree noise: `dist_preview/`, `blog_drafts/`,
   `dist_pypi/`, `.claude/`). Include:
   `veloxquant_mlx/quantizers/curdkv.py`,
   `veloxquant_mlx/cache/curdkv_cache.py`, `veloxquant_mlx/cache/base.py`,
   all new test files, `benchmark_scripts/benchmark_curdkv.py` +
   `curdkv_benchmark_results.json`, `docs-site/docs/algorithms/curdkv.md`,
   `docs-site/sidebars.ts`, `overview.md`, `h2o.md`, `changelog.md`,
   `README.md`, `CHANGELOG.md`, `paper/research/EVIDENCE_TABLE.md`,
   `paper/research/surveys/NEW_METHOD_SURVEY_V19.md`,
   `paper/research/implementation_prompts/IMPLEMENTATION_PROMPT_CURDKV.md`,
   `pyproject.toml`, `landing/index.html`, `landing/assets/main.js`.
3. **Commit:** `git commit -F-` heredoc, **NO Co-Authored-By line**, subject
   `feat(curdkv): CurDKV-adapted value-aware leverage-score eviction —
   v0.36.0`, body covering the mechanism, the honest scope (key-as-query
   proxy; SVD-based leverage-score stand-in, not full CUR sampling; paper
   numbers not ours).
4. **Tag:** `git tag -a v0.36.0 -m "..."`.
5. **Push:** `git push -u origin release/v0.36.0` then `git push origin v0.36.0`.
6. **Build + check:** `rm -rf dist build *.egg-info && python -m build &&
   python -m twine check dist/*`.
7. **PyPI:** `python -m twine upload dist/veloxquant_mlx-0.36.0*`.
8. **GitHub release:** `gh release create v0.36.0 --repo
   rajveer43/VeloxQuant-MLX --title "..." --notes "$(cat <<'EOF' … EOF)"` with
   CurDKV release notes (mechanism table, honest scope, usage snippet).
   Escape code fences in the heredoc as `` \`\`\` ``.

## What we do NOT implement (state plainly)

- The paper's ground-truth `softmax(QK^T)V` attention-output matrix computed
  from real query vectors (key-as-query proxy instead, same limitation as
  H2O/SnapKV/Keyformer/MorphKV/KVzip).
- The paper's specific CUR sketching/sampling algorithm (a standard,
  generically-cited SVD-based leverage-score approximation instead).
- Any per-head budget tuning beyond the repo's existing uniform-budget
  eviction convention.
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers (up to 9.6% higher accuracy than SOTA baselines, up to
  40% latency reduction) are the paper's — not reproduced here.
