# Implementation Prompt — MorphKV-adapted (v0.33.0)

Execute-cold spec for the **36th** method. Chosen because MorphKV was the
standing "best next pick" across surveys V13→V15, deferred every time for one
reason only — **venue unverified**. That reason is now gone:

> **Venue confirmed (re-verify live before citing):** "Dialogue Without Limits:
> Constant-Sized KV Caches for Extended Responses in LLMs," Ghadia, Kumar, Jain,
> Nair, Das — **ICML 2025** (arXiv:2503.00979). Confirmed accepted, not merely
> submitted. Re-check the arXiv abstract + any official code repo live before
> writing it into README/docs/EVIDENCE_TABLE.

Model it on the **H2O pair** (`quantizers/h2o.py`, `cache/h2o_cache.py`) and the
**Keyformer pair** just shipped (`quantizers/keyformer.py`,
`cache/keyformer_cache.py`) — single-layer, no coordinator, no `.bits`, fp16,
lazy per-head state, per-head seed offset scaffolding. Ship it as
**"MorphKV-adapted (VeloxQuant-MLX implementation)," ICML 2025
(arXiv:2503.00979) — NOT a faithful port.**

---

## The mechanism gap (why this is not a duplicate)

The repo has eight proxy-attention scorers. Every one of them scores a stored
token against **either** the cumulative history (H2O accumulates mass forever)
**or** a single most-recent query (TOVA / SnapKV use the latest position).
**None** scores a stored token by its **correlation with the attention pattern
of a sliding *window* of the most recent tokens.** That recent-window-guided
retention is MorphKV's contribution and the genuine, isolable gap.

- **H2O** = cumulative additive mass; early tokens dominate ("early-token bias"
  — the exact failure MorphKV names).
- **TOVA** = one latest query decides; no window, no correlation history.
- **MorphKV-adapted** = keep the tokens whose keys are most **correlated with
  the aggregate attention pattern of the last `window` tokens**, so retention
  tracks what the recent context is actually attending to, and the cache stays
  **constant-size** (a fixed budget refreshed every step, not a monotonically
  growing accumulator).

---

## Non-negotiable honesty constraints (repeat on EVERY surface)

- **Proxy query.** A cache never sees the true query. Like H2O / SnapKV /
  Keyformer-adapted, incoming KEYS stand in for queries when estimating
  attention. Documented substitution, not the paper's math.
- **`morphkv_window=1` collapses onto a TOVA-adapted-style latest-token
  scorer**, and with an all-history correlation it approaches H2O-adapted's
  ranking — state this as the honest reference behavior and **assert the
  `window=1` reduction with a dedicated test** (analogous to Keyformer's
  `tau=0`==H2O collapse). Pick whichever single-arm collapse is cleanest to
  pin exactly; do NOT claim a collapse you cannot assert bit-for-bit.
- **Not the paper's full algorithm.** We implement the *retention rule*
  (recent-window correlation → constant-size keep set), NOT any RoPE remapping,
  NOT per-head adaptive budgets, NOT a trained-model evaluation. The
  "18.2% higher accuracy / 52.9% memory savings" numbers are the PAPER's on
  trained models — **never quote them as ours.**
- **Clean mechanism observable** = a constructed "**topic-shift**" geometry: the
  recent window attends to a *different* region than the early heavy-hitters, so
  a cumulative (H2O-style) keep set retains the stale early tokens while
  MorphKV-adapted swaps toward the region the recent window actually reads.
  Report the **recent-relevant retention rate** as the primary observable;
  report downstream output perturbation as-is (noisy, regime-dependent, with a
  null "stable"/no-shift control where it shows no advantage). Do NOT
  cherry-pick.
- Nothing here is validated on a trained model — offline-synthetic only.

---

## Phase 1 — Survey (write it)

`paper/NEW_METHOD_SURVEY_V16.md`, follow-up to V15. Commit to MorphKV; lead with
"the venue is now verified (ICML 2025), which was the sole deferral reason
V13–V15." State the mechanism gap (recent-window correlation vs
cumulative/latest-only), the honesty crux, the `window=1` reduction, and the
topic-shift observable. Carry the non-chosen rows forward (Scissorhands, MiKV,
NestedKV) and name the **next** fallback explicitly. Re-verify sources live.

## Phase 2 — `veloxquant_mlx/quantizers/morphkv.py`

Mirror the shape and docstring discipline of `quantizers/keyformer.py`.

- `MorphKVState` dataclass: `keys`/`values` (`[n, D]` fp16), `pos` (running
  count), `n_sink`, `budget`, `window` (recent-attention window size),
  `head_dim`. No cumulative score array is *stored* — the whole point is that
  retention is recomputed from the current keep set + recent window each step
  (constant-size, not an accumulator). If you keep a small recency ring buffer
  of the last `window` key rows, store it as `recent_keys: [<=window, D]`.
- `init_morphkv_state(n_sink, budget, head_dim, window=8)` — guards:
  `budget >= 1`, `window >= 1`, `n_sink < budget`, `window <= budget`.
- `_attention_scores(query_proxy, keys)` — reuse Keyformer's exact
  softmax(key-as-query · keys · 1/sqrt(D)); factor to a shared helper or
  duplicate with a comment pointing at the canonical one. Be consistent.
- `_recent_relevance(keys, recent_keys)` — for each stored key, its aggregate
  proxy-attention mass under the **window** of recent key rows (mean over the
  window of `_attention_scores(recent_i, keys)`). This is the MorphKV signal:
  "how much does the recent window attend to this stored token."
- `morphkv_update(state, new_keys, new_values)` — per token: append; push into
  the recent-window ring; if over budget, compute `_recent_relevance` over the
  current keep set, force sinks (leading) and the recent `window` (trailing) to
  survive (+inf), evict the **lowest recent-relevance** non-protected token.
  Constant-size: after each token, `n_kept <= budget`. Kept tokens returned in
  temporal order.
- `morphkv_get_kv`, `morphkv_fp16_bytes` (K+V only; ring/scratch not counted,
  like H2O/Keyformer), `full_morphkv_fp16_bytes`.
- `__all__` exports.

**window=1 reduction:** with `window=1` the recent-relevance is just the latest
token's attention over the keep set — a latest-query (TOVA-adapted-style)
eviction. Design the code so this holds *exactly* and pin it in tests.

## Phase 3 — `veloxquant_mlx/cache/morphkv_cache.py`

`MorphKVKVCache(_MLXKVCache)` modeled on `KeyformerKVCache`:

- Consume `morphkv_budget`(512) / `morphkv_n_sink`(4) / `morphkv_window`(8).
- No `.bits`. Lazy per-head state; per-head independence (no seed needed —
  MorphKV is deterministic — but keep the per-head state list pattern).
- Byte props with MorphKV names: `morphkv_kept_bytes` / `full_seq_bytes` /
  `compression_ratio` / `tokens_seen` / `tokens_kept`.
- Validate at construction (delegate to `init_morphkv_state`).
- Both prefill (S>1) and decode (S==1) through the same update loop.

## Phase 4 — `veloxquant_mlx/cache/base.py`

Add `"morphkv"` to the method `Literal`; config block
(`morphkv_budget=512`, `morphkv_n_sink=4`, `morphkv_window=8`); import
`MorphKVKVCache`; factory branch (`elif config.method == "morphkv"`, no
coordinator); extend the unknown-method error string with `"morphkv"`.

## Phase 5 — Tests (~29, match Keyformer's count/discipline)

`tests/quantizers/test_morphkv.py` (~17):
- init guards (budget/window/sink bounds).
- budget invariant after each token AND across a block (constant-size).
- sink protection; trailing-window protection.
- byte accounting (`morphkv_fp16_bytes`, `full_morphkv_fp16_bytes`).
- **`window=1` reduces to the latest-token eviction** — assert exactly.
- determinism / reproducibility (same input → same keep set, no RNG).
- **`test_topic_shift_retains_recent_relevant`**: the statistical mechanism
  claim — under a planted topic shift, MorphKV retains the recent-relevant
  region at a materially higher rate than a cumulative (H2O-style) baseline.
  Frame as a rate over several planted seeds, not one lucky run.
- a null "stable"/no-shift control where MorphKV shows **no** advantage —
  include it so the win is not overclaimed.

`tests/cache/test_morphkv_cache.py` (~12):
- factory dispatch to `MorphKVKVCache`; no `.bits`.
- construction guards; config propagation via `for_model`.
- budget respected across B/H.
- byte props.
- prefill+decode both within budget (NOT an equivalence claim).
- cache-level `window=1` reduction sanity.

## Phase 6 — Benchmark

`benchmark_scripts/benchmark_morphkv.py` + committed
`morphkv_benchmark_results.json`:
- SEQ_LENS + BUDGETS grid (match Keyformer's scale, e.g. [256,512] × [32,64]).
- GEOMETRIES = `["topic_shift", "stable"]`.
- WINDOWS ∈ {1, 8, 32} (window=1 = latest-token reference arm).
- Arms: MorphKV(window=k), an **H2O-style cumulative cross-check column**, and a
  **random** arm.
- Primary field: **recent-relevant retention rate**; secondary: output
  perturbation (reported as-is). Deterministic in ALL non-timing fields (only
  `_ms` fields may vary — matches Keyformer/qfilters convention). Offline-
  synthetic; loads no model.

## Phase 7 — Docs

- `docs-site/docs/algorithms/morphkv.md` — full page with the honesty crux,
  the window=1 reduction, topic-shift observable, adaptation limitations.
- `docs-site/sidebars.ts` — add `'algorithms/morphkv'` after `keyformer`.
- `docs-site/docs/algorithms/overview.md` — thirty-five→thirty-six + table row
  + bullet.
- `docs-site/docs/changelog.md` — v0.33.0 (Latest); move v0.32.0 down.
- Cross-link from `h2o.md` and `tova.md` (the two methods MorphKV contrasts).

## Phase 8 — README / CHANGELOG / EVIDENCE_TABLE / pyproject

- README: badge 0.32.0→0.33.0; thirty-five→thirty-six; bump the eviction-cache
  count (ten→eleven token-eviction caches, mirror the Keyformer edit); "All 35"→
  36; method-table row (0.33.0); Sources entry (ICML 2025, arXiv:2503.00979,
  official code URL if verified live). Sweep for stale `35`/`0.32.0` — leave the
  one correct Keyformer "since version" cell intact.
- `CHANGELOG.md` `[0.33.0] — 2026-07-10` with **Honest scope** (recent-window
  correlation retention; window=1 reduction; paper numbers not ours).
- `paper/EVIDENCE_TABLE.md` — next contiguous rows (continue after the
  Keyformer 139–148 block).
- `pyproject.toml` — version 0.32.0→0.33.0; description
  "...to Keyformer"→"...to MorphKV", 35→36; **preserve PEP 639 metadata**
  (`license="MIT"` + `license-files=["LICENSE"]`, one-line description,
  name-only author, `requires=["setuptools>=77","wheel"]`).

## Phase 9 — Landing + verify

- `landing/index.html` + `assets/main.js`: hero pill 35→36, roll-call
  +MorphKV, ICML 2025 provenance group, whats-new 0.33.0 li (move the NEW pill
  off Keyformer onto MorphKV), Token Eviction cat-count 11→12, algo card
  `#algo-morphkv`, picker card, code tab button + panel `#tab-morphkv`, hero
  badge text in `initBadgeTyping` → `"v0.33.0 — MorphKV-adapted recent-window
  correlation retention shipped"`. **Read each region before editing** (grep is
  not a Read).
- Verify: full pytest (expect +~29 new; the 5 known vecinfer Metal fp16 flakes
  — `test_vecinfer_fused_sdpa`, `test_vecinfer_metal_parity` — stay flaky, do
  NOT chase); `python -m build` + `twine check` at 0.33.0; inspect wheel
  PKG-INFO (Metadata-Version 2.4, License-Expression MIT, Author: Rajveer
  Rathod, 36-method Summary, `morphkv.py`/`morphkv_cache.py` present in wheel);
  docs `npm run build` SUCCESS; benchmark determinism (non-`_ms` fields stable
  across two runs); grep stale-ref sweep (36 consistent across
  README/overview/pyproject/landing).

---

## Release (provide as CHAT TEXT ONLY — never execute)

Per the standing rule, after implementation give the user the v0.33.0 release
sequence as chat text for them to review and run: branch → `git add` → commit
(via `git commit -F` heredoc, **no** Co-Authored-By line, per the user's prior
instruction for these releases) → tag → push → `python -m build` →
`twine check` → `twine upload` → `gh release create` with notes. Do NOT run any
`git add`/`commit`/`tag`/`push`/`gh release`/`twine` yourself.

## What we do NOT implement (state plainly)

- The paper's real attention logits (key-as-query proxy instead).
- RoPE position-ID remapping after eviction.
- Per-head adaptive budgets / the paper's exact refresh cadence.
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers are the paper's, on trained models — not reproduced here.
