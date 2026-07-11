# Implementation Prompt — Keyformer-adapted (v0.32.0)

Execute-cold spec for the 35th method, chosen in
`NEW_METHOD_SURVEY_V15.md`. Modeled on the H2O pair
(`quantizers/h2o.py`, `cache/h2o_cache.py`) — Keyformer **is** H2O plus a
Gumbel-noise regularizer on the eviction ranking. Ship it as
"Keyformer-adapted (VeloxQuant-MLX implementation)," MLSys 2024
(arXiv:2403.09054), not a faithful port.

**Non-negotiable honesty constraints (repeat on every surface):**
- The paper redraws & anneals Gumbel noise across generation; we draw ONE
  deterministic Gumbel value per token position (seeded by `keyformer_seed` +
  a per-head running position) and FREEZE it. Preserve the intent, not the
  schedule; never claim equivalence.
- `keyformer_tau=0` MUST collapse onto H2O-adapted bit-for-bit — assert it.
- Base score is the key-as-query proxy (no true query at cache level).
- The clean mechanism observable is late-riser SURVIVAL RATE, not downstream
  perturbation (which is noisy/regime-dependent — report as-is).

## Phase 1 — Survey (done: V15)

## Phase 2 — `veloxquant_mlx/quantizers/keyformer.py`
- `KeyformerState` (keys/values/scores/gumbel/pos/n_sink/budget/recent/tau/seed).
- `init_keyformer_state` — guards: `tau >= 0`, `n_sink + recent < budget`.
- `_attention_scores` — softmax(key-as-query · stored keys · scale) (H2O's).
- `_gumbel_at(seed, pos)` — deterministic Gumbel(0,1) via inverse-CDF over a
  seeded MLX key; same (seed,pos) → same value regardless of chunking.
- `keyformer_update` — per token: accumulate proxy mass; append with score 0
  and a frozen Gumbel draw; over budget evict lowest `score + tau·gumbel`,
  sinks (leading) and `recent` (trailing) forced to survive.
- `keyformer_get_kv`, `keyformer_fp16_bytes` (K+V only; scores/gumbel not
  counted, like H2O), `full_keyformer_fp16_bytes`.

## Phase 3 — `veloxquant_mlx/cache/keyformer_cache.py`
- `KeyformerKVCache(_MLXKVCache)` modeled on `H2OKVCache`; consumes
  `keyformer_budget`(512)/`keyformer_n_sink`(4)/`keyformer_recent`(0)/
  `keyformer_tau`(1.0)/`keyformer_seed`(0); per-head seed offset; byte props
  `keyformer_kept_bytes`/`full_seq_bytes`/`compression_ratio`/`tokens_seen`/
  `tokens_kept`. No `.bits`. Validate at construction.

## Phase 4 — `cache/base.py`
- Add `"keyformer"` to method Literal; config block; import; factory branch
  (no coordinator); extend unknown-method error string.

## Phase 5 — Tests (~29)
- `tests/quantizers/test_keyformer.py` (17): init guards, budget invariant
  (token/block), sink & recent protection, byte accounting, Gumbel
  determinism/reproducibility, **`tau=0`==H2O collapse**, `tau=0`
  seed-invariance, positive-tau changes kept set, **late-riser survival rate
  higher with noise on**.
- `tests/cache/test_keyformer_cache.py` (12): factory dispatch, no `.bits`,
  construction guards, config propagation, budget across B/H, byte props,
  prefill+decode both within budget (NOT equivalence), cache-level `tau=0`
  seed-invariance.

## Phase 6 — Benchmark
- `benchmark_scripts/benchmark_keyformer.py` + committed results JSON:
  tau ∈ {0,2,6} (tau=0 seed-invariant), H2O cross-check column, random arm,
  `late_riser` + `stable` geometries, survival-rate field. Deterministic in
  all non-timing fields. Offline-synthetic; loads no model.

## Phase 7 — Docs
- `docs-site/docs/algorithms/keyformer.md`; sidebar entry after qfilters;
  overview count thirty-four→thirty-five + table row + bullet; changelog
  v0.32.0 (Latest); cross-link from h2o.md.

## Phase 8 — README / CHANGELOG / EVIDENCE_TABLE / pyproject
- README: badge 0.31.0→0.32.0; thirty-four→thirty-five; nine→ten... ten→eleven
  eviction caches; "All 34"→35; method-table row; Sources entry (MLSys 2024).
- CHANGELOG.md `[0.32.0]` with "Honest scope".
- EVIDENCE_TABLE rows 139–148.
- pyproject version 0.31.0→0.32.0; description "...to Q-Filters"→"...to
  Keyformer"; PEP 639 metadata preserved.

## Phase 9 — Landing + verify
- `landing/index.html` + `assets/main.js`: hero pill 34→35, roll-call
  +Keyformer, MLSys 2024 provenance group, whats-new 0.32.0 li, Token Eviction
  cat-count 10→11, algo card `#algo-keyformer` (move NEW pill off qfilters),
  picker card, code tab button + panel `#tab-keyformer`, hero badge text.
- Verify: full pytest (expect +29, known vecinfer flakes unchanged),
  `python -m build` + `twine check` at 0.32.0, PKG-INFO metadata, docs
  `npm run build`, benchmark determinism, grep sweep for stale 34/0.31.0 refs.
