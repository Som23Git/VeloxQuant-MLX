# Implementation Prompt — KVzip-adapted (v0.34.0)

Execute-cold spec for the **37th** method. This single prompt covers the whole
release: core logic, tests, benchmark, docs, README + hero-pill/count bumps, a
new landing card, the **git tag + GitHub release + PyPI publish** layer, the
**funding-link fix** (Buy Me a Coffee → GitHub Sponsors), and the **JOSS paper
update**. Do all of it.

> **Venue confirmed (re-verify live before citing):** "KVzip: Query-Agnostic KV
> Cache Compression with Context Reconstruction," Jang-Hyun Kim, Jinuk Kim,
> Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, Hyun Oh Song — **NeurIPS 2025 (Oral)**,
> arXiv:2505.23416, official code `github.com/snu-mllab/KVzip`. Re-check the
> arXiv abstract + the official repo live before writing it into
> README/docs/EVIDENCE_TABLE.

Chosen over the survey's standing pick **NestedKV** because NestedKV
(arXiv:2605.26678) is still a bare preprint with **no verified venue** — the same
condition that correctly deferred MorphKV across V13–V15. KVzip has a
peer-reviewed venue (NeurIPS 2025 Oral) **and** a genuinely new mechanism axis.

Model it on the **Keyformer / MorphKV pair** just shipped
(`quantizers/keyformer.py`+`cache/keyformer_cache.py`,
`quantizers/morphkv.py`+`cache/morphkv_cache.py`) and the **H2O pair** —
single-layer, no coordinator, no `.bits`, fp16, lazy per-head state. Ship it as
**"KVzip-adapted (VeloxQuant-MLX implementation)," NeurIPS 2025
(arXiv:2505.23416) — NOT a faithful port.**

---

## The mechanism gap (why this is not a duplicate)

The repo now has **nine** proxy-attention eviction scorers (SnapKV, H2O, TOVA,
PyramidKV, SqueezeAttention, ChunkKV, CaM, Keyformer, MorphKV). **Every one**
scores a stored token by some form of the **attention it receives** — cumulative
(H2O), latest-query (TOVA/SnapKV), or recent-window (MorphKV). All are variations
on "how much is this token attended to."

KVzip's axis is different: **context reconstructability.** Score a KV pair by how
much the model *relies on it to reconstruct its own context* — importance =
the maximum attention a stored key receives **when the model is prompted to
repeat/reconstruct the cached context** (query-agnostic: measured once against a
reconstruction probe, not against a live user query). Low-reconstruction-value
pairs are evicted. That "keep what the model needs to rebuild the context"
framing is not in the repo.

- **H2O / TOVA / MorphKV** = rank by attention *received from real/proxy
  queries*.
- **KVzip-adapted** = rank by attention *received under a fixed reconstruction
  probe* — a query-agnostic importance profile computed in a one-time pass,
  reused across all future queries. Constant, not accumulated per live token.

### The isolable reduction (must pin exactly)

KVzip's reconstruction probe is, in our proxy, **the cached keys themselves used
as the reconstruction queries** (the model rebuilds the context from what it
stored). Design it so that a `kvzip_probe="latest"` degenerate mode — probe =
only the single most-recent key — reduces **bit-for-bit to the TOVA-adapted
latest-token scorer**, and pin that with a dedicated test (the analogue of
MorphKV's `window=1`==TOVA and Keyformer's `tau=0`==H2O collapses). The default
`kvzip_probe="context"` uses the full kept-set-as-reconstruction-probe (mean/max
attention each stored key receives across all recent context rows). **Do NOT
claim any collapse you cannot assert bit-for-bit** — pin only the `latest`
reduction.

---

## Non-negotiable honesty constraints (repeat on EVERY surface)

- **Proxy reconstruction.** A cache never runs the real model to reconstruct
  text. Like every scorer in the family, incoming/stored KEYS stand in for the
  reconstruction queries. The "importance = reconstruction reliance" is a
  documented **proxy**, not the paper's actual repeat-the-context forward passes.
- **`kvzip_probe="latest"` collapses onto the TOVA-adapted latest-token
  scorer** — state this as the honest reference behavior and **assert it with a
  test** (bit-for-bit kept-set equality vs TOVA-adapted).
- **Query-agnostic, one-pass profile — not accumulated.** The reconstruction
  importance is (re)computed from the live keep set each step against the probe;
  no cumulative score array is stored. State this explicitly (contrast H2O's
  forever-accumulator).
- **Not the paper's full algorithm.** We implement the *reconstruction-reliance
  retention rule*, NOT the paper's actual context-reconstruction forward passes,
  NOT head-level context-independent scoring, NOT DuoAttention-style head
  compression, NOT RoPE remapping, NOT a trained-model evaluation. The paper's
  "3–4× cache reduction, ~2× decode latency, negligible loss on LLaMA3.1 /
  Qwen2.5 / Gemma3 up to 170K tokens" numbers are the PAPER's on trained
  models — **never quote them as ours.**
- **Clean mechanism observable** = a constructed geometry where the
  reconstruction-important region differs from the highest-cumulative-attention
  region, so a cumulative (H2O-style) keep set retains the wrong tokens while
  KVzip-adapted retains the reconstruction-critical ones. Report a
  **reconstruction-critical retention rate** as the primary observable; report
  downstream output perturbation as-is (noisy, regime-dependent, with a null
  control that shows no advantage). Do NOT cherry-pick.
- Nothing here is validated on a trained model — offline-synthetic only.

---

## Phase 1 — Survey (write it)

`paper/NEW_METHOD_SURVEY_V17.md`, follow-up to V16. Commit to KVzip. Lead with:
"NestedKV was V16's named next pick but remains an unverified preprint; KVzip
(NeurIPS 2025 Oral) both satisfies the standing verified-venue rule **and** adds
a new axis (context-reconstruction reliance) orthogonal to the nine
attention-received scorers." State the mechanism gap, the honesty crux, the
`kvzip_probe="latest"`==TOVA reduction, and the reconstruction-critical-retention
observable. Carry the non-chosen rows forward (NestedKV, Scissorhands, MiKV,
Anchor-Direction-Projection) and name the **next** fallback explicitly.
Re-verify sources live.

## Phase 2 — `veloxquant_mlx/quantizers/kvzip.py`

Mirror the shape and docstring discipline of `quantizers/morphkv.py`.

- `KVzipState` dataclass: `keys`/`values` (`[n, D]` fp16), `pos` (running
  count), `n_sink`, `budget`, `probe` (`"context"` | `"latest"`), `head_dim`.
  No cumulative score array stored — importance is recomputed from the current
  keep set against the probe each step (query-agnostic, constant, not an
  accumulator).
- `init_kvzip_state(n_sink, budget, head_dim, probe="context")` — guards:
  `budget >= 1`, `n_sink < budget`, `probe in {"context","latest"}`,
  `n_sink < budget` leaves room for eviction.
- `_attention_scores(query_proxy, keys)` — reuse the exact
  softmax(key-as-query · keys · 1/sqrt(D)) helper shared with
  keyformer/morphkv/tova. Factor to a shared helper or duplicate with a comment
  pointing at the canonical one; be consistent.
- `_reconstruction_importance(keys, probe)` — the KVzip signal. For each stored
  key, the **max** proxy-attention it receives across the reconstruction-probe
  rows (paper uses max-over-probe; use max, not mean, and say so). For
  `probe="context"` the probe rows are the full kept set (or the trailing recent
  rows — pick one, document it); for `probe="latest"` the probe is the single
  most-recent key so the score reduces exactly to TOVA-adapted's latest-token
  attention.
- `kvzip_update(state, new_keys, new_values)` — per token: append; if over
  budget, compute `_reconstruction_importance` over the current keep set, force
  sinks (leading) to survive (+inf), evict the **lowest reconstruction
  importance** non-sink token. Constant-size: after each token,
  `n_kept <= budget`. Kept tokens returned in temporal order.
- `kvzip_get_kv`, `kvzip_fp16_bytes` (K+V only; scratch not counted),
  `full_kvzip_fp16_bytes`.
- `__all__` exports.

**latest reduction:** with `probe="latest"` the importance is exactly the
newest key's attention over the keep set — the TOVA-adapted latest-token
eviction. Make this hold *exactly* and pin it.

## Phase 3 — `veloxquant_mlx/cache/kvzip_cache.py`

`KVzipKVCache(_MLXKVCache)` modeled on `MorphKVKVCache`:

- Consume `kvzip_budget`(512) / `kvzip_n_sink`(4) / `kvzip_probe`("context").
- No `.bits`. Lazy per-head state; per-head independence (deterministic, no seed).
- Byte props with KVzip names: `kvzip_kept_bytes` / `full_seq_bytes` /
  `compression_ratio` / `tokens_seen` / `tokens_kept`.
- Validate at construction (delegate to `init_kvzip_state`).
- Both prefill (S>1) and decode (S==1) through the same update loop.

## Phase 4 — `veloxquant_mlx/cache/base.py`

Add `"kvzip"` to the method `Literal`; config block (`kvzip_budget=512`,
`kvzip_n_sink=4`, `kvzip_probe="context"`); import `KVzipKVCache`; factory branch
(`elif config.method == "kvzip"`, no coordinator); extend the unknown-method
error string with `"kvzip"`. **Read each region before editing.**

## Phase 5 — Tests (~32, match MorphKV's count/discipline)

`tests/quantizers/test_kvzip.py` (~19):
- init guards (budget/sink bounds, invalid `probe`).
- budget invariant after each token AND across a block (constant-size).
- sink protection.
- byte accounting (`kvzip_fp16_bytes`, `full_kvzip_fp16_bytes`).
- **`probe="latest"` reduces to the TOVA-adapted latest-token eviction** —
  assert kept-set equality exactly.
- `test_context_probe_can_differ_from_latest` — the default probe is not
  vacuously equal to latest.
- determinism / reproducibility (same input → same keep set, no RNG).
- **`test_reconstruction_geometry_retains_critical`**: the statistical
  mechanism claim — under a planted geometry where the reconstruction-critical
  region ≠ the highest-cumulative-attention region, KVzip retains the
  reconstruction-critical region at a materially higher rate than a cumulative
  (H2O-style) baseline. A rate over several planted seeds, not one lucky run.
- a null control where KVzip shows **no** advantage — include it.

`tests/cache/test_kvzip_cache.py` (~13):
- factory dispatch to `KVzipKVCache`; no `.bits`.
- construction guards; config propagation via `for_model`.
- budget respected across B/H.
- byte props.
- prefill+decode both within budget (NOT an equivalence claim).
- cache-level `probe="latest"` reduction sanity.

## Phase 6 — Benchmark

`benchmark_scripts/benchmark_kvzip.py` + committed `kvzip_benchmark_results.json`:
- SEQ_LENS + BUDGETS grid (match MorphKV's scale, e.g. [256,512] × [32,64]).
- GEOMETRIES = `["reconstruction_shift", "flat"]` (`reconstruction_shift`:
  critical region ≠ cumulative-attention peak, made weak/noisy so the context
  probe materially beats the latest probe; `flat`: null control, no advantage).
- PROBES ∈ `["latest", "context"]` (latest = TOVA reference arm).
- Arms: KVzip(probe), an **H2O-style cumulative cross-check column**, and a
  **random** arm.
- Primary field: **reconstruction-critical retention rate**; secondary: output
  perturbation (as-is). Deterministic in ALL non-timing fields (only `_ms` may
  vary). Offline-synthetic; loads no model. Verify determinism by diffing two
  runs (non-`_ms` fields identical).

## Phase 7 — Docs

- `docs-site/docs/algorithms/kvzip.md` — full page: honesty crux, the
  `probe="latest"`==TOVA reduction, reconstruction-critical observable,
  adaptation limitations, the paper's numbers labeled as the paper's.
- `docs-site/sidebars.ts` — add `'algorithms/kvzip'` after `morphkv`.
- `docs-site/docs/algorithms/overview.md` — thirty-six→thirty-seven + table row
  + bullet. **Read before editing.**
- `docs-site/docs/changelog.md` — v0.34.0 (Latest); move v0.33.0 down.
- Cross-link from `h2o.md` (cumulative contrast) and `tova.md`
  (`probe="latest"`==TOVA).

## Phase 8 — README / CHANGELOG / EVIDENCE_TABLE / pyproject

- README:
  - changelog badge `0.33.0`→`0.34.0` (line ~25).
  - "**thirty-six** compression strategies"→"**thirty-seven**" (line ~35).
  - eviction-cache count "**eleven** token-eviction caches"→"**twelve**", and
    extend the parenthetical list with "…and KVzip's context-reconstruction
    reliance scorer — NeurIPS 2025" (line ~35).
  - "All **36** methods"→"All **37**" (line ~173).
  - method-table row after the MorphKV row (line ~227): KVzip-adapted / `kvzip`
    / "Context-reconstruction reliance eviction (NeurIPS 2025) — keeps the KV
    pairs the model most relies on to reconstruct its own context;
    `kvzip_probe=latest` == TOVA" / `0.34.0`.
  - Sources entry (NeurIPS 2025, arXiv:2505.23416, official code
    `github.com/snu-mllab/KVzip` if verified live).
  - Sweep for stale `36`/`0.33.0`; leave correct MorphKV "since 0.33.0"
    method-table cell intact.
- `CHANGELOG.md` `[0.34.0] — <today>` with **Honest scope** (context-
  reconstruction reliance; `probe="latest"`==TOVA; paper numbers not ours);
  move `[0.33.0]` down.
- `paper/EVIDENCE_TABLE.md` — next contiguous rows (continue after the MorphKV
  149–158 block).
- `pyproject.toml` — version `0.33.0`→`0.34.0`; description
  "...to MorphKV"→"...to KVzip", 36→37; **preserve PEP 639 metadata**
  (`license="MIT"` + `license-files=["LICENSE"]`, one-line description,
  name-only author, `requires=["setuptools>=77","wheel"]`).

## Phase 9 — Landing page (new card + counts)

`landing/index.html` + `assets/main.js`. **Read each region before editing.**
- `<meta name="description">` (line ~10): append KVzip-adapted to the roll-call;
  change "New in 0.33.0…"→"New in 0.34.0: KVzip-adapted context-reconstruction
  reliance eviction (NeurIPS 2025) — keeps the KV pairs the model most relies on
  to reconstruct its own context; kvzip_probe=latest collapses onto TOVA."
- hero pill (line ~60) "36 algorithms"→"37 algorithms"; "See all 36"→"See all
  37" (line ~66).
- whats-new (line ~211): add a `0.34.0` `<li>` at the top with the honest crux;
  keep the 0.33.0 li below.
- Token Eviction `cat-count` (line ~353) `12`→`13`.
- Roll-call in the "See all" details: +KVzip-adapted.
- New algo card `#algo-kvzip` in the Token Eviction group (clone the
  `#algo-morphkv` card block: `card-meta` → `v0.34.0 · NeurIPS 2025`,
  `data-tags` includes the eviction tag, full-desc with the honest crux).
- Picker card + code tab button + panel `#tab-kvzip` (clone the morphkv ones).
- **Move the NEW pill off the MorphKV card onto `#algo-kvzip`.**
- `assets/main.js` `initBadgeTyping` text → `"v0.34.0 — KVzip-adapted
  context-reconstruction reliance eviction shipped"`.

## Phase 10 — Funding-link fix (Buy Me a Coffee → GitHub Sponsors)

The current `https://buymeacoffee.com/rajveer43` **404s** (handle does not
exist). Replace it with **GitHub Sponsors** as the primary, verifiable link;
leave the other options the user named as commented placeholders in FUNDING.yml
for them to fill in.

- `.github/FUNDING.yml` — replace `buy_me_a_coffee: rajveer43` with:
  ```yaml
  github: [rajveer43]
  # ko_fi:            # add your Ko-fi username
  # patreon:          # add your Patreon username
  # custom: []        # e.g. Buy Me a Chai / Razorpay / Instamojo page URLs
  ```
- `README.md`:
  - badge (line ~28): swap the Buy Me a Coffee shield for a GitHub Sponsors
    shield → `https://github.com/sponsors/rajveer43`
    (`img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?...&logo=githubsponsors`).
  - body CTA (line ~631): "buy me a coffee ☕" → "sponsor on GitHub 💜" →
    `https://github.com/sponsors/rajveer43`.
- `landing/index.html` footer (line ~2525): "☕ Buy me a coffee" →
  "💜 Sponsor on GitHub" → `https://github.com/sponsors/rajveer43`.
- Do NOT invent a Buy-Me-a-Chai/Razorpay/Instamojo URL — those go in FUNDING.yml
  as commented placeholders only.

## Phase 11 — JOSS paper update

`paper/joss/paper.md` is badly stale (it names ~10 methods and says "the newest
addition is KIVI"). Bring it current without inflating claims:
- **Summary** & **State of the field**: update the method count to **37**
  interchangeable strategies; note the suite now spans quantization,
  vector-quantization, cross-layer, sliding-window, and a **token-eviction**
  family (attention-proxy scorers **and** the KVzip-adapted
  context-reconstruction axis). Keep the honest framing: adapted
  re-implementations, memory-footprint benefit (not throughput) on Apple
  Silicon, reproducible-from-committed-results discipline.
- **Software design**: the "newest addition … KIVI … deterministic" paragraph is
  stale — either generalize it (the suite spans deterministic and
  path-dependent methods) or refresh the "newest addition" to reflect the
  eviction family. Do not claim KVzip is deterministic in the paper's strong
  sense if it is path-dependent; state it plainly.
- `paper/joss/paper.bib`: add BibTeX entries for the newly cited methods if you
  reference them (KVzip @kim2025kvzip; MorphKV; Keyformer; etc.) — only cite what
  you actually name in the prose. Re-verify each venue/year live before adding.
- Do **not** regenerate `paper.pdf` unless the JOSS toolchain is available and
  the user asks; leave the PDF and `jats/` alone.
- `date:` → today.

## Phase 12 — Verify

- Full pytest (expect +~32 new; the ~6 known vecinfer Metal fp16 flakes —
  `test_vecinfer_fused_sdpa`, `test_vecinfer_metal_parity`,
  `test_vecinfer_cache::test_reconstruction_error_bounded` — stay flaky, do NOT
  chase). Confirm zero non-vecinfer failures and all new KVzip tests pass.
- `python -m build` + `python -m twine check dist/*` at `0.34.0`; inspect the
  wheel PKG-INFO (Metadata-Version 2.4, Version 0.34.0, License-Expression MIT,
  Author: Rajveer Rathod, **37**-method Summary, `kvzip.py`/`kvzip_cache.py`
  present in the wheel).
- Docs `npm run build` SUCCESS.
- Benchmark determinism (non-`_ms` fields stable across two runs).
- Grep stale-ref sweep (37 consistent across
  README/overview/pyproject/landing/JOSS; no lingering `buymeacoffee`).
- End-to-end factory smoke test: `KVCacheConfig(method="kvzip", …)` →
  `KVCacheBuilder.for_model` → a compression_ratio > 1 at a small budget.

---

## Phase 13 — Release layer (provide as CHAT TEXT ONLY — never execute)

**Standing rule:** the user reviews and runs all git/publish commands
themselves. After implementation, give the v0.34.0 release sequence as **chat
text** for them to run — do NOT execute any `git add`/`commit`/`tag`/`push`/
`gh release`/`gh repo edit`/`twine` yourself. Mirror the exact format used for
the v0.33.0 MorphKV release:

1. **Branch:** `git checkout -b release/v0.34.0`
2. **Stage only KVzip + funding + JOSS paths** (explicit `git add` list — do NOT
   stage the unrelated working-tree noise: `dist_preview/`, `blog_drafts/`,
   `dist_pypi/`, `paper/NEW_METHOD_SURVEY_V2.md`/`V3.md`, `paper/joss/jats/`,
   `.claude/`). Include: `veloxquant_mlx/quantizers/kvzip.py`,
   `veloxquant_mlx/cache/kvzip_cache.py`, `veloxquant_mlx/cache/base.py`, both
   new test files, `benchmark_scripts/benchmark_kvzip.py` +
   `kvzip_benchmark_results.json`, `docs-site/docs/algorithms/kvzip.md`,
   `docs-site/sidebars.ts`, `overview.md`, `h2o.md`, `tova.md`, `changelog.md`,
   `README.md`, `CHANGELOG.md`, `paper/EVIDENCE_TABLE.md`,
   `paper/NEW_METHOD_SURVEY_V17.md`, `paper/IMPLEMENTATION_PROMPT_KVZIP.md`,
   `pyproject.toml`, `landing/index.html`, `landing/assets/main.js`,
   `.github/FUNDING.yml`, `paper/joss/paper.md`, `paper/joss/paper.bib`.
3. **Commit:** `git commit -F-` heredoc, **NO Co-Authored-By line**, subject
   `feat(kvzip): KVzip-adapted context-reconstruction reliance eviction —
   v0.34.0`, body covering the mechanism, the honest scope
   (`probe="latest"`==TOVA; paper numbers not ours), the funding-link fix, and
   the JOSS refresh.
4. **Tag:** `git tag -a v0.34.0 -m "..."`.
5. **Push:** `git push -u origin release/v0.34.0` then `git push origin v0.34.0`.
6. **Build + check:** `rm -rf dist build *.egg-info && python -m build &&
   python -m twine check dist/*`.
7. **PyPI:** `python -m twine upload dist/veloxquant_mlx-0.34.0*`.
8. **GitHub release:** `gh release create v0.34.0 --repo
   rajveer43/VeloxQuant-MLX --title "..." --notes "$(cat <<'EOF' … EOF)"` with
   KVzip release notes (mechanism table, honest scope, usage snippet, funding
   note). Escape code fences in the heredoc as `` \`\`\` ``.

Also provide (chat text) an optional `gh repo edit` only if the user wants the
repo's funding/description touched — do not assume it.

## What we do NOT implement (state plainly)

- The paper's real context-reconstruction forward passes (key-as-reconstruction-
  probe proxy instead).
- Head-level context-independent scoring / DuoAttention-style head compression.
- RoPE position-ID remapping after eviction.
- Per-head adaptive budgets.
- Any trained-model perplexity/throughput/accuracy benchmark; the paper's
  headline numbers (3–4× reduction, ~2× decode, negligible loss up to 170K on
  LLaMA3.1/Qwen2.5/Gemma3) are the paper's — not reproduced here.
