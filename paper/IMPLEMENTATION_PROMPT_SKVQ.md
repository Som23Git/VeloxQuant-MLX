# Implementation Prompt — SKVQ-adapted sliding-window quantization (v0.30.0)

Execute after `paper/NEW_METHOD_SURVEY_V13.md` (Phase 1, done). Ship
`method="skvq"` as v0.30.0: channel-reordered, clip-searched, per-token
group quantization behind a sliding fp16 window with an attention-sink
filter. Label everywhere as **"SKVQ-adapted (VeloxQuant-MLX
implementation)"** — inspired by arXiv:2405.06219 (COLM 2024), not a
faithful port (no offline calibration, no weight fusion, no 1.5-bit values,
no FP8 metadata; deviations documented in the survey and docs page).

House rules (unchanged from previous releases):
- Never fabricate benchmark numbers; report only what a committed results
  JSON actually contains. Two-regime honesty: heterogeneous channels
  (mechanism should win) AND homogeneous control (should buy ~nothing).
- Cache wrappers subclass `mlx_lm.models.cache.KVCache`; round-trip inside
  `update_and_fetch`; never expose `.bits` (SDPA routing); expose
  `assigned_avg_bits`.
- Single-layer method: wire via a `KVCacheFactory.create` branch; no
  `_build_*`, no coordinator, no `__init__.py` exports (xKV precedent).
- Deterministic: no RNG anywhere in the shipped path.
- CHANGELOG gets an "Honest scope" section. EVIDENCE_TABLE rows for every
  claim. Version bump in `pyproject.toml` only (docs read it).

---

## Phase 2 — Quantizer primitives: `veloxquant_mlx/quantizers/skvq.py`

Functional style (knorm/nsnquant precedent), pure MLX, no classes needed.

- `channel_permutation(x: mx.array [N, D]) -> mx.array [D] int32`
  (grouping happens downstream by contiguous cutting, so the permutation
  itself needs no group_size). Per-channel feature = dynamic range
  `max − min` over the N rows;
  permutation = `argsort(range)` (ties broken by argsort's stable order →
  deterministic). Sorting a scalar feature + contiguous cutting is the 1-D
  analogue of the paper's KMeans grouping (documented deviation).
- `invert_permutation(perm) -> mx.array [D] int32`
- `apply_permutation(x [..., D], perm) -> x[..., perm]` (gather on last axis)
- `clipped_group_quant(x [N, D], bits, group_size, alphas) ->
  (codes uint8 [N, D_pad], lo f32 [N, G], scale f32 [N, G])`
  Per-token groups along the channel axis (G = ceil(D / group_size); ragged
  final group padded by repeating the last channel, KIVI precedent). For
  each α in `alphas`: clip window centered on the group midpoint —
  `lo = mid − α·(gmax−gmin)/2`, `scale = max(α·(gmax−gmin)/levels, eps)`,
  `codes = clip(round((x − lo)/scale), 0, levels)`. Reconstruct every arm,
  pick per-group `argmin` MSE. α is *not stored* — it is folded into
  (lo, scale). The *default* grid contains 1.0 so search never loses to
  plain min/max under the search metric; a caller-supplied grid may omit it
  (that is exactly the fixed-α ablation mode).
- `clipped_group_dequant(codes, lo, scale, group_size, d) -> x_hat [N, D] f32`
- `skvq_round_trip(x [N, D], perm, bits, group_size, alphas) -> x_hat`
  permute → clipped_group_quant → dequant → inverse permute; returns input
  dtype.
- `skvq_compressed_bytes(n_tokens, d, bits, group_size) -> int`
  `ceil(n·D·b/8)` code bytes + `n·G·2·2` bytes fp16 (lo, scale).
- `skvq_fp16_bytes(n_tokens, d) -> int` = `n·D·2`.
- `DEFAULT_ALPHA_GRID = (1.0, 0.97, 0.94, 0.90, 0.85)`.
- Guards (ValueError, at call): bits ∉ [1, 8]; group_size < 1; empty
  alphas; any α ∉ (0, 1].

Tests `veloxquant_mlx/tests/quantizers/test_skvq.py` (~11):
permutation validity + inverse round-trip; sorted grouping shrinks
within-group range spread on heterogeneous channels; α=1-only grid equals
plain asymmetric min/max group quant (bit-for-bit vs a manual numpy
reference); clip search per-group MSE never worse than α=1; monotonic error
in bits (2 ≥ 4 ≥ 8); reorder reduces round-trip MSE at 2 bits on
heterogeneous channels and helps ~nothing on homogeneous (assert
heterogeneous improvement strictly greater); determinism; shapes/dtypes;
guards; bytes helpers.

## Phase 3 — Cache wrapper: `veloxquant_mlx/cache/skvq_cache.py` + wiring

`SKVQKVCache(_MLXKVCache)`, modeled on `nsnquant_cache.py` chunk-flush:

- Config fields (add to `KVCacheConfig` after the knorm block):
  `skvq_bits_key=2`, `skvq_bits_value=2`, `skvq_group_size=32`,
  `skvq_window=128` (chunk size / fp16 sliding window),
  `skvq_n_sink=5` (paper's filter: first tokens stay fp16),
  `skvq_reorder=True` (False = ablation), `skvq_clip_search=True`,
  `skvq_clip_alpha=1.0` (used when search off), `skvq_max_ctx=8192`.
- Add `"skvq"` to the method Literal, factory import + branch (comment: no
  coordinator; permutations are per-layer state frozen from the first
  flushed chunk), extend the unknown-method error string.
- Build-time guards: bits in [1,8] each; `skvq_window >= 2`;
  `0 <= skvq_n_sink < skvq_window` (sinks live entirely inside chunk 0);
  `skvq_clip_alpha ∈ (0, 1]`; `skvq_group_size >= 1`.
- Flush loop identical to NSNQuant: while `offset − q_end >= window`,
  round-trip chunk `[q_end, q_end+window)` in place; frontier advances in
  whole chunks (path independence by construction).
- **First flush computes per-head permutations** for K and V independently
  from that chunk (`[B·window, D]` per head), then freezes them for the
  cache's lifetime. `skvq_reorder=False` uses identity.
- **Sink filter:** chunk 0 restores rows `< skvq_n_sink` to their original
  fp16 values after the round-trip (they are also *accounted* as fp16, not
  compressed).
- Byte accounting mirrors NSNQuant: `compressed_key_bytes`,
  `compressed_value_bytes`, `fp16_key_bytes`, `fp16_value_bytes`,
  `residual_fp16_bytes` (snapshot: un-flushed tail + sinks),
  `quantized_tokens`, `assigned_avg_bits`, `tokens_seen`.

Tests `veloxquant_mlx/tests/cache/test_skvq_cache.py` (~14):
build-time validation; short-sequence passthrough is exact; frontier
advances in whole chunks; sink rows bit-exact after flush; **prefill vs
token-by-token decode bit-for-bit equivalence**; frozen permutations
(second chunk reuses chunk-0 perms — expose `key_perms`/`value_perms`
properties); reorder=True beats reorder=False on heterogeneous-channel
data; quantized region actually differs from fp16 input at 2 bits (K and V
both); byte accounting matches the formulas; compression ratio sensible;
no `.bits` attribute; determinism; `skvq_max_ctx` guard; `for_model`
wiring incl. a non-attention layer falling back to plain KVCache.

## Phase 4 — Benchmark: `benchmark_scripts/benchmark_skvq.py` + JSON

Offline-synthetic (no model download). Sweep `SEQ_LENS=[512, 1024]`,
`BITS=[2, 4]`, `REGIMES=["heterogeneous", "homogeneous"]`; head_dim 64,
group 32, window 128, sink 5. Heterogeneous keys: per-channel scales drawn
log-normal (σ≈1.2, a few dominant channels — the KIVI/KVQuant/SKVQ premise);
homogeneous: unit scales. Arms, all at matched bits/group/window:
1. `skvq` (reorder + clip search)
2. `skvq_noreorder` (clip only)
3. `skvq_noclip` (reorder only, α=1)
4. `kivi` (repo's per-channel-key reference at the same bits)

Metrics per row: key reconstruction MSE (quantized region only), attention
output perturbation (1 − cosine, probe queries, same metric family as
knorm/CaM), compressed bytes/token, flush ms. Print an honest-reading
footer: reorder's win exists only under channel heterogeneity; KIVI's
per-channel scheme is a strong baseline for keys and may win rows —
report whatever the numbers say. Save
`benchmark_scripts/skvq_benchmark_results.json`; verify quality fields are
deterministic across two runs (timing fields excluded).

## Phase 5 — Docs site

- `docs-site/docs/algorithms/skvq.md`: mechanism, the two new tricks
  related to the existing family (KIVI = per-channel scales, KVQuant =
  outlier isolation, SKVQ = regroup + clip), usage snippet, adaptation
  notes (all five deviations from the survey), evidence section quoting
  only committed-JSON numbers, when-to-use vs KIVI/NSNQuant table.
- `sidebars.ts`: `'algorithms/skvq'` after `'algorithms/kivi-sink'`
  (quantization cluster, next to KIVI).
- `overview.md`: count 32→33, comparison row, quantization family bullet.
- Cross-links: `kivi.md` (SKVQ as the reorder+clip evolution of the same
  group quantizer) and `nsnquant.md` (shared chunk-flush window idiom).
- `changelog.md`: v0.30.0 — Latest entry above v0.29.0 (demote its badge).

## Phase 6 — Root docs + version

- `CHANGELOG.md`: `[0.30.0]` with Added / Honest scope.
- `README.md`: badge → changelog-0.30.0, "thirty-three compression
  strategies", method table row (after KIVI-sink row), All 33, Sources
  entry (COLM 2024), EVIDENCE_TABLE reference.
- `paper/EVIDENCE_TABLE.md`: rows for every shipped claim.
- `pyproject.toml`: 0.29.0 → 0.30.0.

## Phase 7 — Landing page

`landing/index.html`: hero pill 32→33 + details `· SKVQ`; meta description
"New in 0.30.0"; what's-new 0.30.0 `<li>` (top, others keep order); filter
`All (31)`; Quantization `cat-count` +1; picker card `#algo-skvq`; full algo
card in the quantization group (chip color pick an unused hue); provenance
strip: new `COLM 2024 <em>SKVQ</em>` venue item; code tab `data-tab="skvq"`
+ panel after the nsnquant panel; requirements/tests line updated to the
new totals. `assets/main.js`: hero badge
`"v0.30.0 — SKVQ-adapted sliding-window quantization shipped"`.

## Phase 8 — Final verification

- Full suite: `python3 -m pytest veloxquant_mlx/tests -x -q` — expect only
  the pre-existing vecinfer Metal fp16-tolerance flakes (do not chase).
- `python3 -m build --outdir <scratchpad>/dist_check30` + `twine check`.
- `cd docs-site && npm run build`.
- Benchmark JSON determinism re-run.
- Grep sweep: no stale "0.29.0"/"32 algorithms"/"thirty-two" where the new
  release should be referenced (legit historical mentions stay).

## Phase 9 — Release commands (chat only; the user runs them)

`git add` (explicit file list), `git commit` via a `-F` message file,
`git tag -a v0.30.0`, `git push origin master --follow-tags`,
`gh release create v0.30.0` notes block, PyPI:
`rm -rf dist/ && python3 -m build && twine check dist/* && twine upload dist/*`.
