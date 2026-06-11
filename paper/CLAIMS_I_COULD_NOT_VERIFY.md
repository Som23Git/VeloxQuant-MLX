# Claims I could NOT independently verify — double-check before submitting

These are repository/README/blog claims I could not trace to a measured data
file (or that the measured data contradicts). I either omitted them from the
paper, labeled them as extrapolations/reported-by-authors, or flagged the
inconsistency. **Resolve each before submission.**

## Contradicted by data (must fix in README, already handled in paper)
1. **"243 tests passing"** — `grep def test_` = 231, CHANGELOG 0.5.1 = "212",
   `pytest --collect-only` = **314**. Three numbers. Paper cites "314 collected."
   → Update the README badge to the reproducible figure and run the full suite
   once to record the real pass count.
2. **"13× faster hot path"** (blanket) — only the **quantize** kernel is faster.
   The **dequant** kernel is at/below MLX `mx.take` parity (0.75–1.16×). Paper
   scopes the 13× to quantize. → Reword README.
3. **RaBitQ "speedup" framing** — fp16 baselines read `0.000 ms` in
   `figures/RaBitQ/*/results.json` (i.e. never timed), so any RaBitQ speedup is
   a measurement artifact. → Re-run RaBitQ throughput with a real fp16 timer.
4. **RaBitQ search usable** — `recall_at_10` = **0.0–0.4** in
   `figures/RaBitQ/kernel/results.json`. The ANN search path does not retrieve.
   Paper reports this as a negative result. → Fix or remove the search story.
5. **SpectralQuant compression 5.95× (README headline) vs 5.33× (per-model
   table)** — internal inconsistency. Paper uses 5.33×. → Reconcile.

## Plausible but not traced to a measured row (labeled as estimate/reported)
6. **RaBitQ "103k tokens @ 8 GB vs 17k for fp16"** — not a measured row. My
   KV-only linear extrapolation lands near ~10⁵ tokens, but the 103k/17k pair
   implies a *total-RAM* (weights+OS+cache) budget I could not reproduce.
   Paper presents only the measured KV memory rows + a clearly-labeled
   extrapolation. → Publish the exact memory-budget assumptions, or measure it.
7. **RateQuant "2.7× lower perplexity degradation at 2.0 avg bits vs uniform
   2-bit"** — mechanism is unit-tested, but no PPL result file found. Paper does
   **not** claim this number. → Produce the perplexity comparison or drop it.
8. **SpectralQuant cosine deltas (+7.4pp / +10.4pp)** — only PNG figures exist
   under `figures/spectral_quant_*/`; **no results.json**. Paper reproduces from
   the README table and flags the missing machine-readable source. → Emit a json.
9. **CommVQ "64× key compression"** — an analytic storage ratio from the
   codebook config (D=128, n_cb=4), not an end-to-end quality-validated result.
   Paper labels it analytic. → Add a reconstruction-quality measurement at 64×.

## Citation / external-fact checks needed
10. **arXiv IDs and venues cited in the source** — several (RateQuant
    arXiv:2605.06675, Ascend-RaBitQ arXiv:2605.16007, PolarQuant AISTATS 2026,
    TurboQuant ICLR 2026, CommVQ arXiv:2506.18879) carry **2026/2506/2605**
    identifiers that may be placeholders or mis-transcribed. `references.bib`
    has `note:` flags on the uncertain ones. → Verify every citation against
    the actual published paper before submission; fix author lists (several are
    `Anonymous`/`others`).
11. **"TurboQuant published by Google Research (ICLR 2026)"** and other
    attributions — confirm authorship and venue.

## Setup details to confirm
12. **Chip consistency** — BENCHMARK_RESULTS/OPTIMIZATION_FINDINGS state M4/16GB,
    but not every figure records the chip. → Confirm all reported numbers are
    from the same hardware, or annotate per-figure.
13. **"12 models validated"** = 12 `figures/vecinfer/` dirs ran the
    compression sweep; this is **not** task-accuracy validation. → Make sure the
    paper's wording ("engineering sweep") matches what was actually run.
