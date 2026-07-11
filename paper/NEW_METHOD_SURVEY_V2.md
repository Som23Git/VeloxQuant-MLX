# Phase 1 — New-Method Survey V2 (post-KIVI)

Follow-up to `NEW_METHOD_SURVEY.md` (which led to the KIVI baseline, now shipped
in 0.8.0). The repo now has TurboQuant, RVQ, VecInfer, RaBitQ, CommVQ, QJL,
PolarQuant, RateQuant, SpectralQuant, **KIVI**. The shelf is deep on
*quantization*; the gaps are **attention-sink protection**, **low-rank**, and
**non-uniform datatypes**.

All citations verified to resolve to real papers + venues. Sources at bottom.

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Metal fit | Effort | Verdict |
|---|---|---|---|---|---|
| **KVSink** | arXiv:2508.04257, **COLM 2025** (Su & Yuan) | **Attention-sink preservation** — a plug-and-play layer that improves *any* quantizer, not a 10th competitor | ✅ token-selection logic, no kernel | Low | **CHOSEN** |
| **PALU** | arXiv:2407.21118, **ICLR 2025** (Chang et al.) | **Low-rank projection** of K/V — a genuinely new axis the repo has none of | ⚠️ SVD offline OK; GPU matrix-fusion kernel won't port | Med-High | strong future option |
| **KVQuant** | arXiv:2401.18079, **NeurIPS 2024** (Hooper et al.) | Non-uniform datatype + pre-RoPE keys + dense/sparse outliers | ⚠️ pre-RoPE hook fights immediate-dequant design | Med-High | deferred |
| **KVmix** | arXiv:2506.08018 | Gradient-based per-layer mixed precision | overlaps RateQuant | Med | skip (overlap) |

## Chosen: KVSink (cache-level adaptation)

### What the paper actually does
KVSink ([arXiv:2508.04257](https://arxiv.org/abs/2508.04257), COLM 2025) protects
**attention-sink tokens** — positions that absorb disproportionate attention mass
and are highly sensitive to low-bit quantization. Its mechanism:
1. At a **fixed, model-specific "emergence layer"** (e.g. layer 1–3), certain
   **hidden-state channels** show extreme magnitude outliers.
2. Tokens whose hidden state is in the **top-k** of that outlier channel are the
   sink tokens.
3. Sink tokens are kept in **fp16** (excluded from quantization *and* from
   calibration); everything else is quantized to 2–4 bits.
4. This beats the static "Preserve-First-N" (PFN) baseline because sinks are not
   always the first tokens — KVSink with k=5 often beats PFN with N=20.

### The honest adaptation problem
KVSink's true signal is the **hidden state at the emergence layer**. Our cache
wrappers receive only **per-layer K/V tensors**, not hidden states (by design —
this is what keeps the "3-line, `mlx_lm.generate` unchanged" integration clean).
A literal 1:1 port would require hooking the model forward pass per architecture.

**Adaptation (cache-observable proxy):** sink tokens also exhibit anomalously
large **key L2-norm** — the same outlier-magnitude phenomenon, visible in the K
tensor the cache *does* see. The repo already has `KeyNormObserver`
(`observers/key_norm.py`) built on exactly this signal. So our implementation:

> Identify the top-k highest-key-norm tokens, keep their K/V in fp16, and
> delegate quantization of the rest to a wrapped quantizer.

This is documented as **KVSink-adapted (key-norm sink protection)**, *not* a
faithful port. It preserves KVSink's core idea (dynamic, content-based sink
preservation that beats fixed-prefix) at the cache level, and composes with any
quantizer in the suite. We will measure whether it actually improves
reconstruction over a fixed-prefix baseline; if it does not, we report that as a
negative result.

### Why this is the right pick
1. **Force multiplier, not a competitor** — wraps KIVI/TurboQuant/RVQ/… and
   improves all of them, rather than adding a 10th standalone method.
2. **Fills a real gap** — none of the 10 methods specifically protect sinks, and
   the literature is clear that sinks are where low-bit quantization breaks.
3. **Ports cleanly to Metal** — pure token-selection logic, no exotic kernel, so
   no "memory win but no speed" caveat.
4. **Deterministic and testable** — top-k on key norm, no RNG.

### Planned artifacts (Phases 2–6)
- `veloxquant_mlx/cache/sink_cache.py` — `SinkProtectedKVCache` wrapper around an
  inner cache/quantizer; keeps top-k high-key-norm tokens fp16, with full
  byte-accounting (`compressed_*`, `fp16_*`, `sink_fp16_bytes`).
- Config: `KVCacheConfig(sink_protect=True, n_sink_tokens=k, ...)` or a
  `method="sink"` composition wrapper — decide during implementation to match
  existing composition idioms (`SlidingWindowKVCache` is the precedent).
- Tests: sink-selection correctness (top-k by norm), fp16-preservation of
  selected tokens, byte-accounting, determinism, and a quality comparison vs
  fixed-prefix preservation.
- Benchmark + figures + docs + CHANGELOG + EVIDENCE_TABLE rows, numbers from
  committed `results.json` only.

## Sources (verified)
- KVSink — https://arxiv.org/abs/2508.04257 (COLM 2025)
- PALU — https://arxiv.org/abs/2407.21118 (ICLR 2025); code https://github.com/shadowpa0327/Palu
- KVQuant — https://arxiv.org/abs/2401.18079 (NeurIPS 2024)
- KVmix — https://arxiv.org/abs/2506.08018
