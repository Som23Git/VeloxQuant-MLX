# Phase 1 — New-Method Survey V13 (post-L2Norm)

Follow-up to `NEW_METHOD_SURVEY_V12.md` (L2Norm-adapted intrinsic key-norm
eviction, shipped in 0.29.0). The repo now spans 32 strategies. This survey
widens the lens again (V12 was deliberately narrow — a two-way choice between
V11's deferred candidates) and evaluates six candidates across the eviction
and quantization families.

**Evidence discipline:** every candidate below was verified against its
arXiv abstract and, where a venue is claimed, a corroborating source (the
venue's own listing or the authors' official repository). No venue is
asserted from memory.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **SKVQ** | arXiv:2405.06219 (**COLM 2024** — confirmed via the authors' repo badge, github.com/cat538/SKVQ; Duanmu, Yuan, Li, Duan, Zhang, Lin) | **Channel reordering** (permute head-dim channels so statistically similar ones share a quant group) and **clipped dynamic quantization** (per-group range shrunk by a searched α to tame outliers). Neither mechanism exists anywhere in the repo | ✅ strong — permutation and clip ratio can be derived from cache-observed data (first flushed chunk) instead of the paper's offline calibration; documented deviation | Medium-low | **CHOSEN** |
| Keyformer | arXiv:2403.09054 (**MLSys 2024**) | Gumbel-noise-regularized accumulating eviction score | ✅ (proxy-attention, same as H2O) | Low | Runner-up: would be the repo's 8th proxy-attention eviction scorer — weakest gap-fill of the verified candidates |
| Scissorhands | arXiv:2305.17118 (NeurIPS 2023) | Persistence-of-importance probabilistic retention | ✅ | Low | Reject: overlaps H2O-adapted heavily; venue already on the provenance strip |
| Q-Filters | arXiv:2503.02812 (preprint) | Query-agnostic scoring via a QK-geometry projection | ⚠️ the filter direction is estimated from *query* vectors, which the cache never sees | Medium | Defer: preprint + weakest cache-observability of the six |
| MorphKV | arXiv:2503.00979 ("submitted to ICML 2025"; acceptance unconfirmed) | Constant-size cache with correlation-aware retention | ✅ | Medium | Defer: venue unverified; proxy-attention family again |
| MiKV | arXiv:2402.18096 (preprint) | Retain would-be-evicted tokens at low precision | ⚠️ ZipCache-adapted already ships importance-driven token-wise mixed precision | Low | Reject: duplicate axis |
| NestedKV | arXiv:2605.26678 (preprint; deferred in V11 and V12) | Multi-time-scale cosine-anomaly importance | ✅ mechanically | High | Defer a third time — same reasoning as V12 (fresh four-part preprint, individually unvalidated pieces) |

---

## Chosen: SKVQ (Duanmu et al., COLM 2024)

### What the paper actually does

"SKVQ: Sliding-window Key and Value Cache Quantization for Large Language
Models" ([arXiv:2405.06219](https://arxiv.org/abs/2405.06219); Haojie
Duanmu, Zhihang Yuan, Xiuhong Li, Jiangfei Duan, Xingcheng Zhang, Dahua Lin
— **COLM 2024**, official code at https://github.com/cat538/SKVQ) targets
extremely low-bit KV quantization (2-bit keys / 1.5-bit values) with three
composable mechanisms:

1. **Channel reordering.** Offline, per layer, cluster the head-dim channels
   by their distribution statistics (KMeans over per-channel features
   computed on a calibration set — 256×4096 WikiText-2 samples) and permute
   them so that channels with similar ranges land in the same quantization
   group. Group min/max then fits every member channel tightly. The
   permutation is fused into the attention projection weights at deploy time.
2. **Clipped dynamic quantization.** Per-group scale/zero are computed
   dynamically at runtime (per token group), but the range is shrunk by a
   clip factor α ∈ (0, 1] found offline by minimizing attention-output MSE —
   trading saturation of a few outliers for finer resolution everywhere else.
3. **Sliding window + sink filter.** The most recent ~128 tokens stay fp16;
   tokens leaving the window are quantized once and frozen. The first ~5
   tokens (attention sinks) are retained at high precision.

### Why this is the right pick

1. **Two genuinely missing mechanisms.** The repo has per-channel *scaling*
   (KIVI keys, KVQuant) and fp16 residual windows (KIVI, NSNQuant), but
   nothing that *regroups channels by statistics* and nothing that *clips*
   the dynamic range instead of covering it. Both are general-purpose tricks
   that the docs can relate to the whole quantization family. Keyformer, by
   contrast, adds one noise term to an eviction family that is already seven
   methods deep.
2. **New venue family.** COLM 2024 becomes the landing page's sixth venue
   group (SIGMOD / ICLR / ICML / SIGCOMM / NeurIPS / AISTATS / EMNLP → +COLM).
3. **The infrastructure is already proven.** The NSNQuant chunk-flush
   residual buffer (0.28.0) *is* SKVQ's sliding window semantics: fp16 tail,
   quantize-once-and-freeze on age-out, path-independent chunk boundaries.
   The quantizer core is KIVI's asymmetric min/max group quant with two
   new twists layered on. Risk is concentrated in well-understood code.
4. **Good release cadence.** After two eviction/VQ releases (NSNQuant,
   L2Norm), 0.30.0 returns to the uniform-quantization core of the library.

### The honest adaptation problem

**1. No offline calibration.** The paper computes the channel permutation
(KMeans on WikiText-2 statistics, fused into projection weights) and the
clip factor (attention-output MSE on calibration data, per block) offline.
A cache-side library sees no calibration set and cannot touch projection
weights. Adaptation:

- *Permutation:* computed **from the first flushed chunk** of actual traffic
  (per layer, per head, separately for K and V), then frozen. The feature is
  the per-channel dynamic range (max − min); the "clustering" is a **sort**
  — sorting a scalar feature and cutting into contiguous groups is the
  optimal 1-D grouping, which is what KMeans on a scalar feature converges
  to anyway. Documented as a deviation (multi-feature KMeans → single-feature
  sort; offline corpus → first observed chunk).
- *Clip factor:* a **per-group grid search at flush time** minimizing
  reconstruction MSE over a fixed α grid (the identity α=1.0 is always in
  the grid, so clipping can never lose to not clipping under the search
  metric). The paper's objective is attention-output MSE offline; ours is
  reconstruction MSE online. Documented. A fixed-α mode
  (`skvq_clip_search=False`, `skvq_clip_alpha`) ships for ablation.

**2. No weight fusion.** The paper hides the permutation inside the
projection matrices; we permute explicitly at flush time and invert on
dequantization. Costs two gathers per flushed chunk; the round-trip is
mathematically identical.

**3. No 1.5-bit values.** The paper's 1.5-bit value packing is a CUDA-kernel
artifact; we ship integer bit-widths (`skvq_bits_key`, `skvq_bits_value`,
defaults 2/2). Documented.

**4. No FP8 metadata.** Scale/zero are stored (accounted) as fp16, same as
every group quantizer in the repo. Byte accounting counts them in full.

**5. Synthetic benchmark honesty.** Channel reordering can only help when
channels are *heterogeneous* (real K/V have notorious outlier channels —
that is the paper's premise). The benchmark therefore runs a
heterogeneous-channel regime (log-normal per-channel scales) **and a
homogeneous control where reordering should buy ~nothing**, and reports
both. The claim "real KV caches have heterogeneous channels" remains the
paper's (and KIVI's/KVQuant's), not our benchmark's.

### What we do NOT implement

- Offline calibration (WikiText-2 KMeans permutation, attention-MSE clip
  search) — replaced by first-chunk statistics as above.
- Weight-fused permutation.
- 1.5-bit value packing; FP8(E4M3) metadata.
- The paper's fused CUDA kernels (throughput story). On Apple Silicon the
  win is memory, measured by byte accounting — same caveat as KIVI/NSNQuant.

### Planned artifacts (Phases 2–6)

See `paper/IMPLEMENTATION_PROMPT_SKVQ.md`:
`veloxquant_mlx/quantizers/skvq.py` (channel_permutation /
apply_permutation / clipped_group_quant / clipped_group_dequant /
skvq_round_trip / bytes helpers), `veloxquant_mlx/cache/skvq_cache.py`
(chunk-flush wrapper modeled on `nsnquant_cache.py` with sink filter and
frozen first-chunk permutations), `KVCacheConfig(method="skvq", …)`, tests
(~25 incl. prefill/decode bit-for-bit equivalence, never-worse clip search,
heterogeneous-vs-homogeneous reorder mechanism test),
`benchmark_scripts/benchmark_skvq.py` + committed results JSON (reorder
on/off × clip on/off × KIVI reference, two channel regimes), docs page,
CHANGELOG 0.30.0, README 32→33, EVIDENCE_TABLE rows, landing page (33
algorithms, COLM 2024 provenance item), version bump 0.29.0 → 0.30.0.

---

## Sources (verified this survey)

- SKVQ — https://arxiv.org/abs/2405.06219 (COLM 2024; official code
  https://github.com/cat538/SKVQ)
- Keyformer — https://arxiv.org/abs/2403.09054 (MLSys 2024)
- Scissorhands — https://arxiv.org/abs/2305.17118 (NeurIPS 2023)
- Q-Filters — https://arxiv.org/abs/2503.02812 (preprint)
- MorphKV — https://arxiv.org/abs/2503.00979 (preprint; ICML 2025 submission)
- MiKV — https://arxiv.org/abs/2402.18096 (preprint)
- NestedKV — https://arxiv.org/abs/2605.26678 (preprint; carried from V11/V12)
