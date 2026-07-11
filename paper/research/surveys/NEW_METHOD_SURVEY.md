# Phase 1 — New-Method Survey (KV-Cache Compression candidates for VeloxQuant-MLX)

**Goal:** find a published KV-cache method *not already in the repo* (repo has:
TurboQuant, RVQ, VecInfer, RaBitQ, CommVQ, QJL, PolarQuant, RateQuant,
SpectralQuant) that is implementable in MLX on Apple Silicon, gives the paper a
recognized comparison point, and slots into the existing `Quantizer` /
`*KVCache` / `QuantizerRegistry` architecture.

**Evidence discipline (per repo failure mode):** every citation below was
verified to resolve to a real arXiv paper + venue before listing. No invented
IDs. Sources at the bottom.

## Candidate table

| Method | Paper / arXiv (verified) | Core idea | MLX / unified-memory fit | Impl. difficulty | Expected delta vs repo | CUDA-only dependency? |
|---|---|---|---|---|---|---|
| **KIVI** | arXiv:2402.02750, **ICML 2024** (Liu, Yuan et al.) | **Asymmetric**: keys quantized **per-channel**, values **per-token**, group-wise 2-bit, with a small fp16 **residual** (most-recent R tokens kept full-precision). Tuning-free. | **Excellent.** Pure scalar group quant + min/max scales; no rotation, no codebook training, no exotic kernel. All ops are `mx.min/max/round/clip` — native MLX. Per-channel keys need a transpose, trivially expressible. | **Low.** ~1 quantizer + 1 cache wrapper. Deterministic (no k-means → no flakiness). | The **canonical missing baseline**. Every KV-cache paper compares to KIVI; the repo has none. Gives a 2-bit per-channel/per-token reference the other methods can be measured against. | **No.** The paper's CUDA kernel is for speed; the algorithm is plain arithmetic. (Same Metal-vs-CUDA caveat as VecInfer: we get the *memory* win, not their raw speedup.) |
| **KVQuant** | arXiv:2401.18079, **NeurIPS 2024** (Hooper et al.) | Per-channel **pre-RoPE** key quant + **non-uniform (NUQ)** sensitivity-weighted datatype + per-vector dense-and-sparse **outlier isolation** (1% outliers in fp16). | **Good but heavier.** NUQ datatype + pre-RoPE hook needs reaching into the model's RoPE; outlier sparse path needs gather/scatter. Doable in MLX but more invasive. | **Medium-High.** Pre-RoPE interception is the hard part; the repo's caches operate post-projection. | Best low-bit *quality* of the group; but overlaps conceptually with the repo's existing outlier/rotation machinery. | Sparse outlier kernel is custom (CUDA in ref); MLX path would be slower. |
| **GEAR** | arXiv:2403.05527 (Kang et al.) | Quantize the bulk + **low-rank** matrix for quantization error + **sparse** matrix for outliers (three-component residual). | **Moderate.** SVD for low-rank is in MLX (`mx.linalg.svd`), but a per-step low-rank error fit on the decode path is expensive on unified memory. | **Medium.** Low-rank fit + sparse residual bookkeeping. | Strong quality at 4-bit; but the per-token low-rank fit is throughput-hostile on Metal — likely a *negative* throughput story here. | Not strictly, but the efficient version assumes fused CUDA. |
| **ZipCache** | arXiv:2405.14256, **NeurIPS 2024** (He et al.) | **Salient-token** identification via normalized attention score + **channel-separable** token-wise quant; aggressive bits for non-salient tokens. | **Moderate.** Saliency metric needs attention scores (couples to SDPA); channel-separable quant itself is MLX-friendly. | **Medium-High.** Saliency hook into attention is the friction point with the immediate-dequant cache pattern the repo uses. | Good adaptive compression; but saliency requires attention coupling the repo's "quantize→dequantize inside update_and_fetch" pattern deliberately avoids. | No hard CUDA dep, but FlashAttention-coupled in ref. |
| **(orthogonal) H2O / SnapKV** | arXiv:2306.14048 / 2404.14469 | Token **eviction** (drop low-attention tokens), not quantization. | Fits MLX but is a *different axis* — could stack with any quantizer. | Low-Medium | Not comparable on the compression-quality axis; would muddy the apples-to-apples story. | No. |

## Recommendation: **KIVI**

**Implement KIVI.** It is the highest novelty/effort ratio for this repo:

1. **It's the baseline the repo is conspicuously missing.** A reviewer's first
   question about any KV-cache library is "how does it compare to KIVI?" Right
   now the repo cannot answer. Adding it makes every existing method
   (TurboQuant, RVQ, VecInfer, SpectralQuant) directly comparable to the
   most-cited reference in the field.
2. **It's deterministic.** Pure min/max group quantization — **no k-means, no
   unseeded codebook training**, so it will not add to the VecInfer parity
   flakiness documented in `EVIDENCE_TABLE.md`. (Still seed any tie-breaks.)
3. **It's a clean architectural fit.** The asymmetric per-channel-key /
   per-token-value scheme maps onto exactly one new `Quantizer` + one
   `KIVIKVCache(_MLXKVCache)` wrapper with the same byte-accounting fields
   (`compressed_key_bytes`, `fp16_key_bytes`) the other caches expose.
4. **Honest scope up front:** like VecInfer, KIVI's published *speedup* comes
   from a CUDA kernel that does not port to Metal. On Apple Silicon we expect a
   **memory win and a throughput cost** — which we will measure and report, not
   hide. The fp16-residual window (R recent tokens) is the key correctness
   detail; we will implement it.

**Why not the others:** KVQuant and ZipCache both need to reach into RoPE or
attention scores, fighting the repo's deliberate "quantize+dequantize inside
`update_and_fetch`, downstream SDPA sees fp16" design; GEAR's per-step low-rank
fit is throughput-hostile on unified memory. All three are better as *future*
work once the simpler, higher-value KIVI baseline exists.

## Planned artifacts (Phases 2–6)
- `veloxquant_mlx/quantizers/kivi.py` — `KIVIQuantizer(Quantizer)`, registered `"kivi"`.
- `veloxquant_mlx/cache/kivi_cache.py` — `KIVIKVCache(_MLXKVCache)`, per-channel keys / per-token values, fp16 residual window `R`, full byte-accounting.
- Wire `method="kivi"` into `KVCacheConfig` / `cache/base.py` dispatch.
- Tests: encode/decode shape+dtype, seeded reconstruction cosine/SNR vs a justified tolerance, group-quant round-trip exactness at high bit-width, residual-window correctness.
- `benchmark_scripts/benchmark_kivi.py` → `figures/kivi/<model>/results.json` (≥3 models, real fp16 baseline, `hardware` recorded).
- `scripts/plot_kivi.py` → 4 figures + `figures/kivi/results_summary.json`.
- Docs: README section + "Numbers that matter" row, docs-site page, landing update, CHANGELOG entry, `EVIDENCE_TABLE.md` rows.

## Sources (verified)
- KIVI — https://arxiv.org/abs/2402.02750 (ICML 2024); code https://github.com/jy-yuan/KIVI
- KVQuant — https://arxiv.org/abs/2401.18079 (NeurIPS 2024); code https://github.com/SqueezeAILab/KVQuant
- GEAR — https://arxiv.org/abs/2403.05527; code https://github.com/HaoKang-Timmy/GEAR
- ZipCache — https://arxiv.org/abs/2405.14256 (NeurIPS 2024)
