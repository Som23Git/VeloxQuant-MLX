# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

## [0.3.1] — 2026-05-10

### Changed
- README restructured with TOC, algorithm picker table, per-model benchmark tables,
  and throughput optimization journey. All emojis removed for plain-text rendering.
- Distribution metadata now reflects the new structure.

## [0.3.0] — 2026-05-10

### Added
- **`TurboQuantRVQ`** — two-pass Residual Vector Quantization quantizer that lifts
  2-bit cosine similarity from 0.69 → **0.98** and SNR from −0.5 dB → **13.2 dB**.
  Stage 1 uses N(0, 1/d) Lloyd-Max; stage 2 fits a Laplacian PDF on the per-coordinate
  residual. Total storage 2·b bits/dim. Registered as `turboquant_rvq` in the registry.
- **`AdaptiveScalarCodebook`** — wrapper that refits codebook centroids from observed
  post-rotation distribution after a calibration phase. Plumbed via
  `TurboQuantProd(use_adaptive_codebook=True)` and `TurboQuantProdAdaptive`.
- **Adaptive JL sketch dimension** — `TurboQuantProd.m_default(d, b)` now returns
  `d` at b ≤ 2 and `min(d, 64)` at b ≥ 3, doubling the QJL correction budget at 2-bit.
- **Optimization journey figure** — [`figures/updated_tests/optimization_journey.png`](figures/updated_tests/optimization_journey.png).
- **`OPTIMIZATION_FINDINGS.md`** — full writeup of bottleneck analysis and four-stage
  speedup attribution.
- **`benchmark_mistral7b_v2.py`** and **`benchmark_qwen3_4b_v2.py`** — 5-config v2
  benchmark scripts that include `TurboQuantRVQMLXKVCache` alongside the existing
  fp16/2/3/4-bit configurations.
- **`test_2bit_improvements.py`** — synthetic validation script with asserts for all
  three 2-bit accuracy improvements.

### Changed
- **Throughput parity with fp16** for quantized configs on memory-bound models:
  Mistral 7B RVQ 2-bit at 22.3 tok/s vs fp16 22.1 tok/s. Qwen3 4B RVQ 2-bit at
  36.0 tok/s vs fp16 39.2 tok/s (92% of fp16). Achieved via four sequential changes:
  1. Single shared quantizer with `(B·H·S, D)` flat batching (eliminates per-head Python loop).
  2. Hadamard rotation by default in benchmark wrappers (`use_hadamard=True`).
  3. Boundary-sum `quantize()` in `ScalarCodebook` (replaces broadcast-argmin).
  4. Dropped redundant fp32 ↔ fp16 casts in `update_and_fetch`.
- `ScalarCodebook.__init__` now sorts centroids and precomputes Voronoi boundaries
  in `self._boundaries_mx`. `quantize()` returns 100% index-match output vs the prior
  argmin path.
- `TurboQuantMLXKVCache` and `TurboQuantRVQMLXKVCache` in `benchmark_core.py` use a
  single shared quantizer instance instead of `n_kv_heads` separate ones.

### Performance
- Mistral 7B RVQ 2-bit: **17.7 → 22.3 tok/s** (+26%).
- Qwen3 4B RVQ 2-bit: **24.8 → 36.0 tok/s** (+45%).
- Boundary-sum quantize verified bitwise-identical to broadcast-argmin (100.00% index match on synthetic test).

### Quality
- RVQ 2-bit synthetic cosine **0.9766** preserved through every optimization step.
- Real-model output completeness preserved at every step:
  - Mistral 7B: 201/201 tokens across all 5 configs.
  - Qwen3 4B `<think>` mode: 199/200 tokens for RVQ 2-bit (vs 50/200 for single-pass 4-bit).

## [0.2.0] — 2025-05-07

### Added
- Published to PyPI as `VeloxQuant-MLX`
- `veloxquant` CLI entry point (alias for `mlx-kv-quant`)
- 2-bit quantization support in benchmark suite (11.6× compression ratio)
- Per-model benchmark scripts: Falcon3-7B, Mistral-7B, Qwen3-4B, Qwen3-8B, Qwen2.5-32B, Gemma-4, Phi-4
- `benchmark_core.py` unified benchmark runner with 6-figure report generation
- Validated across 7 models: near-lossless at 3-bit and 4-bit; 2-bit degrades gracefully

### Changed
- Package distribution name renamed from `mlx-kv-quant` → `VeloxQuant-MLX`
- Status classifier updated from Alpha → Beta

## [0.1.0] — 2025-04-01

### Added
- Initial implementation of TurboQuant KV cache quantization for Apple Silicon MLX
- PolarQuant and QJL algorithms
- Chain-of-Responsibility quantization pipeline
- Lloyd-Max scalar codebooks
- Random orthogonal rotation preconditioner
- Builder pattern (`KVCacheBuilder`) for fluent cache construction
- Observer framework (latency, memory, distortion)
- Precompute CLI for offline codebook generation
- Full test suite
