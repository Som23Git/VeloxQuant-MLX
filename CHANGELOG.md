# Changelog

All notable changes to **VeloxQuant-MLX** are documented here.

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
