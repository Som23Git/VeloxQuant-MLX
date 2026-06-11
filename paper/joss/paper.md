---
title: 'VeloxQuant-MLX: A KV-Cache Quantization Suite for LLM Inference on Apple Silicon'
tags:
  - Python
  - machine learning
  - large language models
  - quantization
  - KV cache
  - Apple Silicon
  - MLX
authors:
  - name: Rajveer Rathod
    orcid: 0009-0009-7566-1843
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 11 June 2026
bibliography: paper.bib
---

# Summary

Large language models (LLMs) cache the per-token *key* and *value* projections
of every previous token so that self-attention need not recompute them. This
key–value (KV) cache grows linearly with context length and, on Apple Silicon,
competes with the model weights and the operating system for a single pool of
*unified memory* shared by the CPU, GPU, and Neural Engine. As a result the KV
cache, rather than the model weights, frequently becomes the binding memory
constraint for long-context inference on a Mac. Weight quantization (e.g. GPTQ
[@frantar2022gptq], AWQ [@lin2023awq]) is an offline, one-time operation and
does not bound cache growth, which happens online and changes every generation.

`VeloxQuant-MLX` is an open-source Python library that compresses the KV cache
for `mlx_lm` models on Apple Silicon. It provides a unified interface to a suite
of KV-cache quantization strategies — including re-implementations of published
methods (TurboQuant [@zandieh2026turboquant], QJL [@zandieh2024qjl], VecInfer
[@yao2025vecinfer], RaBitQ [@gao2024rabitq], CommVQ [@commvq2025], KIVI
[@liu2024kivi], RateQuant [@ratequant2026], PolarQuant [@polarquant2026]) and a
spectral-rotation variant (SpectralQuant). Each strategy is exposed behind one
`mlx_lm`-compatible cache interface, selectable by a configuration string and
enabled in three lines without modifying `mlx_lm.generate`. Performance-critical
product-vector-quantization code paths are implemented as hand-written Metal
compute shaders compiled at runtime via `mx.fast.metal_kernel`, with a pure-MLX
fallback for portability and parity testing.

# Statement of need

The local-inference ecosystem on Apple Silicon (`llama.cpp`, Ollama, LM Studio,
and Apple's own `mlx_lm` [@mlx2023]) has optimized weight quantization and
attention kernels but stores the KV cache at full `float16` by default, leaving
context length capped by unified-memory pressure. KV-cache quantization is an
active research area, but published methods ship as research code targeting CUDA
and are not directly usable on Apple's MLX backend, where the kernel fusions
those methods rely on for speed do not port. There is, to our knowledge, no
unified, installable library that brings these methods to MLX with a common API
and Metal-accelerated hot paths.

`VeloxQuant-MLX` fills that gap. It is aimed at ML engineers and researchers who
run LLMs locally on Macs and at researchers who need a reproducible, common
framework in which KV-cache compression methods can be compared on the same
Apple-Silicon hardware. Adding the widely cited KIVI baseline [@liu2024kivi] in
particular lets the other methods in the suite be measured against the field's
reference point. The library ships with a test suite, per-model benchmark
scripts that emit machine-readable result files, and documentation; it is
designed so that compression can be enabled on an existing `mlx_lm` pipeline
with a three-line change.

On Apple Silicon the primary, measured benefit is **memory**: several methods
reduce KV-cache footprint substantially at roughly `float16` generation
throughput, but do not reproduce the raw speedups reported for the original
CUDA implementations. The library reports throughput as measured rather than as
hoped, and includes disclosed negative results (for example, the
approximate-nearest-neighbor search path of the RaBitQ implementation is not
usable for retrieval). Benchmark numbers in the documentation trace to committed
result files produced by the included scripts.

# Acknowledgements

`VeloxQuant-MLX` builds on Apple's MLX framework [@mlx2023] and re-implements
algorithms introduced by the authors of the works cited above; we gratefully
acknowledge that prior research, which this library ports to Apple Silicon
rather than supersedes.

# References
