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
date: 11 July 2026
bibliography: paper.bib
---

# Summary

When a large language model (LLM) generates text, it stores, for every word it
has already processed, two internal vectors — a *key* and a *value* — so that it
does not have to recompute them for each new word. This store is called the
key–value (KV) cache, and it grows steadily as the conversation or document gets
longer. On Apple Silicon Macs, the processor, graphics unit, and neural engine
all share a single pool of memory, so this ever-growing cache competes directly
with the model itself and with everything else running on the machine. In
practice, it is often the KV cache — not the size of the model — that decides how
long a document a Mac can handle before it runs out of memory.

`VeloxQuant-MLX` is an open-source Python library that shrinks this KV cache so
that the same Mac can handle longer inputs. It works with models loaded through
`mlx_lm`, Apple's library for running LLMs on its own hardware, and it can be
switched on by changing three lines of code; the normal text-generation call is
left untouched. The library offers thirty-nine interchangeable compression
strategies, so a user can trade a little accuracy for a lot of memory savings, or
the reverse, by selecting a different option. These span several families:
low-bit *quantization*, *vector quantization* against learned or fixed codebooks,
*cross-layer* sharing that exploits redundancy between network depths, and a
*token-eviction* family that keeps a constant-size cache by dropping (or merging)
the tokens a scoring rule judges least important. Some of the heavy numerical work
is implemented as small programs ("kernels") written for Apple's Metal graphics
interface and compiled on the fly, with a slower but portable pure-Python path
available as a fallback.

# Statement of need

The tooling people use to run LLMs locally on Macs — `llama.cpp`, Ollama, LM
Studio, and Apple's own `mlx_lm` [@mlx2023] — has heavily optimized the
compression of model *weights* and the speed of attention, but stores the KV
cache at full 16-bit precision by default. Because weight compression is an
offline, one-time operation, it does nothing to bound the cache, which is built
up token by token at run time and changes with every generation. As inputs grow,
unified memory becomes the limiting resource.

KV-cache quantization is an active research area, but the published methods are
typically released as standalone research code written for NVIDIA CUDA hardware.
They are not directly usable on Apple's MLX backend, and the custom CUDA kernels
they rely on for their reported speedups do not transfer to Apple's Metal
backend. As a result, Mac users and researchers working in the MLX ecosystem
have no straightforward, installable way to apply or compare these methods.

`VeloxQuant-MLX` addresses this need. Its audience is twofold: engineers who run
LLMs on Macs and want to fit longer contexts into limited memory, and researchers
who need a single reproducible framework in which KV-cache compression methods
can be evaluated on the same Apple-Silicon hardware. The library is installable
from PyPI, integrates with an existing `mlx_lm` pipeline in three lines, and ships
with tests, documentation, and benchmark scripts that emit machine-readable
result files so that every reported number is reproducible.

# State of the field

Published KV-cache compression methods cluster into four families, each
targeting a different point on the compression–quality–speed trade-off.
*Scalar quantization* methods store keys and values at reduced bit-width:
KIVI's tuning-free asymmetric 2-bit quantization [@liu2024kivi], KVQuant's
non-uniform, outlier-aware quantization [@hooper2024kvquant], GEAR's
quantization-plus-low-rank-residual recipe [@kang2024gear], SKVQ's
sliding-window channel reordering [@duanmu2024skvq], and the rotation-and-sketch
approaches of TurboQuant [@zandieh2026turboquant] and QJL [@zandieh2024qjl].
*Vector quantization* methods encode keys or values against a codebook: VecInfer
[@yao2025vecinfer], CommVQ [@commvq2025], the calibration-free universal codebook
of NSNQuant [@son2025nsnquant], and the nearest-neighbor code RaBitQ
[@gao2024rabitq]. *Cross-layer and low-rank* methods exploit redundancy between
network depths or within a head's dimensions instead of within a single token:
PALU's low-rank latent projection [@chang2025palu], MiniCache's cross-layer
SLERP merge of adjacent layers [@liu2024minicache], and xKV's shared-subspace
SVD fit jointly across a layer group [@chang2025xkv]; RateQuant instead
allocates bit-width per layer by rate-distortion theory [@ratequant2026]. A
fourth family keeps the cache at full precision but bounds its *length*,
evicting or merging the tokens a scoring rule deems least useful: from
StreamingLLM's sink-plus-recency window [@xiao2024streamingllm] and the
cumulative-attention heavy-hitter rule of H2O [@zhang2023h2o], through
SnapKV's prefill-window scoring [@yuan2025snapkv], PyramidKV's and
SqueezeAttention's layer-adaptive budgets [@cai2024pyramidkv;
@wang2024squeezeattention], CaM's merge-instead-of-drop rule
[@zhang2024cam], and Keyformer's and MorphKV's refinements to the eviction
signal itself [@adnan2024keyformer; @ghadia2025morphkv], to query-agnostic
reconstruction-reliance scoring as in KVzip [@kim2025kvzip]. These are almost
all distributed as separate research repositories, each with its own interface
and assumptions, and each targeting CUDA.

The build-versus-contribute justification for `VeloxQuant-MLX` is that no existing
package brings these methods to Apple's MLX backend under a common interface, and
none could simply be extended to do so: their performance-critical paths assume
CUDA kernels, and their APIs are not interoperable. Rather than fork one method,
`VeloxQuant-MLX` re-implements a representative set of them against MLX with a
shared cache abstraction, so that they become directly comparable on the same
hardware. Including the widely cited KIVI baseline [@liu2024kivi] is central to
this contribution: it is the reference point most KV-cache papers measure
against, and its presence lets every other method in the suite be assessed
relative to it.

# Software design

The library is organized around a small number of abstractions so that adding or
swapping a method does not disturb the integration layer. A quantizer registry
maps a method name to a `Quantizer` implementation; a cache factory maps a
configuration object to an `mlx_lm`-compatible cache wrapper; and a builder
constructs one cache per model layer, which also enables per-layer mixed-precision
allocation. Each cache wrapper compresses keys and values inside the standard
`update_and_fetch` call and immediately reconstructs them, so that the downstream
attention computation always sees ordinary 16-bit tensors and no changes to the
model or to `mlx_lm.generate` are required.

A deliberate design trade-off concerns where compression pays off. Because MLX
does not expose a sub-byte data type and the published CUDA kernel fusions do not
port to Metal, on Apple Silicon the realized benefit of these methods is reduced
*memory footprint* rather than increased *throughput*. The library is explicit
about this: it reports measured throughput alongside compression so users can see
the actual cost, and it provides a pure-MLX reference path for every accelerated
kernel so that the fast and slow paths can be checked against each other. The
suite deliberately mixes *deterministic* methods — for example KIVI's plain
minimum/maximum group quantization, with no learned codebook and no randomness,
which is reproducible run to run — with *path-dependent* token-eviction methods
whose kept set depends on the order tokens arrive. For the latter the library
documents, and tests, the exact reductions that pin each new scoring rule to an
existing one: KVzip's single-token reconstruction probe reduces bit-for-bit to
the memoryless latest-token TOVA rule, and CaM's cache-merging rule reduces
bit-for-bit to H2O's cumulative-attention eviction when its merge step is
disabled (`cam_merge="drop"`). Each new eviction method added to the suite is
required to state and test such a reduction against an existing method where
one exists, so that a method's behavior is characterized rather than asserted.

# Research impact statement

`VeloxQuant-MLX` is released on PyPI and is accompanied by a documentation site,
per-method benchmark scripts, and committed result files for each benchmarked
model, so that its claims are reproducible rather than aspirational. Benchmarks
are recorded for a range of production 4-bit models (including Llama, Qwen,
Mistral, Phi, Gemma, and Falcon families) on Apple M-series hardware, with each
run writing a machine-readable results file that records the hardware, the
sequence lengths, and the realized compression and throughput. The repository
also documents negative results explicitly — for example, that the
approximate-nearest-neighbor search path of the RaBitQ implementation does not
achieve usable retrieval recall — so that users are not misled about scope. By
providing a common, reproducible framework on Apple Silicon, the library is
intended to lower the barrier to applying and fairly comparing KV-cache
compression methods in the growing MLX ecosystem.

# AI usage disclosure

Generative AI tools (Anthropic's Claude) were used during the development of this
software, its documentation, and this paper: to help draft and refactor code, to
help write documentation and prose, and to help structure benchmarks. All
AI-assisted output was reviewed by the author. Correctness was verified through
the repository's automated test suite and through benchmark scripts whose
numerical results are committed to the repository; the project follows a rule
that no performance or compression figure is reported unless it traces to a
committed result file produced by a script in the repository, and an internal
audit was carried out to remove claims that could not be substantiated from
measured data.

# Acknowledgements

`VeloxQuant-MLX` builds on Apple's MLX framework [@mlx2023] and re-implements
algorithms introduced by the authors of the works cited above; we gratefully
acknowledge that prior research, which this library ports to Apple Silicon rather
than supersedes. The author received no financial support for this work.

# References
