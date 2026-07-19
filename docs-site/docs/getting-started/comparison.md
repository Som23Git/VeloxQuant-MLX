---
id: comparison
title: VeloxQuant-MLX vs. llama.cpp vs. oMLX vs. plain mlx_lm
sidebar_label: vs. llama.cpp / oMLX / mlx_lm
slug: /getting-started/comparison
---

# VeloxQuant-MLX vs. llama.cpp vs. oMLX vs. plain mlx_lm

If you already run local LLMs on Apple Silicon, you've probably used `llama.cpp` (or Ollama/LM Studio, which wrap it), [oMLX](https://omlx.ai/) (a serving layer built on `mlx_lm`), or plain `mlx_lm` directly. All three work today without VeloxQuant-MLX. This page is the honest answer to "why would I add another dependency?"

:::info[Short version]
`llama.cpp` quantizes the KV cache too — but with one fixed scheme applied uniformly to every model and layer, and no eviction. oMLX is a serving layer (continuous batching, OpenAI/Anthropic-compatible API, SSD-tiered cache paging) built on top of `mlx_lm` — a different axis entirely (serving infrastructure, not compression), though it has shipped an experimental, sometimes-toggled TurboQuant-algorithm KV cache feature of its own. Plain `mlx_lm` doesn't quantize the KV cache at all; it stores it fp16, full size, always. VeloxQuant-MLX gives you 41 selectable compression methods (quantization *and* eviction *and* cross-layer merging), each independently tuned per layer, on top of the `mlx_lm` you're probably already using — and it composes with oMLX rather than competing with it, since oMLX consumes `mlx_lm` models the same way VeloxQuant-MLX does.
:::

## The options, side by side

| | Plain `mlx_lm` | `llama.cpp` / Ollama / LM Studio | oMLX | VeloxQuant-MLX |
|---|---|---|---|---|
| What it is | Model loading + generation library | Standalone inference runtime (C/C++) | Serving layer on top of `mlx_lm` (batching, API, paging) | KV cache compression library on top of `mlx_lm` |
| KV cache precision | fp16 (no compression) | Fixed: `q8_0` or `q4_0` via `--cache-type-k` / `--cache-type-v` | fp16 by default; an experimental TurboQuant-algorithm toggle exists, history of being on/off across releases | 1–8 bit, chosen per method |
| Compression scheme | None | One uniform per-tensor quant type, same scheme for every layer | One experimental scheme (RVQ, same family as this library's `turboquant_rvq`, independently implemented) | 41 methods — VQ, RVQ, non-uniform, low-rank, entropy coding, mixed-precision |
| Token eviction (drop stale tokens) | No | No | No | Yes — SnapKV, StreamingLLM, H2O, TOVA, and 8 more |
| Cross-layer compression | No | No | No | Yes — XQuant (code reuse), MiniCache (SLERP merge), xKV (shared subspace) |
| Memory strategy | None | On-device only | SSD-tiered paging (hot RAM / cold SSD blocks) — offloads, doesn't compress | In-memory compression — shrinks what's held, doesn't page to disk |
| Per-layer / per-head tuning | N/A | No — one setting for the whole model | No | Yes — method and bit-width are configurable per layer |
| Calibration step | N/A | None | None | Optional — most methods need none; a few (VecInfer, SpectralQuant) train a codebook once |
| Runtime | Python (MLX, Metal) | C/C++ (Metal backend on macOS) | Python (`mlx_lm` + FastAPI server) | Python (MLX, Metal) — same runtime as plain `mlx_lm` |
| Model format | MLX (safetensors) | GGUF | MLX (safetensors) | MLX (safetensors) — same models plain `mlx_lm` already loads |
| Integration | Native | Native | `brew install omlx` / CLI server | 3 extra lines on top of `mlx_lm` |

:::warning[On oMLX + TurboQuant]
Some articles describe oMLX as shipping "TurboQuant" KV cache support. That refers to Google's TurboQuant paper (arXiv:2504.19874) — the same family this library's `turboquant_rvq` method implements — not a dependency on the `veloxquant-mlx` package. The two are independent implementations of a similar idea; we found no evidence oMLX imports or wraps this library. oMLX's TurboQuant toggle has also been reported broken or temporarily removed across releases (see [omlx#440](https://github.com/jundot/omlx/issues/440), [omlx#1253](https://github.com/jundot/omlx/issues/1253)) — verify current status before relying on it.
:::

## Where llama.cpp actually wins

Be clear-eyed about this: `llama.cpp` is not a strawman.

- **It's the more mature, more portable project.** It runs on everything — Apple Silicon, x86, Linux, Windows, mobile — not just Apple Silicon with Metal.
- **Its KV cache quantization (`q4_0`/`q8_0`) is production-tested at massive scale.** Ollama and LM Studio both build on it, and millions of users run it daily without issue.
- **GGUF is a mature, widely supported model format** with a large pre-quantized model catalog.
- **Zero configuration.** `--cache-type-k q4_0` and you're done — there's no method to choose because there's only one.

If you're not on Apple Silicon, or you want the most battle-tested path with the least decision-making, `llama.cpp`-based tooling is the right default. VeloxQuant-MLX doesn't try to replace that.

## Where oMLX fits — a different axis entirely

oMLX solves a different problem than either of the above: it's a **serving layer**, not a compression scheme. If you need continuous batching, an OpenAI/Anthropic-compatible API endpoint, or SSD-tiered KV paging so a long-idle conversation doesn't sit in RAM, oMLX is the right tool — none of that is in scope for VeloxQuant-MLX at all.

The two are not mutually exclusive, and in principle they compose: oMLX runs `mlx_lm` models the same way VeloxQuant-MLX extends them, so a served model's in-memory KV cache could in theory be compressed by VeloxQuant-MLX while oMLX handles the batching/paging layer around it. That combination hasn't been built or tested by this project — treat it as an open integration opportunity, not a supported path today.

Don't reach for oMLX because you want a smaller KV cache — its own compression story is one experimental, independently-implemented TurboQuant-algorithm toggle (see warning above), not a substitute for VeloxQuant-MLX's 41 methods. Reach for it because you want a local server, not because you want compression.

## Where VeloxQuant-MLX wins

The gap is specifically in **how much control you have over the memory/quality tradeoff**, not raw compatibility:

- **llama.cpp's KV quantization is one fixed scheme for the whole model.** `q4_0` is a uniform 4-bit block quantizer — the same scheme whether the layer is shallow (broad attention) or deep (narrow attention), whether the head is RoPE-sensitive or not. VeloxQuant-MLX's 41 methods exist because no single scheme is optimal everywhere: CommVQ preserves RoPE exactly, PolarQuant fits geometric key clusters, PyramidKV gives early layers a bigger budget than deep ones — none of that is expressible as a `--cache-type-k` flag.
- **llama.cpp has no token eviction.** It quantizes every token's KV pair, forever. VeloxQuant-MLX's eviction methods (StreamingLLM, SnapKV, H2O, and 9 more) drop stale tokens entirely for constant-memory long-context generation — a different lever llama.cpp doesn't expose at all.
- **Compression ceiling is higher.** `q4_0` gets you to 4 bits per element. VeloxQuant-MLX's 1-bit methods (TurboQuant RVQ, RaBitQ, QJL) go further — up to **16× key cache compression** (VecInfer, head_dim=128) — because they're built specifically for the sub-4-bit regime instead of adapting a general block quantizer down to it.
- **You stay in the `mlx_lm` ecosystem.** If you're already loading MLX-format models with `mlx_lm`, VeloxQuant-MLX is three lines, not a runtime switch to GGUF and a different toolchain.

## Real numbers (VeloxQuant-MLX, measured)

These are from this library's own benchmark suite, run against real production models (Llama, Mistral, Qwen, Phi, Gemma 3/4, Falcon) — not projections:

| Metric | Value | Notes |
|---|---|---|
| Max key cache compression | **16×** | VecInfer-1bit, head_dim=128 |
| Metal kernel speedup | **13×** | `quantize_vq` at S=2048 (range 6.9–14.7× across S=128–8192) |
| Peak memory reduction | **98%** | 729 MB → 12 MB, Falcon3-7B shape at the OOM-trigger context length |
| RVQ-1bit compression | **7.5×** | near-zero throughput cost |
| FP16 throughput retained | **100%** | Qwen2.5-7B at 16× compression |
| KIVI-2bit full-KV compression | **~4×** | incl. fp16 residual window; 100–106% of fp16 throughput |
| CommVQ key compression | **64×** | RoPE-commutative VQ, D=128, n_cb=4 |

We don't have head-to-head throughput numbers against `llama.cpp`'s `q4_0`/`q8_0` cache in this repo — different runtime, different hardware paths, and an apples-to-apples run hasn't been published here yet. The honest comparison today is architectural (fixed scheme vs. 41 selectable ones, no eviction vs. 11 eviction methods), not a benchmark race.

## Which one should you use?

```
Are you on Apple Silicon and already using mlx_lm?
├── No  → llama.cpp / Ollama / LM Studio (broader hardware support, zero setup)
└── Yes →
    Do you need a served API endpoint, continuous batching, or SSD-tiered
    paging for many/long-idle conversations?
    ├── Yes → oMLX (serving problem — orthogonal to compression)
    └── No, you're calling mlx_lm.generate() directly →
        Is fp16 KV memory (or a flat q4_0/q8_0 cache) already good enough
        for your context length?
        ├── Yes → Stick with what you have — don't add a dependency you don't need
        └── No, you need more compression, eviction, or per-layer tuning →
            VeloxQuant-MLX
```

In short: reach for oMLX when your problem is *serving* (many requests, long-idle sessions, an API surface). Reach for VeloxQuant-MLX when your problem is *memory* — you've hit a wall a single fixed 4-bit cache can't solve, or you need eviction/cross-layer tricks `llama.cpp` doesn't have. The two questions are independent; you can end up needing both, one, or neither.

## Next steps

- [5-minute quickstart](./quickstart)
- [Algorithm overview](../algorithms/overview) — all 41 methods
- [mlx_lm integration guide](../guides/mlx-lm-integration)
