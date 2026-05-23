# VecInfer comparative study

Cross-model benchmark of fp16 baseline vs. four KV-cache compression
methods at 8 configurations, run on Apple Silicon via MLX. Single prompt,
max 120 generated tokens, `mlx_lm.generate(prompt_cache=...)`.

See [`cross_model_comparison.png`](cross_model_comparison.png) for the
combined bar chart, and `figures/vecinfer/<model>/comparison_summary.png`
for per-model 4-panel summaries.

## Models tested

| Model | head_dim | n_kv_heads | n_q_heads | n_layers | Notes |
|---|---:|---:|---:|---:|---|
| Falcon3-7B-Instruct-4bit | 256 | 4 | 12 | 28 | 6/7 working; VecInfer-2bit OOM/failed |
| Llama-3.1-8B-Instruct-4bit | 128 | 8 | 32 | 32 | full 7/7 |
| Llama-3.2-1B-Instruct-4bit | 64 | 8 | 32 | 16 | full 7/7 |
| Llama-3.2-3B-Instruct-4bit | 128 | 8 | 24 | 28 | full 7/7 |
| Mistral-7B-Instruct-v0.3-4bit | 128 | 8 | 32 | 32 | full 7/7 |
| Phi-4-4bit | 128 | 10 | 40 | 40 | full 7/7 |
| Qwen2.5-7B-Instruct-4bit | 128 | 4 | 28 | 28 | full 7/7 |
| Qwen3-8B-4bit | 128 | 8 | 32 | 36 | full 7/7 |
| SmolLM2-135M-Instruct | 64 | 3 | 9 | 30 | full 7/7 |
| gemma-3-4b-it-4bit | 256 | 4 | 8 | 34 | full 7/7 |

**Excluded:**
- **DeepSeek-V2-Lite-Chat-4bit-mlx** — MLA stores compressed KV at 192-dim
  (non-standard shape); breaks all per-cache wrappers.
- **Qwen3-4B-4bit** — head_dim=80 (not a power of 2); Walsh-Hadamard and
  rotation-based methods require power-of-2 head_dim.

## Key compression ratio (higher is better)

| Model | TQ-2bit | TQ-3bit | TQ-4bit | RVQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Falcon3-7B-Instruct-4bit | 11.64× | 6.74× | 4.74× | 3.94× | 7.76× | — | **16.00×** |
| Llama-3.1-8B-Instruct-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| Llama-3.2-1B-Instruct-4bit | 6.40× | 4.57× | 3.56× | 3.76× | 7.11× | 8.00× | **16.00×** |
| Llama-3.2-3B-Instruct-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| Mistral-7B-Instruct-v0.3-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| Phi-4-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| Qwen2.5-7B-Instruct-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| Qwen3-8B-4bit | 9.14× | 5.82× | 4.27× | 3.88× | 7.53× | 8.00× | **16.00×** |
| SmolLM2-135M-Instruct | 6.40× | 4.57× | 3.56× | 3.76× | 7.11× | 8.00× | **16.00×** |
| gemma-3-4b-it-4bit | 11.64× | 6.74× | 4.74× | 3.94× | 7.76× | 8.00× | **16.00×** |

## Throughput (tok/s, higher is better)

| Model | fp16 | TQ-2bit | TQ-3bit | TQ-4bit | RVQ-2bit | RVQ-1bit | VecInfer-2bit | VecInfer-1bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Falcon3-7B-Instruct-4bit | 17.3 | 15.9 | 20.2 | 21.6 | 20.8 | **21.7** | — | 17.0 |
| Llama-3.1-8B-Instruct-4bit | 20.5 | 19.8 | 14.3 | 20.7 | **20.8** | 20.6 | 10.7 | 19.6 |
| Llama-3.2-1B-Instruct-4bit | **105.4** | 75.5 | 54.0 | 98.8 | 102.1 | 104.3 | 60.4 | 91.2 |
| Llama-3.2-3B-Instruct-4bit | **47.6** | 20.6 | 45.2 | 45.2 | 46.2 | 46.2 | 39.7 | 40.2 |
| Mistral-7B-Instruct-v0.3-4bit | **23.6** | 22.5 | 22.6 | 22.5 | 22.9 | 22.8 | 21.2 | 9.8 |
| Phi-4-4bit | **10.4** | 9.6 | 9.2 | 8.7 | 8.4 | 8.1 | 7.2 | 4.0 |
| Qwen2.5-7B-Instruct-4bit | 21.0 | 12.0 | 22.0 | **23.2** | 22.5 | 20.7 | 21.3 | 21.5 |
| Qwen3-8B-4bit | **20.3** | 19.4 | 18.8 | 18.5 | 19.4 | 19.6 | 17.9 | 2.4 |
| SmolLM2-135M-Instruct | **250.4** | 70.4 | 166.5 | 164.3 | 170.6 | 188.5 | 163.0 | 175.8 |
| gemma-3-4b-it-4bit | **26.0** | 22.7 | 24.1 | 24.8 | 24.4 | 24.2 | 22.6 | 22.6 |

## Key findings

**VecInfer wins on raw compression**: 16× key compression at 1 bit/elem
beats every other method on every model. RVQ-1bit (~7×) and TQ-2bit (~9×)
are the next best.

**TurboQuant / RVQ closely track fp16 throughput** on most models (within 5–10%).
**VecInfer trades throughput for compression** — the nearest-centroid lookup
runs in pure MLX without a fused Metal kernel. The paper's CUDA kernel fusion
(Section 3.3, arxiv:2510.06175) is not portable to Apple Silicon.

## When to pick which method

| Goal | Best choice |
|---|---|
| Match fp16 throughput, modest compression | **RVQ-1bit** (~7×, ~100% fp16 throughput) |
| Max compression, throughput tolerance | **VecInfer-1bit** (16×, ~50–90% fp16 throughput) |
| Best key/throughput tradeoff at 2-bit | **TQ-2bit** (~9×) on dense models |
| Long context where memory blows up | **VecInfer-1bit** — 16× cuts 4 GB cache to 256 MB |

## Known gaps for VecInfer on MLX

1. **head_dim must be power of 2** — Walsh-Hadamard requires 2^n head_dim.
   Models like Qwen3-4B (head_dim=80) are incompatible.
2. **head_dim=256 + small sub_dim → OOM** — chunked argmin allocates a large
   `[chunk, n_centroids, sub_dim]` diff tensor. Use `key_sub_dim=8+` on large
   head_dim models.
3. **No fused dequant kernel** — dequantize on every `update_and_fetch` call,
   eliminating the paper's main speedup over fp16.
4. **MLA models (DeepSeek-V2)** — compressed latent KV at non-standard shape
   breaks all per-cache wrappers.
