---
id: sliding-window
title: Sliding Window Cache
sidebar_label: Sliding Window
slug: /guides/sliding-window
---

# Sliding Window Cache

`SlidingWindowKVCache` wraps any VeloxQuant-MLX cache with a token eviction policy. When the sequence length exceeds the window size, old tokens are evicted from the cache, keeping memory bounded regardless of generation length.

## Why use a sliding window?

Standard KV caches grow linearly with sequence length. Even with compression, a 32k-token conversation can exhaust memory on an M2 MacBook. The sliding window bounds the cache size at the cost of losing access to tokens outside the window.

This is useful for:
- Very long conversations or documents
- Streaming generation where you only need recent context
- Memory-constrained devices (M1 8 GB, M2 8 GB)

## Basic usage

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Create the inner compressed cache
config = KVCacheConfig(method="turboquant_rvq", bits=1)
inner_cache = KVCacheBuilder.build(model, config)

# Wrap it with a sliding window of 2048 tokens
cache = SlidingWindowKVCache(
    inner_cache=inner_cache,
    window_size=2048,
    eviction_strategy="fifo",  # oldest tokens evicted first
)

# Use exactly like a normal cache
response = mlx_lm.generate(
    model, tokenizer,
    prompt="Tell me a very long story...",
    max_tokens=8192,    # can exceed window_size — old tokens evicted automatically
    kv_cache=cache,
)
```

## Eviction strategies

| Strategy | Description | Best for |
|---|---|---|
| `"fifo"` | Evict oldest tokens first | General use — maintains recency |
| `"attention"` | Evict lowest-attention-weight tokens | Preserves semantically important tokens |
| `"fixed"` | Keep first N + last M tokens | Long documents with important prefix |

```python
# Attention-based eviction (requires attention score tracking)
cache = SlidingWindowKVCache(
    inner_cache=inner_cache,
    window_size=2048,
    eviction_strategy="attention",
    eviction_attention_window=64,  # score tokens over last 64 steps
)

# Fixed: keep first 128 (system prompt) + last 1920 (recent context)
cache = SlidingWindowKVCache(
    inner_cache=inner_cache,
    window_size=2048,
    eviction_strategy="fixed",
    fixed_prefix_size=128,
)
```

## Combining with any algorithm

`SlidingWindowKVCache` wraps any inner cache — including calibrated algorithms:

```python
from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

store = NpyArtifactStore("./artifacts/")
inner_cache = VecInferKVCache(
    model=model,
    codebook=store.load("vecinfer_codebook"),
    smooth_factors=store.load("vecinfer_smooth"),
)

cache = SlidingWindowKVCache(
    inner_cache=inner_cache,
    window_size=4096,
    eviction_strategy="attention",
)
```

## Monitoring evictions

```python
cache = SlidingWindowKVCache(inner_cache=inner_cache, window_size=2048)

response = mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=4096, kv_cache=cache)

stats = cache.eviction_stats()
print(f"Total evictions: {stats.total_evictions}")
print(f"Current window usage: {stats.current_size} / {cache.window_size}")
print(f"Tokens evicted: {stats.tokens_evicted}")
```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inner_cache` | `KVCache` | Required | Any VeloxQuant-MLX cache |
| `window_size` | `int` | Required | Maximum tokens to keep in cache |
| `eviction_strategy` | `str` | `"fifo"` | `"fifo"`, `"attention"`, or `"fixed"` |
| `fixed_prefix_size` | `int` | `0` | Tokens to always keep at start (for `"fixed"` strategy) |
| `eviction_attention_window` | `int` | `32` | Steps to score attention over (for `"attention"` strategy) |

## See also

- [mlx_lm integration](/guides/mlx-lm-integration)
- [Observers — memory tracking](/guides/observers)
- [API — Cache classes](/api/cache)
