---
id: observers
title: Observers
sidebar_label: Observers
slug: /guides/observers
---

# Observers

VeloxQuant-MLX includes four observer classes that attach to a cache or quantizer and collect runtime metrics — distortion, latency, memory, and key norms. Observers are non-intrusive: they add no overhead to the quantization logic itself.

## Overview

| Observer | Tracks | Key method |
|---|---|---|
| `DistortionObserver` | Cosine similarity, IP estimation error | `.report()` → `DistortionReport` |
| `LatencyObserver` | Encode/decode timing per layer | `.report()` → `LatencyReport` |
| `MemoryObserver` | Peak compressed vs fp16 memory | `.report()` → `MemoryReport` |
| `KeyNormObserver` | Key magnitude distributions, outlier detection | `.report()` → `KeyNormReport` |

## DistortionObserver

Measures how much the quantized representation deviates from the original:

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.observers.distortion import DistortionObserver

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
config = KVCacheConfig(method="turboquant_rvq", bits=1)
cache = KVCacheBuilder.build(model, config)

observer = DistortionObserver()
observer.attach(cache)

mlx_lm.generate(model, tokenizer, prompt="Hello.", max_tokens=256, kv_cache=cache)

report = observer.report()
print(f"Mean cosine similarity  : {report.mean_cosine_similarity:.4f}")
print(f"Min cosine similarity   : {report.min_cosine_similarity:.4f}")
print(f"IP estimation error     : {report.mean_ip_error:.4f}")
print(f"Per-layer worst layer   : {report.worst_layer}")
```

`DistortionReport` fields:

| Field | Type | Description |
|---|---|---|
| `mean_cosine_similarity` | `float` | Average cosine sim between original and quantized keys |
| `min_cosine_similarity` | `float` | Worst-case cosine sim across all layers and tokens |
| `mean_ip_error` | `float` | Mean absolute inner product estimation error |
| `per_layer_cosine_similarity` | `dict[str, float]` | Per-layer breakdown |
| `worst_layer` | `str` | Layer ID with lowest cosine similarity |

## LatencyObserver

Profiles encode and decode times per layer:

```python
from veloxquant_mlx.observers.latency import LatencyObserver

observer = LatencyObserver()
observer.attach(cache)

mlx_lm.generate(model, tokenizer, prompt="Hello.", max_tokens=512, kv_cache=cache)

report = observer.report()
print(f"Mean encode latency : {report.mean_encode_ms:.2f} ms")
print(f"Mean decode latency : {report.mean_decode_ms:.2f} ms")
print(f"Total encode time   : {report.total_encode_ms:.1f} ms")
print(f"Total decode time   : {report.total_decode_ms:.1f} ms")
print(f"Slowest layer       : {report.slowest_layer}")
```

`LatencyReport` fields:

| Field | Type | Description |
|---|---|---|
| `mean_encode_ms` | `float` | Average ms per encode call |
| `mean_decode_ms` | `float` | Average ms per decode call |
| `total_encode_ms` | `float` | Total encode time across generation |
| `total_decode_ms` | `float` | Total decode time across generation |
| `per_layer_encode_ms` | `dict[str, float]` | Per-layer encode latency |
| `slowest_layer` | `str` | Layer with highest cumulative encode time |

## MemoryObserver

Tracks peak memory usage and computes compression ratio:

```python
from veloxquant_mlx.observers.memory import MemoryObserver

observer = MemoryObserver()
observer.attach(cache)

mlx_lm.generate(model, tokenizer, prompt="Tell me a story.", max_tokens=2048, kv_cache=cache)

report = observer.report()
print(f"Peak compressed memory : {report.peak_compressed_mb:.1f} MB")
print(f"Equivalent fp16 memory : {report.peak_fp16_mb:.1f} MB")
print(f"Compression ratio      : {report.compression_ratio:.1f}×")
print(f"Total tokens processed : {report.total_tokens}")
```

`MemoryReport` fields:

| Field | Type | Description |
|---|---|---|
| `peak_compressed_mb` | `float` | Peak cache memory with compression |
| `peak_fp16_mb` | `float` | What the cache would cost at fp16 |
| `compression_ratio` | `float` | `peak_fp16_mb / peak_compressed_mb` |
| `total_tokens` | `int` | Total tokens written to cache |
| `bytes_per_token` | `float` | Average compressed bytes per token |

## KeyNormObserver

Monitors key magnitude distributions and detects outlier tokens:

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver

observer = KeyNormObserver(
    outlier_threshold=3.0,  # tokens with norm > mean + 3σ are flagged
    window_size=128,         # rolling window for statistics
)
observer.attach(cache)

mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=1024, kv_cache=cache)

report = observer.report()
print(f"Mean key norm        : {report.mean_key_norm:.4f}")
print(f"Std key norm         : {report.std_key_norm:.4f}")
print(f"Outlier count        : {report.outlier_count}")
print(f"Outlier fraction     : {report.outlier_fraction:.2%}")
print(f"Max key norm seen    : {report.max_key_norm:.4f}")
```

## Attaching multiple observers

Multiple observers can be attached to the same cache simultaneously:

```python
from veloxquant_mlx.observers.distortion import DistortionObserver
from veloxquant_mlx.observers.memory import MemoryObserver
from veloxquant_mlx.observers.latency import LatencyObserver

dist_obs = DistortionObserver()
mem_obs = MemoryObserver()
lat_obs = LatencyObserver()

for obs in [dist_obs, mem_obs, lat_obs]:
    obs.attach(cache)

mlx_lm.generate(model, tokenizer, prompt=prompt, max_tokens=1024, kv_cache=cache)

print(f"Distortion: {dist_obs.report().mean_cosine_similarity:.4f}")
print(f"Memory:     {mem_obs.report().compression_ratio:.1f}×")
print(f"Latency:    {lat_obs.report().mean_encode_ms:.2f} ms/encode")
```

## See also

- [Mixed-precision guide — outlier detection](/guides/mixed-precision)
- [Benchmarking guide](/guides/benchmarking)
- [API — Observers](/api/observers-api)
