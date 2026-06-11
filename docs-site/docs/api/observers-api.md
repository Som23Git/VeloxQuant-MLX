---
id: observers-api
title: Observers API
sidebar_label: Observers
slug: /api/observers-api
---

# Observers API

`veloxquant_mlx.observers`

---

## DistortionObserver

```python
from veloxquant_mlx.observers.distortion import DistortionObserver
```

Measures cosine similarity and inner product estimation error between original and quantized keys.

### Constructor

```python
DistortionObserver(sample_rate: float = 1.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sample_rate` | `float` | `1.0` | Fraction of tokens to measure (1.0 = all). Lower values reduce overhead. |

### Methods

```python
def attach(self, cache: KVCache | list[KVCache]) -> None
def report(self) -> DistortionReport
def reset(self) -> None
```

**`attach(cache)`** — Registers the observer with one or more cache instances. Hooks into the encode/decode cycle.

**`report()`** — Returns a `DistortionReport` after generation completes.

**`reset()`** — Clears accumulated statistics. Call between runs.

### DistortionReport

| Field | Type | Description |
|---|---|---|
| `mean_cosine_similarity` | `float` | Average cosine sim across all tokens and layers |
| `min_cosine_similarity` | `float` | Worst-case cosine sim |
| `mean_ip_error` | `float` | Mean absolute inner product estimation error |
| `per_layer_cosine_similarity` | `dict[str, float]` | Per-layer breakdown |
| `worst_layer` | `str` | Layer with lowest cosine sim |
| `total_tokens_measured` | `int` | Total tokens included in statistics |

---

## LatencyObserver

```python
from veloxquant_mlx.observers.latency import LatencyObserver
```

Profiles per-call encode and decode latency.

### Constructor

```python
LatencyObserver(warmup_calls: int = 2)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `warmup_calls` | `int` | `2` | Calls to skip before recording (exclude Metal JIT warmup) |

### Methods

```python
def attach(self, cache: KVCache | list[KVCache]) -> None
def report(self) -> LatencyReport
def reset(self) -> None
```

### LatencyReport

| Field | Type | Description |
|---|---|---|
| `mean_encode_ms` | `float` | Average milliseconds per encode call |
| `mean_decode_ms` | `float` | Average milliseconds per decode call |
| `p99_encode_ms` | `float` | 99th percentile encode latency |
| `total_encode_ms` | `float` | Cumulative encode time |
| `total_decode_ms` | `float` | Cumulative decode time |
| `per_layer_encode_ms` | `dict[str, float]` | Per-layer average encode time |
| `slowest_layer` | `str` | Layer with highest total encode time |
| `num_encode_calls` | `int` | Total encode calls recorded |

---

## MemoryObserver

```python
from veloxquant_mlx.observers.memory import MemoryObserver
```

Tracks peak memory and computes compression ratio vs fp16 baseline.

### Constructor

```python
MemoryObserver()
```

### Methods

```python
def attach(self, cache: KVCache | list[KVCache]) -> None
def report(self) -> MemoryReport
def reset(self) -> None
```

### MemoryReport

| Field | Type | Description |
|---|---|---|
| `peak_compressed_mb` | `float` | Peak compressed cache memory in MB |
| `peak_fp16_mb` | `float` | Equivalent fp16 cache memory in MB |
| `compression_ratio` | `float` | `peak_fp16_mb / peak_compressed_mb` |
| `total_tokens` | `int` | Total tokens written to cache |
| `bytes_per_token` | `float` | Average bytes per token across all layers |
| `per_layer_mb` | `dict[str, float]` | Per-layer peak memory |

---

## KeyNormObserver

```python
from veloxquant_mlx.observers.key_norm import KeyNormObserver
```

Monitors key vector norms and detects outlier tokens.

### Constructor

```python
KeyNormObserver(
    outlier_threshold: float = 3.0,
    window_size: int = 128,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `outlier_threshold` | `float` | `3.0` | Norms above `mean + threshold × std` are outliers |
| `window_size` | `int` | `128` | Rolling window size for computing running statistics |

### Methods

```python
def attach(self, cache: KVCache | list[KVCache]) -> None
def report(self) -> KeyNormReport
def reset(self) -> None
```

### KeyNormReport

| Field | Type | Description |
|---|---|---|
| `mean_key_norm` | `float` | Rolling mean of key L2 norms |
| `std_key_norm` | `float` | Rolling std of key norms |
| `max_key_norm` | `float` | Maximum norm seen |
| `outlier_count` | `int` | Total tokens flagged as outliers |
| `outlier_fraction` | `float` | `outlier_count / total_tokens` |
| `mean_outlier_norm` | `float` | Average norm of outlier tokens |
| `per_layer_outlier_count` | `dict[str, int]` | Outliers per layer |

---

## Example — all observers together

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder
from veloxquant_mlx.observers.distortion import DistortionObserver
from veloxquant_mlx.observers.memory import MemoryObserver
from veloxquant_mlx.observers.latency import LatencyObserver
from veloxquant_mlx.observers.key_norm import KeyNormObserver

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
config = KVCacheConfig(method="turboquant_rvq", bits=1)
cache = KVCacheBuilder.build(model, config)

observers = [
    DistortionObserver(),
    MemoryObserver(),
    LatencyObserver(),
    KeyNormObserver(outlier_threshold=3.0),
]
for obs in observers:
    obs.attach(cache)

mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=1024, kv_cache=cache)

dist, mem, lat, norm = [obs.report() for obs in observers]
print(f"Cosine sim : {dist.mean_cosine_similarity:.4f}")
print(f"Compression: {mem.compression_ratio:.1f}×")
print(f"Encode lat : {lat.mean_encode_ms:.2f} ms")
print(f"Outliers   : {norm.outlier_fraction:.1%}")
```

---

## See also

- [Observers guide](../guides/observers)
- [Mixed-precision — outlier handling](../guides/mixed-precision)
- [Benchmarking guide](../guides/benchmarking)
