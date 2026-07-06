---
id: rabitq
title: RaBitQ
sidebar_label: RaBitQ
slug: /algorithms/rabitq
---

# RaBitQ

RaBitQ achieves **1-bit key compression** using a randomised Hadamard transform followed by binary sign packing with IVF (Inverted File Index) clustering. It delivers 6× total KV compression (keys at 1 bit, values at fp16) with zero calibration.

:::warning Apple Silicon required
Hamming distance scoring uses `rabitq_hamming_score` — a Metal kernel with native XOR+popcount instructions.
:::

## How it works

1. **Randomised Hadamard transform** — Keys are multiplied by a random sign matrix then Walsh-Hadamard transformed. This spreads the energy uniformly and makes 1-bit sign encoding close to optimal (Johnson-Lindenstrauss guarantee).

2. **1-bit sign packing** — Each dimension is encoded as its sign (0 or 1) and packed into uint32 words via `turboquant_bit_pack`, giving 16× memory reduction over fp16 keys.

3. **IVF clustering** — Keys are organised into `num_clusters` Voronoi cells. Each key stores its cluster ID plus the 1-bit residual within the cell — improving inner-product approximation over flat 1-bit encoding.

4. **Hamming distance scoring** — Attention scores are approximated by XOR+popcount Hamming distance between packed query bits and each packed key, run on Metal GPU cores.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 1 (+ cluster ID overhead) |
| Value bits | fp16 (default) |
| Total compression | 6× (keys + values combined) |
| Key-only compression | 16× |
| Metal kernel | `rabitq_hamming_score` |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="rabitq",
    num_clusters=64,   # IVF clusters. More = better quality, more memory.
    value_bits=16,     # values stored in fp16
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Describe the water cycle in detail.",
    max_tokens=512,
    kv_cache=cache,
)
```

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer

quantizer = RaBitQQuantizer(num_clusters=64, seed=42)

keys = mx.random.normal(shape=(1, 8, 1024, 128))  # [batch, heads, seq, dim]

encoded = quantizer.encode(keys)
# encoded.indices: packed uint32 bits  [batch, heads, seq, dim//32]
# encoded.cluster_ids: int16           [batch, heads, seq]

decoded = quantizer.decode(encoded)
```

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_clusters` | `int` | `64` | IVF cluster count. Higher = better accuracy, more memory |
| `value_bits` | `int` | `16` | Value bit rate. `16` = fp16, `2` or `4` for compressed values |
| `seed` | `int` | `0` | Random seed for Hadamard sign matrix |

## Tradeoffs

RaBitQ stores values at full fp16 precision by default. This means:

- **Total compression ≈ 6×** (keys save 16×, values unchanged)
- For maximum total compression: pair with `value_bits=2` (trades some value quality)

Compare with [TurboQuant RVQ](../algorithms/rvq) which compresses both keys and values and typically achieves better quality at equal total compression ratio.

## When to use RaBitQ

**Use RaBitQ when:**
- 1-bit key compression is the goal
- Values can stay at fp16
- Zero calibration is required

**Consider TurboQuant RVQ instead when:**
- You need both keys and values compressed
- Quality at 1-bit is important (RVQ outperforms RaBitQ at equal compression)

## See also

- [QJL — simpler 1-bit method](../algorithms/qjl)
- [NSNQuant — universal-codebook VQ](../algorithms/nsnquant): differs because it adapts the data to a fixed codebook (NSN Gaussianization), not the codebook (or geometry) to the data
- [TurboQuant RVQ — better quality at same bits](../algorithms/rvq)
- [API — RaBitQQuantizer](../api/quantizers)
- [Metal API — rabitq_hamming_score](../api/metal-api)
