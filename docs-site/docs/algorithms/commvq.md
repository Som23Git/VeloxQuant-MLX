---
id: commvq
title: CommVQ
sidebar_label: CommVQ
slug: /algorithms/commvq
---

# CommVQ

CommVQ (Commutative VQ) is designed for models using **Rotary Position Embeddings (RoPE)**. Standard quantization loses positional information because RoPE is applied after keys are written to the cache. CommVQ uses a residual VQ structure that **commutes with RoPE** — so position embeddings can be applied to quantized codes without dequantizing first.

## How it works

Standard KV cache flow with RoPE:
```
k_raw → cache → dequant → apply_rope(position) → attention
```

CommVQ flow:
```
k_raw → apply_rope → CommVQ_encode → cache → CommVQ_decode → attention
```

CommVQ encodes the RoPE-rotated key directly. The codebook is structured so that applying a rotation to the centroid approximates the rotation of the residual — making position-aware decoding possible without storing per-token position metadata.

The Metal kernel `comm_vq_decode_metal` fuses centroid gathering and RoPE application in a single GPU pass.

## Key properties

| Property | Value |
|---|---|
| Calibration | None |
| Key bits | 2–4 |
| Value bits | fp16 (default) |
| Compression | 4–8× keys |
| RoPE compatible | Yes — position applied to codes |
| Metal kernel | `comm_vq_decode_metal` |

## Quickstart

```python
import mlx_lm
from veloxquant_mlx.cache.base import KVCacheConfig, KVCacheBuilder

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

config = KVCacheConfig(
    method="commvq",
    bits=2,        # 2-bit residual VQ = 4× key compression
    value_bits=16, # values at fp16
)
cache = KVCacheBuilder.build(model, config)

response = mlx_lm.generate(
    model, tokenizer,
    prompt="Tell me a story set in ancient Rome.",
    max_tokens=512,
    kv_cache=cache,
)
```

## Using the quantizer directly

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer

quantizer = CommVQQuantizer(bits=2, num_residuals=2)

# Keys should be post-RoPE
keys = mx.random.normal(shape=(1, 8, 512, 128))

encoded = quantizer.encode(keys)
decoded = quantizer.decode(encoded)
```

## Why RoPE compatibility matters

With standard quantization, you must store the position index alongside each key so you can re-apply RoPE after decoding. At long context lengths this metadata overhead adds up. CommVQ eliminates this: the quantized code already encodes positional information, so no position metadata is needed.

This is particularly valuable for:
- Grouped Query Attention (GQA) models where KV heads are shared across many query heads
- Very long contexts (16k+) where position metadata becomes non-trivial
- Deployment scenarios where minimising cache format complexity matters

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bits` | `int` | `2` | Bits per residual pass |
| `num_residuals` | `int` | `2` | Number of residual passes |
| `value_bits` | `int` | `16` | Value bits. `16` = fp16 |

## When to use CommVQ

**Use CommVQ when:**
- The model uses RoPE positional encoding (Llama, Mistral, Qwen, Phi)
- You want to avoid per-token position metadata in the cache
- 2–4 bit key compression is sufficient

**Consider [TurboQuant RVQ](/algorithms/rvq) instead when:**
- Position metadata overhead is acceptable
- You need both key and value compression
- You want higher quality at equal bits

## See also

- [PolarQuant — geometric decomposition](/algorithms/polarquant)
- [Metal API — `comm_vq_decode_metal`](/api/metal-api)
- [API — CommVQQuantizer](/api/quantizers)
