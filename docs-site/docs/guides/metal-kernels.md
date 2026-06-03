---
id: metal-kernels
title: Metal GPU Kernels
sidebar_label: Metal Kernels
slug: /guides/metal-kernels
---

# Metal GPU Kernels

VeloxQuant-MLX compiles nine Metal GPU kernels at runtime using `mx.fast.metal_kernel`. This guide explains what each kernel does, how they are loaded, performance characteristics, and fallback behaviour.

:::warning Apple Silicon required
All Metal kernels require macOS on an M-series chip. On unsupported hardware, VeloxQuant-MLX falls back to MLX Python ops automatically.
:::

## Available kernels

| Kernel module | Functions | Algorithm |
|---|---|---|
| `metal/_vecinfer.py` | `vecinfer_quantize_metal`, `vecinfer_dequant_metal`, `vecinfer_encode_decode_metal` | VecInfer PVQ |
| `metal/_rabitq.py` | `rabitq_hamming_score` | RaBitQ 1-bit |
| `metal/_comm_vq.py` | `comm_vq_decode_metal` | CommVQ RoPE |
| `metal/_scalar_quant.py` | `turboquant_scalar_quantize`, `turboquant_scalar_dequantize`, `turboquant_hadamard_quantize` | TurboQuant RVQ |
| `metal/_rvq_attend.py` | `turboquant_fused_rvq_decode_attend` | RVQ + attention fusion |
| `metal/_qjl.py` | `qjl_encode`, `qjl_inner_product` | QJL |
| `metal/_bit_packing.py` | `turboquant_bit_pack`, `turboquant_bit_unpack` | All algorithms |
| `metal/fused_sdpa.py` | `metal_fused_sdpa` | All (fused attention) |

## How kernels are loaded

Kernels are compiled **lazily on first use** via `mx.fast.metal_kernel`. The first call to any function in a kernel module triggers JIT compilation:

```python
import mlx.core as mx

# This triggers compilation on first call (~200-800ms)
from veloxquant_mlx.metal._scalar_quant import turboquant_scalar_quantize

keys = mx.random.normal(shape=(1, 8, 512, 128))
quantized = turboquant_scalar_quantize(keys, bits=1)  # compilation happens here

# Subsequent calls use the cached compiled kernel
quantized2 = turboquant_scalar_quantize(keys, bits=1)  # fast
```

Compiled kernels are cached in memory for the process lifetime. There is no persistent disk cache — each Python process recompiles on first use.

## Performance characteristics

Benchmarked on M3 Pro, Llama-3.1-8B, 4096 context (source: BENCHMARK_RESULTS.md):

| Operation | MLX Python | Metal kernel | Speedup |
|---|---|---|---|
| VecInfer PVQ quantize | 42 ms | 3.2 ms | **13×** |
| Scalar quantize + Hadamard | 18 ms | 2.1 ms | **8.6×** |
| RaBitQ Hamming score | 31 ms | 2.8 ms | **11×** |
| Bit pack/unpack | 8 ms | 0.9 ms | **8.9×** |
| Fused RVQ decode + attention | 24 ms | 3.5 ms | **6.9×** |

## Fallback behaviour

VeloxQuant-MLX detects Metal availability at import time:

```python
from veloxquant_mlx.metal import metal_available

if metal_available():
    print("Metal kernels active")
else:
    print("Falling back to MLX Python ops")
```

When Metal is unavailable:
- All quantization and dequantization use equivalent pure MLX operations
- Attention scores use standard `mx.matmul`
- Fused SDPA reverts to the unfused path
- Performance is lower but results are numerically identical

## Fused SDPA kernel

The fused scaled dot-product attention kernel (`metal_fused_sdpa`) is the highest-impact optimisation. It combines:

1. Key dequantization
2. Scaled dot-product attention (`Q @ Kᵀ / √d`)
3. Softmax
4. Weighted sum of values

into a single Metal dispatch, avoiding materialising the full dequantized key matrix.

```python
from veloxquant_mlx.metal.fused_sdpa import metal_fused_sdpa, supports_shape

# Check compatibility
ok = supports_shape(batch=1, heads=8, seq_len=4096, head_dim=128)

if ok:
    attn_output = metal_fused_sdpa(
        queries=q,
        encoded_keys=encoded_k,   # compressed format from VecInfer
        values=v,
        scale=1.0 / (head_dim ** 0.5),
    )
```

## Bit packing

Sub-byte indices (1-bit, 2-bit) are packed into uint32 words to minimise memory bandwidth:

```python
from veloxquant_mlx.metal._bit_packing import turboquant_bit_pack, turboquant_bit_unpack
import mlx.core as mx

# indices: int32 in range [0, 2^bits)
indices = mx.array([[0, 1, 0, 1, 1, 0, 0, 1, ...]], dtype=mx.int32)

packed = turboquant_bit_pack(indices, bits=1)
# packed: uint32, 32× smaller than indices

recovered = turboquant_bit_unpack(packed, bits=1, original_length=indices.shape[-1])
```

## Debugging kernel issues

If you see Metal errors, enable verbose kernel output:

```bash
MLX_METAL_DEBUG=1 python your_script.py
```

Common issues:

| Error | Cause | Fix |
|---|---|---|
| `Metal kernel compilation failed` | Xcode CLI tools missing | `xcode-select --install` |
| `Kernel shape mismatch` | head_dim not a multiple of 32 | Use `supports_shape()` to check |
| `Metal device not found` | Running in VM or Rosetta | Run natively on Apple Silicon |

## See also

- [mlx_lm integration](/guides/mlx-lm-integration)
- [API — Metal functions](/api/metal-api)
- [Installation troubleshooting](/getting-started/installation)
