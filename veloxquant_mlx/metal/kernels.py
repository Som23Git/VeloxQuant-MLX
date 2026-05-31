"""Metal kernel wrappers for VeloxQuant-MLX — re-export facade.

Kernels are organized into focused submodules:
  _vecinfer     — VecInfer codebook dequantize, quantize, encode+decode
  _bit_packing  — TurboQuant b-bit index pack/unpack
  _scalar_quant — TurboQuant scalar quantize/dequantize + fused Hadamard
  _qjl          — QJL encode and inner product
  _rvq_attend   — Fused RVQ decode + attention
"""
from __future__ import annotations

from veloxquant_mlx.metal._vecinfer import (
    vecinfer_dequant_metal,
    vecinfer_quantize_metal,
    vecinfer_encode_decode_metal,
    vecinfer_encode_decode_simple_metal,
)
from veloxquant_mlx.metal._bit_packing import (
    turboquant_bit_pack,
    turboquant_bit_unpack,
)
from veloxquant_mlx.metal._scalar_quant import (
    turboquant_scalar_quantize,
    turboquant_scalar_dequantize,
    turboquant_hadamard_quantize,
)
from veloxquant_mlx.metal._qjl import (
    qjl_encode,
    qjl_inner_product,
)
from veloxquant_mlx.metal._rvq_attend import (
    turboquant_fused_rvq_decode_attend,
)
from veloxquant_mlx.metal._comm_vq import (
    comm_vq_decode_metal,
)
from veloxquant_mlx.metal._rabitq import (
    rabitq_hamming_score,
)

__all__ = [
    "vecinfer_dequant_metal",
    "vecinfer_quantize_metal",
    "vecinfer_encode_decode_metal",
    "vecinfer_encode_decode_simple_metal",
    "turboquant_bit_pack",
    "turboquant_bit_unpack",
    "turboquant_scalar_quantize",
    "turboquant_scalar_dequantize",
    "turboquant_hadamard_quantize",
    "qjl_encode",
    "qjl_inner_product",
    "turboquant_fused_rvq_decode_attend",
    "comm_vq_decode_metal",
    "rabitq_hamming_score",
]
