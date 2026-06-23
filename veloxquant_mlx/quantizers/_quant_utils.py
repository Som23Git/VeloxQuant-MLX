"""Shared low-level quantization helpers used by multiple quantizers."""
from __future__ import annotations

import mlx.core as mx


def _group_quant_dequant(x: mx.array, b: int, group_size: int = 32) -> mx.array:
    """Asymmetric min/max group quantization along axis 0. Returns fp16.

    Groups are formed along axis 0 (token/sequence axis). Each group is
    independently scaled and zero-pointed using the group's min/max values.

    Args:
        x: Input array [N, D] fp16 or fp32.
        b: Bit width (1–8).
        group_size: Number of rows per quantization group.

    Returns:
        Quantized-then-dequantized array [N, D] fp16.
    """
    n, d = x.shape
    gs = group_size
    n_groups = (n + gs - 1) // gs
    pad = n_groups * gs - n
    x32 = x.astype(mx.float32)
    if pad:
        x32 = mx.concatenate([x32, mx.broadcast_to(x32[-1:], (pad, d))], axis=0)
    xg = x32.reshape(n_groups, gs, d)
    gmin = mx.min(xg, axis=1, keepdims=True)
    gmax = mx.max(xg, axis=1, keepdims=True)
    levels = (1 << b) - 1
    eps = 1e-8
    scale = mx.maximum((gmax - gmin) / levels, eps)
    codes = mx.clip(mx.round((xg - gmin) / scale), 0, levels)
    recon = codes * scale + gmin
    return recon.reshape(n_groups * gs, d)[:n].astype(mx.float16)


__all__ = ["_group_quant_dequant"]
