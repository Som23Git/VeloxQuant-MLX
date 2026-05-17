"""Bit-allocation strategies for mixed-precision KV cache quantization.

Currently exposes :func:`allocate_bits_ratequant`, the closed-form
reverse-waterfilling allocator from RateQuant (arxiv:2605.06675), and
:func:`calibrate_layer_sensitivities`, a one-pass activation-norm
sensitivity probe that supplies the per-layer weights.

Typical usage::

    from veloxquant_mlx import KVCacheBuilder, KVCacheConfig
    from veloxquant_mlx.allocators import (
        allocate_bits_ratequant,
        calibrate_layer_sensitivities,
    )

    weights = calibrate_layer_sensitivities(model, tokenizer)
    alloc   = allocate_bits_ratequant(weights, target_avg_bits=1.5)
    config  = KVCacheConfig(
        method="turboquant_rvq",
        bit_width_inlier=alloc,   # per-layer list
        seed=42,
    )
    caches = KVCacheBuilder.for_model(model, config)
"""
from __future__ import annotations

from veloxquant_mlx.allocators.ratequant import (
    allocate_bits_ratequant,
    calibrate_layer_sensitivities,
    fit_distortion_curve,
)

__all__ = [
    "allocate_bits_ratequant",
    "calibrate_layer_sensitivities",
    "fit_distortion_curve",
]
