"""mlx_kv_quant — KV cache quantization for Apple Silicon MLX.

Implements TurboQuant, TurboQuantRVQ, PolarQuant, and QJL plus the
RateQuant per-layer bit allocator for production LLM inference.
"""
from __future__ import annotations

from mlx_kv_quant.allocators import (
    allocate_bits_ratequant,
    calibrate_layer_sensitivities,
    fit_distortion_curve,
)
from mlx_kv_quant.cache.base import KVCacheBuilder, KVCacheConfig, KVCacheFactory
from mlx_kv_quant.core.abstractions import (
    ArtifactStore,
    KVCache,
    Quantizer,
    QuantizationObserver,
)
from mlx_kv_quant.core.context import EncodedVector, QuantizationContext, TransformResult
from mlx_kv_quant.core.exceptions import (
    ArtifactNotFoundError,
    CodebookDimensionMismatch,
    CyclicPipelineError,
    QuantizerConfigError,
)
from mlx_kv_quant.observers import KeyNormObserver, KeyNormReport
from mlx_kv_quant.quantizers.base import QuantizerFactory

__all__ = [
    # Configuration & builders
    "KVCacheBuilder",
    "KVCacheConfig",
    "KVCacheFactory",
    # Abstractions
    "ArtifactStore",
    "KVCache",
    "Quantizer",
    "QuantizationObserver",
    # Data types
    "EncodedVector",
    "QuantizationContext",
    "TransformResult",
    # Exceptions
    "ArtifactNotFoundError",
    "CodebookDimensionMismatch",
    "CyclicPipelineError",
    "QuantizerConfigError",
    # Quantizer registry
    "QuantizerFactory",
    # RateQuant allocators (per-layer mixed-precision)
    "allocate_bits_ratequant",
    "calibrate_layer_sensitivities",
    "fit_distortion_curve",
    # Observers
    "KeyNormObserver",
    "KeyNormReport",
]

__version__ = "0.3.5"
