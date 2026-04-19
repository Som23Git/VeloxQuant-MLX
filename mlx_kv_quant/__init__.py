"""mlx_kv_quant — KV cache quantization for Apple Silicon MLX.

Implements TurboQuant, PolarQuant, and QJL for production LLM inference.
"""
from __future__ import annotations

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
from mlx_kv_quant.quantizers.base import QuantizerFactory

__all__ = [
    "KVCacheBuilder",
    "KVCacheConfig",
    "KVCacheFactory",
    "ArtifactStore",
    "KVCache",
    "Quantizer",
    "QuantizationObserver",
    "EncodedVector",
    "QuantizationContext",
    "TransformResult",
    "ArtifactNotFoundError",
    "CodebookDimensionMismatch",
    "CyclicPipelineError",
    "QuantizerConfigError",
    "QuantizerFactory",
]

__version__ = "0.1.0"
