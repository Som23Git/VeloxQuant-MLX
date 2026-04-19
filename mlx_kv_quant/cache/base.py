from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from mlx_kv_quant.core.abstractions import ArtifactStore, KVCache, QuantizationObserver
from mlx_kv_quant.core.exceptions import QuantizerConfigError


@dataclass
class KVCacheConfig:
    """Configuration for a KVCache instance.

    Attributes:
        method: Quantisation algorithm.
        head_dim: Attention head dimension (d).
        bit_width_inlier: Bit-width for inlier channels.
        bit_width_outlier: Bit-width for outlier channels (None → same as inlier).
        jl_dim: JL projection dimension m.
        n_outlier_channels: Number of outlier channels to detect.
        seed: Random seed.
        dtype: MLX dtype for computations.
        capacity: Maximum tokens to store (None → unlimited).
        sliding_window: If set, wrap cache with sliding-window eviction.
        store: ArtifactStore to load precomputed artifacts from.
        observers: List of QuantizationObserver instances.
    """

    method: Literal["turboquant_prod", "turboquant_mse", "polar", "qjl"] = "turboquant_prod"
    head_dim: int = 128
    bit_width_inlier: int = 2
    bit_width_outlier: Optional[int] = None
    jl_dim: Optional[int] = None
    n_outlier_channels: Optional[int] = None
    n_calib_tokens: Optional[int] = None
    enable_vectorized_attend: bool = False
    enable_outlier_two_stream: bool = False
    enable_fused_query_dot: bool = False
    seed: int = 42
    dtype: Any = None
    capacity: Optional[int] = None
    sliding_window: Optional[int] = None
    store: Optional[ArtifactStore] = None
    observers: list = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"KVCacheConfig(method={self.method!r}, d={self.head_dim}, "
            f"b={self.bit_width_inlier}, seed={self.seed})"
        )


class KVCacheFactory:
    """Factory for creating KVCache instances from a KVCacheConfig."""

    @staticmethod
    def create(config: KVCacheConfig) -> KVCache:
        """Instantiate a KVCache from the given configuration.

        Args:
            config: KVCacheConfig instance.

        Returns:
            Configured KVCache.
        """
        from mlx_kv_quant.cache.polar_cache import PolarQuantKVCache
        from mlx_kv_quant.cache.qjl_cache import QJLKVCache
        from mlx_kv_quant.cache.sliding_window_cache import SlidingWindowKVCache
        from mlx_kv_quant.cache.turboquant_cache import TurboQuantKVCache

        d = config.head_dim
        seed = config.seed
        b = config.bit_width_inlier
        m = config.jl_dim if config.jl_dim is not None else d
        store = config.store

        if config.method in ("turboquant_prod", "turboquant_mse"):
            cache: KVCache = TurboQuantKVCache(config)
        elif config.method == "polar":
            cache = PolarQuantKVCache(config)
        elif config.method == "qjl":
            cache = QJLKVCache(config)
        else:
            raise QuantizerConfigError(
                f"KVCacheFactory: unknown method '{config.method}'. "
                f"Choices: turboquant_prod, turboquant_mse, polar, qjl."
            )

        if config.sliding_window is not None:
            cache = SlidingWindowKVCache(cache, window_size=config.sliding_window)

        return cache


class KVCacheBuilder:
    """Fluent builder for KVCache construction with validation.

    Example::

        cache = (
            KVCacheBuilder()
            .with_method("turboquant_prod")
            .with_head_dim(128)
            .with_bit_width(inlier=2, outlier=3)
            .with_jl_dim(128)
            .with_seed(42)
            .build()
        )
    """

    def __init__(self) -> None:
        self._config = KVCacheConfig()

    def with_method(self, method: str) -> "KVCacheBuilder":
        """Set the quantisation method.

        Args:
            method: One of 'turboquant_prod', 'turboquant_mse', 'polar', 'qjl'.
        """
        self._config.method = method  # type: ignore[assignment]
        return self

    def with_head_dim(self, d: int) -> "KVCacheBuilder":
        """Set the attention head dimension.

        Args:
            d: Head dimension (must be a power of 2).
        """
        self._config.head_dim = d
        return self

    def with_bit_width(self, inlier: int, outlier: Optional[int] = None) -> "KVCacheBuilder":
        """Set bit-width(s).

        Args:
            inlier: Bit-width for inlier channels.
            outlier: Bit-width for outlier channels (defaults to inlier if None).
        """
        self._config.bit_width_inlier = inlier
        self._config.bit_width_outlier = outlier
        return self

    def with_jl_dim(self, m: int) -> "KVCacheBuilder":
        """Set the JL projection dimension.

        Args:
            m: Must be <= head_dim.
        """
        self._config.jl_dim = m
        return self

    def with_n_outlier_channels(self, n: int) -> "KVCacheBuilder":
        """Set the number of outlier channels to detect.

        Args:
            n: Must be < head_dim.
        """
        self._config.n_outlier_channels = n
        return self

    def with_n_calib_tokens(self, n: int) -> "KVCacheBuilder":
        """Set calibration token count for outlier activation."""
        self._config.n_calib_tokens = n
        return self

    def with_vectorized_attend(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable vectorized packed-key unpack in attend()."""
        self._config.enable_vectorized_attend = enabled
        return self

    def with_outlier_two_stream(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable outlier/inlier split cache after calibration."""
        self._config.enable_outlier_two_stream = enabled
        return self

    def with_fused_query_dot(self, enabled: bool = True) -> "KVCacheBuilder":
        """Enable fused rotated-query + codebook-dot path."""
        self._config.enable_fused_query_dot = enabled
        return self

    def with_seed(self, seed: int) -> "KVCacheBuilder":
        """Set the random seed.

        Args:
            seed: Integer seed.
        """
        self._config.seed = seed
        return self

    def with_precision(self, dtype: Any) -> "KVCacheBuilder":
        """Set the compute dtype.

        Args:
            dtype: MLX dtype (e.g. mx.float16).
        """
        self._config.dtype = dtype
        return self

    def with_capacity(self, max_tokens: int) -> "KVCacheBuilder":
        """Set the maximum number of tokens to store.

        Args:
            max_tokens: Positive integer.
        """
        self._config.capacity = max_tokens
        return self

    def with_artifact_store(self, store: ArtifactStore) -> "KVCacheBuilder":
        """Provide an ArtifactStore for loading precomputed artifacts.

        Args:
            store: ArtifactStore instance.
        """
        self._config.store = store
        return self

    def with_observer(self, observer: QuantizationObserver) -> "KVCacheBuilder":
        """Attach a QuantizationObserver.

        Args:
            observer: Observer instance.
        """
        self._config.observers.append(observer)
        return self

    def with_sliding_window(self, window_size: int) -> "KVCacheBuilder":
        """Wrap the cache with a sliding-window eviction policy.

        Args:
            window_size: Number of tokens to keep.
        """
        self._config.sliding_window = window_size
        return self

    def build(self) -> KVCache:
        """Validate the configuration and construct the KVCache.

        Returns:
            Configured KVCache instance.

        Raises:
            QuantizerConfigError: If any validation check fails.
        """
        cfg = self._config
        d = cfg.head_dim

        if not (d >= 1 and (d & (d - 1)) == 0):
            raise QuantizerConfigError(
                f"KVCacheBuilder: head_dim={d} must be a power of 2."
            )
        if cfg.bit_width_inlier < 1:
            raise QuantizerConfigError(
                f"KVCacheBuilder: bit_width_inlier={cfg.bit_width_inlier} must be >= 1."
            )
        if cfg.jl_dim is not None and cfg.jl_dim > d:
            raise QuantizerConfigError(
                f"KVCacheBuilder: jl_dim={cfg.jl_dim} must be <= head_dim={d}."
            )
        if cfg.n_outlier_channels is not None and cfg.n_outlier_channels >= d:
            raise QuantizerConfigError(
                f"KVCacheBuilder: n_outlier_channels={cfg.n_outlier_channels} "
                f"must be < head_dim={d}."
            )
        if cfg.n_calib_tokens is not None and cfg.n_calib_tokens < 1:
            raise QuantizerConfigError(
                f"KVCacheBuilder: n_calib_tokens={cfg.n_calib_tokens} must be >= 1."
            )

        return KVCacheFactory.create(cfg)

    def __repr__(self) -> str:
        return f"KVCacheBuilder(config={self._config!r})"
