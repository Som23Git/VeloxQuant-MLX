from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Union

from veloxquant_mlx.core.abstractions import ArtifactStore, KVCache, QuantizationObserver
from veloxquant_mlx.core.exceptions import QuantizerConfigError


@dataclass
class KVCacheConfig:
    """Configuration for a KVCache instance.

    Attributes:
        method: Quantisation algorithm.
        head_dim: Attention head dimension (d).
        bit_width_inlier: Bit-width for inlier channels. Either a single int
            applied uniformly across all layers, OR a ``list[int]`` of length
            ``n_layers`` for per-layer RateQuant-style allocation. When passed
            as a list, ``KVCacheBuilder.for_model()`` consumes element ``i``
            for layer ``i``; ``KVCacheFactory.create()`` requires an int.
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

    method: Literal[
        "turboquant_prod", "turboquant_mse", "turboquant_rvq",
        "polar", "qjl", "vecinfer", "spectral",
    ] = "turboquant_prod"
    head_dim: int = 128
    bit_width_inlier: Union[int, list] = 2
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
    # --- VecInfer-specific configuration -------------------------------
    key_sub_dim: int = 4
    value_sub_dim: int = 8
    key_codebook_bits: int = 12
    value_codebook_bits: int = 8
    residual_length: int = 128
    smooth_factors: Any = None         # mx.array | np.ndarray | None
    key_codebook: Any = None           # mx.array | np.ndarray | None
    value_codebook: Any = None         # mx.array | np.ndarray | None
    # --- SpectralQuant configuration (data-aware, calibration-based) ----
    spectral_key_d_eff: int = 4        # signal dimensions for keys (paper: ~4)
    spectral_val_d_eff: int = 50       # signal dimensions for values (paper: ~50)
    spectral_apply_qjl: bool = True    # apply QJL on signal dims only
    spectral_model_name: str = "model" # identifier for rotation cache on disk
    # --- Metal kernel acceleration (Phase 1, 0.5.1+) -------------------
    # Three-state flag for VecInfer quantize/dequant Metal fast path:
    #   None  → auto-detect (use Metal if available, fall back silently)
    #   True  → require Metal; raise at cache-construction time if missing
    #   False → force pure-MLX path (debug / parity testing)
    use_metal_kernels: Optional[bool] = None
    # --- Fused dequant+SDPA Metal kernel (Phase 2, 0.6.0+) -------------
    # When True (or auto-detected at None), the cache stores K/V as
    # codebook indices only and exposes a fused_sdpa() method that
    # mlx_lm's dispatcher (after patch_mlx_lm_for_fused_sdpa()) routes
    # attention to.  Avoids materializing the fp16 K_hat tensor entirely.
    #   None  → False today (opt-in; will flip to auto-detect later)
    #   True  → require, raise if Metal/shape unsupported
    #   False → run the standard dequant→SDPA path (current 0.5.x default)
    fused_sdpa: Optional[bool] = False
    # Pre-allocated index ring-buffer capacity (in tokens) when
    # fused_sdpa=True.  At construction time we allocate
    # [B, H_kv, fused_sdpa_max_ctx, n_sub] uint32 once and slice-write
    # into it on each update_and_fetch — avoids O(S²) per-step concat.
    # If a generation exceeds this length, the cache raises RuntimeError.
    fused_sdpa_max_ctx: int = 8192

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
        from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
        from veloxquant_mlx.cache.qjl_cache import QJLKVCache
        from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
        from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
        from veloxquant_mlx.cache.turboquant_cache import TurboQuantKVCache
        from veloxquant_mlx.cache.turboquant_rvq_cache import TurboQuantRVQKVCache
        from veloxquant_mlx.cache.vecinfer_cache import VecInferKVCache

        d = config.head_dim
        seed = config.seed
        b = config.bit_width_inlier
        if isinstance(b, list) and config.method != "vecinfer":
            raise QuantizerConfigError(
                "KVCacheFactory.create() requires bit_width_inlier to be a single int. "
                "List-form bit_width_inlier (per-layer allocation) is consumed by "
                "KVCacheBuilder.for_model(), which dispatches to create() once per layer."
            )
        m = config.jl_dim if config.jl_dim is not None else d
        store = config.store

        if config.method in ("turboquant_prod", "turboquant_mse"):
            cache: KVCache = TurboQuantKVCache(config)
        elif config.method == "turboquant_rvq":
            cache = TurboQuantRVQKVCache(config)
        elif config.method == "polar":
            cache = PolarQuantKVCache(config)
        elif config.method == "qjl":
            cache = QJLKVCache(config)
        elif config.method == "vecinfer":
            cache = VecInferKVCache(config)
        elif config.method == "spectral":
            cache = SpectralQuantKVCache(config)
        else:
            raise QuantizerConfigError(
                f"KVCacheFactory: unknown method '{config.method}'. "
                f"Choices: turboquant_prod, turboquant_mse, turboquant_rvq, "
                f"polar, qjl, vecinfer, spectral."
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

    def with_bit_width(self, inlier, outlier: Optional[int] = None) -> "KVCacheBuilder":
        """Set bit-width(s).

        Args:
            inlier: Bit-width for inlier channels. Either an int (uniform across
                all layers) or a list[int] of length n_layers for RateQuant-style
                per-layer allocation. When a list is supplied, this builder
                must be consumed via ``KVCacheBuilder.for_model(model, config)``;
                direct ``.build()`` rejects the list.
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
        if isinstance(cfg.bit_width_inlier, list):
            if not cfg.bit_width_inlier:
                raise QuantizerConfigError(
                    "KVCacheBuilder: bit_width_inlier list must not be empty."
                )
            if not all(isinstance(b, int) and b >= 1 for b in cfg.bit_width_inlier):
                raise QuantizerConfigError(
                    "KVCacheBuilder: every element of bit_width_inlier must "
                    "be an int >= 1."
                )
        elif cfg.bit_width_inlier < 1:
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

    @staticmethod
    def for_model(model, config: "KVCacheConfig") -> list:
        """Build one KVCache per language-model layer, sized per-layer.

        Works for text-only and VLM models (Qwen2-VL, Qwen3-VL, Mistral, etc.).
        Layers without a self_attn attribute (MoE gates, etc.) fall back to a
        standard fp16 KVCache so the list length always matches model.layers.

        Per-layer bit-widths (RateQuant)
        --------------------------------
        If ``config.bit_width_inlier`` is a ``list[int]``, element ``i`` is
        used for layer ``i``. The list length must equal the number of
        attention-bearing layers (layers without self_attn are skipped from
        the count). This lets RateQuant-style mixed-precision allocations
        be passed through the standard API without manual cache wiring.

        Args:
            model: Loaded mlx_lm model instance.
            config: KVCacheConfig specifying method, bit_width_inlier, seed, etc.
                    head_dim is overridden per-layer.

        Returns:
            List of KVCache instances, one per language-model layer.
        """
        from mlx_lm.models.cache import KVCache as _FallbackCache

        # Qwen2-VL exposes model.layers directly; text models expose model.model.layers
        layers = getattr(model, "layers", None) or model.model.layers
        # VLM wrappers (Qwen2-VL) have model.args.text_config only;
        # real attention config lives in model.language_model.args
        args = getattr(model, "args", None)
        if args is not None and not hasattr(args, "hidden_size"):
            lm = getattr(model, "language_model", None)
            if lm is not None:
                args = getattr(lm, "args", args)

        # Resolve per-layer bit-width policy
        b_spec = config.bit_width_inlier
        is_per_layer = isinstance(b_spec, list)
        if is_per_layer:
            # Count attention-bearing layers up-front for validation
            n_attn = sum(1 for L in layers
                         if (getattr(L, "self_attn", None) or getattr(L, "attn", None))
                         is not None)
            if len(b_spec) != n_attn:
                raise QuantizerConfigError(
                    f"KVCacheBuilder.for_model: bit_width_inlier is a list of "
                    f"length {len(b_spec)}, but model has {n_attn} attention "
                    f"layers. The list must have one entry per attention layer."
                )

        caches = []
        attn_idx = 0  # index into b_spec, advances only for attention layers
        for i, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                caches.append(_FallbackCache())
                continue
            hd = getattr(attn, "head_dim", None)
            if hd is None:
                if args is not None:
                    hd = getattr(args, "head_dim", None) or (
                        args.hidden_size // args.num_attention_heads
                    )
            if hd is None:
                caches.append(_FallbackCache())
                continue
            layer_b = b_spec[attn_idx] if is_per_layer else b_spec
            layer_cfg = KVCacheConfig(
                method=config.method,
                head_dim=hd,
                bit_width_inlier=layer_b,
                seed=config.seed + i,
            )
            caches.append(KVCacheFactory.create(layer_cfg))
            attn_idx += 1
        return caches

    def __repr__(self) -> str:
        return f"KVCacheBuilder(config={self._config!r})"
