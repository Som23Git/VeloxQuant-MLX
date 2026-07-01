from __future__ import annotations

import math
from dataclasses import dataclass, field
from dataclasses import replace as dataclasses_replace
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
        "polar", "qjl", "vecinfer", "spectral", "kivi", "kivi_sink", "svdq", "kitty",
        "adakv", "xquant", "kvquant", "palu", "cachegen", "minicache", "gear", "zipcache", "snapkv",
        "streaming_llm", "h2o",
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
    # --- KIVI configuration (asymmetric group quantization) ------------
    kivi_group_size: int = 32          # min/max group size (KIVI default 32)
    # --- SVDq configuration (sub-2-bit key compression via offline SVD) --
    svdq_rank: Optional[int] = None        # explicit rank; None → energy threshold
    svdq_energy_threshold: float = 0.95   # fraction of singular value energy to retain
    svdq_hi_bit: int = 4                  # bits for top-importance latent channels
    svdq_lo_bit: int = 2                  # bits for remaining latent channels
    svdq_hi_fraction: float = 0.25        # fraction of channels routed to hi_bit
    svdq_group_size: int = 32             # group size for latent quantization
    # --- Kitty configuration (dynamic channel-wise mixed-precision) ------
    kitty_hi_fraction: float = 0.25       # fraction of channels routed to hi_bit
    kitty_hi_bit: int = 4                 # bits for high-variance channels
    kitty_lo_bit: int = 2                 # bits for low-variance channels
    kitty_group_size: int = 32            # group size for channel quantization
    # --- AdaKV-proxy configuration (per-head adaptive bit allocation) ----
    adakv_target_avg_bits: float = 2.0    # global average bits/element target
    adakv_lo_bit: int = 2                 # minimum bits any head can get
    adakv_mid_bit: int = 3                # middle tier (set == hi for a 2-tier set)
    adakv_hi_bit: int = 4                 # maximum bits any head can get
    adakv_group_size: int = 32            # group size for per-head quantization
    adakv_update_interval: int = 1        # recompute allocation every N tokens (1 = every step)
    # --- XQuant configuration (cross-layer KV cache reuse) ---------------
    xquant_group_size: int = 2            # layers per anchor/reuse group (2 = pairs)
    xquant_base_bits: int = 2             # anchor quantizer bit-width
    xquant_residual_bits: int = 0         # reuse-layer correction residual (0 = pure reuse)
    xquant_group_quant_size: int = 32     # token group size for quantization
    xquant_max_ctx: int = 8192            # coordinator per-group token budget
    # --- KVQuant-NUQ configuration (non-uniform datatype + outlier isolation) -
    kvquant_bits: int = 3                 # base NUQ bit-width
    kvquant_outlier_fraction: float = 0.01  # top-magnitude fraction kept fp16 (0 = pure NUQ)
    kvquant_group_size: int = 32          # group size for per-channel/per-token fitting
    kvquant_lloyd_iters: int = 8          # Lloyd-Max iterations for level fitting
    kvquant_refit_interval: int = 0       # refit levels every N decode steps (0 = freeze prefill)
    # --- PALU configuration (true-latent low-rank K *and* V) -------------
    palu_rank: Optional[int] = None        # explicit latent rank; None → energy threshold
    palu_energy_threshold: float = 0.90    # singular-value energy to retain
    palu_n_head_groups: int = 4            # group-head low-rank: heads share a projection
    palu_hi_bit: int = 4                   # mixed-bit: top latent channels
    palu_lo_bit: int = 2                   # mixed-bit: remaining latent channels
    palu_hi_fraction: float = 0.25         # fraction of latent channels at hi_bit
    palu_group_size: int = 32              # token group size for latent quantization
    palu_quantize_values: bool = True      # low-rank + mixed-bit values too (False = LR-only)
    # --- CacheGen configuration (entropy-coded byte model over group quant) ----
    cachegen_bits: int = 4                 # base group-quant bit-width
    cachegen_group_size: int = 32          # token group size
    cachegen_use_delta: bool = True        # token-delta transform before entropy coding
    # --- MiniCache configuration (cross-layer depth-dimension SLERP merge) -----
    minicache_start_frac: float = 0.5      # depth fraction below which layers are never merged
    minicache_group_size: int = 2          # layers per merge group (2 = pairs)
    minicache_retention_threshold: float = 0.9  # cosine below which a token pair is kept unmerged
    minicache_slerp_t: float = 0.5         # SLERP interpolation factor
    minicache_max_ctx: int = 8192          # coordinator per-group token budget
    # --- GEAR configuration (error-feedback: residual low-rank + sparse outliers) ---
    gear_bits: int = 2                     # ultra-low base bit-width
    gear_rank: Optional[int] = None        # residual low-rank; None → energy threshold
    gear_energy_threshold: float = 0.90    # residual singular-value energy to retain
    gear_sparse_fraction: float = 0.01     # top-|residual| fraction kept exact (0 = pure low-rank)
    gear_group_size: int = 32              # base group-quant token group size
    gear_quantize_values: bool = True      # apply GEAR to values too (False = keys only)
    # --- ZipCache-adapted configuration (saliency-adaptive per-token mixed-precision) ---
    zipcache_hi_bits: int = 4             # bit-width for salient (high-norm) tokens
    zipcache_lo_bits: int = 2             # bit-width for non-salient tokens
    zipcache_hi_fraction: float = 0.20   # fraction of tokens routed to hi_bits
    zipcache_group_size: int = 32         # token group size for min/max quantization
    zipcache_quantize_values: bool = True # apply mixed-precision to values too
    # --- SnapKV-adapted configuration (prefill observation-window token eviction) ---
    snap_budget: int = 512               # max tokens retained after prefill eviction
    snap_obs_window: int = 32            # trailing key rows used as proxy queries
    snap_n_sink: int = 4                 # initial positions always kept (attention sinks)
    # --- StreamingLLM-adapted configuration (sink + recency-window structural eviction) ---
    stream_n_sink: int = 4               # initial token positions frozen as attention sinks
    stream_window_size: int = 512        # FIFO capacity for recent tokens
    # --- H2O-adapted configuration (cumulative attention-mass heavy-hitter eviction) ---
    h2o_budget: int = 512                # max tokens kept at any time (sinks + non-sinks)
    h2o_n_sink: int = 4                  # initial positions protected from eviction (attention sinks)
    # --- KVSink-adapted sink protection (method="kivi_sink") -----------
    n_sink_tokens: int = 5             # top-k high-key-norm tokens kept fp16
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
        from veloxquant_mlx.cache.adakv_cache import AdaKVCache
        from veloxquant_mlx.cache.xquant_cache import XQuantKVCache
        from veloxquant_mlx.cache.kvquant_cache import KVQuantKVCache
        from veloxquant_mlx.cache.palu_cache import PALUKVCache
        from veloxquant_mlx.cache.cachegen_cache import CacheGenKVCache
        from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
        from veloxquant_mlx.cache.gear_cache import GEARKVCache
        from veloxquant_mlx.cache.zipcache_cache import ZipCacheKVCache
        from veloxquant_mlx.cache.snapkv_cache import SnapKVKVCache
        from veloxquant_mlx.cache.streaming_llm_cache import StreamingLLMKVCache
        from veloxquant_mlx.cache.h2o_cache import H2OKVCache
        from veloxquant_mlx.cache.kitty_cache import KittyKVCache
        from veloxquant_mlx.cache.polar_cache import PolarQuantKVCache
        from veloxquant_mlx.cache.qjl_cache import QJLKVCache
        from veloxquant_mlx.cache.sliding_window_cache import SlidingWindowKVCache
        from veloxquant_mlx.cache.kivi_cache import KIVIKVCache
        from veloxquant_mlx.cache.sink_cache import SinkProtectedKVCache
        from veloxquant_mlx.cache.spectral_cache import SpectralQuantKVCache
        from veloxquant_mlx.cache.svdq_cache import SVDqKVCache
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
        elif config.method == "kivi":
            cache = KIVIKVCache(config)
        elif config.method == "kivi_sink":
            cache = SinkProtectedKVCache(config)
        elif config.method == "svdq":
            cache = SVDqKVCache(config)
        elif config.method == "kitty":
            cache = KittyKVCache(config)
        elif config.method == "adakv":
            cache = AdaKVCache(config)
        elif config.method == "xquant":
            # Single-cache construction yields a degenerate (coordinator-less)
            # anchor. Cross-layer reuse requires KVCacheBuilder.for_model(), which
            # builds the shared XQuantCoordinator and assigns anchor/reuse roles.
            cache = XQuantKVCache(config)
        elif config.method == "kvquant":
            cache = KVQuantKVCache(config)
        elif config.method == "palu":
            cache = PALUKVCache(config)
        elif config.method == "cachegen":
            cache = CacheGenKVCache(config)
        elif config.method == "minicache":
            # Single-cache construction yields a degenerate (coordinator-less)
            # primary that behaves as lossless fp16 passthrough. Cross-layer
            # merging requires KVCacheBuilder.for_model(), which builds the
            # shared MiniCacheCoordinator and assigns primary/merge roles.
            cache = MiniCacheKVCache(config)
        elif config.method == "gear":
            cache = GEARKVCache(config)
        elif config.method == "zipcache":
            cache = ZipCacheKVCache(config)
        elif config.method == "snapkv":
            cache = SnapKVKVCache(config)
        elif config.method == "streaming_llm":
            cache = StreamingLLMKVCache(config)
        elif config.method == "h2o":
            cache = H2OKVCache(config)
        else:
            raise QuantizerConfigError(
                f"KVCacheFactory: unknown method '{config.method}'. "
                f"Choices: turboquant_prod, turboquant_mse, turboquant_rvq, "
                f"polar, qjl, vecinfer, spectral, kivi, kivi_sink, svdq, kitty, "
                f"adakv, xquant, kvquant, palu, cachegen, minicache, gear, zipcache, snapkv, "
                f"streaming_llm, h2o."
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

        # --- XQuant: cross-layer reuse needs a shared coordinator + roles ----
        if config.method == "xquant":
            return KVCacheBuilder._build_xquant(layers, args, config, _FallbackCache)

        # --- MiniCache: cross-layer merge needs a shared coordinator + roles --
        if config.method == "minicache":
            return KVCacheBuilder._build_minicache(layers, args, config, _FallbackCache)

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
            # Preserve every method-specific field (svdq_*, kitty_*, kvquant_*,
            # palu_*, …) from the user's config and override only the per-layer
            # head_dim / bit-width / seed.  Reconstructing the dataclass field by
            # field (as the old code did) silently dropped method hyperparameters.
            layer_cfg = dataclasses_replace(
                config,
                head_dim=hd,
                bit_width_inlier=layer_b,
                seed=config.seed + i,
                store=config.store,
            )
            caches.append(KVCacheFactory.create(layer_cfg))
            attn_idx += 1
        return caches

    @staticmethod
    def _build_xquant(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared XQuantCoordinator and role-assigned caches per layer.

        Anchor/reuse roles are assigned over *attention-bearing* layers only, so
        non-attention layers (MoE gates, etc.) get a plain fallback cache and do
        not consume a group slot.
        """
        from veloxquant_mlx.cache.xquant_cache import XQuantKVCache
        from veloxquant_mlx.cache.xquant_coordinator import XQuantCoordinator
        from veloxquant_mlx.quantizers.xquant import pair_layers

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        roles = pair_layers(len(attn_layer_idx), config.xquant_group_size)
        coordinator = XQuantCoordinator(max_ctx=config.xquant_max_ctx)

        role_by_layer: dict[int, tuple[str, int]] = {
            attn_layer_idx[k]: roles[k] for k in range(len(attn_layer_idx))
        }

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            role, group_id = role_by_layer[i]
            layer_cfg = KVCacheConfig(
                method="xquant",
                head_dim=hd,
                seed=config.seed + i,
                xquant_group_size=config.xquant_group_size,
                xquant_base_bits=config.xquant_base_bits,
                xquant_residual_bits=config.xquant_residual_bits,
                xquant_group_quant_size=config.xquant_group_quant_size,
                xquant_max_ctx=config.xquant_max_ctx,
            )
            caches.append(XQuantKVCache(layer_cfg, role=role, group_id=group_id,
                                        coordinator=coordinator))
        return caches

    @staticmethod
    def _build_minicache(layers, args, config: "KVCacheConfig", fallback_cls) -> list:
        """Build one shared MiniCacheCoordinator and role-assigned caches per layer.

        Primary/merge roles are assigned over *attention-bearing* layers only,
        and only middle-to-deep layers (>= ``minicache_start_frac`` of depth) are
        eligible for merging — earlier layers are standalone primaries.
        """
        from veloxquant_mlx.cache.minicache_cache import MiniCacheKVCache
        from veloxquant_mlx.cache.minicache_coordinator import MiniCacheCoordinator
        from veloxquant_mlx.quantizers.minicache import pair_layers_depth

        def _head_dim(layer):
            attn = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn is None:
                return None
            hd = getattr(attn, "head_dim", None)
            if hd is None and args is not None:
                hd = getattr(args, "head_dim", None) or (
                    args.hidden_size // args.num_attention_heads
                )
            return hd

        attn_layer_idx = [i for i, L in enumerate(layers) if _head_dim(L) is not None]
        roles = pair_layers_depth(
            len(attn_layer_idx),
            start_frac=config.minicache_start_frac,
            group_size=config.minicache_group_size,
        )
        coordinator = MiniCacheCoordinator(max_ctx=config.minicache_max_ctx)

        role_by_layer: dict[int, tuple[str, int]] = {
            attn_layer_idx[k]: roles[k] for k in range(len(attn_layer_idx))
        }

        caches = []
        for i, layer in enumerate(layers):
            hd = _head_dim(layer)
            if hd is None:
                caches.append(fallback_cls())
                continue
            role, group_id = role_by_layer[i]
            layer_cfg = KVCacheConfig(
                method="minicache",
                head_dim=hd,
                seed=config.seed + i,
                minicache_start_frac=config.minicache_start_frac,
                minicache_group_size=config.minicache_group_size,
                minicache_retention_threshold=config.minicache_retention_threshold,
                minicache_slerp_t=config.minicache_slerp_t,
                minicache_max_ctx=config.minicache_max_ctx,
            )
            caches.append(MiniCacheKVCache(layer_cfg, role=role, group_id=group_id,
                                           coordinator=coordinator))
        return caches

    def __repr__(self) -> str:
        return f"KVCacheBuilder(config={self._config!r})"
