"""VecInfer KV cache wrapper for mlx_lm integration.

Wraps :mod:`veloxquant_mlx.allocators.vecinfer` primitives in the standard
``update_and_fetch`` cache protocol expected by mlx_lm. The cache:

* applies a per-(head, channel) smooth scaling + Walsh-Hadamard rotation to
  keys, suppressing outliers before product VQ;
* encodes transformed keys against a pre-trained codebook, immediately
  dequantizes (then inverse-transforms) so the downstream SDPA call sees
  fp16 keys;
* tracks compressed vs fp16 byte counts so benchmarks can report a
  realized compression ratio.

The paper's CUDA kernel fusion (Section 3.3) is NOT portable to MLX/Metal;
the win on Apple Silicon is memory compression, not speedup over fp16.

Per-token storage (keys) at codebook bit-width ``b_k`` and sub-vector
dimension ``d_k``: ``(D / d_k) * b_k / 8`` bytes — plus an amortized
codebook cost of ``2**b_k * d_k * 2`` bytes shared across all tokens.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models.cache import KVCache as _MLXKVCache

from veloxquant_mlx.allocators.vecinfer import (
    apply_dual_transform_keys,
    apply_dual_transform_queries,
    dequantize_vq,
    quantize_vq,
    walsh_hadamard_matrix,
)


class VecInferKVCache(_MLXKVCache):
    """KV cache implementing VecInfer's dual-transform product VQ.

    Args:
        config: :class:`KVCacheConfig` with VecInfer-specific fields populated
            by :class:`KVCacheFactory.create`. Required fields:
            ``head_dim``, ``key_codebook_bits``, ``value_codebook_bits``,
            ``key_sub_dim``, ``value_sub_dim``. Optional: ``smooth_factors``
            (numpy or mx array; identity if absent), ``key_codebook``,
            ``value_codebook`` (random init if absent — for tests only).

    Notes:
        Storage layout deliberately delegates concatenation to mlx_lm's
        base ``_MLXKVCache``. We quantize + dequantize on the way in so
        that subsequent SDPA calls see a standard fp16 key tensor.

        Never exposes ``.bits`` — mlx_lm's SDPA checks
        ``hasattr(cache, "bits")`` to route to a different kernel path.
        We expose ``.assigned_avg_bits`` instead.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self._head_dim = int(config.head_dim)
        self._key_sub_dim = int(getattr(config, "key_sub_dim", 4))
        self._value_sub_dim = int(getattr(config, "value_sub_dim", 8))
        self._key_bits = int(getattr(config, "key_codebook_bits", 12))
        self._value_bits = int(getattr(config, "value_codebook_bits", 8))
        self._residual_length = int(getattr(config, "residual_length", 128))

        if self._head_dim % self._key_sub_dim != 0:
            raise ValueError(
                f"VecInferKVCache: head_dim={self._head_dim} not divisible "
                f"by key_sub_dim={self._key_sub_dim}"
            )
        if self._head_dim % self._value_sub_dim != 0:
            raise ValueError(
                f"VecInferKVCache: head_dim={self._head_dim} not divisible "
                f"by value_sub_dim={self._value_sub_dim}"
            )

        # Smooth factors: [n_heads, head_dim] or [head_dim] or None (identity)
        sm = getattr(config, "smooth_factors", None)
        if sm is None:
            self._smooth = None
        elif isinstance(sm, mx.array):
            self._smooth = sm
        else:
            self._smooth = mx.array(sm)

        # Hadamard matrix (constant)
        self._H = walsh_hadamard_matrix(self._head_dim, dtype=mx.float32)

        # Codebooks
        n_kc = 2 ** self._key_bits
        n_vc = 2 ** self._value_bits
        seed = int(getattr(config, "seed", 42))

        key_cb = getattr(config, "key_codebook", None)
        if key_cb is None:
            # Random init — only useful for shape/wiring tests; real usage
            # supplies a calibrated codebook via the factory.
            rng = mx.random.key(seed)
            key_cb = mx.random.normal(
                shape=(n_kc, self._key_sub_dim), key=rng
            ).astype(mx.float32)
        elif not isinstance(key_cb, mx.array):
            key_cb = mx.array(key_cb)
        self._key_codebook = key_cb.astype(mx.float32)

        val_cb = getattr(config, "value_codebook", None)
        if val_cb is None:
            rng = mx.random.key(seed + 1)
            val_cb = mx.random.normal(
                shape=(n_vc, self._value_sub_dim), key=rng
            ).astype(mx.float32)
        elif not isinstance(val_cb, mx.array):
            val_cb = mx.array(val_cb)
        self._value_codebook = val_cb.astype(mx.float32)

        # Byte accounting
        self._key_bytes_compressed = 0
        self._key_bytes_fp16 = 0
        self._value_bytes_compressed = 0
        self._value_bytes_fp16 = 0
        self._tokens_seen = 0
        self._tokens_quantized = 0

    # ------------------------------------------------------------------
    # mlx_lm protocol
    # ------------------------------------------------------------------
    def update_and_fetch(self, keys, values):
        B, H, S, D = keys.shape
        kdtype = keys.dtype
        vdtype = values.dtype

        # ---- Key path: smooth -> Hadamard -> VQ -> dequant -> inverse ----
        # Promote to fp32 for transformation/quantization stability.
        k32 = keys.astype(mx.float32)
        if self._smooth is not None:
            k_tilde = apply_dual_transform_keys(k32, self._smooth, self._H)
        else:
            k_tilde = k32 @ self._H

        # Quantize -> dequantize on the transformed space
        k_idx = quantize_vq(k_tilde, self._key_codebook, self._key_sub_dim)
        k_hat_tilde = dequantize_vq(k_idx, self._key_codebook)

        # Invert: K_hat = (K_tilde_hat @ H.T) * lambda  (H is orthonormal)
        k_hat = k_hat_tilde @ self._H.T
        if self._smooth is not None:
            sm = self._smooth
            if sm.ndim == 2 and k_hat.ndim >= 4 and k_hat.shape[-3] == sm.shape[0]:
                sm_b = sm[:, None, :].astype(mx.float32)
            elif sm.ndim == 2:
                # GQA mismatch: collapse heads dim to a per-channel mean
                sm_b = mx.mean(sm, axis=0).astype(mx.float32)
            else:
                sm_b = sm.astype(mx.float32)
            k_hat = k_hat * sm_b
        k_dequant = k_hat.astype(kdtype)

        # ---- Value path: VQ directly (no smooth/Hadamard per paper) ------
        v32 = values.astype(mx.float32)
        v_idx = quantize_vq(v32, self._value_codebook, self._value_sub_dim)
        v_hat = dequantize_vq(v_idx, self._value_codebook).astype(vdtype)

        # ---- Byte accounting -------------------------------------------
        # Indices per token = D / sub_dim, each stored at b bits
        k_bits_per_tok = (D // self._key_sub_dim) * self._key_bits
        v_bits_per_tok = (D // self._value_sub_dim) * self._value_bits
        k_bytes_per_tok = math.ceil(k_bits_per_tok / 8) * H * B
        v_bytes_per_tok = math.ceil(v_bits_per_tok / 8) * H * B
        self._key_bytes_compressed += k_bytes_per_tok * S
        self._value_bytes_compressed += v_bytes_per_tok * S
        self._key_bytes_fp16 += H * B * S * D * 2
        self._value_bytes_fp16 += H * B * S * D * 2
        self._tokens_seen += S
        self._tokens_quantized += S

        return super().update_and_fetch(k_dequant, v_hat)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @property
    def compressed_key_bytes(self) -> int:
        return self._key_bytes_compressed

    @property
    def fp16_key_bytes(self) -> int:
        return self._key_bytes_fp16

    @property
    def compressed_value_bytes(self) -> int:
        return self._value_bytes_compressed

    @property
    def fp16_value_bytes(self) -> int:
        return self._value_bytes_fp16

    @property
    def codebook_bytes(self) -> int:
        """Static codebook overhead in bytes (fp16 storage)."""
        kb = (2 ** self._key_bits) * self._key_sub_dim * 2
        vb = (2 ** self._value_bits) * self._value_sub_dim * 2
        return kb + vb

    @property
    def assigned_avg_bits(self) -> float:
        """Effective bits/element averaged over keys and values.

        Excludes codebook overhead (amortized across many tokens); for an
        end-to-end byte ratio use compressed_*_bytes / fp16_*_bytes.
        """
        k_bits = (self._head_dim // self._key_sub_dim) * self._key_bits / self._head_dim
        v_bits = (self._head_dim // self._value_sub_dim) * self._value_bits / self._head_dim
        return (k_bits + v_bits) / 2.0


__all__ = ["VecInferKVCache"]
