from __future__ import annotations

import math
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_kv_quant.codebooks.base import CodebookFactory
from mlx_kv_quant.math.rotation import make_hadamard_diagonal, make_rotation_matrix
from mlx_kv_quant.preconditioners.rotation import HadamardPreconditioner, RotationPreconditioner


class QuantizedLinear(nn.Module):
    """Drop-in replacement for nn.Linear with TurboQuant weight compression.

    Weight matrix W (out, in) is quantized row-by-row offline:
        1. Rotate each row: w_rot = preconditioner.apply(w_row)
        2. Quantize to b-bit Lloyd-Max indices
        3. Store packed uint8 indices + codebook centroids

    Forward pass dequantizes on the fly:
        w_hat = centroids[indices]          # (out, in) fp16
        out   = x @ w_hat.T + bias

    Memory: (out * in * b / 8) bytes vs (out * in * 2) for fp16.
    Compression: 16/b x (e.g. 4x at 4-bit, 5.3x at 3-bit).

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bits: Bit-width for weight quantization (2, 3, or 4).
        use_hadamard: Use randomized Hadamard (O(d log d)) instead of QR (O(d²)).
        bias: Whether the layer has a bias term.
        seed: Random seed for the rotation matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        use_hadamard: bool = True,
        bias: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self._in = in_features
        self._out = out_features
        self._bits = bits
        self._seed = seed

        # Build rotation preconditioner for the weight row dimension (in_features)
        if use_hadamard:
            D_np = make_hadamard_diagonal(in_features, seed=seed)
            self._preconditioner = HadamardPreconditioner(mx.array(D_np))
        else:
            Pi_np = make_rotation_matrix(in_features, seed=seed)
            self._preconditioner = RotationPreconditioner(mx.array(Pi_np.astype(np.float16)))

        # Codebook for N(0, 1/in_features) — same distribution as post-rotation coords
        distribution = "gaussian" if in_features >= 64 else "beta"
        self._codebook = CodebookFactory.create(distribution, b=bits, d=in_features)
        centroids = self._codebook.centroids_mx()   # (2^bits,) fp16

        # Packed weight indices — filled by quantize_weights(), zeros until then
        # Shape: (out_features, in_features) uint8 — one index per coordinate
        self._w_indices: mx.array = mx.zeros((out_features, in_features), dtype=mx.uint8)
        self._centroids: mx.array = centroids       # (2^bits,) fp16, shared across rows

        # Bias (optional, kept in fp16 uncompressed — tiny vs. weight matrix)
        self._bias: Optional[mx.array] = None
        self._has_bias = bias

    def quantize_weights(self, weight: mx.array, bias: Optional[mx.array] = None) -> None:
        """Compress a weight matrix into this layer.

        Call once after construction (or after loading a pretrained model).

        Args:
            weight: Weight tensor of shape (out_features, in_features), fp16 or fp32.
            bias: Optional bias of shape (out_features,).
        """
        w = weight.astype(mx.float32)   # (out, in)

        # Rotate all rows in one batched call: (out, in) treated as batch of rows
        w_rot = self._preconditioner.apply(w)

        # Quantize: broadcast argmin over codebook → (out, in) uint8
        c = self._centroids.astype(mx.float32)   # (k,)
        dists = mx.abs(w_rot[:, :, None] - c[None, None, :])  # (out, in, k)
        self._w_indices = mx.argmin(dists, axis=-1).astype(mx.uint8)

        if bias is not None:
            self._bias = bias.astype(mx.float16)
        mx.eval(self._w_indices)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: dequantize weights then linear projection.

        Args:
            x: Input of shape (..., in_features).

        Returns:
            Output of shape (..., out_features).
        """
        # Dequantize: centroid gather → (out, in) fp16
        w_rot_hat = self._centroids[self._w_indices]   # (out, in) fp16

        # Unrotate: apply inverse rotation to each row
        w_hat = self._preconditioner.apply_inverse(w_rot_hat)  # (out, in) fp16

        # Linear: x @ w_hat.T
        out = x.astype(mx.float16) @ w_hat.T
        if self._bias is not None:
            out = out + self._bias
        return out

    @property
    def memory_bytes(self) -> int:
        """Memory used by compressed weights (indices only, not centroids)."""
        return math.ceil(self._out * self._in * self._bits / 8)

    @property
    def fp16_bytes(self) -> int:
        """Memory the equivalent fp16 layer would use."""
        return self._out * self._in * 2

    def compression_ratio(self) -> float:
        return self.fp16_bytes / self.memory_bytes

    def __repr__(self) -> str:
        ratio = self.compression_ratio()
        return (
            f"QuantizedLinear(in={self._in}, out={self._out}, "
            f"bits={self._bits}, compression={ratio:.1f}x)"
        )
