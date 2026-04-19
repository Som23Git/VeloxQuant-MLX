from __future__ import annotations

from typing import Any

import numpy as np

from mlx_kv_quant.core.abstractions import Preconditioner
from mlx_kv_quant.core.registry import PreconditionerRegistry


@PreconditionerRegistry.register("rotation")
class RotationPreconditioner(Preconditioner):
    """Orthogonal rotation preconditioner Π ∈ ℝ^(d×d).

    Forward:  y = x @ Π^T
    Inverse:  x = y @ Π       (since Π^T Π = I)

    Args:
        Pi: Orthogonal rotation matrix of shape (d, d), fp16 MLX array.
    """

    def __init__(self, Pi: Any) -> None:
        self._Pi = Pi

    def apply(self, x: Any) -> Any:
        """Rotate x: y = x @ Π^T.

        Args:
            x: Array of shape (batch, d).

        Returns:
            Rotated array of shape (batch, d).
        """
        import mlx.core as mx
        return x @ self._Pi.T

    def apply_inverse(self, y: Any) -> Any:
        """Rotate back: x = y @ Π.

        Args:
            y: Rotated array of shape (batch, d).

        Returns:
            Reconstructed array of shape (batch, d).
        """
        import mlx.core as mx
        return y @ self._Pi

    @property
    def dim(self) -> int:
        """Matrix dimension d."""
        return int(self._Pi.shape[0])

    def __repr__(self) -> str:
        return f"RotationPreconditioner(d={self.dim})"
