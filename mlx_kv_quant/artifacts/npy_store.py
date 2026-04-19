from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from mlx_kv_quant.core.abstractions import ArtifactStore
from mlx_kv_quant.core.exceptions import ArtifactNotFoundError


class NpyArtifactStore(ArtifactStore):
    """Artifact store that reads and writes ``.npy`` files from a local directory.

    File naming conventions:
        rotation_d{d}_seed{seed}.npy
        codebook_{distribution}_b{b}_d{d}.npy
        jl_d{d}_m{m}_seed{seed}.npy

    Args:
        root_dir: Path to the directory where artifacts are stored.
            Created automatically on first save if absent.
    """

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rotation matrix
    # ------------------------------------------------------------------

    def _rotation_path(self, d: int, seed: int) -> Path:
        return self._root / f"rotation_d{d}_seed{seed}.npy"

    def load_rotation_matrix(self, d: int, seed: int) -> Any:
        path = self._rotation_path(d, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Rotation matrix not found at {path}. "
                f"Run `python -m mlx_kv_quant precompute --head_dim {d}` first."
            )
        import mlx.core as mx
        return mx.array(np.load(path).astype(np.float16))

    def save_rotation_matrix(self, Pi: Any, d: int, seed: int) -> None:
        path = self._rotation_path(d, seed)
        arr = np.array(Pi, dtype=np.float16)
        np.save(path, arr)

    # ------------------------------------------------------------------
    # Codebook
    # ------------------------------------------------------------------

    def _codebook_path(self, distribution: str, b: int, d: int) -> Path:
        return self._root / f"codebook_{distribution}_b{b}_d{d}.npy"

    def load_codebook(self, distribution: str, b: int, d: int) -> Any:
        path = self._codebook_path(distribution, b, d)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"Codebook not found at {path}. "
                f"Run `python -m mlx_kv_quant precompute --head_dim {d} --bits {b}` first."
            )
        import mlx.core as mx
        return mx.array(np.load(path).astype(np.float16))

    def save_codebook(self, cb: Any, distribution: str, b: int, d: int) -> None:
        path = self._codebook_path(distribution, b, d)
        arr = np.array(cb, dtype=np.float16)
        np.save(path, arr)

    # ------------------------------------------------------------------
    # JL matrix
    # ------------------------------------------------------------------

    def _jl_path(self, d: int, m: int, seed: int) -> Path:
        return self._root / f"jl_d{d}_m{m}_seed{seed}.npy"

    def load_jl_matrix(self, d: int, m: int, seed: int) -> Any:
        path = self._jl_path(d, m, seed)
        if not path.exists():
            raise ArtifactNotFoundError(
                f"JL matrix not found at {path}. "
                f"Run `python -m mlx_kv_quant precompute --head_dim {d} --jl_dim {m}` first."
            )
        import mlx.core as mx
        return mx.array(np.load(path).astype(np.float16))

    def save_jl_matrix(self, S: Any, d: int, m: int, seed: int) -> None:
        path = self._jl_path(d, m, seed)
        arr = np.array(S, dtype=np.float16)
        np.save(path, arr)

    # ------------------------------------------------------------------
    # Existence check
    # ------------------------------------------------------------------

    def exists(self, artifact_type: str, **kwargs: Any) -> bool:
        if artifact_type == "rotation":
            return self._rotation_path(kwargs["d"], kwargs["seed"]).exists()
        if artifact_type == "codebook":
            return self._codebook_path(
                kwargs["distribution"], kwargs["b"], kwargs["d"]
            ).exists()
        if artifact_type == "jl":
            return self._jl_path(
                kwargs["d"], kwargs["m"], kwargs["seed"]
            ).exists()
        return False

    def __repr__(self) -> str:
        return f"NpyArtifactStore(root={self._root!r})"
