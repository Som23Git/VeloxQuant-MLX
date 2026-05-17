"""Shared pytest fixtures for veloxquant_mlx tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def d() -> int:
    """Default head dimension."""
    return 128


@pytest.fixture(scope="session")
def seed() -> int:
    return 42


@pytest.fixture(scope="session")
def small_vectors(d: int) -> np.ndarray:
    """100 random unit-norm fp32 vectors of dimension d."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    return x


@pytest.fixture(scope="session")
def rotation_matrix(d: int, seed: int) -> np.ndarray:
    """d×d orthogonal rotation matrix."""
    from veloxquant_mlx.math.rotation import make_rotation_matrix
    return make_rotation_matrix(d, seed=seed)


@pytest.fixture(scope="session")
def jl_matrix_128(d: int, seed: int) -> np.ndarray:
    """128×d JL projection matrix."""
    from veloxquant_mlx.math.rotation import make_jl_matrix
    return make_jl_matrix(d, m=min(128, d), seed=seed)


@pytest.fixture(scope="session")
def codebook_b2(d: int) -> object:
    """Gaussian Lloyd-Max codebook at b=2."""
    from veloxquant_mlx.codebooks.base import CodebookFactory
    return CodebookFactory.create("gaussian", b=2, d=d)


@pytest.fixture(scope="session")
def in_memory_store():
    """InMemoryArtifactStore pre-populated with d=64 artifacts."""
    from veloxquant_mlx.artifacts.memory_store import InMemoryArtifactStore
    from veloxquant_mlx.math.rotation import make_jl_matrix, make_rotation_matrix

    store = InMemoryArtifactStore()
    d = 64
    Pi = make_rotation_matrix(d, seed=42)
    S = make_jl_matrix(d, m=d, seed=42)
    store.save_rotation_matrix(Pi, d=d, seed=42)
    store.save_jl_matrix(S, d=d, m=d, seed=42)
    return store
