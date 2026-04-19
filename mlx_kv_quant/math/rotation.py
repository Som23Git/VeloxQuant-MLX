from __future__ import annotations

import numpy as np


def make_rotation_matrix(d: int, seed: int = 42) -> np.ndarray:
    """Generate a d×d random orthogonal rotation matrix via QR decomposition.

    Draws G ~ N(0, 1)^{d×d} and returns Q from G = QR (economy QR).
    The resulting Q is a Haar-distributed orthogonal matrix.

    Args:
        d: Matrix dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (d, d) with orthonormal rows (Q @ Q.T ≈ I).

    Raises:
        ValueError: If d < 1.
    """
    if d < 1:
        raise ValueError(f"make_rotation_matrix: d must be >= 1, got {d}")
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float64)
    Q, _ = np.linalg.qr(G)
    return Q.astype(np.float64)


def make_jl_matrix(d: int, m: int, seed: int = 42) -> np.ndarray:
    """Generate an m×d Gaussian JL projection matrix.

    Each row is drawn i.i.d. from N(0, I_d). This is the correct
    construction for the QJL sign-based inner product estimator to be
    unbiased:

        E[sqrt(pi/2)/m * ||k|| * sum_i sign(s_i·k)(s_i·q)] = <q, k>

    Unlike orthogonal JL, Gaussian JL allows m > d.

    Args:
        d: Input dimension (must be >= 1).
        m: Sketch dimension (must be >= 1).
        seed: NumPy random seed for reproducibility.

    Returns:
        Float64 array of shape (m, d) with i.i.d. N(0,1) entries.

    Raises:
        ValueError: If d < 1 or m < 1.
    """
    if d < 1:
        raise ValueError(f"make_jl_matrix: d must be >= 1, got {d}")
    if m < 1:
        raise ValueError(f"make_jl_matrix: m must be >= 1, got {m}")

    rng = np.random.default_rng(seed + 1)
    S = rng.standard_normal((m, d)).astype(np.float64)
    return S
