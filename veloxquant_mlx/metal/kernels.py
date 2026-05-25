"""Metal kernel wrappers for VeloxQuant-MLX hot paths.

Phase 1 ships two kernels:

* :func:`vecinfer_dequant_metal` — drop-in for
  :func:`veloxquant_mlx.allocators.vecinfer.dequantize_vq`. Bit-exact
  with the pure-MLX path. Performance parity with MLX's tuned
  ``mx.take``; included for completeness and as a building block for
  Phase 2's fused dequant+SDPA.

* :func:`vecinfer_quantize_metal` — drop-in for
  :func:`veloxquant_mlx.allocators.vecinfer.quantize_vq`. The pure-MLX
  path allocates a ``[chunk, n_centroids, sub_dim]`` diff tensor that
  OOMs on ``head_dim=256`` models (Falcon3-7B VecInfer-2bit). The
  Metal kernel keeps squared distance accumulation in thread-local
  registers — peak memory drops to ``O(N)`` instead of
  ``O(N * n_centroids * sub_dim)``.

Both kernels JIT-compile on first use and cache the compiled binary
for the rest of the process.
"""
from __future__ import annotations

from typing import Optional

import mlx.core as mx


# ===========================================================================
# 1. dequantize — codebook gather
# ===========================================================================
#
# Signature (MLX generates this from input/output_names):
#   const device uint32_t* indices  [[buffer(0)]]   // [N_total]
#   const device T*        codebook [[buffer(1)]]   // [n_centroids, sub_dim]
#   device       T*        out      [[buffer(2)]]   // [N_total * sub_dim]
#
# One thread per input index; thread copies sub_dim contiguous components.
_VECINFER_DEQUANT_SRC = r"""
    uint flat_idx = thread_position_in_grid.x;
    uint N_total  = indices_shape[0];
    if (flat_idx >= N_total) {
        return;
    }

    uint sub_dim = codebook_shape[1];
    uint code_idx = indices[flat_idx];
    uint cb_base  = code_idx * sub_dim;
    uint out_base = flat_idx * sub_dim;

    for (uint i = 0; i < sub_dim; ++i) {
        out[out_base + i] = codebook[cb_base + i];
    }
"""


# ===========================================================================
# 2. quantize — fused nearest-centroid argmin
# ===========================================================================
#
# Signature:
#   const device T*        x        [[buffer(0)]]   // [N_total, sub_dim]
#   const device T*        codebook [[buffer(1)]]   // [n_centroids, sub_dim]
#   device       uint32_t* out      [[buffer(2)]]   // [N_total]
#
# One thread per sub-vector. The thread iterates over all centroids,
# computing accumulated squared distance in a register, and writes only
# the index of the minimum. No O(N*n_centroids*sub_dim) intermediate.
#
# sub_dim is small (4/8/16) so the inner accumulation loop is cheap
# even when sub_dim isn't known at compile time.
_VECINFER_QUANTIZE_SRC = r"""
    uint vec_idx = thread_position_in_grid.x;
    uint N_total = x_shape[0];
    if (vec_idx >= N_total) {
        return;
    }

    uint n_centroids = codebook_shape[0];
    uint sub_dim     = codebook_shape[1];
    uint x_base      = vec_idx * sub_dim;

    // Track running argmin in registers — never materialize the diff matrix.
    float best_dist = INFINITY;
    uint  best_idx  = 0;

    for (uint c = 0; c < n_centroids; ++c) {
        uint cb_base = c * sub_dim;
        float dist = 0.0f;
        for (uint i = 0; i < sub_dim; ++i) {
            float d = float(x[x_base + i]) - float(codebook[cb_base + i]);
            dist += d * d;
        }
        if (dist < best_dist) {
            best_dist = dist;
            best_idx  = c;
        }
    }

    out[vec_idx] = best_idx;
"""


# ---------------------------------------------------------------------------
# Compiled-kernel cache — one entry per (kernel_name, input_dtype).
# ---------------------------------------------------------------------------
_kernel_cache: dict = {}


def _get_dequant_kernel(dtype: mx.Dtype):
    """Lazily compile the dequant kernel for the given codebook dtype."""
    key = ("dequant", str(dtype))
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_dequant_{str(dtype).replace('.', '_')}",
            input_names=["indices", "codebook"],
            output_names=["out"],
            source=_VECINFER_DEQUANT_SRC,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def _get_quantize_kernel(dtype: mx.Dtype):
    """Lazily compile the quantize kernel for the given input dtype.

    Both ``x`` and ``codebook`` must share the same dtype at the kernel
    boundary; the Python wrapper casts as needed.
    """
    key = ("quantize", str(dtype))
    if key not in _kernel_cache:
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_quantize_{str(dtype).replace('.', '_')}",
            input_names=["x", "codebook"],
            output_names=["out"],
            source=_VECINFER_QUANTIZE_SRC,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


# ===========================================================================
# Public API
# ===========================================================================

def vecinfer_dequant_metal(
    indices: mx.array,
    codebook: mx.array,
    out_dtype: Optional[mx.Dtype] = None,
) -> mx.array:
    """Drop-in Metal replacement for
    :func:`veloxquant_mlx.allocators.vecinfer.dequantize_vq`.

    Args:
        indices: Codebook indices, ``[..., n_sub]``. Promoted to ``uint32``
            for the kernel.
        codebook: Centroid table, ``[n_centroids, sub_dim]``. Any float
            dtype; output matches ``out_dtype`` (default: codebook dtype).
        out_dtype: Output dtype. Defaults to ``codebook.dtype``.

    Returns:
        Reconstruction of shape ``[..., n_sub * sub_dim]``. Bit-equivalent
        to::

            mx.take(codebook, indices.flatten()).reshape(..., n_sub * sub_dim)
    """
    if codebook.ndim != 2:
        raise ValueError(
            f"vecinfer_dequant_metal: codebook must be 2D, got {codebook.shape}"
        )
    sub_dim = codebook.shape[1]
    *leading, n_sub = indices.shape
    n_total = 1
    for d in indices.shape:
        n_total *= d

    flat_indices = indices.reshape(-1).astype(mx.uint32)

    if out_dtype is None:
        out_dtype = codebook.dtype

    kernel = _get_dequant_kernel(out_dtype)
    tg_x = min(256, n_total)
    cb = codebook.astype(out_dtype) if codebook.dtype != out_dtype else codebook

    outputs = kernel(
        inputs=[flat_indices, cb],
        output_shapes=[(n_total * sub_dim,)],
        output_dtypes=[out_dtype],
        grid=(n_total, 1, 1),
        threadgroup=(tg_x, 1, 1),
    )
    return outputs[0].reshape(*leading, n_sub * sub_dim)


def vecinfer_quantize_metal(
    x: mx.array,
    codebook: mx.array,
    sub_dim: int,
) -> mx.array:
    """Drop-in Metal replacement for
    :func:`veloxquant_mlx.allocators.vecinfer.quantize_vq`.

    The pure-MLX path materializes a ``[chunk, n_centroids, sub_dim]``
    diff tensor on every chunk, which OOMs when ``head_dim`` and
    ``n_centroids`` are both large (Falcon3-7B VecInfer-2bit). This
    kernel computes squared distance in thread-local registers and
    emits only the winning index — peak memory is ``O(N)``.

    Args:
        x: Input vectors ``[..., D]`` with ``D`` divisible by ``sub_dim``.
        codebook: ``[n_centroids, sub_dim]``.
        sub_dim: Sub-vector dimension; must match ``codebook.shape[1]``.

    Returns:
        ``[..., D // sub_dim]`` int32 indices. Equivalent to::

            quantize_vq(x, codebook, sub_dim)
    """
    *leading, D = x.shape
    if D % sub_dim != 0:
        raise ValueError(
            f"vecinfer_quantize_metal: D={D} not divisible by sub_dim={sub_dim}"
        )
    if codebook.ndim != 2 or codebook.shape[1] != sub_dim:
        raise ValueError(
            f"vecinfer_quantize_metal: codebook must be [n_centroids, {sub_dim}], "
            f"got {codebook.shape}"
        )

    n_sub = D // sub_dim
    # Reshape to flat sub-vector layout [N_total, sub_dim] where
    # N_total = prod(leading) * n_sub.
    x_sub = x.reshape(*leading, n_sub, sub_dim)
    flat_x = x_sub.reshape(-1, sub_dim)
    n_total = flat_x.shape[0]

    # Kernel reads x and codebook at the same dtype — promote to fp32
    # if either is integer/bf16 mismatch. fp16 is fine.
    work_dtype = flat_x.dtype if flat_x.dtype in (mx.float16, mx.float32) else mx.float32
    flat_x_w = flat_x.astype(work_dtype) if flat_x.dtype != work_dtype else flat_x
    cb_w = codebook.astype(work_dtype) if codebook.dtype != work_dtype else codebook

    kernel = _get_quantize_kernel(work_dtype)
    tg_x = min(256, max(1, n_total))

    outputs = kernel(
        inputs=[flat_x_w, cb_w],
        output_shapes=[(n_total,)],
        output_dtypes=[mx.uint32],
        grid=(n_total, 1, 1),
        threadgroup=(tg_x, 1, 1),
    )
    # Cast to int32 to match pure-MLX path's contract.
    out_flat = outputs[0].astype(mx.int32)
    return out_flat.reshape(*leading, n_sub)


__all__ = ["vecinfer_dequant_metal", "vecinfer_quantize_metal"]
