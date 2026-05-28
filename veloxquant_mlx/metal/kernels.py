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

Phase 2 adds two fused encode+decode kernels:

* :func:`vecinfer_encode_decode_metal` — fused key path: applies
  per-channel smooth scaling and Walsh-Hadamard rotation in threadgroup
  shared memory, then quantizes (nearest-centroid argmin) and
  immediately dequantizes + inverse-transforms, emitting both the
  reconstructed fp16 ``k_hat`` tensor and the uint32 codebook index
  tensor in a single Metal dispatch.  Replaces 7 MLX graph nodes
  (fp32 cast, smooth, matmul WHT, quantize, dequantize, matmul inv-WHT,
  smooth-inverse) with 1 kernel launch.

* :func:`vecinfer_encode_decode_simple_metal` — value path: no
  smooth/Hadamard, just quantize+dequantize in one pass.  Values skip
  the dual-transform per the VecInfer paper.

All kernels JIT-compile on first use and cache the compiled binary for
the rest of the process.
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


# ===========================================================================
# 3. Fused encode+decode — key path  (smooth + WHT + VQ + dequant + inv-WHT)
# ===========================================================================
#
# Grid:        (B * H * S, 1, 1)    — one threadgroup per (batch, head, token)
# Threadgroup: (D, 1, 1)            — D threads share one threadgroup
#
# Each threadgroup processes one input key vector of length D.
# Thread ``lane`` owns output element ``lane``.
#
# Pipeline:
#   Phase A: load k into buf_in (with optional smooth divide)
#   Phase B: WHT = H_mat @ buf_in via dot-product per thread → buf_tg
#   Phase C: quantize (leader per sub-vec scans all centroids via buf_tg)
#   Phase D: dequantize (gather centroid into buf_in)
#   Phase E: inv-WHT = H_mat.T @ buf_in (H is symmetric, T = original)
#   Phase F: smooth multiply (element-wise)
#   Phase G: write k_hat_out[fp16] + idx_out[uint32]
#
# Using the precomputed H_mat (passed as input) ensures bit-exact parity
# with the Python walsh_hadamard_matrix path; no butterfly ordering issues.
#
# Threadgroup memory:
#   float buf_in[D]   — 512 B   input vector (after smooth)
#   float buf_tg[D]   — 512 B   WHT output / dequant staging
#   uint  idx[n_sub]  — 64 B    centroid indices

_VECINFER_ENCODE_DECODE_FULL_SRC = r"""
    threadgroup float buf_in[MAX_D];
    threadgroup float buf_tg[MAX_D];
    threadgroup uint  idx[MAX_N_SUB];

    uint tg_idx  = threadgroup_position_in_grid.x;
    uint lane    = thread_position_in_threadgroup.x;

    uint H           = params[1];
    uint S           = params[2];
    uint D           = params[3];
    uint n_sub       = params[4];
    uint sub_dim     = params[5];
    uint n_cents     = params[6];
    uint has_smooth  = params[7];
    uint smooth_rows = params[8];

    uint s_idx = tg_idx % S;
    uint h_idx = (tg_idx / S) % H;
    uint b_idx = tg_idx / (S * H);

    uint key_base    = ((b_idx * H + h_idx) * S + s_idx) * D;
    uint smooth_base = (has_smooth ? (h_idx % smooth_rows) * D : 0);

    // Phase A: load + optional smooth divide
    float val = float(keys[key_base + lane]);
    if (has_smooth) {
        float s = float(smooth[smooth_base + lane]);
        val = (s > 1e-8f) ? val / s : val;
    }
    buf_in[lane] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase B: WHT via matvec — thread lane computes dot(H_mat[lane, :], buf_in)
    {
        float dot = 0.0f;
        uint row_base = lane * D;
        for (uint c = 0; c < D; ++c) {
            dot += float(H_mat[row_base + c]) * buf_in[c];
        }
        buf_tg[lane] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase C: quantize — one leader per sub-vector scans all centroids
    uint my_sub  = lane / sub_dim;
    uint my_comp = lane % sub_dim;

    if (my_comp == 0 && my_sub < n_sub) {
        float best_dist = INFINITY;
        uint  best_c    = 0;
        uint  x_off     = my_sub * sub_dim;

        for (uint c = 0; c < n_cents; ++c) {
            float dist = 0.0f;
            uint  cb_base = c * sub_dim;
            for (uint i = 0; i < sub_dim; ++i) {
                float d = buf_tg[x_off + i] - float(k_codebook[cb_base + i]);
                dist += d * d;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_c    = c;
            }
        }
        idx[my_sub] = best_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase D: dequantize — gather winning centroid into buf_in
    {
        uint  c       = idx[my_sub];
        uint  cb_base = c * sub_dim;
        buf_in[lane] = float(k_codebook[cb_base + my_comp]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase E: inv-WHT via matvec — H_mat.T[lane, c] = H_mat[c, lane]
    // H is symmetric orthogonal so H.T = H (but we use transpose indexing for clarity)
    {
        float dot = 0.0f;
        for (uint c = 0; c < D; ++c) {
            dot += float(H_mat[c * D + lane]) * buf_in[c];
        }
        buf_tg[lane] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase F: inverse smooth multiply
    float out_val = buf_tg[lane];
    if (has_smooth) {
        float s = float(smooth[smooth_base + lane]);
        out_val *= s;
    }

    // Phase G: write outputs
    k_hat_out[key_base + lane] = half(out_val);

    if (my_comp == 0 && my_sub < n_sub) {
        uint idx_base = ((b_idx * H + h_idx) * S + s_idx) * n_sub;
        idx_out[idx_base + my_sub] = idx[my_sub];
    }
"""


# ===========================================================================
# 4. Fused encode+decode — value path  (VQ only, no smooth/WHT)
# ===========================================================================
#
# Grid:        (B * H * S, 1, 1)
# Threadgroup: (D, 1, 1)
#
# Simplified: no Hadamard, no smooth.  Values are quantized directly in
# their original space per the VecInfer paper.
#
# Threadgroup memory: float buf[D] + uint idx[n_sub]

_VECINFER_ENCODE_DECODE_SIMPLE_SRC = r"""
    threadgroup float buf[MAX_D];
    threadgroup uint  idx[MAX_N_SUB];

    uint tg_idx  = threadgroup_position_in_grid.x;
    uint lane    = thread_position_in_threadgroup.x;

    uint H       = params[1];
    uint S       = params[2];
    uint D       = params[3];
    uint n_sub   = params[4];
    uint sub_dim = params[5];
    uint n_cents = params[6];

    uint s_idx = tg_idx % S;
    uint h_idx = (tg_idx / S) % H;
    uint b_idx = tg_idx / (S * H);

    uint val_base = ((b_idx * H + h_idx) * S + s_idx) * D;

    // Load value element into threadgroup buffer
    buf[lane] = float(values[val_base + lane]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Quantize: thread with (lane % sub_dim == 0) scans all centroids
    uint my_sub  = lane / sub_dim;
    uint my_comp = lane % sub_dim;

    if (my_comp == 0 && my_sub < n_sub) {
        float best_dist = INFINITY;
        uint  best_c    = 0;
        uint  x_off     = my_sub * sub_dim;

        for (uint c = 0; c < n_cents; ++c) {
            float dist = 0.0f;
            uint  cb_base = c * sub_dim;
            for (uint i = 0; i < sub_dim; ++i) {
                float d = buf[x_off + i] - float(v_codebook[cb_base + i]);
                dist += d * d;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_c    = c;
            }
        }
        idx[my_sub] = best_c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Dequantize: gather centroid back into buf
    if (my_sub < n_sub) {
        uint  c       = idx[my_sub];
        uint  cb_base = c * sub_dim;
        buf[lane] = float(v_codebook[cb_base + my_comp]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write outputs
    v_hat_out[val_base + lane] = half(buf[lane]);

    if (my_comp == 0 && my_sub < n_sub) {
        uint idx_base = ((b_idx * H + h_idx) * S + s_idx) * n_sub;
        idx_out[idx_base + my_sub] = idx[my_sub];
    }
"""


# ---------------------------------------------------------------------------
# Compiled-kernel factories for Phase 2
# ---------------------------------------------------------------------------

def _get_encode_decode_full_kernel(D: int, n_sub: int, sub_dim: int, n_centroids: int):
    """Lazily compile the fused key encode-decode kernel."""
    key = ("enc_dec_full", D, n_sub, sub_dim, n_centroids)
    if key not in _kernel_cache:
        # Compile-time constants injected via header — used for threadgroup
        # array sizes (must be compile-time constants in Metal).
        header = (
            "#pragma METAL fp math_mode(relaxed)\n"
            f"#define MAX_D {D}\n"
            f"#define MAX_N_SUB {n_sub}\n"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_enc_dec_full_d{D}_ns{n_sub}_sd{sub_dim}_nc{n_centroids}",
            input_names=["keys", "k_codebook", "smooth", "H_mat", "params"],
            output_names=["k_hat_out", "idx_out"],
            header=header,
            source=_VECINFER_ENCODE_DECODE_FULL_SRC,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


def _get_encode_decode_simple_kernel(D: int, n_sub: int, sub_dim: int, n_centroids: int):
    """Lazily compile the fused value encode-decode kernel."""
    key = ("enc_dec_simple", D, n_sub, sub_dim, n_centroids)
    if key not in _kernel_cache:
        header = (
            "#pragma METAL fp math_mode(relaxed)\n"
            f"#define MAX_D {D}\n"
            f"#define MAX_N_SUB {n_sub}\n"
        )
        _kernel_cache[key] = mx.fast.metal_kernel(
            name=f"vecinfer_enc_dec_simple_d{D}_ns{n_sub}_sd{sub_dim}_nc{n_centroids}",
            input_names=["values", "v_codebook", "params"],
            output_names=["v_hat_out", "idx_out"],
            header=header,
            source=_VECINFER_ENCODE_DECODE_SIMPLE_SRC,
            ensure_row_contiguous=True,
        )
    return _kernel_cache[key]


# ===========================================================================
# Public API — Phase 2
# ===========================================================================

def vecinfer_encode_decode_metal(
    keys: mx.array,
    k_codebook: mx.array,
    sub_dim: int,
    H_mat: mx.array,
    smooth: Optional[mx.array] = None,
) -> tuple:
    """Fused key encode+decode: smooth → WHT → VQ → dequant → inv-WHT → smooth.

    Replaces the 7-node MLX graph (fp32-cast, smooth-divide, matmul-WHT,
    quantize, dequantize, matmul-inv-WHT, smooth-multiply) with a single
    Metal dispatch.  Both outputs are produced in one kernel launch.

    Args:
        keys:       ``[B, H, S, D]`` fp16 or fp32 key tensors.
        k_codebook: ``[n_centroids, sub_dim]`` fp32 centroids.
        sub_dim:    Sub-vector dimension; must evenly divide ``D``.
        H_mat:      ``[D, D]`` orthogonal Walsh-Hadamard matrix (fp32).
                    Must be the same matrix used for calibration, typically
                    from :func:`veloxquant_mlx.allocators.vecinfer.walsh_hadamard_matrix`.
        smooth:     ``[H, D]`` or ``[D,]`` smooth factors, or ``None``
                    for identity (no smooth).

    Returns:
        ``(k_hat, k_indices)`` where:

        * ``k_hat``     — ``[B, H, S, D]`` fp16 reconstructed keys
        * ``k_indices`` — ``[B, H, S, n_sub]`` int32 codebook indices
    """
    if keys.ndim != 4:
        raise ValueError(f"vecinfer_encode_decode_metal: keys must be 4D, got {keys.shape}")
    B, H_dim, S, D = keys.shape
    n_sub = D // sub_dim
    n_centroids = k_codebook.shape[0]

    if D % sub_dim != 0:
        raise ValueError(f"vecinfer_encode_decode_metal: D={D} not divisible by sub_dim={sub_dim}")
    if D > 512:
        raise ValueError(f"vecinfer_encode_decode_metal: D={D} > 512 (threadgroup size limit)")

    # Ensure inputs are fp32 for kernel
    keys_f32 = keys.astype(mx.float32) if keys.dtype != mx.float32 else keys
    cb_f32 = k_codebook.astype(mx.float32) if k_codebook.dtype != mx.float32 else k_codebook
    H_f32 = H_mat.astype(mx.float32) if H_mat.dtype != mx.float32 else H_mat

    # Build smooth tensor — always [H_eff, D] float32
    has_smooth = 0
    smooth_rows = 1
    if smooth is not None:
        has_smooth = 1
        if smooth.ndim == 1:
            smooth_2d = smooth.reshape(1, D).astype(mx.float32)
        else:
            smooth_2d = smooth.astype(mx.float32)
        smooth_rows = smooth_2d.shape[0]
    else:
        smooth_2d = mx.ones((1, D), dtype=mx.float32)

    n_tokens = B * H_dim * S
    params = mx.array(
        [B, H_dim, S, D, n_sub, sub_dim, n_centroids, has_smooth, smooth_rows],
        dtype=mx.uint32,
    )

    kernel = _get_encode_decode_full_kernel(D, n_sub, sub_dim, n_centroids)

    outputs = kernel(
        inputs=[keys_f32, cb_f32, smooth_2d, H_f32, params],
        output_shapes=[(B, H_dim, S, D), (B, H_dim, S, n_sub)],
        output_dtypes=[mx.float16, mx.uint32],
        grid=(n_tokens, 1, 1),
        threadgroup=(D, 1, 1),
    )
    k_hat = outputs[0]
    k_idx = outputs[1].astype(mx.int32)
    return k_hat, k_idx


def vecinfer_encode_decode_simple_metal(
    values: mx.array,
    v_codebook: mx.array,
    sub_dim: int,
) -> tuple:
    """Fused value encode+decode: VQ quantize + dequantize in one pass.

    Values skip the dual-transform per the VecInfer paper, so no smooth
    or Hadamard is needed.

    Args:
        values:     ``[B, H, S, D]`` fp16 or fp32 value tensors.
        v_codebook: ``[n_centroids, sub_dim]`` fp32 centroids.
        sub_dim:    Sub-vector dimension; must evenly divide ``D``.

    Returns:
        ``(v_hat, v_indices)`` where:

        * ``v_hat``     — ``[B, H, S, D]`` fp16 reconstructed values
        * ``v_indices`` — ``[B, H, S, n_sub]`` uint32 codebook indices
    """
    if values.ndim != 4:
        raise ValueError(f"vecinfer_encode_decode_simple_metal: values must be 4D, got {values.shape}")
    B, H, S, D = values.shape
    n_sub = D // sub_dim
    n_centroids = v_codebook.shape[0]

    if D % sub_dim != 0:
        raise ValueError(f"vecinfer_encode_decode_simple_metal: D={D} not divisible by sub_dim={sub_dim}")
    if D > 512:
        raise ValueError(f"vecinfer_encode_decode_simple_metal: D={D} > 512 (threadgroup size limit)")

    values_f32 = values.astype(mx.float32) if values.dtype != mx.float32 else values
    cb_f32 = v_codebook.astype(mx.float32) if v_codebook.dtype != mx.float32 else v_codebook

    n_tokens = B * H * S
    params = mx.array(
        [B, H, S, D, n_sub, sub_dim, n_centroids],
        dtype=mx.uint32,
    )

    kernel = _get_encode_decode_simple_kernel(D, n_sub, sub_dim, n_centroids)

    outputs = kernel(
        inputs=[values_f32, cb_f32, params],
        output_shapes=[(B, H, S, D), (B, H, S, n_sub)],
        output_dtypes=[mx.float16, mx.uint32],
        grid=(n_tokens, 1, 1),
        threadgroup=(D, 1, 1),
    )
    return outputs[0], outputs[1]  # v_hat [fp16], v_indices [uint32]


__all__ = [
    "vecinfer_dequant_metal",
    "vecinfer_quantize_metal",
    "vecinfer_encode_decode_metal",
    "vecinfer_encode_decode_simple_metal",
]
