# KV Cache Quantization on Apple Silicon M4
## A Unified MLX Implementation Guide for TurboQuant, PolarQuant, and QJL

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Paper Breakdown](#2-paper-breakdown)
3. [Mathematical to Practical Mapping](#3-mathematical-to-practical-mapping)
4. [Apple Silicon–First Implementation Plan](#4-apple-siliconfirst-implementation-plan)
5. [Apple Silicon Optimization Strategy](#5-apple-silicon-optimization-strategy)
6. [Local Execution Constraints and Trade-offs](#6-local-execution-constraints-and-trade-offs)
7. [Step-by-Step Build Guide](#7-step-by-step-build-guide)
8. [Code Snippets](#8-code-snippets)
9. [Evaluation and Metrics](#9-evaluation-and-metrics)
10. [Risks, Limitations, and Challenges](#10-risks-limitations-and-challenges)
11. [Extensions and Improvements](#11-extensions-and-improvements)

---

## 1. High-Level Overview

### What Problem These Papers Solve

Large language models store Key-Value (KV) embeddings for every generated token in a KV cache. At 16-bit precision, this cache grows to gigabytes for long-context models — a severe bottleneck on memory-constrained hardware like a MacBook Air M4. The three papers in this collection form a progression toward theoretically-grounded, online, hardware-friendly KV cache quantization:

**QJL** (2406.03482) introduces the foundational primitive: a 1-bit quantized Johnson-Lindenstrauss transform that produces an *unbiased* inner product estimator with zero per-token quantization overhead (no stored scale/zero-point).

**PolarQuant** (2502.02617) builds on QJL by transforming KV vectors into polar coordinates after random preconditioning. It exploits the analytically predictable angle distribution to design level-specific codebooks, achieving ~4.2× KV compression at near-lossless quality.

**TurboQuant** (2504.19874) is the most general and theoretically complete system. It proves near-optimal MSE and inner-product distortion bounds for any bit-width, constructs optimal scalar quantizers via the Lloyd-Max algorithm on Beta-distributed coordinates after random rotation, and provides a two-stage scheme (MSE quantizer + QJL residual) for unbiased inner product estimation. It achieves quality-neutral KV compression at 3.5 bits/channel and only marginal degradation at 2.5 bits/channel.

### Why the Approaches Work

All three methods share a core insight: **random preconditioning (rotation/projection) transforms arbitrary worst-case vectors into statistically well-behaved ones** — coordinates become nearly independent and their distribution becomes analytically tractable (Beta/Gaussian for TurboQuant/QJL; Gamma-angle distributions for PolarQuant). This allows optimal scalar quantizers to be precomputed once and applied per-coordinate at runtime, making the algorithms online, data-oblivious, and vectorization-friendly.

### Key Innovations

**QJL:** An asymmetric estimator — sign-quantize the key, apply a full JL transform to the query — yields an unbiased inner product estimate. No normalization constants needed.

**PolarQuant:** Recursive polar decomposition exposes a structured, level-dependent angle distribution that concentrates tightly around π/4 at higher levels. This removes the need for explicit data normalization (the biggest source of memory overhead in classical quantization).

**TurboQuant:** Proves the information-theoretic lower bound for any vector quantizer, and shows TurboQuant is within a factor of √(3π)/2 ≈ 2.7 of optimal at any bit-width. The two-stage MSE+QJL composition resolves the bias problem in MSE-optimal quantizers.

### Suitability for Apple Silicon

**Excellent.** These algorithms are:

- **Online and data-oblivious**: no k-means training on input data, works in streaming token generation.
- **Vectorization-friendly**: all operations are matrix multiplications, element-wise sign operations, codebook lookups — all MLX-native.
- **Memory-light**: the precomputed rotation matrix and codebook are small (d×d float16 and a few hundred floats).
- **No CUDA dependency**: nothing in the algorithm requires GPU-specific primitives unavailable on Apple Neural Engine or Metal-backed MLX.

A MacBook Air M4 with 16–32 GB unified memory can comfortably run 7–8B parameter models while integrating TurboQuant/QJL KV compression for context lengths up to ~50K tokens.

---

## 2. Paper Breakdown

### 2.1 QJL — The Core Primitive

**Problem:** Each key token k ∈ ℝ^d is expensive to store at fp16. Traditional quantization needs per-block scale+zeropoint overhead (~1–2 extra bits per number).

**Method:**
1. Draw a shared random Gaussian matrix S ∈ ℝ^(m×d) once, orthogonalize its rows via QR.
2. For each new key token k: compute sign(S·k) ∈ {-1,+1}^m and store ‖k‖₂ as a scalar.
3. At attention time, estimate ⟨q, k⟩ using: `ProdQJL(q, k) = sqrt(π/2) / m * ‖k‖₂ * ⟨S·q, sign(S·k)⟩`

The value cache uses standard per-token quantization (normalize, round to int8).

**Key theoretical result:** The estimator is exactly unbiased (E[ProdQJL] = ⟨q,k⟩), and with m ≥ (4/3)·(1+ε)/ε² · log(2/δ), the error is bounded by ε‖q‖‖k‖ with probability 1-δ.

**Outlier handling:** In deeper layers, ~4 fixed channels exhibit large magnitudes. Identify them once during prefill; quantize outlier channels and inlier channels using two independent QJL instances with different compression rates.

### 2.2 PolarQuant — Polar Coordinate Quantization

**Problem:** Cartesian coordinate distributions vary wildly across tokens; explicit normalization adds memory overhead.

**Method:**
1. Apply a shared random rotation matrix S ∈ ℝ^(d×d) to each KV vector.
2. Run a recursive polar transformation for L=4 levels:
   - Level 1: group pairs (x_{2j-1}, x_{2j}), compute θ_j = atan2(x_{2j}, x_{2j-1}) and r_j = ‖(x_{2j-1}, x_{2j})‖
   - Levels 2–4: repeat on the radius vectors from the previous level.
3. After 4 levels, you have 15d/16 angles and d/16 final radii.
4. Quantize angles to b bits using precomputed optimal codebooks (different per level).
5. Store the final scalar radius in fp16.

**Bit budget:** Level-1 angles use 4 bits (range [0, 2π)), levels 2–4 use 2 bits (range [0, π/2]). Total: fp16 + 32 + 8 + 4 + 2 = fp16 + 46 bits per 16 coordinates = 3.875 bits/coord at fp16.

**Key theoretical result:** After random preconditioning, angles at level ℓ follow f_Ψ(ψ) ∝ sin^(2^(ℓ-1)-1)(2ψ). This distribution is independent of the input vector. At higher levels it concentrates sharply around π/4, enabling accurate quantization with very few bits.

### 2.3 TurboQuant — Near-Optimal Vector Quantization

**TurboQuantMSE:**
1. Apply a random rotation Π ∈ ℝ^(d×d) (orthogonal matrix via QR decomposition of random Gaussian).
2. Each coordinate of Π·x follows a Beta distribution (Lemma 1). In high dimensions, this converges to N(0, 1/d).
3. Precompute optimal scalar quantizer codebooks c₁,...,c_{2^b} for the Beta distribution by solving the continuous 1D k-means problem (Lloyd-Max algorithm). Store these for b = 1,2,3,4.
4. **Quantize:** for each coordinate y_j of Π·x, find and store the index of the nearest centroid.
5. **Dequantize:** retrieve centroids, apply Π^T.

**MSE Bounds:** D_mse ≤ (√(3π)/2) · 4^(-b). Empirically: b=1→0.36, b=2→0.117, b=3→0.03, b=4→0.009.

**TurboQuantProd (two-stage, unbiased inner product):**
1. Apply TurboQuantMSE at bit-width b-1 to get x̃_mse and residual r = x - x̃_mse.
2. Apply QJL to the residual r: store sign(S·r) and ‖r‖₂.
3. **Dequantize:** x̃ = x̃_mse + ‖r‖₂ · (√(π/2)/d) · S^T · sign(S·r).

**Inner product distortion:** D_prod ≤ (√(3π)/2) · ‖y‖²/d · 4^(-b). Unbiased by construction.

**Lower bounds (Theorem 3):** For any randomized quantizer, D_mse ≥ 4^(-b) and D_prod ≥ (‖y‖²/d) · 4^(-b). TurboQuant is within √(3π)/2 ≈ 2.7× of information-theoretically optimal.

**Outlier handling (practical extension):** Split d channels into k outlier channels (large magnitude) and d-k inlier channels. Apply two independent TurboQuant instances with different bit allocations. For 2.5-bit effective precision with d=128: 32 outlier channels at 3 bits + 96 inlier channels at 2 bits.

---

## 3. Mathematical to Practical Mapping

### 3.1 The Rotation Matrix Π

**Math:** Π ∈ ℝ^(d×d), orthogonal. Generated by QR decomposition of a random Gaussian matrix.

**Code:** Generate once, store in float16.
```python
import mlx.core as mx
import numpy as np

def make_rotation_matrix(d: int, seed: int = 42) -> mx.array:
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    return mx.array(Q.astype(np.float16))
```

**MLX note:** Store as float16. The matrix-vector product Π·x for a batch of n vectors is just `x @ Pi.T` (shape n×d), which is a single fused GEMM kernel on the Metal GPU — maximally efficient on Apple Silicon.

### 3.2 The Beta Distribution Codebook

**Math:** Coordinates of Π·x follow f_X(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2). In high dimensions, this converges to N(0, 1/d). The Lloyd-Max optimality condition gives codebook centroids c₁,...,c_{2^b} that minimize ∑∫|x-cᵢ|² f_X(x)dx.

**Practical mapping:** In high dimensions (d ≥ 64), the Beta distribution is well-approximated by N(0, 1/d), so you can precompute Lloyd-Max centroids for a standard Gaussian and rescale. For small d, solve numerically.

```python
from scipy.optimize import minimize
from scipy.stats import norm
import numpy as np

def lloyd_max_gaussian(n_levels: int, sigma: float, n_iter: int = 300):
    """Solve Lloyd-Max for N(0, sigma^2) scalar quantization."""
    # Initialize centroids uniformly
    centroids = np.linspace(-3*sigma, 3*sigma, n_levels)
    for _ in range(n_iter):
        # Boundaries are midpoints
        bounds = np.concatenate([[-np.inf],
                                  (centroids[:-1] + centroids[1:]) / 2,
                                  [np.inf]])
        # Update centroids: E[X | X in interval]
        new_centroids = []
        for i in range(n_levels):
            lo, hi = bounds[i], bounds[i+1]
            p_lo = norm.cdf(lo / sigma)
            p_hi = norm.cdf(hi / sigma)
            p = p_hi - p_lo
            if p < 1e-12:
                new_centroids.append(centroids[i])
            else:
                # E[X | lo < X < hi] for Gaussian
                e = sigma * (norm.pdf(lo/sigma) - norm.pdf(hi/sigma)) / p
                new_centroids.append(e)
        centroids = np.array(new_centroids)
    return centroids

# Precompute for b = 1, 2, 3, 4
# sigma = 1/sqrt(d) but we store normalized; rescale at runtime
CODEBOOKS = {}
for b in [1, 2, 3, 4]:
    CODEBOOKS[b] = lloyd_max_gaussian(2**b, sigma=1.0)
```

**The scaling:** Since Π·x has coordinates ~N(0, 1/d), scale the unit-sigma codebook by 1/√d at runtime. Alternatively, normalize x to unit norm first (standard assumption), then scale by 1/√d.

### 3.3 Quantization as Nearest-Centroid Lookup

**Math:** idx_j = argmin_k |y_j - c_k| for each coordinate j.

**MLX:** This is a broadcast distance computation — no loop needed.

```python
def quantize_coords(y: mx.array, codebook: mx.array) -> mx.array:
    # y: (batch, d), codebook: (2^b,)
    # returns indices: (batch, d) in uint8
    dists = mx.abs(y[:, :, None] - codebook[None, None, :])  # (batch, d, 2^b)
    return mx.argmin(dists, axis=-1).astype(mx.uint8)

def dequantize_coords(idx: mx.array, codebook: mx.array) -> mx.array:
    # idx: (batch, d), codebook: (2^b,)
    return codebook[idx]  # (batch, d) — simple gather
```

**Memory:** At b=2 bits, indices are uint8 (8 bits on disk). Pack 4 indices into one byte for actual 2-bit storage. In MLX, uint8 is the smallest practical type.

### 3.4 QJL Sign Quantization

**Math:** QJL(k) = sign(S·k), S ∈ ℝ^(m×d) with Sᵢⱼ ~ N(0,1), orthogonalized.

**MLX:**
```python
def qjl_quantize(k: mx.array, S: mx.array) -> tuple[mx.array, mx.array]:
    # k: (batch, d), S: (m, d)
    Sk = k @ S.T  # (batch, m)
    signs = mx.sign(Sk).astype(mx.int8)  # {-1, +1}^m, stored as int8
    norms = mx.norm(k, axis=-1, keepdims=True)  # (batch, 1)
    return signs, norms

def qjl_inner_product(q: mx.array, k_signs: mx.array,
                       k_norms: mx.array, S: mx.array) -> mx.array:
    # q: (1, d), k_signs: (n, m) int8, k_norms: (n, 1)
    Sq = q @ S.T  # (1, m)
    ip = (Sq * k_signs.astype(mx.float16)).sum(axis=-1)  # (n,)
    m = S.shape[0]
    return (mx.sqrt(mx.array(mx.pi / 2)) / m) * k_norms.squeeze() * ip
```

**Key insight for Apple Silicon:** The inner product `Sq @ k_signs` where k_signs is int8 maps naturally to a matrix-vector multiply with integer accumulation. MLX supports mixed-precision GEMMs. For large n (long context), this is the hot path — profile and optimize this first.

### 3.5 TurboQuantProd Two-Stage Pipeline

**Math:** Qprod(x) = (Qmse(x), sign(S·r), ‖r‖₂) where r = x - Qmse⁻¹(Qmse(x)).

**MLX assembly:**
```python
def turboquant_prod_encode(x: mx.array, Pi: mx.array, codebook: mx.array,
                            S: mx.array, b: int):
    # Stage 1: MSE quantize at b-1 bits
    y = x @ Pi.T
    idx = quantize_coords(y, codebook)      # (batch, d) uint8
    y_hat = dequantize_coords(idx, codebook)
    x_hat_mse = y_hat @ Pi                  # (batch, d)
    # Stage 2: QJL on residual
    r = x - x_hat_mse
    r_signs, r_norms = qjl_quantize(r, S)   # (batch, m), (batch, 1)
    return idx, r_signs, r_norms.squeeze(-1)

def turboquant_prod_decode(idx: mx.array, r_signs: mx.array,
                            r_norms: mx.array, Pi: mx.array,
                            codebook: mx.array, S: mx.array) -> mx.array:
    y_hat = dequantize_coords(idx, codebook)
    x_hat_mse = y_hat @ Pi
    x_hat_qjl = (r_norms[:, None] * (mx.sqrt(mx.array(mx.pi / 2)) / S.shape[0])
                 * (r_signs.astype(mx.float16) @ S))
    return x_hat_mse + x_hat_qjl
```

### 3.6 Polar Transformation (PolarQuant)

**Math:** Recursive: ψ_j^(1) = atan2(x_{2j}, x_{2j-1}), r_j^(1) = ‖(x_{2j-1}, x_{2j})‖; repeat on radii for L levels.

**MLX:**
```python
def polar_transform(x: mx.array, n_levels: int = 4):
    """x: (batch, d), d must be power of 2."""
    batch, d = x.shape
    angles = []
    r = x
    for ell in range(n_levels):
        r_pairs = r.reshape(batch, -1, 2)  # (batch, d/2^ell, 2)
        a = mx.arctan2(r_pairs[:, :, 1], r_pairs[:, :, 0])  # (batch, d/2^ell)
        r = mx.sqrt(r_pairs[:, :, 0]**2 + r_pairs[:, :, 1]**2)  # new radii
        angles.append(a)
    final_radius = r.squeeze(-1)  # scalar per vector
    return angles, final_radius

def polar_inverse(angles: list, final_radius: mx.array,
                  codebooks: dict, n_levels: int = 4) -> mx.array:
    r = final_radius[:, None]  # (batch, 1)
    for ell in range(n_levels - 1, -1, -1):
        theta = dequantize_coords(angles[ell], codebooks[ell])
        r_prev = mx.concatenate(
            [r * mx.cos(theta), r * mx.sin(theta)], axis=-1
        )
        # interleave: [cos0, sin0, cos1, sin1, ...]
        batch = r.shape[0]
        r = r_prev.reshape(batch, -1)
    return r
```

---

## 4. Apple Silicon–First Implementation Plan

### 4.1 Module Structure

```
kv_quant/
├── __init__.py
├── rotation.py          # Random rotation matrix generation and caching
├── codebook.py          # Lloyd-Max codebook precomputation (scipy offline)
│                        # and MLX-based quantize/dequantize
├── qjl.py               # QJL: encode keys, estimate inner products
├── turboquant.py        # TurboQuantMSE and TurboQuantProd
├── polarquant.py        # PolarQuant recursive polar transform
├── kv_cache.py          # Drop-in KV cache with quantization backend
├── outlier.py           # Outlier channel detection and split quantization
├── config.py            # Hyperparameters: d, b, m, n_levels, n_outlier
└── benchmarks/
    ├── distortion.py    # Measure MSE and inner product error vs. theory
    ├── memory.py        # Track memory footprint
    └── latency.py       # Token generation throughput
```

### 4.2 Data Flow During Inference

```
Token arrives → compute q, k, v (fp16)
                       ↓
              Detect outlier channels (once, at prefill)
                       ↓
         Split k into k_outlier (k_o) and k_inlier (k_i)
                       ↓
    Quantize k_o with TurboQuantProd(b=3)  ←── higher bits for outliers
    Quantize k_i with TurboQuantProd(b=2)  ←── lower bits for inliers
                       ↓
         Store: (idx_o, signs_o, norm_o, idx_i, signs_i, norm_i)
         + fp16 norm of full k (for QJL scaling)
                       ↓
    Standard per-token int8 quantization for v cache
                       ↓
         Attention score computation:
         Decode k̃ from stored tokens (or compute inner product directly)
         Compute softmax(q·K̃^T / √d) · Ṽ
```

### 4.3 Key Design Decisions for MLX

**Shared rotation matrix:** All layers and heads share one rotation matrix Π (d×d). With d=128, Π is 128×128×2 bytes = 32 KB — trivially fits in L1 cache on M4.

**Codebook as MLX constant:** Store precomputed codebooks as `mx.array` constants. Index lookups become gather operations, which are fast on Apple Silicon.

**Batching:** Process all attention heads simultaneously. If model has H heads with head_dim d_h, reshape the KV vectors to (n_tokens × H, d_h) and apply quantization in a single batched call.

**Lazy evaluation:** MLX uses lazy evaluation by default. Chain quantization operations without explicit `.eval()` calls until the KV cache must be stored. This amortizes kernel launch overhead.

**Precision:** Use fp16 throughout. The rotation matrix, codebook centroids, and residual norms are all fp16. The only int operations are the index/sign arrays. MLX's Metal backend handles fp16 matrix multiplies efficiently on M4.

### 4.4 KV Cache Class Design

```python
class TurboQuantKVCache:
    def __init__(self, config: QuantConfig):
        self.config = config
        # Precomputed, shared state
        self.Pi = load_rotation_matrix(config.d, config.seed)    # (d, d) fp16
        self.S  = load_jl_matrix(config.d, config.m, config.seed) # (m, d) fp16
        self.codebook_b = load_codebook(config.b - 1)             # (2^(b-1),) fp16
        self.codebook_bo = load_codebook(config.b_outlier)        # outlier codebook
        # Dynamic cache state
        self.k_idx:    list[mx.array] = []  # (d,) uint8 per token
        self.k_signs:  list[mx.array] = []  # (m,) int8 per token
        self.k_norms:  list[mx.array] = []  # scalar per token
        self.v_cache:  list[mx.array] = []  # (d,) int8 per token
        self.v_scales: list[mx.array] = []  # scalar per token
        self.outlier_channels: mx.array | None = None  # set at prefill

    def append(self, k: mx.array, v: mx.array):
        """Called once per new token."""
        # k, v: (d,) fp16
        k_q = turboquant_prod_encode(k[None], self.Pi, self.codebook_b, self.S,
                                      self.config.b)
        self.k_idx.append(k_q[0].squeeze(0))
        self.k_signs.append(k_q[1].squeeze(0))
        self.k_norms.append(k_q[2])
        # Simple int8 for values
        v_scale = mx.max(mx.abs(v)) / 127.0
        self.v_scales.append(v_scale)
        self.v_cache.append((v / v_scale).astype(mx.int8))

    def attend(self, q: mx.array) -> mx.array:
        """Compute attention output for query q: (d,) fp16."""
        n = len(self.k_idx)
        # Stack cached data
        k_idx   = mx.stack(self.k_idx)    # (n, d) uint8
        k_signs = mx.stack(self.k_signs)  # (n, m) int8
        k_norms = mx.stack(self.k_norms)  # (n,)
        # Decode key estimates (or compute inner products directly)
        k_hat = turboquant_prod_decode(k_idx, k_signs, k_norms,
                                        self.Pi, self.codebook_b, self.S)
        # Attention scores
        scores = mx.softmax(k_hat @ q / mx.sqrt(mx.array(q.shape[-1])), axis=0)
        # Decode and weight values
        v_scales = mx.stack(self.v_scales)  # (n,)
        v_int    = mx.stack(self.v_cache)   # (n, d) int8
        v_hat    = v_int.astype(mx.float16) * v_scales[:, None]
        return (scores[:, None] * v_hat).sum(axis=0)
```

---

## 5. Apple Silicon Optimization Strategy

### 5.1 Memory Layout

The M4 has 120 GB/s memory bandwidth and 16 GB (base) or 32 GB unified memory. The KV cache is the main memory consumer. With TurboQuantProd at b=2.5 bits effective:

| Config | Per-token per-head storage | 8B model, 50K tokens |
|--------|---------------------------|----------------------|
| fp16 baseline | 2 × d × 2 bytes = 512 B (d=128) | ~12.8 GB |
| TurboQuant 2.5-bit | ~80 B | ~2.0 GB |
| TurboQuant 3.5-bit | ~112 B | ~2.8 GB |

This makes 50K-token context feasible on a 16 GB M4 MacBook Air alongside the model weights (~8 GB for an 8B model in int4).

### 5.2 Operation Mapping to MLX Primitives

| Algorithm step | MLX operation | Notes |
|---|---|---|
| Rotation `x @ Pi.T` | `mx.matmul` | Fused GEMM, fp16 |
| Nearest centroid | `mx.argmin(abs(y - codebook))` | Vectorized broadcast |
| Codebook lookup | `codebook[idx]` | Gather, very fast |
| QJL `S @ k` | `mx.matmul` | GEMM, fp16 |
| Sign | `mx.sign` | Element-wise Metal kernel |
| Residual norm | `mx.norm` | Single reduction |
| Inner product decode | `(signs @ S)` | GEMM with int8 accumulation |
| Polar atan2 | `mx.arctan2` | Element-wise |
| Softmax | `mx.softmax` | Numerically stable built-in |

### 5.3 Precision Choices

Use **float16** for all rotation matrices, codebooks, and intermediate values. The M4's AMX (Apple Matrix Extension) coprocessor processes fp16 GEMM at peak throughput. Avoid float32 for any inner-loop operations.

For the sign/index arrays: `mx.int8` for signs and `mx.uint8` for codebook indices (even if b < 8 — packing into true 2/3-bit storage is optional and complex to implement in MLX without custom Metal kernels).

### 5.4 Batch Sizes

During token generation (decoding), the batch dimension is 1 (single token). The bottleneck is the attention score computation: a matrix-vector multiply of shape (n_tokens, d) × (d,). With TurboQuant, this becomes (n_tokens, d) reconstruction + dot product.

For prefill (prompt encoding), process tokens in chunks of 512–1024 for cache efficiency. Avoid processing the full prompt as one matrix if it would exceed ~2 GB of activation memory.

### 5.5 Unified Memory Advantages

Unlike CUDA systems where data must be explicitly transferred between CPU and GPU memory, MLX on Apple Silicon accesses all data from unified memory. This means:

- The rotation matrix and codebooks are always "on device" — no transfer overhead.
- The KV cache can be large without penalizing access speed (DRAM bandwidth is shared, not duplicated).
- Intermediate activations (residuals, projections) don't need explicit deallocation between operations.

### 5.6 When to Train vs. Fine-tune vs. Inference-only

**Inference-only** (recommended for MacBook Air M4): Load a pre-trained 7–8B model (e.g., Llama-3.1-8B in int4 via mlx-lm), integrate TurboQuant/QJL as a drop-in KV cache replacement, and run inference. No training needed — quantization is applied online.

**Fine-tuning** with quantized KV cache: Possible via QLoRA + TurboQuant KV compression, but gradient memory for activations will be the bottleneck, not the KV cache. Not recommended on MacBook Air M4 for >3B parameter models.

**Training from scratch**: Out of scope for local M4 hardware.

### 5.7 The Neural Engine

The M4's Neural Engine (ANE) is optimized for small feed-forward networks in Core ML's compute graph. MLX does not target the ANE directly — it targets the GPU via Metal. The GPU is the right compute target for the large GEMM operations involved in KV cache quantization.

---

## 6. Local Execution Constraints and Trade-offs

### 6.1 What You Give Up vs. the Papers

| Trade-off | Paper setting | MacBook Air M4 setting |
|---|---|---|
| Model size | Llama-3.1-8B / Ministral-7B | Same (feasible in int4) |
| Context length | Up to 104K tokens | Practical limit ~32–50K at 2.5-bit KV |
| Batch size | Multi-request server batching | Single-request only |
| JL matrix m | m = d (1:1 sketch) | Same — m=128 for d=128 |
| Codebook precision | Float32 | Float16 (negligible quality loss) |
| Quantization time | 0.0007s (paper, A100) | ~0.01s (M4 GPU, estimated) |

### 6.2 No CUDA Required

Every operation — QR decomposition for rotation matrices, Lloyd-Max iteration, quantization/dequantization, inner product estimation — maps cleanly to MLX or numpy/scipy on CPU. The only paper-claimed advantage of CUDA (custom CUDA kernels for batched sign-quantized GEMM) can be approximated in MLX with int8 matrix multiplications.

### 6.3 Codebook Precomputation

The Lloyd-Max iterations to compute optimal codebooks require scipy on CPU and take less than 1 second for all bit-widths. Run this once offline and save the results as `.npy` files loaded at startup.

### 6.4 Rotation Matrix Storage

For a transformer with L layers and H heads each with head_dim d_h: you need one shared rotation matrix of size d_h × d_h. For Llama-3.1-8B (d_h=128): 128×128×2 bytes = 32 KB. Trivially small. If using per-layer or per-head rotations (possible but not recommended), this scales to L×H×32KB — still manageable.

### 6.5 Approximations Made

One notable simplification for Apple Silicon: instead of true 2-bit or 2.5-bit integer packing (which requires custom Metal kernels), store codebook indices as uint8. This wastes 6 bits per index at b=2, meaning actual memory savings are ~2× instead of the theoretical ~6.4×. To get the full compression benefit, implement a packing/unpacking step:

```python
def pack_2bit(indices: np.ndarray) -> np.ndarray:
    """Pack 4 2-bit indices into one uint8."""
    assert indices.max() < 4
    packed = np.zeros(len(indices) // 4, dtype=np.uint8)
    for i in range(4):
        packed |= (indices[i::4] & 0x3) << (2 * i)
    return packed

def unpack_2bit(packed: np.ndarray, n: int) -> np.ndarray:
    indices = np.zeros(n, dtype=np.uint8)
    for i in range(4):
        indices[i::4] = (packed >> (2 * i)) & 0x3
    return indices
```

---

## 7. Step-by-Step Build Guide

### Step 1: Environment Setup

```bash
# Create a dedicated virtual environment
python3 -m venv ~/mlx-kv-quant
source ~/mlx-kv-quant/bin/activate

# Core dependencies
pip install mlx mlx-lm
pip install numpy scipy matplotlib tqdm
pip install transformers huggingface_hub

# For model loading and tokenization
pip install sentencepiece protobuf

# Optional: for benchmarking
pip install psutil py-cpuinfo
```

Verify MLX and Metal access:
```python
import mlx.core as mx
print(mx.default_device())  # Should print: Device(gpu, 0)
a = mx.random.normal((128, 128))
b = mx.random.normal((128, 128))
print(mx.matmul(a, b).shape)  # (128, 128)
```

### Step 2: Precompute Shared State (Offline)

Run this once and save artifacts:

```bash
python -m kv_quant.precompute \
    --head_dim 128 \
    --jl_dim 128 \
    --bit_widths 1 2 3 4 \
    --seed 42 \
    --output_dir ./artifacts/
```

This script: (1) generates and QR-orthogonalizes the rotation matrix Π and JL matrix S; (2) runs Lloyd-Max to compute codebooks for each bit-width; (3) saves everything as float16 `.npy` files.

### Step 3: Implement Core Quantization Modules

Implement in order: `rotation.py` → `codebook.py` → `qjl.py` → `turboquant.py` → `outlier.py`

Validate each module in isolation with unit tests checking theoretical bounds:
```python
# Example: verify QJL unbiasedness
def test_qjl_unbiased(d=128, m=128, n_trials=10000):
    S = make_jl_matrix(d, m)
    q = mx.random.normal((d,)) 
    k = mx.random.normal((d,))
    true_ip = float((q * k).sum())
    estimates = []
    for _ in range(n_trials):
        signs, norms = qjl_quantize(k[None], S)
        est = qjl_inner_product(q[None], signs, norms, S)
        estimates.append(float(est))
    print(f"True: {true_ip:.4f}, Mean estimate: {np.mean(estimates):.4f}")
    assert abs(np.mean(estimates) - true_ip) < 0.01, "QJL is biased!"
```

### Step 4: Implement the KV Cache Wrapper

Build `kv_cache.py` with `TurboQuantKVCache` as described in Section 4.4. The key contract: it must be a drop-in replacement for a standard list-based KV cache in any MLX-compatible attention implementation.

Test with synthetic attention:
```python
cache = TurboQuantKVCache(QuantConfig(d=128, b=3, m=128))
for t in range(100):
    k = mx.random.normal((128,)).astype(mx.float16)
    v = mx.random.normal((128,)).astype(mx.float16)
    cache.append(k, v)
q = mx.random.normal((128,)).astype(mx.float16)
out = cache.attend(q)
print(f"Attention output shape: {out.shape}")  # (128,)
```

### Step 5: Integrate with a Real Model

Use `mlx-lm` to load Llama-3.1-8B (or a smaller model like Llama-3.2-3B for faster iteration) and patch the attention module to use the quantized KV cache:

```python
from mlx_lm import load
from kv_quant import TurboQuantKVCache, QuantConfig, patch_model_kv_cache

model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
config = QuantConfig(d=128, b=3, m=128, n_outlier_channels=4)
patch_model_kv_cache(model, config)
```

The `patch_model_kv_cache` function replaces each attention layer's KV cache with a `TurboQuantKVCache` instance.

### Step 6: Validation and Evaluation

Run the Needle-In-A-Haystack test using a simple Python reimplementation (no need for the full evaluation harness):

```python
# Generate a long document (haystack)
# Insert a unique sentence (needle) at a random position
# Run the model with TurboQuant KV cache
# Check if the model can retrieve the needle
```

Also run LongBench tasks using the Hugging Face `datasets` library and `mlx-lm` generation API.

### Step 7: Measure and Optimize

Profile memory and latency:
```python
import psutil, time

def benchmark_generation(model, prompt, n_tokens=50):
    proc = psutil.Process()
    mem_before = proc.memory_info().rss / 1e9
    t0 = time.time()
    output = generate(model, prompt, max_tokens=n_tokens)
    t1 = time.time()
    mem_after = proc.memory_info().rss / 1e9
    print(f"Throughput: {n_tokens/(t1-t0):.1f} tokens/s")
    print(f"Memory delta: {mem_after - mem_before:.2f} GB")
```

Optimization passes to try in order:
1. Replace uint8 index storage with 2/3-bit packed storage using numpy bit manipulation.
2. Fuse the `x @ Pi.T → quantize → dequantize → Pi @` pipeline into a single MLX graph.
3. Profile with `mx.metal.start_capture()` to identify bottleneck Metal kernels.
4. Try reducing m (JL dimension) from d to d/2 — check if quality degrades.

---

## 8. Code Snippets

### 8.1 Complete TurboQuant MSE Module

```python
# turboquant.py
import mlx.core as mx
import numpy as np
from typing import NamedTuple

class TurboQuantMSEState(NamedTuple):
    Pi: mx.array        # (d, d) rotation matrix, fp16
    codebook: mx.array  # (2^b,) centroids, fp16
    b: int

def make_turboquant_mse(d: int, b: int, seed: int = 42) -> TurboQuantMSEState:
    # Build rotation matrix
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, d)).astype(np.float32)
    Q, _ = np.linalg.qr(G)
    Pi = mx.array(Q.astype(np.float16))
    # Load precomputed codebook (scaled for unit-variance Gaussian)
    codebook = mx.array(_lloyd_max_centroids(b, sigma=1.0/np.sqrt(d)))
    return TurboQuantMSEState(Pi=Pi, codebook=codebook, b=b)

def turboquant_mse_encode(x: mx.array, state: TurboQuantMSEState) -> mx.array:
    """x: (batch, d) fp16 -> indices: (batch, d) uint8"""
    y = x @ state.Pi.T  # rotate
    # broadcast distance to all centroids
    diff = mx.abs(y[:, :, None] - state.codebook[None, None, :])
    return mx.argmin(diff, axis=-1).astype(mx.uint8)

def turboquant_mse_decode(idx: mx.array, state: TurboQuantMSEState) -> mx.array:
    """idx: (batch, d) uint8 -> x_hat: (batch, d) fp16"""
    y_hat = state.codebook[idx]  # gather
    return y_hat @ state.Pi      # rotate back

def _lloyd_max_centroids(b: int, sigma: float) -> np.ndarray:
    """Lloyd-Max for N(0, sigma^2)."""
    from scipy.stats import norm as sp_norm
    k = 2 ** b
    c = np.linspace(-3*sigma, 3*sigma, k)
    for _ in range(500):
        bounds = np.concatenate([[-np.inf],
                                  (c[:-1] + c[1:]) / 2,
                                  [np.inf]])
        new_c = []
        for i in range(k):
            lo, hi = bounds[i] / sigma, bounds[i+1] / sigma
            p = sp_norm.cdf(hi) - sp_norm.cdf(lo)
            if p < 1e-14:
                new_c.append(c[i])
                continue
            mu = sigma * (sp_norm.pdf(lo) - sp_norm.pdf(hi)) / p
            new_c.append(mu)
        c = np.array(new_c, dtype=np.float16)
    return c
```

### 8.2 QJL Inner Product Estimation

```python
# qjl.py
import mlx.core as mx
import numpy as np
import math

def make_jl_matrix(d: int, m: int, seed: int = 42) -> mx.array:
    """Orthogonalized JL matrix S: (m, d) fp16."""
    rng = np.random.default_rng(seed + 1)
    G = rng.standard_normal((m, d)).astype(np.float32)
    # Orthogonalize rows (QR of G^T, take Q^T)
    Q, _ = np.linalg.qr(G.T)
    S = Q.T[:m]  # (m, d)
    return mx.array(S.astype(np.float16))

def qjl_encode_key(k: mx.array, S: mx.array) -> tuple[mx.array, mx.array]:
    """k: (batch, d) -> signs: (batch, m) int8, norms: (batch,) fp16"""
    Sk = k @ S.T       # (batch, m)
    signs = mx.sign(Sk).astype(mx.int8)
    norms = mx.sqrt((k * k).sum(axis=-1))  # (batch,)
    return signs, norms

def qjl_decode_ip(q: mx.array, k_signs: mx.array,
                   k_norms: mx.array, S: mx.array) -> mx.array:
    """
    Estimate inner product ⟨q, k⟩ for each cached key.
    q: (d,) fp16
    k_signs: (n, m) int8
    k_norms: (n,) fp16
    Returns: (n,) estimated inner products
    """
    m = S.shape[0]
    Sq = (q @ S.T)  # (m,)
    # (n, m) int8 @ (m,) fp16 -> need to cast
    ip = (k_signs.astype(mx.float16) @ Sq)  # (n,)
    scale = math.sqrt(math.pi / 2) / m
    return scale * k_norms * ip
```

### 8.3 Outlier Channel Detection

```python
# outlier.py
import mlx.core as mx
import numpy as np

def detect_outlier_channels(k_cache_sample: np.ndarray,
                              n_outliers: int = 4) -> np.ndarray:
    """
    k_cache_sample: (n_tokens, d) numpy array from the first few hundred tokens.
    Returns: indices of the n_outliers channels with highest mean absolute magnitude.
    """
    mean_mag = np.abs(k_cache_sample).mean(axis=0)  # (d,)
    return np.argsort(mean_mag)[-n_outliers:]  # top n_outliers

def split_channels(x: mx.array, outlier_idx: np.ndarray) -> tuple:
    inlier_idx = np.setdiff1d(np.arange(x.shape[-1]), outlier_idx)
    return x[:, outlier_idx], x[:, inlier_idx], inlier_idx
```

### 8.4 PolarQuant Angle Codebook Construction

```python
# polarquant.py — codebook building for each level
import numpy as np
from scipy.stats import norm as sp_norm
from scipy.special import gamma

def polar_angle_pdf(psi: np.ndarray, level: int) -> np.ndarray:
    """
    PDF of angle at polar level ell >= 2: f(psi) ∝ sin^(2^(ell-1)-1)(2*psi)
    on [0, pi/2].
    Level 1 is uniform on [0, 2*pi).
    """
    if level == 1:
        return np.ones_like(psi) / (2 * np.pi)
    k = 2 ** (level - 1)
    # normalizing constant
    from scipy.integrate import quad
    Z, _ = quad(lambda t: np.sin(2*t)**(k-1), 0, np.pi/2)
    return np.sin(2*psi)**(k-1) / Z

def build_polar_codebook(level: int, b: int, n_samples: int = 100_000) -> np.ndarray:
    """
    Sample from angle distribution, run 1D k-means to find optimal codebook.
    """
    from sklearn.cluster import KMeans
    # Sample angles from the distribution
    if level == 1:
        samples = np.random.uniform(0, 2*np.pi, n_samples)
    else:
        # Rejection sampling from f(psi) ∝ sin^(2^(l-1)-1)(2*psi) on [0, pi/2]
        k = 2 ** (level - 1)
        max_pdf = 1.0  # sin^k(2*pi/4) = 1 at psi=pi/4
        samples = []
        while len(samples) < n_samples:
            psi = np.random.uniform(0, np.pi/2, 2*n_samples)
            u   = np.random.uniform(0, 1, 2*n_samples)
            accept = u < np.sin(2*psi)**(k-1)
            samples.extend(psi[accept].tolist())
        samples = np.array(samples[:n_samples])

    km = KMeans(n_clusters=2**b, n_init=10)
    km.fit(samples.reshape(-1, 1))
    return np.sort(km.cluster_centers_.flatten()).astype(np.float32)
```

### 8.5 Memory-Efficient Context Generation Loop

```python
# generation.py — token generation with TurboQuant KV cache
import mlx.core as mx
from mlx_lm.utils import generate_step

def generate_with_turboquant(model, tokenizer, prompt: str,
                               max_tokens: int = 200,
                               quant_config=None) -> str:
    tokens = tokenizer.encode(prompt, return_tensors="mlx")
    kv_caches = [TurboQuantKVCache(quant_config)
                 for _ in range(model.config.num_hidden_layers)]
    generated = []
    for step in range(max_tokens):
        # MLX-LM compatible generation step
        logits, kv_caches = model(tokens, cache=kv_caches)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        tokens = next_token[:, None]
        generated.append(int(next_token))
        if int(next_token) == tokenizer.eos_token_id:
            break
        # Force evaluation every 10 tokens to avoid graph growing unbounded
        if step % 10 == 0:
            mx.eval(next_token)
    return tokenizer.decode(generated)
```

---

## 9. Evaluation and Metrics

### 9.1 Paper Metrics and How to Compute Them

**MSE Distortion (D_mse):**
```python
def compute_mse_distortion(x_orig: mx.array, x_recon: mx.array) -> float:
    """Expected over many vectors. x: (n, d) normalized to unit norm."""
    return float(mx.mean(mx.sum((x_orig - x_recon)**2, axis=-1)))
# Expected per paper: b=2 → ~0.117, b=3 → ~0.030, b=4 → ~0.009
```

**Inner Product Distortion (D_prod):**
```python
def compute_ip_distortion(x: mx.array, x_recon: mx.array, y: mx.array) -> float:
    """y: (d,) query vector. x: (n, d) database."""
    true_ip = (x @ y).reshape(-1)
    est_ip  = (x_recon @ y).reshape(-1)
    return float(mx.mean((true_ip - est_ip)**2))
# Should scale as ‖y‖²/d · 4^(-b)
```

**Bias check for QJL/TurboQuantProd:**
```python
def check_unbiasedness(encoder, decoder, x, y, n_trials=1000):
    estimates = []
    true_ip = float((x * y).sum())
    for _ in range(n_trials):
        encoded = encoder(x[None])
        decoded = decoder(*encoded)
        estimates.append(float((decoded.squeeze() * y).sum()))
    bias = abs(np.mean(estimates) - true_ip)
    print(f"Bias: {bias:.6f} (should be ~0 for TurboQuantProd)")
```

**Recall@1@k for nearest neighbor search:**
```python
def recall_at_k(q: np.ndarray, db: np.ndarray, db_quant: np.ndarray, k: int) -> float:
    true_nn = np.argmax(q @ db.T, axis=-1)  # (n_queries,)
    approx_scores = q @ db_quant.T
    top_k = np.argsort(-approx_scores, axis=-1)[:, :k]
    hits = np.any(top_k == true_nn[:, None], axis=-1)
    return float(hits.mean())
```

### 9.2 Local Benchmarking on MacBook Air M4

```python
import time, psutil, mlx.core as mx

def benchmark_kv_operations(d=128, m=128, b=3, seq_len=1000):
    state = make_turboquant_mse(d, b)
    S = make_jl_matrix(d, m)
    # Simulate n_tokens key vectors
    keys = mx.random.normal((seq_len, d)).astype(mx.float16)
    mx.eval(keys)
    # Benchmark encode
    t0 = time.perf_counter()
    idx = turboquant_mse_encode(keys, state)
    r_signs, r_norms = qjl_encode_key(keys - turboquant_mse_decode(idx, state), S)
    mx.eval(idx, r_signs, r_norms)
    t1 = time.perf_counter()
    encode_ms = (t1 - t0) * 1000
    # Benchmark decode + inner product
    q = mx.random.normal((d,)).astype(mx.float16)
    t0 = time.perf_counter()
    for _ in range(100):
        ip = qjl_decode_ip(q, r_signs, r_norms, S)
        mx.eval(ip)
    t1 = time.perf_counter()
    decode_ms = (t1 - t0) * 10  # 100 runs / 10 = avg per run
    print(f"Encode {seq_len} tokens: {encode_ms:.2f} ms")
    print(f"Decode + IP estimate ({seq_len} keys): {decode_ms:.2f} ms")
    # Memory estimate
    bits_per_token = b * d + m + 16  # idx + signs + norm
    print(f"Bits per token: {bits_per_token}")
    print(f"Memory for {seq_len} tokens: {bits_per_token * seq_len / 8 / 1024:.1f} KB")
```

### 9.3 Theoretical Bound Verification

Plot empirical vs. theoretical distortion across bit-widths (reproducing Figure 3 from TurboQuant):

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_distortion_bounds(d=1536, n_samples=10000):
    bit_widths = [1, 2, 3, 4, 5]
    empirical_mse, empirical_ip = [], []
    theoretical_upper = [np.sqrt(3*np.pi)/2 * 4**(-b) for b in bit_widths]
    theoretical_lower = [4**(-b) for b in bit_widths]

    x = mx.random.normal((n_samples, d)).astype(mx.float16)
    x = x / mx.norm(x, axis=-1, keepdims=True)  # normalize to unit sphere
    y = mx.random.normal((d,)).astype(mx.float16)
    y = y / mx.norm(y)

    for b in bit_widths:
        state = make_turboquant_mse(d, b)
        x_hat = turboquant_mse_decode(turboquant_mse_encode(x, state), state)
        mse = float(mx.mean(mx.sum((x - x_hat)**2, axis=-1)))
        ip_err = float(mx.mean((x @ y - x_hat @ y)**2))
        empirical_mse.append(mse)
        empirical_ip.append(ip_err * d)  # normalized by d

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].semilogy(bit_widths, empirical_mse, 'b-o', label='TurboQuant (empirical)')
    ax[0].semilogy(bit_widths, theoretical_upper, 'r--', label='Upper bound')
    ax[0].semilogy(bit_widths, theoretical_lower, 'g--', label='Lower bound')
    ax[0].set_title('MSE Distortion vs. Bit-width')
    ax[0].legend(); ax[0].set_xlabel('Bit-width b')
    plt.tight_layout()
    plt.savefig('distortion_bounds_m4.pdf')
```

---

## 10. Risks, Limitations, and Challenges

### 10.1 Algorithm Ambiguities

**Rotation matrix sharing:** TurboQuant and PolarQuant both say the rotation is "shared" but don't specify if it's shared across layers, heads, or both. The original QJL paper shares across all layers and heads. For a faithful MLX implementation, use one rotation matrix per model (simplest and smallest memory footprint). Per-layer rotations may marginally improve quality but require L×32KB additional storage.

**Outlier channel count:** The paper uses 32 outlier channels out of d=128 (25%). The optimal split depends on the model. In practice, detect channels empirically on a calibration prompt (100–200 tokens).

**Codebook for small d:** In low dimensions (d < 64), the Beta distribution deviates significantly from Gaussian. Use numerical integration (scipy.integrate.quad) to compute the true Beta-distribution Lloyd-Max codebook rather than the Gaussian approximation.

### 10.2 MLX-Specific Challenges

**No native 2-bit integer type:** MLX's smallest integer type is int8/uint8. To achieve true 2-bit storage, implement bit-packing in numpy before storing and bit-unpacking after loading. This adds ~0.1ms per operation — negligible for token generation but worth doing for long-context applications where memory is the binding constraint.

**int8 × fp16 GEMM:** MLX supports mixed-precision matmul but the exact behavior depends on the Metal backend version. Test that `(signs.astype(mx.float16) @ Sq)` gives correct results vs. a float32 reference.

**Graph size for long contexts:** MLX builds a computation graph lazily. For contexts > 10K tokens, explicitly call `mx.eval()` periodically (every 100–200 tokens) to prevent the graph from growing unbounded in memory.

**arctan2 precision:** MLX's `mx.arctan2` operates in float32 by default. For PolarQuant, the angle computation should be done in float32 and then quantized — avoid doing the polar transform in float16 directly.

### 10.3 Performance Bottlenecks

The main bottleneck is the inner product estimation at decode time: for n cached tokens and m = d = 128, you need n matrix-vector products. At 50K tokens, this is a (50000, 128) × (128,) operation — about 12.8M multiply-accumulates per attention head per token. With 32 layers × 32 heads, this is ~13B operations per generated token. At M4 GPU's ~14 TFLOPS fp16, this takes ~1ms per token — comparable to the compute time for the FFN layers. Acceptable.

### 10.4 Quality vs. Theory Gap

The paper proves bounds for uniformly distributed vectors on the unit sphere. Real KV embeddings from transformer models are not uniformly distributed, especially in shallower layers. The random rotation brings them closer to the theoretical setting, but quality guarantees are somewhat looser in practice. Empirically the papers show this is fine, but be prepared for model-specific tuning of bit-widths by layer.

---

## 11. Extensions and Improvements

### 11.1 Layer-Adaptive Bit-Width

Not all layers need equal compression. Shallow layers have simple, near-isotropic distributions (low outlier frequency per QJL Figure 2). Deeper layers have strong outliers. A simple heuristic: use b=2 for layers 0–L/3, b=3 for L/3–2L/3, b=4 for 2L/3–L. This keeps average bit-width at ~3 while preserving quality where it matters most.

```python
def get_bitwidth_for_layer(layer_idx: int, n_layers: int) -> int:
    frac = layer_idx / n_layers
    if frac < 0.33:
        return 2
    elif frac < 0.67:
        return 3
    else:
        return 4
```

### 11.2 Head-Adaptive Compression

Different attention heads have different distributions. Compute per-head variance of key magnitudes on a calibration set; allocate more bits to high-variance heads. This can reduce total bits by 10–15% with no quality loss.

### 11.3 Streaming-Friendly Sliding Window

For very long contexts (> 100K tokens), memory becomes the limit even with compression. Combine TurboQuant with a sliding window: keep only the last W tokens in the quantized cache (W = 8192 or 16384), and evict old tokens. This is compatible with TurboQuant because it's online — no data-dependent preprocessing needed.

### 11.4 Lighter Alternatives for Very Constrained Hardware

If running on an 8 GB M4 MacBook Air with a large model:

- Use QJL alone (no TurboQuantMSE stage) for b=1 bit keys: ~16× compression, higher distortion but nonzero quality on long-context tasks.
- Reduce m (JL dimension) to d/2 = 64: halves memory for signs with ~1.4× quality degradation.
- Use PolarQuant with L=2 levels instead of 4: faster polar transform, slightly lower quality.

### 11.5 Productionizing

To build a production local inference server with this compression:

1. Implement as an `mlx-lm` plugin or monkey-patch the model's attention class.
2. Add a REST API (FastAPI) that exposes `generate()` with the quantized KV cache.
3. Use `mlx.stream` for async generation.
4. Add context-length monitoring: automatically evict or compress older tokens when the cache approaches a memory limit.
5. Consider persisting the KV cache to disk between sessions for long-running conversations (compress with zstd before writing).

### 11.6 Combining TurboQuant + PolarQuant

An interesting hybrid: use TurboQuant for keys (inner product matters for attention score computation) and PolarQuant for values (norm-preserving reconstruction matters for the output). This is theoretically motivated — TurboQuantProd is optimal for inner product queries (keys), while PolarQuant is optimal in an MSE/reconstruction sense (values).

---

## Quick Reference: Hyperparameter Choices for MacBook Air M4

| Parameter | Value | Rationale |
|---|---|---|
| d (head dim) | 128 (Llama-3.1-8B) | Fixed by model |
| b (bit-width, inliers) | 2 | 4× compression, good quality |
| b (bit-width, outliers) | 3 | Protects high-magnitude channels |
| n_outlier_channels | 4 | Matches empirical observation in QJL paper |
| m (JL dimension) | 128 | Equal to d; higher m = better quality |
| n_polar_levels | 4 | Standard per PolarQuant |
| rotation seed | 42 | Fixed; shared across all layers/heads |
| Effective bits/channel | 2.5 | = (4×3 + 124×2) / 128 |
| fp16 storage for norms | yes | One fp16 per token per head |
| Value cache quantization | int8, per-token | Standard, proven effective |
| Recommended context limit | 32K tokens | ~1.5 GB KV cache at 2.5-bit |

---

*This document is intended as a complete engineering specification for implementing TurboQuant, PolarQuant, and QJL on Apple Silicon using MLX. All algorithms can be implemented without CUDA, run efficiently on the M4's Metal GPU, and integrate with off-the-shelf models via mlx-lm.*