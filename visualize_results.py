"""
TurboQuant benchmark visualizations.
Produces 5 figures saved as PNGs + one combined report figure.

Run: python visualize_results.py
"""
import math
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {
    "fp16":      "#4C72B0",
    "3bit":      "#DD8452",
    "4bit":      "#55A868",
    "4bit_out":  "#C44E52",
    "accent":    "#8172B2",
}
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Benchmark data (from actual runs) ─────────────────────────────────────────
CONFIGS   = ["fp16\nbaseline", "TurboQuant\n3-bit", "TurboQuant\n4-bit", "4-bit +\noutliers"]
COLORS    = [PALETTE["fp16"], PALETTE["3bit"], PALETTE["4bit"], PALETTE["4bit_out"]]
COMPRESS  = [1.00,  5.82,  4.27,  3.51]   # key compression ratio
THROUGHPUT= [47.2,  25.8,  24.9,   5.2]   # tok/s
TOKENS_OUT= [201,    89,   201,   201]     # tokens generated (89 = repetition crash)
KEY_KB    = [8120,  1396,  3360,  4088]   # compressed key bytes in KB
TIME_S    = [4.3,    3.4,   8.1,  38.5]   # wall-clock seconds

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Benchmark Summary (2×2 grid)
# ═══════════════════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle(
    "TurboQuant KV Cache Benchmark\nLlama-3.2-3B · Apple M4 · head_dim=128",
    fontsize=16, fontweight="bold", y=1.01,
)

x = np.arange(len(CONFIGS))
bar_w = 0.55


def _bar(ax, values, ylabel, title, color_override=None, hline=None, fmt=".1f"):
    cols = color_override if color_override else COLORS
    bars = ax.bar(x, values, width=bar_w, color=cols, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(CONFIGS, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:{fmt}}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.25)
    sns.despine(ax=ax, left=False, bottom=False)


# 1a — Compression ratio
_bar(axes[0, 0], COMPRESS, "Compression Ratio (×)", "Key Compression Ratio",
     hline=1.0, fmt=".2f")
axes[0, 0].annotate("← fp16 baseline", xy=(0, 1.04), fontsize=9, color="grey")

# 1b — Throughput
_bar(axes[0, 1], THROUGHPUT, "Tokens / second", "Generation Throughput (tok/s)",
     hline=THROUGHPUT[0])
axes[0, 1].annotate("fp16 reference", xy=(0.05, THROUGHPUT[0] + 1), fontsize=9, color="grey")

# 1c — Tokens output (shows 3-bit generation failure)
qual_colors = [PALETTE["fp16"], "#DD8452", PALETTE["4bit"], PALETTE["4bit_out"]]
_bar(axes[1, 0], TOKENS_OUT, "Tokens Generated", "Tokens Generated (max 200)",
     color_override=qual_colors, hline=200, fmt="d")
axes[1, 0].annotate(
    "3-bit stopped at 89\n(repetition loop)", xy=(1, 95), fontsize=9,
    color=PALETTE["3bit"], ha="center",
)

# 1d — Compressed key size
_bar(axes[1, 1], KEY_KB, "Key Cache Size (KB)", "Compressed Key Cache Size",
     fmt=".0f")

fig1.tight_layout()
fig1.savefig(f"{OUT_DIR}/fig1_benchmark_summary.png", dpi=150, bbox_inches="tight")
print("Saved fig1_benchmark_summary.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TurboQuant Pipeline Visualization (2D toy example)
# ═══════════════════════════════════════════════════════════════════════════════
np.random.seed(42)
N = 80
d = 2

# Simulate key vectors: random directions with varying magnitudes
angles_orig = np.random.uniform(0, 2 * np.pi, N)
mags = np.abs(np.random.randn(N)) + 0.5
X = np.column_stack([mags * np.cos(angles_orig), mags * np.sin(angles_orig)])

# Step 1: normalize
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_unit = X / norms   # unit circle

# Step 2: rotate by 45°
theta = np.pi / 4
R = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
X_rot = X_unit @ R.T

# Step 3: quantize — 2-bit (4 centroids per dim); simulate Lloyd-Max centroids
centroids_1d = np.array([-0.798, -0.266,  0.266,  0.798])  # typical 2-bit Lloyd-Max

def quantize_1d(v, cents):
    idx = np.argmin(np.abs(v[:, None] - cents[None, :]), axis=1)
    return cents[idx]

X_q = np.column_stack([
    quantize_1d(X_rot[:, 0], centroids_1d),
    quantize_1d(X_rot[:, 1], centroids_1d),
])

# Step 4: unrotate
X_recon_unit = X_q @ R   # R^T inverse = R for orthogonal
# Rescale by original norms
X_recon = X_recon_unit * norms

fig2, axes2 = plt.subplots(1, 4, figsize=(18, 5))
fig2.suptitle(
    "TurboQuant Key Vector Pipeline (2D Toy, d=2, 2-bit MSE)",
    fontsize=14, fontweight="bold",
)

step_data = [
    (X,          "Step 1: Original Key Vectors\n(varying magnitude, any direction)",
     PALETTE["fp16"], True),
    (X_unit,     "Step 2: Normalize to Unit Sphere\n(magnitude stored separately)",
     PALETTE["accent"], False),
    (X_rot,      "Step 3: Random Rotation\n(spreads information uniformly)",
     PALETTE["3bit"], False),
    (X_recon,    "Step 5: Reconstruct\n(unrotate → rescale by stored norm)",
     PALETTE["4bit"], True),
]

for ax, (data, title, color, show_mag) in zip(axes2, step_data):
    ax.scatter(data[:, 0], data[:, 1], c=color, alpha=0.55, s=30, edgecolors="none")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    if not show_mag:
        # draw unit circle
        theta_c = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta_c), np.sin(theta_c), "--", color="grey",
                linewidth=1, alpha=0.5, label="unit circle")
    sns.despine(ax=ax)

# Insert codebook visualization in the 4th panel area (between step 3 and 5)
# Override axes2[3] with quantized step
ax_q = axes2[3]
ax_q.scatter(X_rot[:, 0], X_rot[:, 1], c="lightgrey", alpha=0.4, s=20,
             edgecolors="none", label="rotated (pre-quant)")
ax_q.scatter(X_q[:, 0], X_q[:, 1], c=PALETTE["3bit"], alpha=0.7, s=30,
             edgecolors="none", label="quantized (post-quant)")

# Draw grid of codebook centroids
for cx in centroids_1d:
    for cy in centroids_1d:
        ax_q.scatter(cx, cy, marker="x", s=80, c="black", linewidths=2, zorder=5)
ax_q.set_title("Step 4: Lloyd-Max Quantize\n(× = codebook centroids, 4×4 grid = 2-bit×2-bit)",
               fontsize=10, fontweight="bold")
ax_q.set_aspect("equal")
ax_q.axhline(0, color="black", linewidth=0.5, alpha=0.3)
ax_q.axvline(0, color="black", linewidth=0.5, alpha=0.3)
ax_q.legend(fontsize=8)
sns.despine(ax=ax_q)

fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig2_vector_pipeline.png", dpi=150, bbox_inches="tight")
print("Saved fig2_vector_pipeline.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Quality vs Bits across head dimensions
# ═══════════════════════════════════════════════════════════════════════════════
import mlx.core as mx
from mlx_kv_quant.quantizers.turboquant_prod import TurboQuantProd

head_dims = [64, 128, 256]
bit_range  = [2, 3, 4, 5, 6]
dim_colors = [PALETTE["3bit"], PALETTE["4bit"], PALETTE["fp16"]]
dim_markers= ["o", "s", "^"]

snr_results = {}
cos_results = {}

print("Computing SNR/cosine curves (takes ~30s)...")
for d in head_dims:
    snrs, coss = [], []
    np.random.seed(42)
    x_raw = np.random.randn(64, d)
    x_unit = (x_raw / np.linalg.norm(x_raw, axis=1, keepdims=True)).astype(np.float16)
    x_mx = mx.array(x_unit)
    for b in bit_range:
        try:
            q = TurboQuantProd(d=d, b=b, m=min(d, 64), seed=0)
            ev = q.encode(x_mx)
            x_hat = q.decode(ev)
            mse = float(mx.mean((x_mx - x_hat) ** 2))
            var = float(mx.mean(x_mx ** 2))
            snr = 10 * np.log10(max(var / mse, 1e-10))
            cos = float(mx.mean(
                mx.sum(x_mx * x_hat, axis=1) /
                (mx.linalg.norm(x_mx, axis=1) * mx.linalg.norm(x_hat, axis=1))
            ))
        except Exception:
            snr, cos = float("nan"), float("nan")
        snrs.append(snr)
        coss.append(cos)
    snr_results[d] = snrs
    cos_results[d] = coss

fig3, (ax_snr, ax_cos) = plt.subplots(1, 2, figsize=(14, 6))
fig3.suptitle(
    "TurboQuant Quality vs Bit-width (unit-norm key vectors)",
    fontsize=14, fontweight="bold",
)

for d, col, mkr in zip(head_dims, dim_colors, dim_markers):
    ax_snr.plot(bit_range, snr_results[d], color=col, marker=mkr,
                linewidth=2, markersize=7, label=f"head_dim={d}")
    ax_cos.plot(bit_range, cos_results[d], color=col, marker=mkr,
                linewidth=2, markersize=7, label=f"head_dim={d}")

# Quality thresholds
ax_snr.axhline(0,  color="red",    linestyle="--", linewidth=1.5, alpha=0.7,
               label="0 dB (noise = signal)")
ax_snr.axhline(6,  color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
               label="6 dB (usable)")
ax_snr.axhline(10, color="green",  linestyle="--", linewidth=1.5, alpha=0.7,
               label="10 dB (near-lossless)")

ax_cos.axhline(0.90, color="green",  linestyle="--", linewidth=1.5, alpha=0.7,
               label="0.90 cosine (near-lossless)")
ax_cos.axhline(0.80, color="orange", linestyle="--", linewidth=1.5, alpha=0.7,
               label="0.80 cosine (degraded)")

ax_snr.set_xlabel("Bit-width (b)", fontsize=12)
ax_snr.set_ylabel("SNR (dB)", fontsize=12)
ax_snr.set_title("Signal-to-Noise Ratio", fontsize=12, fontweight="bold")
ax_snr.legend(fontsize=9)
ax_snr.set_xticks(bit_range)

ax_cos.set_xlabel("Bit-width (b)", fontsize=12)
ax_cos.set_ylabel("Mean Cosine Similarity", fontsize=12)
ax_cos.set_title("Cosine Similarity (Original vs Reconstructed)", fontsize=12, fontweight="bold")
ax_cos.legend(fontsize=9)
ax_cos.set_xticks(bit_range)
ax_cos.set_ylim(0, 1.05)

# Annotate our benchmark configs
ax_cos.annotate(
    "Our 3-bit\n(repetition)", xy=(3, cos_results[128][1]),
    xytext=(3.3, 0.72), fontsize=9, color=PALETTE["3bit"],
    arrowprops=dict(arrowstyle="->", color=PALETTE["3bit"]),
)
ax_cos.annotate(
    "Our 4-bit\n(near-lossless)", xy=(4, cos_results[128][2]),
    xytext=(4.3, 0.84), fontsize=9, color=PALETTE["4bit"],
    arrowprops=dict(arrowstyle="->", color=PALETTE["4bit"]),
)

sns.despine(ax=ax_snr)
sns.despine(ax=ax_cos)
fig3.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig3_quality_vs_bits.png", dpi=150, bbox_inches="tight")
print("Saved fig3_quality_vs_bits.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — KV Cache Memory Growth at Scale (Llama-3.2-3B numbers)
# ═══════════════════════════════════════════════════════════════════════════════
# Llama-3.2-3B: 28 layers, 8 KV heads, head_dim=128
n_layers, n_kv, hd = 28, 8, 128
token_counts = np.array([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])

# Key-only sizes
fp16_keys   = token_counts * n_layers * n_kv * hd * 2          # bytes
fp16_full   = token_counts * n_layers * n_kv * hd * 2 * 2      # K+V fp16

def tq_key_bytes(tokens, bits):
    b_mse = max(bits - 1, 1)
    m = min(hd, 64)
    per_token = (
        math.ceil(hd * b_mse / 8)  # MSE indices
        + math.ceil(m / 8)          # QJL signs
        + 2                          # residual norm
        + 2                          # per-vec norm
    ) * n_kv * n_layers
    return tokens * per_token

tq3_keys = np.array([tq_key_bytes(t, 3) for t in token_counts])
tq4_keys = np.array([tq_key_bytes(t, 4) for t in token_counts])
val_fp16  = token_counts * n_layers * n_kv * hd * 2            # values stay fp16

fig4, (ax_abs, ax_ratio) = plt.subplots(1, 2, figsize=(14, 6))
fig4.suptitle(
    "KV Cache Memory at Scale — Llama-3.2-3B (28 layers, head_dim=128)",
    fontsize=14, fontweight="bold",
)

to_mb = lambda b: b / 1024**2

# Absolute memory
ax_abs.plot(token_counts, to_mb(fp16_full),   color=PALETTE["fp16"],  marker="o",
            linewidth=2.5, markersize=6, label="fp16 full (K+V)")
ax_abs.plot(token_counts, to_mb(fp16_keys),   color=PALETTE["fp16"],  marker="o",
            linewidth=2.5, markersize=6, linestyle="--", label="fp16 keys only")
ax_abs.plot(token_counts, to_mb(tq3_keys + val_fp16), color=PALETTE["3bit"], marker="s",
            linewidth=2.5, markersize=6, label="TQ 3-bit keys + fp16 values")
ax_abs.plot(token_counts, to_mb(tq4_keys + val_fp16), color=PALETTE["4bit"], marker="^",
            linewidth=2.5, markersize=6, label="TQ 4-bit keys + fp16 values")

ax_abs.set_xscale("log", base=2)
ax_abs.set_xticks(token_counts)
ax_abs.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_counts])
ax_abs.set_xlabel("Context Length (tokens)", fontsize=12)
ax_abs.set_ylabel("KV Cache Memory (MB)", fontsize=12)
ax_abs.set_title("Absolute KV Cache Memory", fontsize=12, fontweight="bold")
ax_abs.legend(fontsize=9)
ax_abs.axhline(16384, color="red", linestyle=":", linewidth=1.5, alpha=0.6,
               label="16 GB limit")

# Savings ratio
ratio_3b = fp16_full / (tq3_keys + val_fp16)
ratio_4b = fp16_full / (tq4_keys + val_fp16)

ax_ratio.plot(token_counts, ratio_3b, color=PALETTE["3bit"], marker="s",
              linewidth=2.5, markersize=6, label="TQ 3-bit")
ax_ratio.plot(token_counts, ratio_4b, color=PALETTE["4bit"], marker="^",
              linewidth=2.5, markersize=6, label="TQ 4-bit")
ax_ratio.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
ax_ratio.set_xscale("log", base=2)
ax_ratio.set_xticks(token_counts)
ax_ratio.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_counts])
ax_ratio.set_xlabel("Context Length (tokens)", fontsize=12)
ax_ratio.set_ylabel("Compression Ratio vs fp16 (higher = better)", fontsize=12)
ax_ratio.set_title("Compression Ratio vs fp16 Full KV Cache", fontsize=12, fontweight="bold")
ax_ratio.legend(fontsize=9)
ax_ratio.fill_between(token_counts, ratio_4b, 1.0, alpha=0.12, color=PALETTE["4bit"],
                      label="memory saved 4-bit")
ax_ratio.fill_between(token_counts, ratio_3b, ratio_4b, alpha=0.10, color=PALETTE["3bit"],
                      label="extra saved 3-bit")

sns.despine(ax=ax_abs)
sns.despine(ax=ax_ratio)
fig4.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig4_memory_at_scale.png", dpi=150, bbox_inches="tight")
print("Saved fig4_memory_at_scale.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Attention Score Distortion (fp16 vs 3-bit vs 4-bit)
# ═══════════════════════════════════════════════════════════════════════════════
from mlx_kv_quant.quantizers.turboquant_prod import TurboQuantProd

np.random.seed(7)
N_keys = 32
d = 128

# Generate a realistic query + key set (unit-norm)
q_np = np.random.randn(d).astype(np.float32)
q_np /= np.linalg.norm(q_np)
keys_np = np.random.randn(N_keys, d).astype(np.float32)
keys_unit = keys_np / np.linalg.norm(keys_np, axis=1, keepdims=True)

q_mx = mx.array(q_np.astype(np.float16)).reshape(1, -1)
k_mx = mx.array(keys_unit.astype(np.float16))

# True attention scores (fp16)
scores_true = np.array(k_mx @ q_mx.T).flatten()
softmax_true = np.exp(scores_true) / np.exp(scores_true).sum()

def get_attn(bits):
    qt = TurboQuantProd(d=d, b=bits, m=min(d, 64), seed=0)
    ev = qt.encode(k_mx)
    k_hat = qt.decode(ev)
    scores = np.array(k_hat @ q_mx.T).flatten()
    sm = np.exp(scores) / np.exp(scores).sum()
    return sm

sm_3b = get_attn(3)
sm_4b = get_attn(4)

token_ids = np.arange(N_keys)
fig5, axes5 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig5.suptitle(
    "Attention Score Distortion: fp16 vs TurboQuant\n"
    "Query dot-product attention over 32 key vectors (d=128, unit-norm)",
    fontsize=14, fontweight="bold",
)

configs_attn = [
    (softmax_true, "fp16 Baseline (reference)", PALETTE["fp16"]),
    (sm_3b,        "TurboQuant 3-bit  (cosine ~0.85)", PALETTE["3bit"]),
    (sm_4b,        "TurboQuant 4-bit  (cosine ~0.95)", PALETTE["4bit"]),
]

for ax, (scores, label, col) in zip(axes5, configs_attn):
    ax.bar(token_ids, scores, color=col, alpha=0.75, edgecolor="white", linewidth=0.5)
    ax.plot(token_ids, softmax_true, color=PALETTE["fp16"], linewidth=1.5,
            alpha=0.5, linestyle="--", label="fp16 reference")
    # Highlight max attention token
    peak = np.argmax(softmax_true)
    ax.axvline(peak, color="red", linestyle=":", linewidth=1.5, alpha=0.6)

    mse_attn = np.mean((scores - softmax_true) ** 2)
    cos_attn = np.dot(scores, softmax_true) / (np.linalg.norm(scores) * np.linalg.norm(softmax_true))
    ax.set_ylabel("Attention weight", fontsize=10)
    ax.set_title(
        f"{label}   |   MSE vs fp16 = {mse_attn:.2e},   cosine = {cos_attn:.4f}",
        fontsize=11, fontweight="bold",
    )
    ax.set_ylim(0, max(softmax_true) * 1.4)
    sns.despine(ax=ax)

axes5[-1].set_xlabel("Key Token Index", fontsize=12)
fig5.tight_layout()
fig5.savefig(f"{OUT_DIR}/fig5_attention_distortion.png", dpi=150, bbox_inches="tight")
print("Saved fig5_attention_distortion.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Combined report figure (all 5 panels, blog-ready)
# ═══════════════════════════════════════════════════════════════════════════════
fig6 = plt.figure(figsize=(20, 24))
fig6.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(3, 2, figure=fig6, hspace=0.42, wspace=0.35)

# Panel A — Compression ratio
ax_a = fig6.add_subplot(gs[0, 0])
bars = ax_a.bar(CONFIGS, COMPRESS, color=COLORS, edgecolor="white", linewidth=1.2)
ax_a.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
for b, v in zip(bars, COMPRESS):
    ax_a.text(b.get_x() + b.get_width()/2, v + 0.05, f"{v:.2f}×",
              ha="center", fontsize=11, fontweight="bold")
ax_a.set_title("A  Key Compression Ratio", fontsize=13, fontweight="bold", loc="left")
ax_a.set_ylabel("Ratio vs fp16")
sns.despine(ax=ax_a)

# Panel B — Throughput
ax_b = fig6.add_subplot(gs[0, 1])
bars = ax_b.bar(CONFIGS, THROUGHPUT, color=COLORS, edgecolor="white", linewidth=1.2)
ax_b.axhline(THROUGHPUT[0], color="grey", linestyle="--", linewidth=1, alpha=0.7)
for b, v in zip(bars, THROUGHPUT):
    ax_b.text(b.get_x() + b.get_width()/2, v + 0.8, f"{v:.1f}",
              ha="center", fontsize=11, fontweight="bold")
ax_b.set_title("B  Generation Throughput (tok/s)", fontsize=13, fontweight="bold", loc="left")
ax_b.set_ylabel("Tokens / second")
sns.despine(ax=ax_b)

# Panel C — Quality (cosine) vs bits for d=128
ax_c = fig6.add_subplot(gs[1, 0])
ax_c.plot(bit_range, cos_results[128], color=PALETTE["4bit"], marker="s",
          linewidth=2.5, markersize=8)
ax_c.fill_between(bit_range, 0.90, [min(c, 1.0) for c in cos_results[128]],
                  where=[c >= 0.90 for c in cos_results[128]],
                  alpha=0.2, color=PALETTE["4bit"], label="lossless zone (cosine ≥ 0.90)")
ax_c.axhline(0.90, color="green",  linestyle="--", linewidth=1.5, alpha=0.7)
ax_c.axhline(0.80, color="orange", linestyle="--", linewidth=1.5, alpha=0.7, label="degraded threshold")
ax_c.annotate("3-bit\n(broken)", xy=(3, cos_results[128][1]),
              xytext=(2.4, 0.72), fontsize=10, color=PALETTE["3bit"],
              arrowprops=dict(arrowstyle="->", color=PALETTE["3bit"]))
ax_c.annotate("4-bit\n(lossless)", xy=(4, cos_results[128][2]),
              xytext=(4.2, 0.88), fontsize=10, color=PALETTE["4bit"],
              arrowprops=dict(arrowstyle="->", color=PALETTE["4bit"]))
ax_c.set_xlabel("Bit-width")
ax_c.set_ylabel("Cosine Similarity")
ax_c.set_xticks(bit_range)
ax_c.set_ylim(0.5, 1.05)
ax_c.set_title("C  Quality vs Bits (head_dim=128)", fontsize=13, fontweight="bold", loc="left")
ax_c.legend(fontsize=9)
sns.despine(ax=ax_c)

# Panel D — Memory at scale
ax_d = fig6.add_subplot(gs[1, 1])
ax_d.plot(token_counts, to_mb(fp16_full), color=PALETTE["fp16"], linewidth=2.5,
          marker="o", markersize=5, label="fp16 (K+V)")
ax_d.plot(token_counts, to_mb(tq3_keys + val_fp16), color=PALETTE["3bit"],
          linewidth=2.5, marker="s", markersize=5, label="TQ 3-bit")
ax_d.plot(token_counts, to_mb(tq4_keys + val_fp16), color=PALETTE["4bit"],
          linewidth=2.5, marker="^", markersize=5, label="TQ 4-bit")
ax_d.set_xscale("log", base=2)
ax_d.set_xticks(token_counts)
ax_d.set_xticklabels([f"{t//1024}K" if t >= 1024 else str(t) for t in token_counts], fontsize=9)
ax_d.set_xlabel("Context Length (tokens)")
ax_d.set_ylabel("KV Cache Memory (MB)")
ax_d.set_title("D  KV Cache Memory at Scale", fontsize=13, fontweight="bold", loc="left")
ax_d.legend(fontsize=9)
sns.despine(ax=ax_d)

# Panel E — Attention distortion side-by-side (3-bit vs 4-bit vs fp16)
ax_e = fig6.add_subplot(gs[2, :])
w = 0.28
ax_e.bar(token_ids - w, softmax_true, width=w, color=PALETTE["fp16"],
         alpha=0.85, label="fp16 baseline")
ax_e.bar(token_ids,     sm_3b,        width=w, color=PALETTE["3bit"],
         alpha=0.85, label="TQ 3-bit")
ax_e.bar(token_ids + w, sm_4b,        width=w, color=PALETTE["4bit"],
         alpha=0.85, label="TQ 4-bit")
ax_e.set_xlabel("Key Token Index", fontsize=12)
ax_e.set_ylabel("Attention Weight (softmax)", fontsize=11)
ax_e.set_title(
    "E  Attention Score Distortion — fp16 vs TurboQuant  "
    f"(d=128, 32 keys)",
    fontsize=13, fontweight="bold", loc="left",
)
ax_e.legend(fontsize=10)
sns.despine(ax=ax_e)

fig6.suptitle(
    "TurboQuant KV Cache on Apple Silicon — Full Benchmark Report\n"
    "Llama-3.2-3B · M4 MacBook · mlx_kv_quant library",
    fontsize=17, fontweight="bold", y=1.005,
)
fig6.savefig(f"{OUT_DIR}/fig6_full_report.png", dpi=150, bbox_inches="tight")
print("Saved fig6_full_report.png")

print(f"\nAll figures saved to ./{OUT_DIR}/")
print("  fig1_benchmark_summary.png  — bar charts (compression, throughput, tokens, cache size)")
print("  fig2_vector_pipeline.png    — 2D visualization of the TurboQuant pipeline")
print("  fig3_quality_vs_bits.png    — SNR and cosine similarity curves")
print("  fig4_memory_at_scale.png    — KV cache memory at 256–32K tokens")
print("  fig5_attention_distortion.png — attention weight comparison fp16/3-bit/4-bit")
print("  fig6_full_report.png        — combined blog-ready report")
