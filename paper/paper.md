# VeloxQuant-MLX: A Metal-Accelerated KV-Cache Quantization Suite for LLM Inference on Apple Silicon Unified Memory

*Rajveer Rathod · 2026 · Markdown mirror of `paper.tex` for review.*

## Abstract
The KV cache, not the model weights, is the binding memory constraint for long-context LLM inference on Apple Silicon, where CPU/GPU/Neural-Engine share one unified memory pool. Weight quantization—the ecosystem's focus—does nothing to bound cache growth, which is linear in context length. We present **VeloxQuant-MLX**, an open-source library that ports a suite of KV-cache quantization algorithms (TurboQuant, RVQ, VecInfer, RaBitQ, CommVQ, QJL, PolarQuant, RateQuant, and a new SpectralQuant) to Apple's MLX with hand-written, runtime-compiled Metal kernels on the hot path. The contribution is primarily *systems*: a unified, Metal-accelerated cache-quantization layer that plugs into `mlx_lm` in three lines without changing `mlx_lm.generate`. Our Metal product-VQ quantize kernel is **6.9–14.7× faster** than pure-MLX (**13.2× at S=2048**) and cuts peak memory **98.4%** (729 MB → 12 MB) at the shape that otherwise OOMs. VecInfer-1bit reaches **16× key compression on Qwen2.5-7B while matching fp16 throughput**; RaBitQ + 4-bit values gives **5.95× full-KV compression on Falcon3-7B**. We disclose negative results: the RaBitQ ANN **search path has recall@10 ≤ 0.4** and is unusable for retrieval, and quality is measured by reconstruction fidelity + a coherence proxy, not downstream accuracy.

## 1. Introduction
KV-cache memory is `L·H_kv·d_h·S·2·2` bytes (fp16) — linear in S. Llama-3.1-8B: 4.2 GB @ 8k, 16.8 GB @ 32k. On Apple Silicon's **unified memory**, weights + cache + OS + apps contend for the same 16/24 GB, so **cache size, not weight size, caps context length**. `llama.cpp`/Ollama/LM Studio/`mlx_lm` all default to an fp16 cache; weight quantization (GGUF/GPTQ/AWQ) is offline and cannot help. Cache quantization runs online, per token, on tensors that change every generation.

**Contributions (stated honestly — most quantizers are re-implementations):**
- A unified Apple-Silicon KV-cache suite: nine strategies behind one `mlx_lm`-compatible interface, three-line enable.
- Hand-written Metal kernels (JIT via `mx.fast.metal_kernel`); the quantize kernel is verified at **13.2× / 98.4%**.
- **SpectralQuant** — SVD-eigenbasis signal/noise codebook split (most plausibly novel piece).
- A **12-model** engineering sweep with reproducible scripts and per-model result files.
- Disclosed negatives: non-functional RaBitQ search; Qwen2.5-32B OOM.

## 2. Background & Related Work
Weight vs. cache quantization; KIVI, KVQuant, GEAR, QJL, TurboQuant, RaBitQ, CommVQ, VecInfer, RateQuant. MLX provides `mx.fast.hadamard_transform` (O(d log d) Metal rotation) and `mx.fast.metal_kernel` (runtime shader compilation) — the primitives this work builds on. No prior library unifies these KV methods under MLX with custom Metal kernels.

## 3. Method (summary)
- **TurboQuant**: randomized-Hadamard rotation + unit-norm + Lloyd–Max (b−1 bits) + QJL 1-bit residual. MSE bound `D(b) ≤ √(3π)/2 · 4^(−b)`, **verified** in `test_distortion_bounds.py` for b=1..4.
- **RVQ**: two 1-bit passes, analytical Gaussian/Laplacian codebooks, zero calibration (synthetic cosine 0.69→0.98).
- **VecInfer**: per-channel smooth + Hadamard dual transform + product VQ (256-entry codebooks). The Metal-accelerated path.
- **SpectralQuant**: SVD rotation → high-res codebook on signal dims, coarse on noise dims.
- **RaBitQ**: randomized Hadamard + sign-packing + IVF + top-M fp16 re-rank; paired with 4-bit values.
- **CommVQ**: centroids constrained to RoPE-commuting 2×2 form after each EM step.
- **RateQuant**: closed-form reverse-waterfilling per-layer bit allocation; D(b)=αβ^(−b), β≈3.5.

## 4. System & Implementation
- **Metal kernels**: `vecinfer_quantize_metal` accumulates squared distance in registers, never materializing the `[chunk, n_centroids, sub_dim]` diff tensor (the OOM cause). JIT-compiled; `metal_available()` probe; three-state `use_metal_kernels` flag preserves a pure-MLX path. *Dequant kernel is at MLX parity — the win is quantize only.*
- **`mlx_lm` integration** (three lines):
  ```python
  config = KVCacheConfig(method="turboquant_rvq", bit_width_inlier=1, seed=42)
  caches = KVCacheBuilder.for_model(model, config)
  model.make_cache = lambda *_a, **_k: caches
  ```
  Quantize→dequantize inside `update_and_fetch`, so SDPA sees fp16. Byte-accurate accounting via tracked fields.
- **Architecture**: quantizer registry + cache factory + `for_model` per-layer list path (RateQuant).

## 5. Evaluation
Hardware: **Apple M4, 16 GB**, MLX ≥ 0.18, Python 3.11/3.12; 4-bit mlx-community models. Quality = reconstruction fidelity + tokens-generated proxy (no downstream task accuracy).

**Table 1 — Metal quantize kernel (D=128, n_centroids=256, sub_dim=8).** `figures/metal/results.json`

| S | pure (ms) | metal (ms) | speedup |
|---|---|---|---|
| 128 | 3.64 | 0.53 | 6.9× |
| 512 | 13.47 | 1.26 | 10.7× |
| 2048 | 55.10 | 4.18 | **13.2×** |
| 8192 | 228.56 | 15.57 | 14.7× |

Peak memory at OOM-trigger shape (H=4, S=4096, D=256, sub_dim=4): **729.3 MB → 12.0 MB (98.4%)**.

**Table 2 — Qwen2.5-7B-Instruct-4bit (d_h=128, H_kv=4, S≈120).** `figures/vecinfer/Qwen2.5-7B-Instruct-4bit/results.json`

| Config | tok/s | key comp. | tokens |
|---|---|---|---|
| fp16 | 21.0 | 1.0× | 120 |
| RVQ-1bit | 20.7 | 7.5× | 112 |
| VecInfer-2bit | 21.3 | 8.0× | 120 |
| VecInfer-1bit | **21.5** | **16.0×** | 121 |

VecInfer-1bit matches fp16 throughput at 16× (strong GQA: 4 KV / 28 query heads). Caveat: TQ-2bit emitted only 66 tokens (early stop).

**Full-KV (RaBitQ, Falcon3-7B, d_h=256):** key ratio 11.6×; with 4-bit MSE values **5.95× full-KV**, constant across S. KV-only memory @ 32k: 3.67 GB → 0.616 GB. The "~10⁵ tokens @ 8 GB" figure is a **KV-only linear extrapolation**, not a measured row; the README's "103k vs 17k" mixes in total-RAM accounting we did not reproduce.

**RateQuant:** Falcon3-7B (sensitivity 6.48×) → 14×b2+14×b1, 100% fp16 throughput @ 5.22×; Gemma3-4B (14.39×) → 3×b3+11×b2+20×b1, 91% @ 5.22×. The "2.7× lower PPL degradation" figure lacks a source file → not claimed as measured.

**SpectralQuant** (vs TurboQuant 3-bit, matched bits): cosine 0.8329→0.9072 on Qwen2.5-0.5B (+7.4pp); 0.7581→0.8625 on Gemma-4-4B (+10.4pp); ~5.3× compression. No machine-readable result file accompanies the PNGs — reproduced from reported tables.

**Optimization journey:** four dispatch-overhead removals (head-batching; Hadamard; boundary-sum quantize, 100% index-match to argmin; cast cleanup) → 1.2–2.6× throughput, zero quality regression. Qwen3-4B RVQ-2bit reaches 92% of fp16.

## 6. Discussion & Limitations
- **No downstream-task eval** (no LongBench / WikiText sweep / NIAH) — chief threat to validity.
- **RaBitQ search non-functional**: recall@10 0.0–0.4; fp16 baselines read 0.000 ms (untimed) so "speedup" is an artifact. Only its memory story holds.
- **Unified-memory specificity**: published CUDA kernel fusions don't port to Metal; several methods trade throughput for memory here.
- **Honest negative**: Qwen2.5-32B (~17.5 GB weights) OOMs on 16 GB M4; cache quant can't recover weight headroom.
- **Reporting fixes**: "13× hot path" → quantize only; "243 tests" badge ≠ 314 collected; SpectralQuant 5.95× headline ≠ 5.33× table. We use reproducible figures.
- **Breadth vs depth**: one chip (M4), short generation lengths.

## 7. Conclusion
A three-line, Metal-accelerated KV-cache quantization option for Apple-Silicon inference, with a verified 13×/98% quantize kernel, framed honestly as integration + acceleration + a 12-model sweep + SpectralQuant, with disclosed negatives.

## Appendix — Reproducibility
`python -m pytest veloxquant_mlx/tests` (314 collected). Key runs:
```bash
PYTHONPATH=. python benchmark_scripts/benchmark_vecinfer.py --model mlx-community/Qwen2.5-7B-Instruct-4bit
python scripts/metal_quantize_proof.py
python benchmark_scripts/benchmark_kv.py
python -m pytest veloxquant_mlx/tests/integration/test_distortion_bounds.py
```
Seeds fixed (`seed=42`; per-test RNG seeded by b).
