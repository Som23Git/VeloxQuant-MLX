# Phase 1 — New-Method Survey V4 (post-KVQuant-NUQ)

Follow-up to `NEW_METHOD_SURVEY_V3.md` (which led to SVDq, now shipped). Since
V3, the repo has shipped **every** candidate from both prior surveys: KVSink
(`kivi_sink`, 0.9.0), SVDq (0.10.0), Kitty (0.11.0), AdaKV-proxy (0.12.0),
XQuant (0.13.0), and KVQuant-NUQ (0.14.0). The V2/V3 candidate shelf is
exhausted. This survey picks the next method from the **one major axis the repo
still lacks**: genuine low-rank projection of *both* tensors with the cache held
in latent form.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper. No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Metal fit | Effort | Verdict |
|---|---|---|---|---|---|
| **PALU** | arXiv:2407.21118, **ICLR 2025** (Chang et al.) | **True low-rank latent cache for K *and* V** — stores `[S, r]` codes directly (not full fp16), group-head shared projections; a structurally new axis vs SVDq (keys-only, reconstructs full fp16) | ⚠️ offline SVD fits cleanly; fused recon-attention CUDA kernel won't port — reconstruct then MLX SDPA | Med-High | **CHOSEN** |
| **KVLinC** | arXiv:2510.05373 (Oct 2025) | Hadamard rotation + trainable linear correction adapters | ❌ adapters need 2–11 h training; full-precision keys at runtime | High | skip (training; re-confirmed from V3) |
| **ThinKV** | arXiv:2510.01290, **ICLR 2026 Oral** | Thought-adaptive hybrid quant+eviction for reasoning | ❌ needs attention sparsity patterns (attention scores); CoT-specific | High | skip (attn scores; re-confirmed from V3) |
| **KVmix** | arXiv:2506.08018 | Gradient-based per-layer mixed precision | overlaps RateQuant (already shipped) | Med | skip (overlap; re-confirmed from V2) |

---

## Chosen: PALU (true low-rank latent storage for keys *and* values)

### What the paper actually does

PALU ([arXiv:2407.21118](https://arxiv.org/abs/2407.21118), ICLR 2025)
compresses the KV cache by **low-rank projection**. Rather than quantizing in
the original `D`-dimensional space, it projects keys and values into a rank-`r`
latent space and stores the *latent* `[S, r]` — the cache itself is
low-dimensional. Its mechanism:

1. **Offline (calibration):** for each KV projection, factor the weight/output
   into a low-rank product via SVD. PALU explores three granularities:
   whole-matrix (M-LRD, max compression), per-head (H-LRD, best fidelity), and
   **group-head (G-LRD)** — heads partitioned into groups that share a
   projection, the recommended middle ground.
2. **Runtime:** the cache stores the latent codes; attention reconstructs the
   full K/V on the fly (PALU fuses this reconstruction into the attention
   kernel). PALU additionally low-bit quantizes the latents.

The paper reports large KV memory reductions at small quality cost, and notes
the latent representation is itself more quantization-friendly than the original
space.

### The honest adaptation problem

Three decisions for the VeloxQuant-MLX port:

**1. SVD timing.** The paper fits projections offline on a calibration set. As
with SVDq, the `update_and_fetch` contract gives us the full prefill batch when
`S > 1`, so we fit group-head SVD from that batch — no separate calibration step.

**2. True latent storage (the load-bearing decision).** SVDq, already in the
repo, reconstructs full fp16 keys and hands them to the parent `mlx_lm` cache —
its win is byte-accounting/bandwidth only. To make PALU *different*, the cache
must keep `[S, r]` and bypass the parent fp16 ring buffer. We do exactly that:
`PALUKVCache` overrides storage, tracks its own `offset`, and reconstructs fp16
only at attend time. The latent buffers genuinely hold `[S, r]`; a test asserts
`cache.keys is None`.

**3. Group rank uniformity + mixed-bit compose.** Each group's energy-threshold
SVD may retain a different rank; we use the minimum so latent buffers share one
`r` (clean shapes). The stored latents are mixed-bit quantized by reusing the
already-tested SVDq latent coder (top-25% channels at 4-bit, rest at 2-bit),
giving a full-KV rate of `(r/D) · avg_bits` — below 1 bit/element on low-rank
data.

**What we do NOT implement:** PALU's fused low-rank-reconstruction attention
CUDA kernel. We reconstruct fp16 then call MLX SDPA. Consequence: storage is
low-rank, but the working set *during attention* is briefly the reconstructed
fp16 K/V — peak memory at attend time is not reduced, only the stored cache
size. Documented as a known simplification.

### Why this is the right pick

1. **Fills the one genuinely missing axis.** SVDq does keys-only low-rank and
   reconstructs to fp16. PALU does both tensors and keeps the cache latent.
   Together the repo now has two SVD-based methods on different points: SVDq
   (keys, bandwidth) and PALU (K+V, storage).
2. **Cache-only access — no model surgery.** Group-head SVD is computed from the
   K/V the cache already holds at prefill. No hidden-state hooks, no
   attention-score coupling. A first-class `update_and_fetch` citizen.
3. **Composes with existing quantizers.** The latent quantization delegates to
   the SVDq mixed-bit coder, inheriting its tests.
4. **Honest uncertainty:** ICLR 2025 paper, but our *true-latent storage* +
   *prefill-batch SVD* + *no-fused-kernel* combination is an adaptation. Labeled
   "PALU-adapted (VeloxQuant-MLX implementation)"; numbers come from committed
   `results.json`, not paper claims.

### Why the alternatives were not chosen

- **KVLinC** and **ThinKV** were already hard-rejected in V3 (training required;
  attention scores required) — re-confirmed, nothing changed.
- **KVmix** overlaps RateQuant's per-layer mixed-precision axis, already in the
  repo since 0.x.

### Planned artifacts (Phases 2–6)

- `veloxquant_mlx/quantizers/palu.py` — `head_group_bounds`, `group_head_svd`,
  `project_to_latent`, `reconstruct_from_latent`, `quantize_latent` (reuses the
  SVDq mixed-bit coder).
- `veloxquant_mlx/cache/palu_cache.py` — `PALUKVCache` with **true latent
  storage** (parent fp16 buffer bypassed, own offset bookkeeping) and a
  `_TensorLowRank` helper per tensor.
- Config: `KVCacheConfig(method="palu", palu_rank, palu_energy_threshold,
  palu_n_head_groups, palu_hi_bit, palu_lo_bit, palu_hi_fraction,
  palu_group_size, palu_quantize_values)`.
- Tests: latent-storage assertion, both-tensors-beat-naive-2bit, decode
  accumulation, byte accounting, group SVD subspace recovery, determinism.
- `benchmark_scripts/benchmark_palu.py` — vs SVDq/KIVI/fp16 + offline full-KV
  reconstruction MSE. No model-level numbers claimed until `results.json` exists.
- Docs page, sidebar, overview, CHANGELOG (root + docs-site), EVIDENCE_TABLE
  rows, landing-page featured card.
- **Incidental fix:** `KVCacheBuilder.for_model()` now propagates all
  method-specific config fields via `dataclasses.replace` (it previously dropped
  them, affecting svdq/kitty/kvquant/palu built through `for_model`).

---

## Sources (verified)

- PALU — https://arxiv.org/abs/2407.21118 (ICLR 2025); code https://github.com/shadowpa0327/Palu
- KVLinC — https://arxiv.org/abs/2510.05373 (re-confirmed reject from V3)
- ThinKV — https://arxiv.org/abs/2510.01290 (ICLR 2026 Oral; re-confirmed reject from V3)
- KVmix — https://arxiv.org/abs/2506.08018 (re-confirmed reject from V2)
