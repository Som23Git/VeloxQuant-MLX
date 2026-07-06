# Phase 1 — New-Method Survey V10 (post-CaM)

Follow-up to `NEW_METHOD_SURVEY_V3.md`–`V9.md` (most recently CaM-adapted cache
merging, shipped in 0.26.0). The repo now spans 29 strategies across: scalar/
group quant (KIVI, KVQuant-NUQ, TurboQuant, RateQuant), vector quant (RVQ,
VecInfer, CommVQ, RaBitQ, QJL, PolarQuant, SpectralQuant), low-rank projection
(SVDq — keys only, latent+mixed-bit; PALU — full K+V), cross-layer reuse
(XQuant — odd/even rematerialization; MiniCache — SLERP depth-merge), entropy
coding (CacheGen), error-feedback (GEAR), saliency-adaptive mixed precision
(ZipCache), attention-proxy adaptive schemes (KIVI-Sink, AdaKV-proxy, Kitty),
score-based prefill eviction (SnapKV), structural positional eviction
(StreamingLLM, sink, sliding-window, TOVA), layer-adaptive budgets (PyramidKV),
2D layer×token budgets (SqueezeAttention), chunk-level eviction (ChunkKV), and
merge-vs-drop eviction (CaM).

The open gap: **cross-layer *shared-subspace* compression**. XQuant reuses one
layer's quantized cache for its pair (rematerialization — no dedicated shared
basis). MiniCache merges two layers' *directions* via SLERP (a per-token-pair
geometric merge, not a joint factorization). Neither computes a **shared
low-rank subspace fit jointly across a *group* of layers** — a structurally
different compression axis with a different failure mode (subspace misalignment
across layers) and a different byte-accounting story (one shared basis
amortized over N layers, not one pairwise operation).

**Evidence discipline:** every arXiv ID below was verified to resolve to a real
paper (fetched abstract + author list, cross-checked against the claim) before
inclusion. No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds (not already in repo) | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **xKV** | arXiv:2503.18893 (Mar 2025, preprint) | **Cross-layer shared-subspace compression**: jointly factorizes grouped-layer KV-caches into *one* shared low-rank basis via CKA-aligned SVD, plus a decode-time "Selective Reconstruction" step | ✅ operates purely on cached K/V tensors post-prefill; SVD is `mx.linalg.svd`-native; no attention scores, no training | Med | **CHOSEN** |
| **SemShareKV** | arXiv:2509.24832 (Sep 2025, preprint) | Cross-*request* semantic cache sharing via token-embedding LSH | ❌ needs token embeddings (outside pure K/V tensors) + cross-session store; breaks single-sequence `update_and_fetch` contract | High | reject (architecture mismatch) |
| **HeadInfer** | arXiv:2502.12574 (Feb 2025, preprint) | Head-wise CPU/GPU offload scheduling | ⚠️ no attention/training needed, but the CPU/GPU bandwidth asymmetry it exploits doesn't exist on Apple Silicon's unified memory — motivation collapses | Med-High | reject (no hardware motivation on target platform) |
| **RDKV** | arXiv:2605.08317 (May 2026, preprint) | Rate-distortion joint eviction+quantization bit allocation | ❌ distortion weights derived from attention scores | — | hard reject (needs attn scores) |
| **CommonKV** | arXiv:2508.16134 (Aug 2025, preprint) | Cross-layer SVD sharing with cosine-similarity adaptive budget | ⚠️ same axis as xKV, weaker verification of calibration-free claim | Med | reject (redundant with xKV; xKV better verified) |

---

## Chosen: xKV (cross-layer shared-subspace KV compression)

### What the paper actually does

xKV ([arXiv:2503.18893](https://arxiv.org/abs/2503.18893), Chang, Lin, Lin,
Chiang, Akhauri, Dai, Jiang, Li, Ceze, Wu, Abdelfattah — preprint, code at
https://github.com/abdelfattah-lab/xKV) starts from an empirical observation:
using Centered Kernel Alignment (CKA), the **dominant singular subspaces of
per-layer KV-caches are well aligned across groups of nearby layers** — far
more aligned than raw per-token cosine similarity (the signal MiniCache-style
methods rely on) would suggest. The method:

1. **Grouping (offline, architecture-level):** partition the model's layers
   into fixed-size contiguous groups (e.g. groups of 2–4 consecutive layers).
2. **Joint factorization (once per group, per sequence, after prefill):**
   stack the group's per-layer key matrices along the token axis (or
   concatenate along a new axis) into one tall matrix, and compute a **single
   truncated SVD** over the stacked matrix. This yields one shared basis `V_g`
   (and mean) for the *entire group*, rather than one basis per layer.
3. **Per-layer latent coding:** each layer in the group projects its own keys
   into the shared basis (`L_i = (K_i - mean) @ V_g`) and stores only its
   latent codes — the basis itself is amortized across all layers in the
   group, which is where the memory win compounds (O(group_size × D × r) basis
   cost shared by `group_size` layers' worth of tokens, vs one basis per
   layer).
4. **Selective Reconstruction (decode-time optimization):** rather than fully
   reconstructing every layer's keys every step, the paper reconstructs a
   subset of layers exactly and derives the rest via the shared basis — an
   inference-latency optimization on top of the memory-compression mechanism.

The paper reports up to 8× KV-cache compression preserving long-context
accuracy, and (combined with Selective Reconstruction) up to 4.23× end-to-end
speedup on their hardware target — not applicable claims for us to inherit,
since our target is Apple Silicon bandwidth, not their GPU setup.

### The honest adaptation problem

**1. Grouping granularity.** The paper's grouping is decided empirically per
model architecture (which layers CKA-align best). We do not have per-model CKA
profiling infrastructure. We adapt by defaulting to **fixed contiguous groups
of `xkv_group_size` consecutive attention layers** (default 2, matching
XQuant's default pairing granularity for consistency), with no CKA validation
step — documented plainly as a simplification. A future version could add an
optional CKA-based grouping pass as a calibration step.

**2. Joint SVD requires seeing all group members' keys at the same token
range simultaneously — but each layer's cache only sees its own
`update_and_fetch` call.** This is architecturally identical to the problem
XQuant and MiniCache already solved with a **coordinator**: one layer in the
group (the "leader", lowest index) publishes its raw prefill keys to a shared
`XKVCoordinator`; the other group members ("followers") fetch the leader's
keys, concatenate with their own along a new leading axis, run the joint SVD,
and all members then project into the resulting shared basis. This exactly
follows the `MiniCacheCoordinator`/`XQuantCoordinator` pattern already proven
in this codebase (`veloxquant_mlx/cache/minicache_coordinator.py`,
`xquant_coordinator.py`) — same single-threaded, sequential-forward-pass
assumption, same `(group_id, token_start)`-keyed store.

**3. Decode-time incremental projection.** After the group's shared basis is
fixed at prefill, decode-time new keys from *any* group member project
independently into the already-computed `V_g` — no further coordination
needed per decode step (unlike MiniCache, which must coordinate every step).
This makes xKV's *decode* path simpler than MiniCache's despite prefill being
more involved.

**4. Selective Reconstruction — deferred, not implemented.** The paper's decode
latency optimization (reconstruct only a subset of group layers exactly, derive
the rest) is a compute/latency trick orthogonal to the memory-compression
mechanism. We implement full reconstruction for every layer on every fetch (as
every existing wrapper does) and document Selective Reconstruction as future
work — consistent with how SVDq already deferred the paper's optional sparsity
layer.

**5. Mixed-precision latent coding.** Reuse the exact `SVDqQuantizer` machinery
(`quantize_latents_mixed`, `_group_quant_dequant`) for the per-layer latent
codes — xKV's paper doesn't specify a quantization scheme for the latents (it
assumes fp16 latents and gets its compression purely from the rank
reduction + basis sharing), but this repo's design philosophy is byte-accounted
compression, so we add group-quantization on top as `xkv_hi_bit`/`xkv_lo_bit`
fields, defaulting to a **conservative single-bit-width** latent quantizer
(`xkv_latent_bits=4`) rather than SVDq's mixed-bit routing, to keep xKV's
distinguishing feature (the shared basis, not novel bit-allocation) the star of
the show. Mixed-bit is exposed as an optional extension, off by default.

### What we do NOT implement

- CKA-based automatic grouping (fixed contiguous groups only).
- Selective Reconstruction (decode-latency optimization — memory-compression
  mechanism only).
- Cross-model group-size tuning — one default (`xkv_group_size=2`), overridable.
- Any coordination across *values* — like SVDq, xKV compresses keys only in
  our adaptation (the paper does cover values, but keeping the values path
  fp16 mirrors SVDq's existing precedent in this repo and keeps the wrapper
  auditable).

### Why this is the right pick

1. **Fills the genuine last cross-layer gap.** XQuant = rematerialization
   (pairwise reuse). MiniCache = pairwise geometric merge (SLERP of two
   directions). xKV = joint multi-layer subspace factorization (one shared
   basis for N layers). Three structurally distinct cross-layer mechanisms,
   not three variations on one idea.
2. **Reuses proven infrastructure twice over.** The coordinator pattern
   (`MiniCacheCoordinator`/`XQuantCoordinator`) and the latent SVD+quant
   pattern (`SVDqKVCache`/`svdq.py`) both transfer almost directly — this is
   the least novel-code-per-feature addition of any candidate, which lowers
   implementation risk.
3. **Cache-only, no model surgery, no training, no attention scores.** Passes
   every hard constraint that has governed every method in this repo.
4. **Verified, real, matches claims exactly** (title/authors/abstract checked
   directly against arXiv metadata — see Sources).
5. **Honest uncertainty flagged up front:** xKV is a preprint (no venue as of
   Jul 2026). Label as "xKV-adapted (VeloxQuant-MLX implementation)" throughout;
   report only committed `results.json` numbers, never paper claims, exactly as
   done for SVDq/CaM/ChunkKV.

### Why the alternatives were not chosen

- **SemShareKV** — genuinely new axis (cross-*request* cache dedup), but its
  LSH mechanism needs token embeddings, which is outside pure K/V tensor
  manipulation, and cross-session state management doesn't fit the
  single-sequence `update_and_fetch` contract without a substantial
  architecture change (a persistent multi-session store). Deferred, not
  rejected outright — worth revisiting if the library ever adds multi-session
  serving support.
- **HeadInfer** — real, disqualifier-free mechanically, but its entire premise
  (hide slow CPU↔GPU transfer behind compute) doesn't apply to Apple Silicon's
  unified memory architecture. Porting it would produce a diluted
  reimplementation with no real motivation on the target hardware — the repo's
  own stated bar ("Apple Silicon bandwidth story") would not be met honestly.
- **RDKV** — hard reject, needs attention-derived distortion weights.
- **CommonKV** — same axis as xKV (cross-layer SVD sharing) but weaker
  verification of its calibration-free claims on a first read; xKV is the
  better-evidenced, cleaner mechanism covering the same gap, so implementing
  both would be redundant.

### Planned artifacts (Phases 2–6)

- `veloxquant_mlx/cache/xkv_coordinator.py` — `XKVCoordinator`, modeled
  directly on `MiniCacheCoordinator`: leader publishes raw prefill keys keyed
  by `(group_id, token_start)`; followers fetch, stack, jointly SVD.
- `veloxquant_mlx/quantizers/xkv.py` — pure primitives: `joint_svd_compress`
  (stack N layers' centered keys along token axis, truncated SVD → shared
  `V_g`, `K_mean_g`, singular values), `project_into_shared_basis`,
  `reconstruct_from_shared_basis` — deliberately mirrors `svdq.py`'s function
  signatures where the shapes allow, for reviewer familiarity.
- `veloxquant_mlx/cache/xkv_cache.py` — `XKVCache(role="leader"|"follower",
  group_id, coordinator)`. Leader path: publish own prefill keys, wait
  (same-forward-pass) for all followers to arrive is not needed — because
  unlike MiniCache's SLERP (needs both directions simultaneously), xKV's joint
  SVD can be computed incrementally: leader computes provisional basis from
  its own keys first, refines when followers' keys become available, OR
  (simpler, chosen default) leader waits to compute the shared SVD until all
  group members have published for the same token range, then broadcasts
  `V_g`/`K_mean_g` back through the coordinator for every member to consume,
  including itself. This costs one extra round-trip at prefill only.
  Decode path: any member projects new keys into the frozen `V_g` directly, no
  coordination needed.
- Config: `KVCacheConfig(method="xkv", xkv_group_size=2, xkv_rank=None,
  xkv_energy_threshold=0.95, xkv_latent_bits=4, xkv_max_ctx=8192)`.
- `KVCacheBuilder._build_xkv` — modeled on `_build_minicache`/`_build_xquant`:
  assign contiguous groups over attention-bearing layers, build one shared
  `XKVCoordinator`, instantiate one `XKVCache` per layer with its role/group.
- Tests:
  - Joint SVD correctness: shared basis reconstructs each group member's keys
    within tolerance ε (compare vs each layer's *own* independent SVD — shared
    basis should be close but not identical).
  - Group-of-1 degeneracy: `xkv_group_size=1` should reduce to (numerically
    close to) standalone per-layer SVDq-style compression — an analogous
    equivalence check to CaM's `cam_merge="drop" == H2O` and XQuant's
    `residual_bits=0` pure-reuse check.
  - Coordinator round-trip: leader publishes, all followers correctly receive
    the *same* `V_g`/`K_mean_g` object (or numerically identical arrays).
  - Byte accounting: shared basis storage amortized correctly across group
    size (basis bytes / group_size charged per layer, not full basis per
    layer).
  - Decode accumulation: sequential post-prefill keys project correctly into
    the frozen basis without re-triggering joint SVD.
  - Non-attention layer / fallback cache path unaffected (mirrors XQuant test).
- `benchmark_scripts/benchmark_xkv.py` — perplexity/reconstruction-error proxy
  + throughput vs XQuant and MiniCache at matched compression ratios, offline
  synthetic harness on Apple Silicon (consistent with every prior benchmark in
  this repo — no paper-number borrowing).
- CHANGELOG entry, `paper/EVIDENCE_TABLE.md` row, `docs-site/docs/algorithms/xkv.md`,
  README method count bump (29 → 30), landing page Method Library card +
  code panel tab + version banner.

---

## Sources (verified)

- xKV — https://arxiv.org/abs/2503.18893 (preprint, submitted Mar 2025, revised
  May 2026); code https://github.com/abdelfattah-lab/xKV
- SemShareKV — https://arxiv.org/abs/2509.24832 (preprint, Sep 2025)
- HeadInfer — https://arxiv.org/abs/2502.12574 (preprint, Feb 2025)
- RDKV — https://arxiv.org/abs/2605.08317 (preprint, submitted May 2026)
- CommonKV — https://arxiv.org/abs/2508.16134 (preprint, Aug 2025)
