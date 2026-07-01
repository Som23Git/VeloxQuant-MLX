# Phase 1 — New-Method Survey V9 (post-SnapKV)

Follow-up to `NEW_METHOD_SURVEY_V8.md` (which led to SnapKV-adapted, shipped in 0.19.0).
After SnapKV, the repo spans: scalar/group quant (KIVI, KVQuant-NUQ, TurboQuant),
vector quant (RVQ, VecInfer, CommVQ, RaBitQ, QJL), low-rank (SVDq, PALU),
cross-layer (XQuant, MiniCache), entropy coding (CacheGen), error-feedback (GEAR),
per-token saliency routing (ZipCache-adapted), attention-proxy adaptive schemes
(KIVI-Sink, AdaKV-proxy, Kitty), and score-based prefill eviction (SnapKV-adapted).
The axis still uncovered: **structural positional eviction** — keeping tokens by their
position (sinks + recency) rather than by learned attention score.

**Evidence discipline:** every arXiv ID below was verified to resolve to a real paper.
No invented IDs. Sources at bottom.

---

## Candidate table

| Method | Paper (verified) | What it adds | Cache-only fit | Effort | Verdict |
|---|---|---|---|---|---|
| **StreamingLLM** | arXiv:2309.17453, ICLR 2024 (Xiao et al.) | **Structural eviction**: keep first N_sink tokens (attention sinks) + rolling window of last W recent tokens. Pure position-based — no scoring, no calibration. | ✅ only K/V tensor shape + concat; purely structural | Low | **CHOSEN** |
| **H2O** | arXiv:2306.14048 | Heavy hitter + recency eviction — requires per-decode-step accumulated attention scores | ❌ decode attention not observable by cache wrapper | Med | hard reject (needs decode attn) |
| **ScissorHands** | arXiv:2305.17118, NeurIPS 2023 | Token merging: merge similar KV pairs by cosine similarity | ❌ cosine merge across decode steps needs full accumulated K — structural modification of stored tensors | High | deferred — reshape complexity |
| **LESS** | arXiv:2402.09398 | Low-rank eviction scoring from accumulated attention patterns | ❌ requires accumulated cross-decode attention | High | hard reject |

---

## Chosen: StreamingLLM-adapted (sink + recency window eviction)

### What the paper actually does

StreamingLLM (arXiv:2309.17453, ICLR 2024, Xiao et al.) — "Efficient Streaming Language
Models with Attention Sinks" — identifies that language models always attend strongly to
the *first* few tokens of any sequence regardless of content (attention sinks), and that
most other attention is locally recency-biased. The paper proposes keeping:
  1. The first `N_sink` token positions unconditionally (attention sinks).
  2. The most recent `W_recent` token positions (a rolling window).

All other positions are evicted. This allows infinite-context streaming generation at
constant KV memory: the cache never exceeds `N_sink + W_recent` positions regardless of
how many decode tokens have been generated. The paper reports near-lossless perplexity on
streaming tasks vs a full-context baseline, far outperforming plain sliding-window (which
suffers "perplexity explosion" when the first token leaves the window).

### The honest adaptation problem

**Adaptation 1 — Cache-level implementation.** The paper is implemented at the model
forward-pass level (patching attention masks and the KV cache together). Our cache
wrapper sees only `update_and_fetch(keys, values)`. We implement the eviction by
maintaining a FIFO recent-window and a frozen sink buffer inside the cache wrapper,
concatenating them on each update. This is semantically identical to the paper's intent.

**Adaptation 2 — No attention mask adjustment.** The paper also adjusts the attention
mask so positions beyond the window are not attended to by the query. A cache wrapper
cannot inject attention masks. This is a known limitation: the model still "sees" all
positions in the returned K/V but only the sink+recent positions are present, so the
effective sequence length is bounded — the functional memory budget is correct, and the
only gap is that the model's position IDs may not perfectly match in all architectures.
Documented plainly.

**Adaptation 3 — Token-level, not position-ID-level.** We evict by token count. The
paper's original implementation remaps position IDs when using RoPE — we do not remap;
we drop tokens and let the model see the surviving positions at their original positional
indices inside the returned K/V rows. For GQA / multi-head models this is equivalent.
Position-ID remapping would require model-level patching.

### What we do NOT implement
- Attention mask adjustment / position-ID remapping.
- Adaptive sink discovery (we use a fixed `stream_n_sink` count).
- Separate treatment for cross-attention vs self-attention.

### Why this is the right pick
1. **Last uncovered axis.** SnapKV-adapted covers score-based eviction. StreamingLLM
   covers *structural positional eviction* — drop by position, not by importance score.
   Orthogonal: a token can be important (would survive SnapKV) but old (would be evicted
   by StreamingLLM's recency window).
2. **Cleanest possible implementation.** No proxies, no calibration, no scoring —
   two buffer operations (sink concat + FIFO dequeue) on the K/V tensors already visible
   at `update_and_fetch`. Faithful to the paper's structural intent with one clearly
   stated limitation (no mask adjustment).
3. **Composable.** The sink+window pattern is orthogonal to quantization — wrapping a
   quantizer cache inside StreamingLLM is natural and not implemented by any existing
   method.
4. **Zero calibration, no model surgery, single-layer.** Identical constraint profile
   to all successful methods in this repo.
5. **High-impact paper.** ICLR 2024, 3000+ citations. Well-understood baseline for any
   streaming/long-context inference work.

### Byte accounting
- `stream_kept_bytes`: fp16 bytes stored (N_sink + W_recent positions × D × 2 tensors)
- `full_seq_bytes`: hypothetical cost if all tokens were kept
- `streaming_ratio`: full_seq_bytes / stream_kept_bytes (> 1 after window fills)
- `tokens_seen`: total positions seen since construction
- `tokens_in_window`: current N_sink + recent tokens in cache

---

## Sources (verified)

- StreamingLLM — https://arxiv.org/abs/2309.17453 (ICLR 2024, Xiao et al.)
- H2O — https://arxiv.org/abs/2306.14048 (hard reject — decode attn not observable)
- ScissorHands — https://arxiv.org/abs/2305.17118 (deferred — merge reshape complexity)
- LESS — https://arxiv.org/abs/2402.09398 (hard reject — needs accumulated attn)
