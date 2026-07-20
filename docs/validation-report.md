# Validation Report: KV Cache Quantization on Apple Silicon

This report explains what VeloxQuant-MLX does, how to measure it honestly, and
how to reproduce numbers with `scripts/validate_kv_memory.py`.

**Writing rule:** no marketing softeners. Every claim below is either a
definition, a formula, or a number produced by a committed script.

## Measured run (committed results)

Source: [`figures/validation/Llama-3.2-3B-Instruct-4bit/results.json`](../figures/validation/Llama-3.2-3B-Instruct-4bit/results.json)

| Field | Value |
| --- | --- |
| Hardware | Apple M4 Pro, 48 GB unified RAM |
| Model | `mlx-community/Llama-3.2-3B-Instruct-4bit` |
| Workload | short prompt, `max_tokens=64` |
| Tokens in cache | 120 (all arms) |

| Arm | tok/s | MLX peak MB | Keys fp16 (acct) | Keys compressed (acct) | Compression claim | Cache tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp16-baseline | 58.2 | 1954 | n/a (no counters) | n/a | 1.00× | 120 |
| RVQ-1bit | 81.1 | 1855 | **6.562 MB** | **0.872 MB** | **7.53×** | 120 |
| VecInfer-1bit | 49.2 | 1926 | **6.562 MB** | **0.410 MB** | **16.00×** | 120 |

How to read the compression columns:

```text
compression claim = fp16_key_bytes / compressed_key_bytes
RVQ:      6,881,280 / 913,920  ≈ 7.53×
VecInfer: 6,881,280 / 430,080  = 16.00×
```

Raw sizes are **key-byte accounting** (what the packed format would use), not
Activity Monitor RSS. Peak MB stayed near ~1.9 GB for all arms because weights
dominate at this short context (honesty rule in section 3).

---

## 1. What the KV cache is

During autoregressive generation, each transformer layer computes **key** (K)
and **value** (V) vectors for every token. Without a cache, generating token
`t` would recompute K/V for all previous tokens.

The **KV cache** stores those K/V tensors so each new token only computes its
own K/V and attends over the stored history.

Rough memory for an fp16 cache:

```text
bytes ≈ 2 × n_layers × n_kv_heads × head_dim × seq_len × 2
```

The leading `2` is keys + values. The trailing `2` is bytes per fp16 element.
`seq_len` is the number of tokens currently in the cache (prompt + generated).

---

## 2. What quantization does here

**Quantization** stores fewer bits per element (or a compact codebook index)
instead of full fp16. That reduces the *packed* size of the cache so the same
RAM budget can hold a larger `seq_len`.

**Eviction** methods (H2O, StreamingLLM, and others) instead drop or merge
tokens so the cache length stays bounded. They are a different axis from
bit-width compression.

**Cross-layer** methods share codes, directions, or subspaces across layers.

VeloxQuant plugs into `mlx_lm` by building a list of cache objects and passing
them as `prompt_cache` (or via `model.make_cache`). The generate API stays the
same.

---

## 3. Where reduction happens (and where it does not)

| Axis | What shrinks | Typical methods |
| --- | --- | --- |
| Keys only | Packed key bytes | RVQ, VecInfer (keys), SpectralQuant |
| Keys + values | Full KV packed bytes | RaBitQ, KIVI, NSNQuant |
| Token count | `seq_len` in cache | H2O, StreamingLLM, SnapKV |
| Across layers | Shared state | XQuant, MiniCache, xKV |

**Important honesty rule for default RVQ and VecInfer:**

1. Encode (quantize) incoming keys (and values for VecInfer).
2. Immediately decode (dequantize) back to fp16.
3. Store the dequantized tensors in the parent `mlx_lm` `KVCache`.
4. Report `compressed_key_bytes` as if the packed format were retained.

So the headline ratios (about **7.5×** for RVQ-1bit at `head_dim=128`, about
**16×** for VecInfer-1bit with `sub_dim=8`) are **key-byte accounting**, not
guaranteed Activity Monitor RSS savings at short context.

Packed / fused paths (for example RaBitQ fused attend, or VecInfer
`fused_sdpa=True`) are the place to look for true resident compression. This
validation harness measures accounting + MLX peak, and says so in JSON.

---

## 4. Before / after metrics you can observe

| Metric | How to read it | Source in harness |
| --- | --- | --- |
| Tokens in cache | Context length held per layer | `tokens_in_cache_max` from `cache.offset` |
| Keys fp16 size | Uncompressed key bytes (accounting) | `fp16_key_bytes` / `fp16_key_mb` |
| Keys compressed size | Packed key bytes (accounting) | `compressed_key_bytes` / `compressed_key_mb` |
| Compression claim | `fp16_key_bytes / compressed_key_bytes` | `key_compression` |
| Value accounting | Same for values when present | `value_compression` |
| Full KV accounting | Includes residual fp16 windows if any | `full_kv_compression` |
| MLX peak MB | Peak allocator use during generate | `peak_mb` via `mx.get_peak_memory()` |
| Throughput | Generated tokens / wall seconds | `throughput_tok_s` |
| Text preview | Spot-check quality | `output_preview` |

At short generation (for example 128 tokens), **weights dominate** peak MB.
Do not expect peak MB to fall by 7.5× or 16×. To stress the cache term, grow
prefill with `--prompt-repeat` or a long document, or run until fp16 OOMs.

---

## 5. Method comparison (practical defaults)

| Goal | Method | Why |
| --- | --- | --- |
| Everyday long context, zero calibration | `turboquant_rvq` b=1 | ~7.5× key accounting; usually near fp16 tok/s |
| Max key accounting | `vecinfer` 1-bit | ~16× key accounting; needs codebook calibration |
| Best quality at moderate rate | `spectral` | Calibration; stronger reconstruction vs same bits |
| More context in tight RAM (full KV) | `rabitq` | Compresses keys and values; fused Metal path exists |
| Constant memory / unbounded length | `streaming_llm` or `h2o` | Eviction, not bit packing |

Mac + RAM picking is automated by the Mac recommender CLI (see Phase 1c /
`veloxquant recommend` once that lands). Rule of thumb: on 8–16 GB prefer
eviction or full-KV methods for long context; on 32 GB+ everyday RVQ is fine
for 7–8B models.

---

## 6. How to reproduce every number

```bash
cd ~/Downloads/Projects/VeloxQuant-MLX
source .venv/bin/activate
pip install -e ".[dev]" mlx-lm   # if needed

PYTHONPATH=. python scripts/validate_kv_memory.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 128
```

Output:

```text
figures/validation/Llama-3.2-3B-Instruct-4bit/results.json
```

JSON fields to cite:

- `hardware.chip`, `hardware.unified_ram_gb`
- `results[].key_compression` (accounting)
- `results[].tokens_in_cache_max` (context in cache)
- `results[].peak_mb` (MLX peak, not RSS)
- `honesty` block (metric definitions)

Optional longer prefill:

```bash
PYTHONPATH=. python scripts/validate_kv_memory.py \
  --model mlx-community/Llama-3.2-3B-Instruct-4bit \
  --max-tokens 128 \
  --prompt-repeat 8
```

Smoke test without VecInfer calibration:

```bash
PYTHONPATH=. python scripts/validate_kv_memory.py --skip-vecinfer
```

---

## 7. Expected accounting (formula check)

For RVQ at bit-width `b` and head dim `d`:

```text
bytes_per_key_vector ≈ ceil(d * 2 * b / 8) + 2
fp16_bytes_per_key_vector = d * 2
ratio ≈ (d * 2) / (ceil(d * 2 * b / 8) + 2)
```

At `d=128`, `b=1`: `(256) / (32 + 2) = 7.53` → about **7.5×**.

For VecInfer product VQ with codebook index bits `b_k` and sub-vector size
`d_k`:

```text
bytes_per_key ≈ (d / d_k) * b_k / 8
ratio ≈ (d * 2) / bytes_per_key
```

At `d=128`, `d_k=8`, `b_k=8`: `16` bytes vs `256` → **16×**.

If your JSON shows those ratios for RVQ-1bit / VecInfer-1bit, the counters
match the design. That still does not by itself prove resident RAM fell by
the same factor.

---

## 8. What this report does not claim

- It does not claim 16× lower Activity Monitor memory at short context.
- It does not reproduce paper task scores (RULER, LongBench, and similar).
- It does not replace per-method algorithm pages under `docs-site/docs/algorithms/`.

For Metal kernel peak-memory claims (temporary tensors during quantize), use
the Metal-specific benches and guides, not this harness alone.
