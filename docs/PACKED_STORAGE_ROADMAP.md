# Packed storage roadmap

## Problem

Several methods report large **key-byte accounting** ratios while the default
runtime path still materializes **fp16** tensors in the parent `mlx_lm`
`KVCache` (quantize → dequantize → store). Users then expect Activity Monitor
RSS to drop by 7.5× or 16× and do not see it at short context.

## Goal

Track, per method, whether compressed state is **actually retained** in
resident memory during decode, versus accounting-only counters.

## Status matrix (starting point)

| Method | Default stores packed? | Notes |
| --- | --- | --- |
| `turboquant_rvq` | No | Dequant into parent fp16 cache; counters are accounting |
| `vecinfer` | No (default) | Optional `fused_sdpa` / index ring buffer exists |
| `rabitq` | Partial / fused path | Fused encode/attend Metal path aims to avoid materializing K/V |
| Eviction methods | N/A | Reduce token count (`offset`), not bit-width |

Update this table as methods change. Prefer linking a `results.json` that
measures RSS or cache `nbytes` for packed paths.

## Engineering work items

1. Inventory each cache class: what `update_and_fetch` writes to parent state.
2. Add a shared reporting helper: `accounting_bytes` vs `resident_cache_bytes`.
3. Extend `scripts/validate_kv_memory.py` with optional OS RSS sampling.
4. Document fused/packed flags clearly in quickstart (when they help, when
   launch overhead hurts).
5. Treat "resident compression" as a first-class claim type in PR template
   checkboxes (already sketched in CONTRIBUTING honesty rule).

## Success metric

A user can run one script and see side-by-side:

- accounting ratio
- resident cache bytes (or RSS delta at long context)
- tok/s

without reading source to learn that dequantization hides the savings.
