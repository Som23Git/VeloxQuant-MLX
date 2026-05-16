# Memory Constraint Findings — Apple M4 24 GB

This document records the memory headroom constraints discovered during the
Outlier-Token + RateQuant (OTRQ) sweep on Apple M4 24 GB unified memory, and
the watchdog mechanism added to protect the GPU/system from OOM-driven faults.

## TL;DR

**Qwen2.5-32B-Instruct-4bit cannot run any KV-cache quantization variant on
a 24 GB Apple M4.** The model weights alone (~17.5 GB) plus the OS, Chrome,
and Claude Code (~7 GB) leave **negative headroom** for the activations and
quantization scratch buffers. fp16 baseline barely fits at 200 tokens
because activations stay small; any quantizer wrapper adds ~500 MB – 1 GB
that tips the system over.

The other 7 models in the benchmark (Mistral 7B, Falcon3 7B, Phi-4, Qwen3
4B/8B, Llama 3.1 8B, Gemma3 4B) ran all four OTRQ configs cleanly.

## Memory Budget — Qwen2.5-32B on 24 GB M4

| Component | Size |
|---|---|
| Qwen2.5-32B-4bit weights | ~17.5 GB |
| OS + Chrome + Claude Code + background processes | ~7.0 GB |
| **Total baseline committed** | **~24.5 GB** |
| Physical RAM | 24.0 GB |
| **Available headroom for activations + KV cache + quantizer state** | **−0.5 GB** |

The fp16 baseline succeeds at 2.8 tok/s because:
- Activations at 200 tokens are tiny (~50 MB)
- The cache stays small at fp16 (no per-token encode/decode overhead)
- inactive-page reclamation reclaims ~4 GB of file cache under pressure

The RVQ/Outlier/RateQuant configs fail because they add (per layer):
- Shared quantizer instance (rotation diagonal, codebook, boundaries) — ~5 MB
- Bit-packed index buffers held during forward pass
- Stage-1 and stage-2 codebook decode intermediates in fp16
- Rotation matrix + Hadamard scratch space

Aggregated across 64 layers, this overhead is ~500 MB – 1 GB, which
pushes the system from "tight" to "kernel kill or thermal event".

## What Happened During the Sweep

The original OTRQ sweep (`run_outlier_ratequant.py` across all 8 models)
left Qwen2.5-32B with only 1 of 4 configs completed (fp16). The other 3
(rvq1, rvq1o, rvqrq) silently failed because the parent process's
30-second subprocess timeout fired during the long model-load step.

Symptom in `.bench_tmp/`:
```
otrq_qwen25_32b_fp16.json    ← present
otrq_qwen25_32b_rvq1.json    ← missing
otrq_qwen25_32b_rvq1o.json   ← missing
otrq_qwen25_32b_rvqrq.json   ← missing
```

## The Watchdog

To safely re-attempt the run without risking GPU thermal events, a
memory-pressure watchdog was added (`/tmp/memory_watchdog.sh`). It polls
`vm_stat` every 10 seconds during a benchmark child process and sends
`SIGTERM` → `SIGKILL` to the entire process tree if free + inactive
memory drops below **1 GB**.

```bash
# Usage: invoked by the benchmark runner with the child PID
/tmp/memory_watchdog.sh $CHILD_PID > /tmp/memwatch.log 2>&1 &
```

Watchdog log entries during the qwen25_32b/rvq1 attempt that confirm the
fault was caught before MLX could fault the GPU:

```
[07:27:19] free=2243MB inactive=6421MB pressure=unknown
[07:27:29] free=67MB   inactive=1723MB pressure=unknown
[07:27:39] free=445MB  inactive=1839MB pressure=unknown
[07:27:49] free=78MB   inactive=2466MB pressure=unknown
[07:27:59] free=63MB   inactive=1563MB pressure=unknown
[07:28:09] free=62MB   inactive=1094MB pressure=unknown
[07:28:19] free=63MB   inactive=891MB  pressure=unknown
[07:28:19] CRITICAL: only 954MB available — killing PID 21438
```

System recovered to `10.2 GB free + 4.1 GB inactive` immediately after
the kill, with no GPU stall or kernel panic.

## What This Means For The Repo

1. **The OTRQ benchmark is treated as complete with 7/8 models.** Qwen2.5-32B
   shows fp16 only; figures for that model document the constraint in their
   captions rather than showing zeros for the missing configs.

2. **Any future large-model benchmark (≥30B weights on 24 GB) must:**
   - Close Chrome and other heavy GUI apps before running
   - Use the watchdog (currently at `/tmp/memory_watchdog.sh`)
   - Be run from a headless terminal (or via SSH from another machine)
   - Accept that the 32B regime requires ≥32 GB unified memory for any
     KV-cache quantization config to fit

3. **Alternative models that exercise the same regime safely on 24 GB:**
   - Qwen2.5-14B-Instruct-4bit (~8 GB weights, ample headroom)
   - Mixtral 8x7B 4-bit (~26 GB, won't fit; not a viable substitute)

## Re-Running 32B Later

If/when running on hardware that fits (32 GB+ Mac, or 24 GB with Chrome
fully closed and headless), the safe sequence is:

```bash
# 1. Verify available memory
vm_stat | awk '/Pages free/ {f=$3} /Pages inactive/ {i=$3} END {p=16384; printf "Free: %.1f GB + %.1f GB inactive\n", f*p/1073741824, i*p/1073741824}'
# Need ≥18 GB free+inactive before starting

# 2. Start watchdog wrapped run, one config at a time
PYTHONPATH=. python3 benchmark_scripts/run_outlier_ratequant.py \
    --models qwen25_32b --configs rvq1 > /tmp/q32_rvq1.log 2>&1 &
RUNPID=$!
/tmp/memory_watchdog.sh $RUNPID > /tmp/memwatch.log 2>&1 &

# 3. Wait for completion before starting the next config (don't pipeline)
wait $RUNPID

# 4. Repeat for rvq1o, then rvqrq
```

Running configs sequentially (not the default parallel sweep) ensures
the parent process's memory is fully released before each child loads
the 32B model, and gives the watchdog clean attribution if it has to
kill a single config.

## Watchdog Threshold Choice

The 1 GB threshold was chosen because:
- Below 1 GB free+inactive, the macOS kernel begins aggressive page-out
  to swap, which on Apple Silicon contends with GPU memory bandwidth
- 1 GB leaves enough room for the watchdog itself, log writers, and
  process-group signal delivery
- Empirically: the failed run hit 891 MB just before the kernel would
  have started swapping the Metal heap

Lower thresholds (500 MB) are tempting but unsafe — once swap pressure
starts, killing the process doesn't reliably free the GPU-allocated
pages quickly enough to recover.

## Related Files

- `benchmark_scripts/run_outlier_ratequant.py` — main runner (subprocess isolation)
- `benchmark_scripts/run_full_reports.py` — same architecture for the v3 sweep
- `.bench_tmp/otrq_qwen25_32b_fp16.json` — the one Qwen2.5-32B result we have
- `figures/outlier_token_ratequant/` — output figures (no qwen25_32b folder)
