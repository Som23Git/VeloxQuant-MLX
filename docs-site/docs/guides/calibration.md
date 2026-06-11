---
id: calibration
title: Calibration Guide
sidebar_label: Calibration
slug: /guides/calibration
---

# Calibration Guide

Some VeloxQuant-MLX algorithms require a calibration step before inference. This guide explains when calibration is needed, how to collect activations, and how to save and reuse calibration artifacts.

## Which algorithms need calibration?

| Algorithm | Calibration needed | Time | What is calibrated |
|---|---|---|---|
| TurboQuant RVQ | No | — | Fixed analytical codebooks |
| QJL | No | — | Fixed random projection |
| RaBitQ | No | — | Fixed Hadamard + IVF init |
| PolarQuant | No | — | Fixed decomposition |
| CommVQ | No | — | Fixed codebook structure |
| **VecInfer** | **Yes** | ~2 min | Smooth factors + product codebook |
| **RateQuant** | **Yes** | ~90 sec | Per-layer sensitivity curves |
| **SpectralQuant** | **Yes** | ~3 min | SVD rotation matrices |

## Calibration data

All calibration functions accept a model, tokenizer, and a `num_samples` argument. Internally, they pass short sequences through the model's forward pass to collect key/value activations.

**Recommended calibration data:**
- Use domain-representative text (ideally the same distribution as your inference prompts)
- 64–128 samples of 256–1024 tokens each is sufficient for all algorithms
- Any text works as a fallback — the algorithms are robust to calibration distribution shift

```python
# You can use any text dataset; here we use random prompts as a placeholder
calibration_prompts = [
    "The history of machine learning began...",
    "In physics, the concept of entropy...",
    # ... 62 more prompts
]
```

## VecInfer calibration

```python
import mlx_lm
from veloxquant_mlx.allocators.vecinfer import calibrate_smooth_factors, train_codebook
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Step 1: smooth factors (fast, ~10 sec)
smooth_factors = calibrate_smooth_factors(
    model, tokenizer,
    num_samples=64,
    sequence_length=256,
)

# Step 2: codebook training (~2 min)
codebook = train_codebook(
    model, tokenizer,
    smooth_factors=smooth_factors,
    num_samples=128,
    num_centroids=256,
    num_subspaces=8,
)

# Save both artifacts
store = NpyArtifactStore("./artifacts/")
store.save("vecinfer_smooth", smooth_factors)
store.save("vecinfer_codebook", codebook)
```

## RateQuant calibration

```python
from veloxquant_mlx.allocators.ratequant import (
    calibrate_layer_sensitivities,
    fit_distortion_curve,
    allocate_bits_ratequant,
)
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore

sensitivities = calibrate_layer_sensitivities(
    model, tokenizer,
    num_samples=32,
    sequence_length=512,
)

distortion_curves = fit_distortion_curve(sensitivities)

bit_allocation = allocate_bits_ratequant(
    distortion_curves,
    target_bits=2.0,
    min_bits=1,
    max_bits=4,
)

store = NpyArtifactStore("./artifacts/")
store.save("ratequant_allocation", bit_allocation)
```

## SpectralQuant calibration

```python
from veloxquant_mlx.spectral.calibrate import calibrate_spectral_rotation, save_rotations

rotations = calibrate_spectral_rotation(
    model, tokenizer,
    num_samples=64,
    sequence_length=1024,
)

save_rotations(rotations, "./artifacts/spectral/")
```

## Loading calibration artifacts

```python
from veloxquant_mlx.artifacts.npy_store import NpyArtifactStore
from veloxquant_mlx.spectral.calibrate import load_cached_rotations

store = NpyArtifactStore("./artifacts/")

# VecInfer
smooth_factors = store.load("vecinfer_smooth")
codebook = store.load("vecinfer_codebook")

# RateQuant
bit_allocation = store.load("ratequant_allocation")

# SpectralQuant
rotations = load_cached_rotations("./artifacts/spectral/")
```

## Using the CLI precompute command

VeloxQuant-MLX includes a CLI that wraps the calibration steps:

```bash
# VecInfer calibration
python -m veloxquant_mlx precompute \
    --method vecinfer \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --output ./artifacts/ \
    --num-samples 128

# SpectralQuant calibration
python -m veloxquant_mlx precompute \
    --method spectral \
    --model mlx-community/Qwen2.5-7B-Instruct-4bit \
    --output ./artifacts/ \
    --num-samples 64

# RateQuant calibration
python -m veloxquant_mlx precompute \
    --method ratequant \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --output ./artifacts/ \
    --target-bits 2.0
```

## Artifact reuse across sessions

Calibration artifacts are model-specific but session-agnostic. They can be:

- Reused indefinitely for the same model (rotation matrices do not expire)
- Shared across machines with the same Apple Silicon generation
- Versioned alongside your model checkpoints

:::tip
Create one `./artifacts/<model-name>/` directory per model and point `NpyArtifactStore` to it. This keeps calibration data organised when working with multiple models.
:::

## Recalibration when to recalibrate

| Situation | Recalibrate? |
|---|---|
| Same model, new prompt domain | Optional — usually not needed |
| Updated model weights (fine-tune) | Yes |
| Different model family | Yes |
| Different quantization bit rate | RateQuant only |
| Updated VeloxQuant-MLX version | Check changelog |

## See also

- [VecInfer algorithm](../algorithms/vecinfer)
- [RateQuant algorithm](../algorithms/ratequant)
- [SpectralQuant algorithm](../algorithms/spectral)
- [API — Allocators](../api/allocators)
- [API — SpectralQuant](../api/spectral-api)
