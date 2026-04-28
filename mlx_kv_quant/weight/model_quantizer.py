from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_kv_quant.weight.quantized_linear import QuantizedLinear


def quantize_model(
    model: nn.Module,
    bits: int = 4,
    use_hadamard: bool = True,
    skip_embeddings: bool = True,
    seed: int = 42,
) -> nn.Module:
    """Replace all nn.Linear layers in a model with QuantizedLinear.

    Walks the module tree, replaces each Linear with a QuantizedLinear,
    copies the pretrained weights, and compresses them in-place.

    Args:
        model: Any mlx.nn.Module (e.g. a loaded mlx-lm model).
        bits: Bit-width for compression (2, 3, or 4).
        use_hadamard: Use Metal-accelerated Hadamard rotation (recommended).
        skip_embeddings: Skip nn.Embedding layers (they have different structure
            and are often a large fraction of small-model params — see blog note).
        seed: Base random seed. Each layer gets seed + layer_index for independence.

    Returns:
        The same model with Linear layers replaced (in-place mutation + return).

    Example::

        import mlx_lm
        model, tokenizer = mlx_lm.load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        model = quantize_model(model, bits=3, use_hadamard=True)
    """
    _replace_linears(model, bits=bits, use_hadamard=use_hadamard, seed=seed, counter=[0])
    mx.eval(model.parameters())
    return model


def _replace_linears(
    module: nn.Module,
    bits: int,
    use_hadamard: bool,
    seed: int,
    counter: list,
) -> None:
    """Recursively walk module tree and replace Linear layers in-place."""
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue

        # Skip the embedding projection if it happens to be an nn.Linear
        # (some architectures use tied embeddings implemented as Linear)
        if _is_embedding_like(name):
            continue

        weight = child.weight    # (out, in) — mlx-lm convention
        bias = getattr(child, "bias", None)

        layer_seed = seed + counter[0]
        counter[0] += 1

        q_layer = QuantizedLinear(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
            bits=bits,
            use_hadamard=use_hadamard,
            bias=bias is not None,
            seed=layer_seed,
        )
        q_layer.quantize_weights(weight, bias=bias)

        # Replace the child in the parent module
        _set_nested_attr(module, name, q_layer)


def _is_embedding_like(name: str) -> bool:
    """Heuristic: skip layers whose name suggests they are embedding projections."""
    lower = name.lower()
    return any(kw in lower for kw in ("embed", "lm_head", "tok_emb"))


def _set_nested_attr(root: nn.Module, dotted_name: str, value: nn.Module) -> None:
    """Set a nested attribute on a module given a dotted path like 'layers.0.mlp.gate'."""
    parts = dotted_name.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def compression_report(model: nn.Module) -> dict:
    """Summarise memory savings across all QuantizedLinear layers.

    Args:
        model: Model after quantize_model() has been called.

    Returns:
        Dict with total_compressed_bytes, total_fp16_bytes, ratio, n_layers.
    """
    total_compressed = 0
    total_fp16 = 0
    n_layers = 0

    for _, child in model.named_modules():
        if isinstance(child, QuantizedLinear):
            total_compressed += child.memory_bytes
            total_fp16 += child.fp16_bytes
            n_layers += 1

    ratio = total_fp16 / total_compressed if total_compressed > 0 else 0.0
    return {
        "n_layers": n_layers,
        "total_compressed_mb": total_compressed / 1024 ** 2,
        "total_fp16_mb": total_fp16 / 1024 ** 2,
        "compression_ratio": ratio,
    }
