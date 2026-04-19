"""Monkey-patches an mlx-lm model's attention layers to use a quantized KV cache.

Usage::

    from mlx_lm import load
    from mlx_kv_quant.integration.mlx_lm_patch import patch_model_kv_cache
    from mlx_kv_quant.cache import KVCacheConfig

    model, tokenizer = load("mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    config = KVCacheConfig(method="turboquant_prod", head_dim=128, bit_width_inlier=2)
    patch_model_kv_cache(model, config)
"""
from __future__ import annotations

from typing import Any

from mlx_kv_quant.cache.base import KVCacheConfig, KVCacheFactory


def patch_model_kv_cache(model: Any, config: KVCacheConfig) -> None:
    """Replace each attention layer's KV cache with a quantized implementation.

    This function monkey-patches the model in-place. It iterates over all
    sub-modules that have a ``cache`` attribute (the mlx-lm convention) and
    replaces them with a new KVCache instance built from config.

    After patching, the model's KV cache is automatically used during the
    next forward pass.

    Args:
        model: An mlx-lm model object with ``.layers`` attribute containing
               attention sub-modules.
        config: KVCacheConfig describing the quantization scheme.

    Raises:
        AttributeError: If the model does not have the expected structure.

    Example::

        config = KVCacheConfig(
            method="turboquant_prod",
            head_dim=128,
            bit_width_inlier=2,
            seed=42,
        )
        patch_model_kv_cache(model, config)
    """
    n_patched = 0

    def _patch_module(module: Any) -> None:
        nonlocal n_patched
        # Patch any sub-module that has a 'cache' attribute
        if hasattr(module, "cache"):
            module.cache = KVCacheFactory.create(config)
            n_patched += 1
        # Recurse into children
        for child in _get_children(module):
            _patch_module(child)

    def _get_children(module: Any) -> list:
        """Return iterable of child sub-modules."""
        children = []
        if hasattr(module, "layers"):
            children.extend(module.layers)
        if hasattr(module, "self_attn"):
            children.append(module.self_attn)
        if hasattr(module, "attention"):
            children.append(module.attention)
        return children

    _patch_module(model)

    if n_patched == 0:
        import warnings
        warnings.warn(
            "patch_model_kv_cache: no attention layers with a 'cache' attribute "
            "were found. The model may not follow the mlx-lm convention or may "
            "not yet have been called (cache is created lazily). "
            "Consider calling the model once before patching.",
            stacklevel=2,
        )
    else:
        print(f"[mlx_kv_quant] Patched {n_patched} attention cache(s) with {config.method!r}.")
