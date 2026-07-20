from __future__ import annotations

from veloxquant_mlx.integration.mlx_lm_patch import patch_model_kv_cache
from veloxquant_mlx.integration.mlx_vlm_patch import patch_vlm_kv_cache

__all__ = ["patch_model_kv_cache", "patch_vlm_kv_cache"]
