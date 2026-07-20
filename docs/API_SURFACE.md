# API surface notes

## Canonical import

Prefer:

```python
from veloxquant_mlx import KVCacheBuilder, KVCacheConfig
```

Older docs or blog posts may mention `mlx_kv_quant`. Treat that as a legacy
name. New code and docs should use `veloxquant_mlx` only.

## CLI entry points

`pyproject.toml` registers:

- `veloxquant`
- `mlx-kv-quant` (alias)

Both call `veloxquant_mlx.__main__:main`.

Commands: `precompute`, `benchmark`, `recommend`.

## Config stability

`KVCacheConfig` carries many method-specific fields. When adding a method:

1. Add fields with defaults (do not break existing callers).
2. Wire `KVCacheFactory` / `KVCacheBuilder.for_model`.
3. Document fields on the algorithm page.
4. Avoid renaming public fields without a deprecation note in CHANGELOG.

## Dual-package cleanup (follow-up)

If any remaining `mlx_kv_quant` import shims exist, mark them deprecated in
CHANGELOG and remove after one minor release cycle with a clear migration
note on the docs site.
