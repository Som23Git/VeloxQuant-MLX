---
id: quantizers
title: Quantizers API
sidebar_label: Quantizers
slug: /api/quantizers
---

# Quantizers API

`veloxquant_mlx.quantizers`

All quantizers implement the `Quantizer` abstract base class. See [Core API](/api/core-api) for the interface definition.

---

## QuantizerFactory

```python
from veloxquant_mlx.quantizers.base import QuantizerFactory
```

### `QuantizerFactory.create`

```python
@staticmethod
def create(name: str, **kwargs) -> Quantizer
```

Create a quantizer by name. Registered names: `"turboquant_rvq"`, `"turboquant_mse"`, `"turboquant_prod"`, `"rabitq"`, `"commvq"`, `"polarquant"`, `"qjl"`, `"composite"`.

```python
quantizer = QuantizerFactory.create("turboquant_rvq", bits=1, num_residuals=2)
```

---

## TurboQuantRVQ

```python
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
```

Two-pass Residual VQ with Gaussian + Laplacian analytical codebooks.

### Constructor

```python
TurboQuantRVQ(
    bits: int = 1,
    num_residuals: int = 2,
    use_hadamard: bool = True,
    value_bits: int = 2,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `bits` | `int` | `1` | Bits per residual pass |
| `num_residuals` | `int` | `2` | Number of RVQ passes |
| `use_hadamard` | `bool` | `True` | Apply Walsh-Hadamard transform |
| `value_bits` | `int` | `2` | Value quantization bits |

### Methods

```python
def encode(self, keys: mx.array) -> EncodedVector: ...
def decode(self, encoded: EncodedVector) -> mx.array: ...
def encode_values(self, values: mx.array) -> EncodedVector: ...
def decode_values(self, encoded: EncodedVector) -> mx.array: ...
```

**`encode(keys)`**: Takes `keys` of shape `[batch, heads, seq, head_dim]`. Returns `EncodedVector` containing packed bit indices and residual codes.

**`decode(encoded)`**: Reconstructs approximate keys from `EncodedVector`. Shape: `[batch, heads, seq, head_dim]`.

```python
import mlx.core as mx
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ

q = TurboQuantRVQ(bits=1, num_residuals=2)
keys = mx.random.normal(shape=(1, 8, 512, 128))
encoded = q.encode(keys)
decoded = q.decode(encoded)
```

---

## TurboQuantMSE

```python
from veloxquant_mlx.quantizers.turboquant_mse import TurboQuantMSE
```

MSE-optimal scalar quantization via Lloyd-Max algorithm with rotation. No residual pass.

### Constructor

```python
TurboQuantMSE(bits: int = 2, use_hadamard: bool = True)
```

---

## TurboQuantProd

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProd
```

Product VQ with QJL residual correction. Combines Lloyd-Max scalar centroids with a JL sign sketch for the residual.

### Constructor

```python
TurboQuantProd(
    bits: int = 2,
    residual_sketch_dim: int = 64,
    use_hadamard: bool = True,
)
```

### TurboQuantProdAdaptive

```python
from veloxquant_mlx.quantizers.turboquant_prod import TurboQuantProdAdaptive
```

Adaptive version of `TurboQuantProd` that dynamically increases bits when observed distortion exceeds a threshold.

```python
TurboQuantProdAdaptive(
    base_bits: int = 2,
    max_bits: int = 4,
    distortion_threshold: float = 0.05,
    observer: DistortionObserver | None = None,
)
```

---

## RaBitQQuantizer

```python
from veloxquant_mlx.quantizers.rabitq import RaBitQQuantizer
```

Randomised Hadamard + 1-bit sign packing with IVF clustering.

### Constructor

```python
RaBitQQuantizer(num_clusters: int = 64, seed: int = 0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_clusters` | `int` | `64` | Number of IVF clusters |
| `seed` | `int` | `0` | Random seed for Hadamard sign matrix |

### Methods

```python
def encode(self, keys: mx.array) -> EncodedVector: ...
def decode(self, encoded: EncodedVector) -> mx.array: ...
```

`EncodedVector.indices` — packed uint32 bit fields, shape `[batch, heads, seq, head_dim // 32]`
`EncodedVector.metadata["cluster_ids"]` — int16 cluster assignments, shape `[batch, heads, seq]`

---

## CommVQQuantizer

```python
from veloxquant_mlx.quantizers.comm_vq import CommVQQuantizer
```

RoPE-commutative residual VQ.

### Constructor

```python
CommVQQuantizer(bits: int = 2, num_residuals: int = 2)
```

---

## PolarQuantizer

```python
from veloxquant_mlx.quantizers.polarquant import PolarQuantizer
```

Recursive polar coordinate decomposition.

### Constructor

```python
PolarQuantizer(norm_bits: int = 8)
```

---

## QJLQuantizer

```python
from veloxquant_mlx.quantizers.qjl import QJLQuantizer
```

Johnson-Lindenstrauss 1-bit sign sketch.

### Constructor

```python
QJLQuantizer(sketch_dim: int = 64, seed: int = 0)
```

---

## CompositeQuantizer

```python
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
```

Chains multiple quantizers in sequence. First quantizer encodes the input; each subsequent quantizer encodes the residual of the previous.

### Constructor

```python
CompositeQuantizer(quantizers: list[Quantizer])
```

```python
from veloxquant_mlx.quantizers.composite import CompositeQuantizer
from veloxquant_mlx.quantizers.turboquant_rvq import TurboQuantRVQ
from veloxquant_mlx.quantizers.qjl import QJLQuantizer

q = CompositeQuantizer([TurboQuantRVQ(bits=1), QJLQuantizer(sketch_dim=32)])
encoded = q.encode(keys)
decoded = q.decode(encoded)
```

---

## See also

- [Algorithm pages](/algorithms/overview)
- [API — Cache](/api/cache)
- [API — Core abstractions](/api/core-api)
