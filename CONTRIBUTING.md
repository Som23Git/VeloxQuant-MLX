# Contributing to VeloxQuant-MLX

Thanks for your interest in contributing. VeloxQuant-MLX is a KV-cache
quantization library for `mlx_lm` on Apple Silicon, and contributions of all
kinds are welcome: bug reports, new quantization methods, benchmarks on
additional models, documentation, and performance work.

## Reporting bugs and requesting features

Please open an issue at
<https://github.com/rajveer43/veloxquant-mlx/issues>. For bugs, include:

- your hardware (chip + RAM) and macOS version,
- `python`, `mlx`, and `mlx_lm` versions,
- the model id you were running,
- a minimal snippet that reproduces the problem, and
- the full error output.

## Getting set up

Requires Apple Silicon (M1 or later), Python ≥ 3.11, and MLX ≥ 0.18.

```bash
git clone https://github.com/rajveer43/veloxquant-mlx
cd VeloxQuant-MLX
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running the tests

```bash
python -m pytest veloxquant_mlx/tests -q
```

New code should come with tests that mirror the existing conventions in
`veloxquant_mlx/tests/` (shape/dtype preservation, reconstruction-quality
bounds on seeded synthetic data, and — for any Metal kernel — a parity test
against the pure-MLX path). Seed all randomness: determinism is treated as a
correctness requirement.

## Submitting changes

1. Fork the repository and create a feature branch.
2. Make your change with accompanying tests and documentation.
3. Ensure the full test suite passes locally.
4. Open a pull request describing the change and, for any performance or
   compression claim, the committed `results.json` it traces to. We do not
   merge numbers that are not reproducible from a script in the repository.

## Adding a new quantization method

A new method typically consists of:

- a `Quantizer` subclass in `veloxquant_mlx/quantizers/`, registered with
  `QuantizerRegistry`,
- an `mlx_lm`-compatible cache wrapper in `veloxquant_mlx/cache/` with byte
  accounting,
- wiring into `KVCacheConfig` / `KVCacheFactory`,
- tests, and
- a benchmark script emitting `figures/<method>/<model>/results.json`.

See `paper/NEW_METHOD_SURVEY.md` for an example of how a method is scoped and
chosen before implementation.

## Commit conventions

Use short imperative subject lines (`Add sink cache`, `Fix dtype mismatch in
KIVI path`). Reference the relevant issue number when one exists (`Closes #12`).
Avoid commits that mix unrelated changes.

## Code of conduct

Please be respectful and constructive. We follow the
[Contributor Covenant v2.1](CODE_OF_CONDUCT.md).
