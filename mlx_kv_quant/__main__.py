"""Entry point for `python -m mlx_kv_quant <command>`."""
from __future__ import annotations

import sys


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: veloxquant {precompute|benchmark}")
        sys.exit(1)

    command = sys.argv[1]
    # Remove the subcommand so sub-parsers see argv correctly
    sys.argv = [f"mlx_kv_quant {command}"] + sys.argv[2:]

    if command == "precompute":
        from mlx_kv_quant.cli.precompute import main as _main
        _main()
    elif command == "benchmark":
        from mlx_kv_quant.cli.benchmark import main as _main
        _main()
    else:
        print(f"Unknown command: {command!r}. Choices: precompute, benchmark")
        sys.exit(1)


if __name__ == "__main__":
    main()
