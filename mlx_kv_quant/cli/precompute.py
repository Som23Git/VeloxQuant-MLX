"""Entry point: python -m mlx_kv_quant precompute"""
from __future__ import annotations

import argparse

from mlx_kv_quant.codebooks.precompute import precompute


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx_kv_quant precompute",
        description="Precompute rotation matrices, JL matrices, and codebooks.",
    )
    parser.add_argument("--head_dim", type=int, default=128, help="Attention head dimension.")
    parser.add_argument("--bits", type=int, nargs="+", default=[1, 2, 3, 4], help="Bit-widths.")
    parser.add_argument("--jl_dim", type=int, default=128, help="JL projection dimension.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="./artifacts/", help="Output directory.")
    args = parser.parse_args()

    precompute(
        head_dim=args.head_dim,
        bits=args.bits,
        jl_dim=args.jl_dim,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
