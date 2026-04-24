#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate R-Sparse search files for new target sparsities by reusing the "
            "per-module alpha allocation from an existing search file and replacing "
            "the per-module target sparsity entries."
        )
    )
    parser.add_argument(
        "--source_search_file",
        type=Path,
        required=True,
        help="Existing search file containing interleaved (alpha, s) pairs.",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        required=True,
        help="Target sparsities to generate, e.g. 0.25 0.40 0.65",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to place generated search files.",
    )
    parser.add_argument(
        "--prefix",
        default="llama31_sparsity",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def validate_source(values, source_path: Path):
    if values.ndim != 1 or values.size % 2 != 0:
        raise ValueError(f"Unexpected search file shape in {source_path}: {values.shape}")
    alpha = values[0::2]
    s = values[1::2]
    if np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError(f"Alpha values out of range [0, 1] in {source_path}")
    return alpha, s


def main():
    args = parse_args()
    values = np.loadtxt(args.source_search_file, dtype=np.float64)
    alpha, source_s = validate_source(values, args.source_search_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {args.source_search_file}")
    print(f"Modules: {alpha.size}")
    print(f"Source target sparsity values: {sorted(set(np.round(source_s, 6).tolist()))}")
    print(f"Alpha mean={alpha.mean():.6f}, min={alpha.min():.6f}, max={alpha.max():.6f}")

    for target in args.targets:
        if not (0.0 < target < 1.0):
            raise ValueError(f"Target sparsity must be in (0, 1), got {target}")
        out = np.empty_like(values)
        out[0::2] = alpha
        out[1::2] = target
        tag = int(round(target * 100))
        out_path = args.output_dir / f"{args.prefix}_{tag:02d}_heuristic_from50_alpha_search.npy"
        np.savetxt(out_path, out, fmt="%.18e")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
