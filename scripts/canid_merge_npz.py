"""Merge nonoverlapping chromosome chunks produced by canid VCF extraction."""

from __future__ import annotations

import argparse

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    archives = [np.load(path, allow_pickle=True) for path in args.inputs]
    first = archives[0]
    for archive in archives[1:]:
        if archive["gm"].shape[0] != first["gm"].shape[0]:
            raise ValueError("genotype row counts differ among chunks")
        if not np.array_equal(archive["sample_ids"], first["sample_ids"]):
            raise ValueError("sample order differs among chunks")
    order = np.argsort([int(archive["pos"][0]) for archive in archives])
    positions = np.concatenate([archives[index]["pos"] for index in order])
    if np.any(np.diff(positions) <= 0):
        raise ValueError("chunk positions overlap or are not strictly increasing")
    matrix = np.concatenate([archives[index]["gm"] for index in order], axis=1)
    values = {name: first[name] for name in first.files
              if name not in {"gm", "pos", "pop"}}
    np.savez(args.output, gm=matrix, pos=positions, pop=args.key, **values)
    print(f"{args.key}: {matrix.shape} -> {args.output}")


if __name__ == "__main__":
    main()
