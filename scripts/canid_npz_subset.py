"""Subset a canid genotype archive while preserving diploid sample pairs."""

from __future__ import annotations

import argparse

import numpy as np


def read_ids(path: str) -> list[str]:
    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("source_ids")
    parser.add_argument("subset_ids")
    parser.add_argument("output")
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    source_ids = read_ids(args.source_ids)
    subset_ids = read_ids(args.subset_ids)
    archive = np.load(args.source, allow_pickle=True)
    matrix = archive["gm"]
    if matrix.shape[0] != 2 * len(source_ids):
        raise ValueError("source ID count does not match diploid genotype rows")
    lookup = {sample: index for index, sample in enumerate(source_ids)}
    missing = [sample for sample in subset_ids if sample not in lookup]
    if missing:
        raise ValueError(f"subset samples absent from source IDs: {missing}")

    rows = np.ravel([[2 * lookup[sample], 2 * lookup[sample] + 1]
                     for sample in subset_ids])
    values = {name: archive[name] for name in archive.files
              if name not in {"gm", "n_ind", "pop", "sample_ids"}}
    np.savez(
        args.output,
        gm=matrix[rows],
        n_ind=len(subset_ids),
        pop=args.key,
        sample_ids=np.asarray(subset_ids),
        **values,
    )
    print(f"{args.key}: {matrix[rows].shape} -> {args.output}")


if __name__ == "__main__":
    main()
