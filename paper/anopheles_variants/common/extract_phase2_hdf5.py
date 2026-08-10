#!/usr/bin/env python3
"""Extract fixed Phase 2 panels from released arm-level phased HDF5 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


def find_dataset(handle: h5py.File, candidates: tuple[str, ...]) -> h5py.Dataset:
    found: list[h5py.Dataset] = []

    def visit(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and any(name.endswith(candidate) for candidate in candidates):
            found.append(obj)

    handle.visititems(visit)
    if len(found) != 1:
        names = [dataset.name for dataset in found]
        raise ValueError(f"expected one dataset ending in {candidates}, found {names}")
    return found[0]


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--arm", choices=("2R", "2L", "3R", "3L", "X"), required=True)
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-variants", type=int, default=100_000)
    parser.add_argument("--only-cohort")
    args = parser.parse_args()

    panels: dict[str, list[str]] = {}
    with args.selected_samples.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            panels.setdefault(row["cohort"], []).append(row["sample_id"])
    if args.only_cohort:
        panels = {args.only_cohort: panels[args.only_cohort]}
    args.out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.source, "r") as source:
        samples_ds = find_dataset(source, ("/samples",))
        pos_ds = find_dataset(source, ("/variants/POS", "/variants/position"))
        gt_ds = find_dataset(source, ("/calldata/genotype", "/calldata/GT"))
        samples = decode(samples_ds[:])
        sample_index = {sample: index for index, sample in enumerate(samples)}
        if len(sample_index) != len(samples):
            raise ValueError("duplicate sample identifiers in released HDF5")
        if gt_ds.ndim != 3 or gt_ds.shape[2] != 2:
            raise ValueError(f"unexpected genotype shape {gt_ds.shape}")

        for cohort, selected in sorted(panels.items()):
            missing = sorted(set(selected) - set(sample_index))
            if missing:
                raise ValueError(f"{cohort}/{args.arm}: missing released samples {missing[:5]}")
            indices = np.asarray([sample_index[sample] for sample in selected], dtype=int)
            if not np.all(indices[:-1] <= indices[1:]):
                order = np.argsort(indices)
                indices = indices[order]
                selected = [selected[int(index)] for index in order]

            target = args.out / f"{cohort}__{args.arm}.h5"
            with h5py.File(target, "w") as output:
                gm_out = output.create_dataset(
                    "gm",
                    shape=(2 * len(selected), 0),
                    maxshape=(2 * len(selected), None),
                    chunks=(2 * len(selected), min(args.chunk_variants, 100_000)),
                    compression="gzip",
                    compression_opts=1,
                    dtype="i1",
                )
                pos_out = output.create_dataset(
                    "pos",
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(min(args.chunk_variants, 100_000),),
                    compression="gzip",
                    compression_opts=1,
                    dtype="i8",
                )
                written = 0
                for start in range(0, gt_ds.shape[0], args.chunk_variants):
                    stop = min(start + args.chunk_variants, gt_ds.shape[0])
                    genotype = np.asarray(gt_ds[start:stop, indices, :], dtype=np.int8)
                    matrix = genotype.transpose(1, 2, 0).reshape(2 * len(selected), stop - start)
                    allele_sum = matrix.sum(axis=0)
                    keep = (matrix >= 0).all(axis=0) & (allele_sum > 0) & (allele_sum < matrix.shape[0])
                    if not np.any(keep):
                        continue
                    positions = np.asarray(pos_ds[start:stop], dtype=np.int64)[keep]
                    kept = matrix[:, keep]
                    next_written = written + kept.shape[1]
                    gm_out.resize((gm_out.shape[0], next_written))
                    pos_out.resize((next_written,))
                    gm_out[:, written:next_written] = kept
                    pos_out[written:next_written] = positions
                    written = next_written
                output.create_dataset("sample_id", data=np.asarray(selected, dtype="S"))
                output.attrs.update(
                    cohort=cohort,
                    arm=args.arm,
                    release="Ag1000G Phase 2 AR1",
                    reference_assembly="AgamP4",
                    n_diploid=len(selected),
                    n_hap=2 * len(selected),
                    n_snp=written,
                    source=str(args.source),
                )
            if written == 0:
                raise ValueError(f"{cohort}/{args.arm}: no segregating complete SNPs")
            record = {
                "cohort": cohort,
                "arm": args.arm,
                "n_diploid": len(selected),
                "n_hap": 2 * len(selected),
                "n_snp": written,
                "source_sha256": sha256(args.source),
                "output_sha256": sha256(target),
            }
            target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
            print(f"{cohort} {args.arm}: {written:,} SNPs -> {target}")


if __name__ == "__main__":
    main()
