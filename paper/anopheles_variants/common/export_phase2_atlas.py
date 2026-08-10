#!/usr/bin/env python3
"""Export Phase 2 maps to release BED files and a checksum-bound cohort manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ("2R", "2L", "3R", "3L", "X")
WINDOW = 50_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selected-samples", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    bed_dir = args.out / "bed"
    bed_dir.mkdir(parents=True, exist_ok=True)

    with args.selection.open(encoding="utf-8", newline="") as handle:
        selections = list(csv.DictReader(handle, delimiter="\t"))
    with args.selected_samples.open(encoding="utf-8", newline="") as handle:
        sample_rows = list(csv.DictReader(handle, delimiter="\t"))
    with args.metadata.open(encoding="utf-8", newline="") as handle:
        metadata = {row["ox_code"]: row for row in csv.DictReader(handle, delimiter="\t")}
    samples_by_cohort: dict[str, list[str]] = {}
    for row in sample_rows:
        samples_by_cohort.setdefault(row["cohort"], []).append(row["sample_id"])

    manifest_rows = []
    for selection in selections:
        cohort = selection["cohort"]
        n_snp = 0
        n_hap = 0
        ne = []
        bed_rows = []
        map_hashes = {}
        for arm in ARMS:
            path = args.maps / f"{cohort}__{arm}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            map_hashes[arm] = sha256(path)
            with np.load(path, allow_pickle=True) as data:
                starts = np.asarray(data[f"starts_{WINDOW}"], dtype=float)
                rates = np.asarray(data[f"r_{WINDOW}"], dtype=float)
                rhos = np.asarray(data[f"rho_{WINDOW}"], dtype=float)
                n_snp += int(data["n_snp"])
                n_hap = int(data["n_hap"])
                ne.append(float(data["Ne_est"]))
            ends = np.r_[starts[1:], starts[-1] + WINDOW]
            for start, end, rate, rho in zip(starts, ends, rates, rhos):
                if np.isfinite(rate):
                    bed_rows.append(
                        f"{arm}\t{int(start)}\t{int(end)}\t{rate:.6e}\t{rate * 1e8:.4f}\t{rho:.6e}"
                    )
        bed = bed_dir / f"{cohort}.bed"
        bed.write_text(
            "# chrom\tstart\tend\trate_per_bp\tcM_per_Mb\trho_per_bp\n"
            + "\n".join(bed_rows)
            + "\n",
            encoding="utf-8",
        )
        selected = samples_by_cohort[cohort]
        latitudes = [float(metadata[sample]["latitude"]) for sample in selected]
        longitudes = [float(metadata[sample]["longitude"]) for sample in selected]
        manifest_rows.append(
            {
                "cohort": cohort,
                "release_population": selection["release_population"],
                "species": selection["species"],
                "country": selection["country"],
                "lat": f"{np.mean(latitudes):.6f}",
                "lon": f"{np.mean(longitudes):.6f}",
                "n_hap": n_hap,
                "n_snp": n_snp,
                "Ne_est": f"{np.mean(ne):.0f}",
                "win_bp": WINDOW,
                "bed_sha256": sha256(bed),
                "map_set_sha256": hashlib.sha256(
                    "".join(f"{arm}:{map_hashes[arm]}\n" for arm in ARMS).encode()
                ).hexdigest(),
            }
        )

    fields = [
        "cohort", "release_population", "species", "country", "lat", "lon",
        "n_hap", "n_snp", "Ne_est", "win_bp", "bed_sha256", "map_set_sha256",
    ]
    manifest = args.out / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)
    provenance = {
        "schema_version": 1,
        "variant": "phase2",
        "release": "Ag1000G Phase 2 AR1",
        "reference_assembly": "AgamP4",
        "cohort_count": len(manifest_rows),
        "arms": list(ARMS),
        "window_bp": WINDOW,
        "selection_sha256": sha256(args.selection),
        "selected_samples_sha256": sha256(args.selected_samples),
        "metadata_sha256": sha256(args.metadata),
        "manifest_sha256": sha256(manifest),
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
