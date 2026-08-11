"""Create a deterministic, cross-map-blind Arabis panel resampling plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sheet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.sample_sheet.open(newline=""), delimiter="\t"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    plan = []

    nem = sorted((r["accession"] for r in rows if r["species"] == "nemorensis"), key=int)
    for omitted in nem:
        name = f"nemorensis_loo_{omitted}"
        chosen = [x for x in nem if x != omitted]
        (args.output_dir / f"{name}.samples").write_text("\n".join(chosen) + "\n")
        plan.append((name, "nemorensis", "leave_one_out", omitted, *chosen))

    groups = {
        pop: sorted((r["accession"] for r in rows
                     if r["species"] == "sagittata" and r["real_population"] == pop), key=int)
        for pop in ("Rhine", "Lob", "Adl-1")
    }
    seen: set[tuple[str, ...]] = set()
    while len(seen) < 12:
        chosen = tuple(sorted(
            [*rng.choice(groups["Rhine"], 7, replace=False),
             *rng.choice(groups["Lob"], 3, replace=False),
             *rng.choice(groups["Adl-1"], 2, replace=False)], key=int))
        if chosen in seen:
            continue
        seen.add(chosen)
        index = len(seen) - 1
        name = f"sagittata_n12_{index:02d}"
        (args.output_dir / f"{name}.samples").write_text("\n".join(chosen) + "\n")
        plan.append((name, "sagittata", "stratified_n12", "", *chosen))

    manifest = args.output_dir / "panels.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("panel", "species", "design", "omitted", "accessions"))
        for name, species, design, omitted, *chosen in plan:
            writer.writerow((name, species, design, omitted, ",".join(chosen)))
    print(f"panels={len(plan)} manifest={manifest}")


if __name__ == "__main__":
    main()
