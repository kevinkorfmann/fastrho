#!/usr/bin/env python3
"""Freeze Phase 2 cohort eligibility and deterministic 40-diploid panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

PRIMARY = {
    "AOcol": ("colu_AO", "Anopheles coluzzii", "Angola"),
    "BFcol": ("colu_BF", "Anopheles coluzzii", "Burkina Faso"),
    "BFgam": ("gamb_BF", "Anopheles gambiae", "Burkina Faso"),
    "CIcol": ("colu_CI", "Anopheles coluzzii", "Cote d'Ivoire"),
    "CMgam": ("gamb_CM", "Anopheles gambiae", "Cameroon"),
    "GAgam": ("gamb_GA", "Anopheles gambiae", "Gabon"),
    "GHcol": ("colu_GH", "Anopheles coluzzii", "Ghana"),
    "GNgam": ("gamb_GN", "Anopheles gambiae", "Guinea"),
    "UGgam": ("gamb_UG", "Anopheles gambiae", "Uganda"),
}
UNCERTAIN = {
    "GM": "Far-west samples have uncertain species status in the Phase 2 resource paper.",
    "GW": "Far-west samples have uncertain species status in the Phase 2 resource paper.",
    "KE": "Kenyan samples have uncertain species status in the Phase 2 resource paper.",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--eligible-samples",
        type=Path,
        help="Optional newline-delimited sample IDs present on every requested chromosome arm.",
    )
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    with args.metadata.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    eligible = None
    if args.eligible_samples:
        eligible = {
            line.strip() for line in args.eligible_samples.read_text().splitlines() if line.strip()
        }
    by_population: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_population.setdefault(row["population"], []).append(row)

    audit = []
    selections = []
    selected_samples = []
    for population in sorted(by_population):
        members = sorted(by_population[population], key=lambda row: row["ox_code"])
        count = len(members)
        eligible_members = members if eligible is None else [
            row for row in members if row["ox_code"] in eligible
        ]
        eligible_count = len(eligible_members)
        primary = population in PRIMARY and eligible_count >= args.n
        if population in PRIMARY:
            cohort, species, country = PRIMARY[population]
            reason = "eligible established species-population cohort" if primary else f"fewer than {args.n} arm-complete samples"
        else:
            cohort = ""
            species = "uncertain" if population in UNCERTAIN else "not selected"
            country = members[0]["country"]
            reason = UNCERTAIN.get(population, "not a prespecified primary species-population cohort")
        audit.append(
            {
                "release_population": population,
                "cohort": cohort,
                "species": species,
                "country": country,
                "n_release": count,
                "n_arm_complete": eligible_count,
                "primary_eligible": str(primary).lower(),
                "reason": reason,
            }
        )
        if not primary:
            continue
        sample_ids = [row["ox_code"] for row in eligible_members]
        if len(sample_ids) > args.n:
            rng = np.random.default_rng(args.seed)
            indices = np.sort(rng.choice(len(sample_ids), args.n, replace=False))
            sample_ids = [sample_ids[int(index)] for index in indices]
        if len(sample_ids) != args.n or len(set(sample_ids)) != args.n:
            raise ValueError(f"invalid frozen panel for {population}")
        selections.append(
            {
                "cohort": cohort,
                "release_population": population,
                "species": species,
                "country": country,
                "n_diploid": len(sample_ids),
                "selection_seed": args.seed,
                "sample_set_sha256": hashlib.sha256(("\n".join(sample_ids) + "\n").encode()).hexdigest(),
            }
        )
        selected_samples.extend(
            {
                "cohort": cohort,
                "release_population": population,
                "sample_id": sample_id,
                "selection_order": order,
            }
            for order, sample_id in enumerate(sample_ids)
        )

    if len(selections) != len(PRIMARY):
        raise ValueError(f"expected {len(PRIMARY)} primary cohorts, found {len(selections)}")
    if Counter(row["cohort"] for row in selected_samples) != Counter({value[0]: args.n for value in PRIMARY.values()}):
        raise ValueError("selected-sample counts do not match the frozen design")

    write_tsv(
        args.out / "cohort_audit.tsv",
        audit,
        ["release_population", "cohort", "species", "country", "n_release", "n_arm_complete", "primary_eligible", "reason"],
    )
    write_tsv(
        args.out / "selection.tsv",
        selections,
        ["cohort", "release_population", "species", "country", "n_diploid", "selection_seed", "sample_set_sha256"],
    )
    write_tsv(
        args.out / "selected_samples.tsv",
        selected_samples,
        ["cohort", "release_population", "sample_id", "selection_order"],
    )
    provenance = {
        "schema_version": 2,
        "source": str(args.metadata),
        "source_sha256": digest(args.metadata),
        "eligible_samples_source": str(args.eligible_samples) if args.eligible_samples else None,
        "eligible_samples_sha256": digest(args.eligible_samples) if args.eligible_samples else None,
        "eligibility_rule": "sample is present in all five released chromosome-arm HDF5 files" if args.eligible_samples else "metadata membership only",
        "selection_seed": args.seed,
        "diploids_per_cohort": args.n,
        "primary_cohort_count": len(selections),
        "primary_species": ["Anopheles gambiae", "Anopheles coluzzii"],
        "excluded_uncertain_populations": sorted(UNCERTAIN),
    }
    (args.out / "cohort_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(args.out / "selection.tsv")


if __name__ == "__main__":
    main()
