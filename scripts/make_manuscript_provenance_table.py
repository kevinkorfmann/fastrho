"""Generate the supplemental data-source table from the provenance ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "paper" / "data_provenance.yaml"
OUTPUT = ROOT / "paper" / "manuscript" / "generated" / "data_sources_table.tex"
PHASE2_OUTPUT = ROOT / "paper" / "anopheles_variants" / "phase2" / "manuscript" / "generated" / "data_sources_table.tex"
COHORT_OUTPUT = ROOT / "paper" / "manuscript" / "generated" / "cohort_table.tex"


def escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def main() -> None:
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    rows = [
        r"\begin{longtable}{@{}p{0.27\textwidth}p{0.12\textwidth}p{0.13\textwidth}p{0.19\textwidth}p{0.21\textwidth}@{}}",
        r"\caption{Datasets and simulation resources used in the study. Versions and stable access routes are reported for each source.}\label{tab:data-sources}\\",
        r"\toprule Source and version & Manuscript scope & Organism & Stable source & Citations \\",
        r"\midrule\endfirsthead",
        r"\toprule Source and version & Manuscript scope & Organism & Stable source & Citations \\",
        r"\midrule\endhead",
    ]
    for source in ledger["datasets"]:
        organisms = ", ".join(source["organisms"])
        citations = ",".join(source["citation_keys"])
        version = source["version"]
        if "/" in version and len(version) > 55:
            version = "versions listed in the cited sources"
        version = version.replace("qualified snapshot", "qualified set")
        label = escape(f"{source['name']} ({version})")
        repository = escape(source["repository"])
        url = source["accession_or_url"]
        rows.append(
            f"{label} & {escape(source['manuscript_scope'])} & {escape(organisms)} & "
            f"\\href{{{url}}}{{{repository}}} & \\citep{{{citations}}} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{longtable}", ""])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(rows)
    OUTPUT.write_text(payload, encoding="utf-8")
    PHASE2_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    PHASE2_OUTPUT.write_text(payload, encoding="utf-8")

    validation = json.loads(
        (ROOT / "paper" / "anopheles_variants" / "phase2" / "results" / "phase2_2la.json").read_text(encoding="utf-8")
    )
    full_cohort = {row["pop"]: row for row in validation["rows"]}
    cohort_rows = [
        r"\begin{longtable}{@{}lllrrrrr@{}}",
        r"\caption{Ag1000G Phase 2 AR1 populations included in the active atlas. Map panels contained 80 haplotypes from 40 diploid mosquitoes. $H_{40}$ is the expected 2La heterokaryotype frequency in that map panel; $n_{\mathrm{full}}$ and $H_{\mathrm{full}}$ use all eligible released Phase 2 tag-SNP calls.}\label{tab:cohorts}\\",
        r"\toprule Cohort & Species & Country & Haplotypes & SNPs & $H_{40}$ & $n_{\mathrm{full}}$ & $H_{\mathrm{full}}$ \\",
        r"\midrule\endfirsthead",
        r"\toprule Cohort & Species & Country & Haplotypes & SNPs & $H_{40}$ & $n_{\mathrm{full}}$ & $H_{\mathrm{full}}$ \\",
        r"\midrule\endhead",
    ]
    with (ROOT / "paper" / "anopheles_variants" / "phase2" / "release" / "atlas_anopheles" / "manifest.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            species = r"\textit{A. " + row["species"].split()[-1] + "}"
            full = full_cohort[row["cohort"]]
            cohort_rows.append(
                f"{escape(row['cohort'])} & {species} & {escape(row['country'])} & "
                f"{int(row['n_hap']):,} & {int(row['n_snp']):,} & "
                f"{float(full['panel_het_expected']):.3f} & {int(full['n_samples']):,} & "
                f"{float(full['het_expected']):.3f} \\\\"
            )
    cohort_rows.extend([r"\bottomrule", r"\end{longtable}", ""])
    COHORT_OUTPUT.write_text("\n".join(cohort_rows), encoding="utf-8")


if __name__ == "__main__":
    main()
