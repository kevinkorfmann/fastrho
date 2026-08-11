#!/usr/bin/env python3
"""Run the CPU documentation examples from an isolated installed environment."""

from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
import tempfile
from pathlib import Path

import msprime
import numpy as np


def run(*arguments: object, cwd: Path) -> str:
    result = subprocess.run(
        [str(argument) for argument in arguments],
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def write_example_vcf(path: Path) -> None:
    content = "\n".join(
        [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=100000>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts0\ts1",
            "chr1\t101\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\t0|0",
            "chr1\t501\t.\tC\tT\t.\tPASS\t.\tGT\t1|1\t0|1",
            "chr1\t901\t.\tG\tA\t.\tPASS\t.\tGT\t0|.\t0|1",
        ]
    )
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(content + "\n")


def smoke_package(work: Path) -> None:
    import fastrho

    vcf = work / "cohort.vcf.gz"
    write_example_vcf(vcf)
    assert fastrho.vcf_contigs(vcf) == ["chr1"]
    gm, positions, metadata = fastrho.read_vcf(
        vcf,
        contig="chr1",
        missing="drop-site",
        return_metadata=True,
    )
    assert gm.shape == (4, 2)
    assert positions.tolist() == [100.0, 500.0]
    assert metadata == {"contig": "chr1", "phased": True, "dropped_missing_sites": 1}

    prediction = {
        "pos_left": np.array([100.0, 500.0]),
        "pos_right": np.array([500.0, 900.0]),
        "rho_per_bp": np.array([4e-4, 8e-4]),
        "r_per_bp": np.array([1e-8, 2e-8]),
        "r_ci_lo": np.array([0.5e-8, 1.0e-8]),
        "r_ci_hi": np.array([1.5e-8, 3.0e-8]),
        "Ne_used": 10_000.0,
        "Ne_estimated": 12_000.0,
        "r_interval_is_conditional_on_Ne": True,
    }
    frame = fastrho.to_dataframe(prediction, chrom="chr1")
    frame["cM_per_Mb"] = frame["r_per_bp"] * 1e8
    assert frame["cM_per_Mb"].tolist() == [1.0, 2.0]
    assert frame.attrs["Ne_used"] == 10_000.0

    rate_map = msprime.RateMap(
        position=[0, 700_000, 950_000, 1_300_000, 2_000_000],
        rate=[1e-8, 8e-8, 1e-8, 3e-8],
    )
    ancestry = msprime.sim_ancestry(
        samples=50,
        ploidy=2,
        population_size=10_000,
        sequence_length=2_000_000,
        recombination_rate=rate_map,
        random_seed=17,
    )
    mutated = msprime.sim_mutations(
        ancestry,
        rate=1.5e-8,
        model=msprime.BinaryMutationModel(),
        random_seed=18,
    )
    assert mutated.num_sites > 0


def smoke_manuscript_presets(repository: Path, work: Path) -> None:
    data_script = repository / "examples" / "manuscript_species" / "data.py"
    infer_script = repository / "examples" / "manuscript_species" / "infer.py"
    listed = run(sys.executable, data_script, "list", cwd=work)
    assert "anopheles_gambiae" in listed
    shown = run(sys.executable, data_script, "show", "human", cwd=work)
    assert "Homo sapiens" in shown and "1.29e-08" in shown
    download = run(
        sys.executable,
        data_script,
        "download",
        "human",
        "--dry-run",
        "--include-companions",
        cwd=work,
    )
    assert "human.chr2.vcf.gz" in download and "human.phase3.panel.tsv" in download
    prepared = run(
        sys.executable,
        data_script,
        "prepare",
        "human",
        "--vcf",
        "raw.vcf.gz",
        "--out",
        "prepared.vcf.gz",
        "--dry-run",
        cwd=work,
    )
    assert "bcftools view" in prepared and 'GT="mis"' in prepared
    inferred = run(
        sys.executable,
        infer_script,
        "--species",
        "human",
        "--vcf",
        "cohort.vcf.gz",
        "--checkpoint",
        "model.ckpt",
        "--stats",
        "feat_stats.npz",
        "--out",
        "map.bed",
        "--dry-run",
        cwd=work,
    )
    assert "unpolarized" in inferred and "100000 bp" in inferred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    args = parser.parse_args()
    repository = args.repo_root.resolve()
    with tempfile.TemporaryDirectory(prefix="fastrho-examples-") as directory:
        work = Path(directory)
        smoke_package(work)
        smoke_manuscript_presets(repository, work)
    print("fresh-install CPU examples passed")
    print("CUDA inference examples require a compatible NVIDIA runner and released model bundle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
