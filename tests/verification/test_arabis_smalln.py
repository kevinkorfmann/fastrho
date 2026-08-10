from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def test_freeze_uses_exact_metric_not_rounded_checkpoint_name(tmp_path: Path) -> None:
    root = tmp_path / "campaign" / "seeds" / "seed0"
    checkpoints = root / "logs" / "fastrho" / "version_0" / "checkpoints"
    checkpoints.mkdir(parents=True)
    for epoch in (1, 2):
        (checkpoints / f"epoch={epoch}-val_pearson=0.800.ckpt").write_bytes(bytes([epoch]))
    with (checkpoints.parent / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_pearson"])
        writer.writeheader()
        writer.writerows([
            {"epoch": 1, "val_pearson": 0.8004},
            {"epoch": 2, "val_pearson": 0.7996},
        ])
    shards = root / "shards"
    shards.mkdir()
    np.savez(shards / "feat_stats.npz", feat_mean=np.arange(3), label=np.array("hap"))
    output = tmp_path / "frozen.json"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/arabis_smalln_freeze.py"),
         "--campaign", str(tmp_path / "campaign"), "--seeds", "1", "--output", str(output)],
        check=True,
    )
    record = json.loads(output.read_text())["members"][0]
    assert "epoch=1-" in record["checkpoint"]
    assert record["validation_pearson"] == 0.8004


def test_ensemble_averages_blind_members(tmp_path: Path) -> None:
    input_root = tmp_path / "maps"
    panels = ("nemorensis", "sagittata", "nemorensis_no_cross_parent",
              "sagittata_no_cross_parent")
    for seed in range(5):
        directory = input_root / f"seed{seed}"
        directory.mkdir(parents=True)
        for panel in panels:
            for chrom in range(1, 9):
                np.savez(
                    directory / f"{panel}.chr{chrom}.npz",
                    pos_left=np.array([0.0, 1.0]),
                    pos_right=np.array([1.0, 2.0]),
                    rho_per_bp=np.full(2, seed + 1.0),
                    log_rho=np.full(2, np.log(seed + 1.0)),
                    sigma_log_rho=np.full(2, 0.1),
                    Ne_estimated=np.array(100.0 + seed),
                    Ne_used=np.array(100.0 + seed),
                    n_accessions=np.array(12),
                    n_snps=np.array(2),
                    checkpoint_sha256=np.array(str(seed)),
                    stats_sha256=np.array(str(seed)),
                )
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"frozen": true}\n')
    output = tmp_path / "ensemble"
    subprocess.run(
        [sys.executable, str(REPO / "scripts/arabis_ensemble_maps.py"),
         "--input-root", str(input_root), "--manifest", str(manifest),
         "--output-dir", str(output)],
        check=True,
    )
    with np.load(output / "sagittata.chr3.npz") as result:
        expected_rate = float(np.exp(np.mean(np.log(np.arange(1.0, 6.0)))))
        np.testing.assert_allclose(result["rho_per_bp"], expected_rate)
        expected_ne = float(np.exp(np.mean(np.log(np.arange(100.0, 105.0)))))
        np.testing.assert_allclose(result["Ne_estimated"], expected_ne)
        assert int(result["ensemble_members"]) == 5
        assert str(result["checkpoint_sha256"]).startswith("ensemble_manifest:")
