from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from fastrho.evaluate import score_rates

ROOT = Path(__file__).parents[1]
DESIGN = json.loads((ROOT / "research/demography_matched/design.json").read_text())


def load_script(name: str):
    path = ROOT / "scripts" / name
    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_design_freezes_paired_demographic_arms() -> None:
    assert DESIGN["evaluation_data"]["configurations"] == [
        "bottleneck_n20",
        "expansion_n20",
    ]
    assert DESIGN["ReLERNN"]["arms"] == ["constant", "matched"]
    assert DESIGN["pyrho"]["arms"] == ["constant", "matched"]
    assert DESIGN["endpoints"]["primary"].endswith("at 25 kb")
    assert DESIGN["analysis_rules"][-1] == "Use Betty Slurm allocations for all Betty computation."


def test_matched_history_files_equal_frozen_events() -> None:
    expected = {
        "bottleneck": [(0, 10000), (1000, 1000), (3000, 10000)],
        "expansion": [(0, 10000), (2000, 1000)],
    }
    for scenario, rows in expected.items():
        path = ROOT / f"research/demography_matched/demographies/{scenario}_smcpp.csv"
        observed = []
        for line in path.read_text().splitlines()[1:]:
            _, time, size = line.split(",")
            observed.append((int(time), int(size)))
        assert observed == rows


def test_auto_uprtr_rule() -> None:
    module = load_script("run_relernn_paired.py")
    assert module.math.ceil(1.15 * 1.5e-7 / 1.5e-8) == 12


def test_auto_uprtr_uses_physical_length_weighting(tmp_path: Path) -> None:
    module = load_script("run_relernn_paired.py")
    np.savez(
        tmp_path / "region_000.npz",
        map_position=np.array([0, 999_000, 1_000_000]),
        map_rate=np.array([1.0e-8, 2.0e-7]),
    )
    assert module.auto_uprtr(tmp_path, 1.0e-8) == 2


def test_explicit_uprtr_bypasses_analysis_specific_rule(tmp_path: Path) -> None:
    module = load_script("run_relernn_paired.py")
    value, source = module.resolve_uprtr(tmp_path, 1.0e-8, 1.0)
    assert value == 1.0
    assert source == "explicit_cli"


def test_native_window_scoring_averages_truth_on_prediction_intervals(tmp_path: Path) -> None:
    module = load_script("score_relernn_native.py")
    position = np.array([0, 25, 100], dtype=float)
    rate = np.array([1.0, 3.0], dtype=float)
    observed = module.mean_rate_between(position, rate, np.array([0, 50, 100]))
    assert observed == pytest.approx([2.0, 3.0])


def test_combine_vcfs_rejects_different_sample_panels(tmp_path: Path) -> None:
    module = load_script("run_relernn_paired.py")
    template = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr{index},length=100>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample}\n"
        "chr{index}\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0|1\n"
    )
    (tmp_path / "region_000.vcf").write_text(template.format(index=1, sample="S1"))
    (tmp_path / "region_001.vcf").write_text(template.format(index=2, sample="S2"))
    with pytest.raises(ValueError, match="sample columns differ"):
        module.combine_vcfs(tmp_path)


def test_collator_reports_relernn_native_resolution(tmp_path: Path) -> None:
    module = load_script("collate_demography_benchmark.py")
    path = (
        tmp_path
        / "arms"
        / "bottleneck_relernn_constant"
        / "relernn_project"
        / "networks"
        / "windowSizes.txt"
    )
    path.parent.mkdir(parents=True)
    path.write_text("chr1\t20\t123000\nchr2\t20\t189000\nchr3\t20\t241000\n")
    assert module.relernn_native_windows(tmp_path, "bottleneck_relernn_constant") == {
        "minimum": 123000,
        "median": 189000,
        "maximum": 241000,
        "n_regions": 3,
    }


def _write_collator_fixture(root: Path) -> tuple[Path, Path]:
    design = root / "design.json"
    design.write_text("{}\n")
    (root / "input_manifest.json").write_text("{}\n")
    (root / "results").mkdir()
    truth = 1.0e-8 * np.array(
        [1.0, 2.5, 1.5, 4.0, 2.0, 5.0, 3.0, 6.0, 3.5, 1.2, 4.5, 2.2, 5.5, 1.8, 6.5, 2.8]
    )
    for scenario_index, scenario in enumerate(("bottleneck", "expansion")):
        reference_arm = root / "arms" / f"{scenario}_pyrho_constant"
        reference_arm.mkdir(parents=True)
        (reference_arm / "config.json").write_text(json.dumps({"seq_len": 400_000}))
        np.savez(
            reference_arm / "region_000.npz",
            map_position=np.arange(0, 400_001, 25_000),
            map_rate=truth,
        )
        for method_index, method in enumerate(("relernn", "pyrho")):
            for history_index, history in enumerate(("constant", "matched")):
                value = 0.1 * (1 + scenario_index + method_index + history_index)
                record = {
                    "scales": {
                        scale: {
                            method: {
                                "pearson": value,
                                "spearman": value - 0.01,
                                "bias_ratio": 0.8 + value,
                                "n": 8 if scale == "25kb" else 2,
                            }
                        }
                        for scale in ("25kb", "100kb")
                    }
                }
                name = f"{scenario}_{method}_{history}"
                (root / "results" / f"{name}.json").write_text(json.dumps(record))
                arm = root / "arms" / name
                arm.mkdir(parents=True, exist_ok=True)
                prediction = truth * (1.0 + 0.1 * history_index)
                prediction = prediction.copy()
                prediction[history_index] = 0.0
                np.savez(arm / f"pred_{method}.npz", region_000=prediction)
                if method == "relernn":
                    windows = arm / "relernn_project" / "networks"
                    windows.mkdir(parents=True)
                    (windows / "windowSizes.txt").write_text("chr1\t20\t125000\nchr2\t20\t225000\n")
    reference = root / "fastrho_reference.json"
    reference.write_text(
        json.dumps(
            {
                "scenarios": {
                    scenario: {"25kb": {"pearson": 0.75}, "100kb": {"pearson": 0.85}}
                    for scenario in ("bottleneck", "expansion")
                }
            }
        )
    )
    return design, reference


def test_collator_reports_all_arms_and_exact_paired_differences(tmp_path: Path) -> None:
    module = load_script("collate_demography_benchmark.py")
    design, reference = _write_collator_fixture(tmp_path)
    observed = module.collate(tmp_path, design, reference)
    assert observed["design_sha256"] == hashlib.sha256(design.read_bytes()).hexdigest()
    assert (
        observed["fastrho_reference_sha256"] == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    for scenario in ("bottleneck", "expansion"):
        assert observed["scenarios"][scenario]["fastrho_reference"]["25kb"]["pearson"] == 0.75
        for method in ("relernn", "pyrho"):
            record = observed["scenarios"][scenario][method]
            assert set(record["arms"]) == {"constant", "matched"}
            assert record["arms"]["constant"]["25kb"]["n"] == 14
            assert record["arms"]["matched"]["25kb"]["n"] == 14
            assert record["matched_minus_constant"]["25kb"]["bias_ratio"] == pytest.approx(
                0.1
            )
        assert (
            observed["scenarios"][scenario]["relernn"]["arms"]["matched"]["native_window_bp"][
                "median"
            ]
            == 175_000
        )


def test_collator_ignores_stage_counts_and_rescores_joint_support(tmp_path: Path) -> None:
    module = load_script("collate_demography_benchmark.py")
    design, reference = _write_collator_fixture(tmp_path)
    path = tmp_path / "results" / "bottleneck_pyrho_matched.json"
    record = json.loads(path.read_text())
    record["scales"]["25kb"]["pyrho"]["n"] = 7
    path.write_text(json.dumps(record))
    observed = module.collate(tmp_path, design, reference)
    arms = observed["scenarios"]["bottleneck"]["pyrho"]["arms"]
    assert arms["constant"]["25kb"]["n"] == arms["matched"]["25kb"]["n"] == 14


def test_prediction_archive_keeps_every_arm_on_common_grid(tmp_path: Path) -> None:
    module = load_script("archive_demography_predictions.py")
    root = tmp_path / "run"
    expected = np.linspace(1.0e-8, 2.0e-8, 80)
    for scenario in ("bottleneck", "expansion"):
        reference = root / "arms" / f"{scenario}_pyrho_constant"
        reference.mkdir(parents=True)
        (reference / "config.json").write_text(json.dumps({"seq_len": 2_000_000}))
        np.savez(
            reference / "region_000.npz",
            map_position=np.arange(0, 2_000_001, 25_000),
            map_rate=expected,
        )
        for method in ("relernn", "pyrho"):
            for history in ("constant", "matched"):
                arm = root / "arms" / f"{scenario}_{method}_{history}"
                arm.mkdir(parents=True, exist_ok=True)
                np.savez(arm / f"pred_{method}.npz", region_000=expected)
        fixed = root / "reference" / f"{scenario}_pred_fastrho.npz"
        fixed.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fixed, region_000=expected)

    (root / "fastrho_reference.json").write_text(
        json.dumps(
            {
                "scenarios": {
                    scenario: {
                        "file": f"reference/{scenario}_pred_fastrho.npz",
                        "sha256": hashlib.sha256(
                            (root / "reference" / f"{scenario}_pred_fastrho.npz").read_bytes()
                        ).hexdigest(),
                    }
                    for scenario in ("bottleneck", "expansion")
                }
            }
        )
    )

    output = root / "results" / "paired_demography_predictions.npz"
    metadata = module.archive(root, output)
    expected_competitors = {
        f"{scenario}_{method}_{history}"
        for scenario in ("bottleneck", "expansion")
        for method in ("relernn", "pyrho")
        for history in ("constant", "matched")
    }
    assert set(metadata["arms"]) == expected_competitors | {
        "bottleneck_fastrho_fixed",
        "expansion_fastrho_fixed",
    }
    with np.load(output) as archived:
        assert len(archived.files) == 13
        np.testing.assert_allclose(archived["truth__bottleneck__region_000"], expected)
        np.testing.assert_allclose(
            archived["pred__expansion_relernn_matched__region_000"], expected
        )
    exporter = load_script("export_paper_data.py")
    exporter.DEMOGRAPHY_PREDICTIONS = output
    columns, rows = exporter._demography_matched()
    assert columns[-2:] == ("true_rate_per_bp", "predicted_rate_per_bp")
    assert len(rows) == 10 * 80
    assert {row[1] for row in rows} == {"fastrho", "pyrho", "relernn"}


def test_committed_demography_diagnostics_are_exact_and_interpretable() -> None:
    module = load_script("demography_matched_diagnostics.py")
    observed = module.build()
    committed = json.loads(
        (ROOT / "paper/results_snapshot/demography_matched_diagnostics.json").read_text()
    )
    assert observed == committed
    assert observed["float_rounding_digits"] == module.FLOAT_DIGITS

    benchmark = json.loads(
        (ROOT / "paper/results_snapshot/demography_matched.json").read_text()
    )
    for scenario in ("bottleneck", "expansion"):
        for method in ("relernn", "pyrho"):
            diagnostic = observed["scenarios"][scenario][method]
            expected = benchmark["scenarios"][scenario][method]
            for history in ("constant", "matched"):
                assert diagnostic[history]["pearson"] == pytest.approx(
                    expected["arms"][history]["25kb"]["pearson"]
                )
                assert diagnostic[history]["spearman"] == pytest.approx(
                    expected["arms"][history]["25kb"]["spearman"]
                )
                assert diagnostic[history]["median_estimated_to_true"] == pytest.approx(
                    expected["arms"][history]["25kb"]["bias_ratio"]
                )

    expansion = observed["scenarios"]["expansion"]["pyrho"]
    interval = expansion["pearson_delta_region_bootstrap"]
    assert interval["lower_95"] < 0 < interval["upper_95"]
    assert expansion["matched_minus_constant"]["spearman"] > 0
    assert abs(1 - expansion["matched"]["median_estimated_to_true"]) < abs(
        1 - expansion["constant"]["median_estimated_to_true"]
    )

    def assert_stable_floats(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values():
                assert_stable_floats(child)
        elif isinstance(value, float):
            assert value == round(value, module.FLOAT_DIGITS)

    assert_stable_floats(observed)


def test_fixed_fastrho_reference_is_hash_checked_and_rescored(tmp_path: Path) -> None:
    module = load_script("score_fastrho_reference.py")
    root = tmp_path / "run"
    true_rate = np.linspace(1.0e-8, 2.0e-8, 80)
    scenarios = {}
    for scenario in ("bottleneck", "expansion"):
        reference = root / "arms" / f"{scenario}_pyrho_constant"
        reference.mkdir(parents=True)
        (reference / "config.json").write_text(json.dumps({"seq_len": 2_000_000}))
        np.savez(
            reference / "region_000.npz",
            map_position=np.arange(0, 2_000_001, 25_000),
            map_rate=true_rate,
        )
        prediction = root / "reference" / f"{scenario}_pred_fastrho.npz"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prediction, region_000=0.8 * true_rate)
        scenarios[scenario] = {
            "file": str(prediction.relative_to(root)),
            "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        }
    manifest = root / "fastrho_reference.json"
    manifest.write_text(json.dumps({"scenarios": scenarios}))

    result = module.score(root, manifest)
    for scenario in ("bottleneck", "expansion"):
        assert result["scenarios"][scenario]["n_regions"] == 1
        assert result["scenarios"][scenario]["25kb"]["n"] == 80
        assert np.isclose(result["scenarios"][scenario]["25kb"]["pearson"], 1)
        assert np.isclose(result["scenarios"][scenario]["25kb"]["bias_ratio"], 0.8)


def test_fixed_fastrho_reference_rejects_a_modified_prediction(tmp_path: Path) -> None:
    module = load_script("score_fastrho_reference.py")
    root = tmp_path / "run"
    scenarios = {}
    for scenario in ("bottleneck", "expansion"):
        reference = root / "arms" / f"{scenario}_pyrho_constant"
        reference.mkdir(parents=True)
        (reference / "config.json").write_text(json.dumps({"seq_len": 50_000}))
        np.savez(
            reference / "region_000.npz",
            map_position=np.array([0, 25_000, 50_000]),
            map_rate=np.array([1.0e-8, 2.0e-8]),
        )
        prediction = root / "reference" / f"{scenario}_pred_fastrho.npz"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        np.savez(prediction, region_000=np.array([1.0e-8, 2.0e-8]))
        scenarios[scenario] = {
            "file": str(prediction.relative_to(root)),
            "sha256": hashlib.sha256(prediction.read_bytes()).hexdigest(),
        }
    manifest = root / "fastrho_reference.json"
    manifest.write_text(json.dumps({"scenarios": scenarios}))
    with (root / "reference" / "bottleneck_pred_fastrho.npz").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        module.score(root, manifest)


def test_relernn_compatibility_patch_matches_design() -> None:
    path = ROOT / DESIGN["ReLERNN"]["compatibility_patch"]
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == DESIGN["ReLERNN"]["compatibility_patch_sha256"]
    )
    text = path.read_text()
    assert text.count('modelSave = os.path.join(networkDir,"model.keras")') == 2
    assert text.count('modelSave = os.path.join(networkDir,"model.weights.h5")') == 2
    assert "save_weights_only=True" in text
    assert "model.save_weights(network)" in text
    assert "model.load_weights(network)" in text


def test_frozen_input_archive_matches_design() -> None:
    path = ROOT / DESIGN["evaluation_data"]["archive"]
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == DESIGN["evaluation_data"]["archive_sha256"]
    )
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
    for scenario in ("bottleneck_n20", "expansion_n20"):
        assert f"{scenario}/config.json" in names
        assert f"{scenario}/genome.bed" in names
        for region in range(24):
            stem = f"{scenario}/region_{region:03d}"
            assert {f"{stem}.vcf", f"{stem}.npz", f"{stem}.trees"} <= names


def test_frozen_archive_prepares_byte_identical_pairs(tmp_path: Path) -> None:
    root = tmp_path / "run"
    input_root = root / "input"
    input_root.mkdir(parents=True)
    archive_path = ROOT / DESIGN["evaluation_data"]["archive"]
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(input_root)
    module = load_script("prepare_demography_benchmark.py")
    manifest = module.prepare(root)
    assert {
        scenario: manifest["scenarios"][scenario]["dropped_vcf_record_count"]
        for scenario in ("bottleneck", "expansion")
    } == {"bottleneck": 14, "expansion": 6}
    for scenario in ("bottleneck", "expansion"):
        recorded = manifest["scenarios"][scenario]["validated_checksums"]["region_000.vcf"]
        for method in ("relernn", "pyrho"):
            paths = [
                root / "arms" / f"{scenario}_{method}_{history}" / "region_000.vcf"
                for history in ("constant", "matched")
            ]
            assert all(hashlib.sha256(path.read_bytes()).hexdigest() == recorded for path in paths)
            assert paths[0].stat().st_ino == paths[1].stat().st_ino
            for path in paths:
                for line in path.read_text().splitlines():
                    if line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    assert module.invalid_gt_reason(fields) is None


def test_slurm_jobs_guard_cluster_execution() -> None:
    common = (ROOT / "research/demography_matched/slurm/common.sh").read_text()
    assert "SLURM_JOB_ID" in common
    for path in (ROOT / "research/demography_matched/slurm").glob("*.sbatch"):
        text = path.read_text()
        assert "#SBATCH" in text
        assert "common.sh" in text
    submit = (ROOT / "research/demography_matched/slurm/submit.sh").read_text()
    for stage in range(8):
        assert f"/{stage:02d}_" in submit
    training = (
        ROOT / "research/demography_matched/slurm/05_relernn_train_score.sbatch"
    ).read_text()
    assert "#SBATCH --exclude=dgx015" in training


def _pool_archived_scale(
    archive: np.lib.npyio.NpzFile,
    scenario: str,
    arm: str,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    truth_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    truth_prefix = f"truth__{scenario}__"
    for truth_key in sorted(key for key in archive.files if key.startswith(truth_prefix)):
        region = truth_key.removeprefix(truth_prefix)
        truth = np.asarray(archive[truth_key], dtype=float)
        prediction = np.asarray(archive[f"pred__{arm}__{region}"], dtype=float)
        assert truth.shape == prediction.shape
        assert truth.size % block_size == 0
        assert np.all(np.isfinite(truth)) and np.all(np.isfinite(prediction))
        truth_parts.append(truth.reshape(-1, block_size).mean(axis=1))
        prediction_parts.append(prediction.reshape(-1, block_size).mean(axis=1))
    assert len(truth_parts) == 24
    return np.concatenate(prediction_parts), np.concatenate(truth_parts)


def test_archived_predictions_exactly_rederive_every_reported_paired_metric() -> None:
    """Make the manuscript's paired benchmark independent of its summary JSON."""

    result_path = ROOT / "paper/results_snapshot/demography_matched.json"
    archive_path = ROOT / "paper/figdata/demography_matched_predictions.npz"
    assert result_path.exists(), "retrieve the completed cross-map-blind Slurm result"
    assert archive_path.exists(), "retrieve the completed cross-map-blind prediction archive"
    result = json.loads(result_path.read_text())
    with np.load(archive_path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["_metadata"]))
        assert metadata["grid_bp"] == 25_000
        assert len(metadata["arms"]) == 10
        for scenario in ("bottleneck", "expansion"):
            fastrho_arm = f"{scenario}_fastrho_fixed"
            assert metadata["arms"][fastrho_arm]["n_regions"] == 24
            assert metadata["arms"][fastrho_arm]["n_windows"] == 1_920
            for scale, block_size in (("25kb", 1), ("100kb", 4)):
                predicted, truth = _pool_archived_scale(
                    archive, scenario, fastrho_arm, block_size
                )
                observed = score_rates(predicted, truth)
                reported = result["scenarios"][scenario]["fastrho_reference"][scale]
                assert reported["n"] == observed["n"]
                for metric in ("pearson", "spearman", "bias_ratio"):
                    assert reported[metric] == pytest.approx(
                        observed[metric], rel=1e-12, abs=1e-12
                    )

            for method in ("relernn", "pyrho"):
                arm_names = {
                    history: f"{scenario}_{method}_{history}" for history in ("constant", "matched")
                }
                for arm in arm_names.values():
                    assert metadata["arms"][arm]["n_regions"] == 24
                    assert metadata["arms"][arm]["n_windows"] == 1_920
                for scale, block_size in (("25kb", 1), ("100kb", 4)):
                    values = {
                        history: _pool_archived_scale(archive, scenario, arm, block_size)
                        for history, arm in arm_names.items()
                    }
                    truth = values["constant"][1]
                    np.testing.assert_allclose(values["matched"][1], truth)
                    joint = np.isfinite(truth) & (truth > 0)
                    for predicted, _truth in values.values():
                        joint &= np.isfinite(predicted) & (predicted > 0)
                    for history, (predicted, _truth) in values.items():
                        observed = score_rates(predicted[joint], truth[joint])
                        reported = result["scenarios"][scenario][method]["arms"][history][scale]
                        assert reported["n"] == observed["n"]
                        for metric in ("pearson", "spearman", "bias_ratio"):
                            assert reported[metric] == pytest.approx(
                                observed[metric], rel=1e-12, abs=1e-12
                            )
