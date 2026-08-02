import json

import numpy as np
import pytest

from fastrho.benchmark import score_prediction_archive
from fastrho.evaluate import absolute_rate_view


def _write_shard(path, truth):
    np.savez(
        path,
        interval_target=np.asarray(truth, dtype=np.float32),
        meta=json.dumps({"Ne": 10_000}),
    )


def test_absolute_rate_view_is_conditional_on_ne():
    pred = {
        "rho_per_bp": np.array([4e-4, 8e-4]),
        "rho_ci_lo": np.array([2e-4, 4e-4]),
        "rho_ci_hi": np.array([6e-4, 12e-4]),
    }
    view = absolute_rate_view(pred, Ne=10_000)
    assert np.allclose(view["r_per_bp"], [1e-8, 2e-8])
    assert view["Ne_used"] == 10_000
    with pytest.raises(ValueError, match="positive"):
        absolute_rate_view(pred, Ne=0)


def test_external_prediction_archive_uses_identical_shards(tmp_path):
    truth_a = np.array([1e-8, 2e-8, 3e-8])
    truth_b = np.array([2e-8, 3e-8, 4e-8])
    _write_shard(tmp_path / "ts_00000000.npz", truth_a)
    _write_shard(tmp_path / "ts_00000001.npz", truth_b)
    archive = tmp_path / "external.npz"
    np.savez(archive, ts_00000000=truth_a, ts_00000001=truth_b)
    result = score_prediction_archive(str(archive), str(tmp_path))
    assert result["status"] == "scored"
    assert result["n_shards"] == 2
    assert result["pearson"] == pytest.approx(1.0)


def test_external_prediction_archive_rejects_missing_or_misaligned(tmp_path):
    _write_shard(tmp_path / "ts_00000000.npz", [1e-8, 2e-8])
    missing = tmp_path / "missing.npz"
    np.savez(missing, another_region=np.array([1e-8, 2e-8]))
    with pytest.raises(KeyError, match="no prediction"):
        score_prediction_archive(str(missing), str(tmp_path))

    wrong = tmp_path / "wrong.npz"
    np.savez(wrong, ts_00000000=np.array([1e-8]))
    with pytest.raises(ValueError, match="shape mismatch"):
        score_prediction_archive(str(wrong), str(tmp_path))
