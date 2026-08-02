import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
RegionTokenDataset = pytest.importorskip("fastrho.dataset").RegionTokenDataset


def _write_shard(path, length, n_features=3, n_haplotypes=None):
    meta = {"Ne": 10_000, "mutation_rate": 1.5e-8, "n_samples": 10}
    if n_haplotypes is not None:
        meta["n_haplotypes"] = n_haplotypes
    np.savez(
        path,
        tokens=np.arange(length * n_features, dtype=np.float32).reshape(length, n_features),
        interval_target=np.full(max(0, length - 1), 1e-8, dtype=np.float32),
        meta=json.dumps(meta),
    )


def test_validation_crops_cover_the_tail(tmp_path):
    _write_shard(tmp_path / "ts_00000000.npz", length=10)
    ds = RegionTokenDataset(str(tmp_path), context_len=4, train=False)
    assert ds._eval_items == [(0, 0), (0, 4), (0, 6)]
    assert len(ds) == 3
    tail = ds[2]["tokens"].numpy()
    assert tail[0, 0] == 6 * 3


def test_explicit_haplotype_count_overrides_diploid_sample_fallback(tmp_path):
    _write_shard(tmp_path / "ts_00000000.npz", length=4, n_haplotypes=7)
    ds = RegionTokenDataset(str(tmp_path), context_len=4, train=False)
    assert np.isclose(ds[0]["cond"].numpy()[1], np.log10(7))
