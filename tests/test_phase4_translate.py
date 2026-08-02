"""Phase-4/5: inference plumbing (translate) + BED writing (GPU-only; Mamba2 needs CUDA)."""

import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba2 requires CUDA")


def test_predictive_interval_is_destandardized_and_conditional_on_ne():
    from types import SimpleNamespace

    from fastrho.translate import predict_from_tokens

    class FixedModel:
        def __call__(self, tokens, cond, mask):
            length = tokens.shape[1]
            rho = torch.empty((1, length, 2), dtype=torch.float32)
            rho[..., 0] = 0.5
            rho[..., 1] = np.log(0.25)
            return {"rho": rho, "log_Ne": torch.tensor([np.log(20_000.0)])}

    stats = {
        "feat_mean": np.zeros(2, np.float32),
        "feat_std": np.ones(2, np.float32),
        "tgt_mean": np.float32(-3.0),
        "tgt_std": np.float32(2.0),
        "ne_mean": np.float32(0.0),
        "ne_std": np.float32(1.0),
    }
    prediction = predict_from_tokens(
        FixedModel(),
        SimpleNamespace(context_len=8),
        stats,
        np.zeros((3, 2), np.float32),
        np.array([10.0, 20.0, 35.0]),
        n_hap=20,
        mutation_rate=1.5e-8,
        Ne=10_000.0,
        device="cpu",
    )

    expected_log_rho = -2.0
    expected_sigma_log_rho = 1.0
    expected_r = np.exp(expected_log_rho) / (4.0 * 10_000.0)
    np.testing.assert_allclose(prediction["log_rho"], expected_log_rho)
    np.testing.assert_allclose(
        prediction["sigma_log_rho"], expected_sigma_log_rho, rtol=1e-6
    )
    np.testing.assert_allclose(prediction["r_per_bp"], expected_r)
    np.testing.assert_allclose(
        prediction["r_ci_lo"],
        np.exp(expected_log_rho - 1.96 * expected_sigma_log_rho)
        / (4.0 * 10_000.0),
    )
    np.testing.assert_allclose(
        prediction["r_ci_hi"],
        np.exp(expected_log_rho + 1.96 * expected_sigma_log_rho)
        / (4.0 * 10_000.0),
    )
    assert prediction["r_interval_is_conditional_on_Ne"] is True


@cuda
def test_predict_from_ts_and_bed(tmp_path):
    import msprime

    from fastrho.config import PRESETS
    from fastrho.features import n_features
    from fastrho.model import FastRhoModel
    from fastrho.simulate import make_recombination_map
    from fastrho.translate import predict_map_from_ts, rebin_to_windows, write_bed

    cfg = PRESETS["small"]
    model = FastRhoModel(cfg).to("cuda:0").eval()
    F = n_features()
    stats = {  # identity-ish stats (untrained model; we only check plumbing)
        "feat_mean": np.zeros(F, np.float32), "feat_std": np.ones(F, np.float32),
        "tgt_mean": np.float32(-7.0), "tgt_std": np.float32(2.0),
        "ne_mean": np.float32(9.0), "ne_std": np.float32(0.7),
    }

    L = 300_000
    rm = make_recombination_map(L, np.random.default_rng(0), kind="gp", mean_rate=1e-8)
    ts = msprime.sim_ancestry(samples=30, population_size=1e4, recombination_rate=rm,
                              sequence_length=L, random_seed=1)
    ts = msprime.sim_mutations(ts, rate=1.5e-8, random_seed=2)

    pred = predict_map_from_ts(ts, model, cfg, stats, mutation_rate=1.5e-8, device="cuda:0")
    S1 = len(pred["pos_left"])
    assert S1 > 10
    for k in ("rho_per_bp", "r_per_bp", "r_ci_lo", "r_ci_hi"):
        assert pred[k].shape == (S1,)
        assert np.isfinite(pred[k]).all()
    assert (pred["r_ci_hi"] >= pred["r_ci_lo"]).all()
    assert pred["Ne_used"] > 0

    centers, binned = rebin_to_windows(pred, 50_000, "r_per_bp")
    assert len(centers) == len(binned) and np.isfinite(binned).all()

    out = tmp_path / "map.bed"
    write_bed(pred, str(out), chrom="chr1", window_size=50_000)
    assert os.path.exists(out)
    assert sum(1 for _ in open(out)) > 1
    print(f"predict OK: {S1} intervals, Ne_used={pred['Ne_used']:.0f}, "
          f"{len(binned)} windows")
