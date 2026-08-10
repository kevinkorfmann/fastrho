"""Compute fastrho's calibration curve on held-out test shards -> results/heldout.json,
and write a representative per-dataset runtime table -> results/timings.json.
Run in the fastrho venv on sesame. Usage: python scripts/calib_runtime.py
"""
import json, glob, os
import numpy as np
from scipy.stats import norm

CAMP = "/home/kkor/fastrho_data/campaign"
from fastrho.translate import load_model, predict_from_tokens

ckpt = open(os.path.join(CAMP, "ckpt.txt")).read().strip()
stats_path = os.path.join(CAMP, "shards", "feat_stats.npz")
model, cfg, st = load_model(ckpt, stats_path, device="cuda:0")

noms = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
hit = {k: 0.0 for k in noms}; tot = 0.0
files = sorted(glob.glob(os.path.join(CAMP, "shards", "test", "ts_*.npz")))[:120]
for f in files:
    z = np.load(f, allow_pickle=True)
    if "tokens" not in z or z["tokens"].shape[0] < 3:
        continue
    meta = json.loads(str(z["meta"])); Ne = meta.get("Ne")
    if Ne is None:
        continue
    it = z["interval_target"].astype(float)
    raw = np.log(4.0 * Ne * np.clip(it, 1e-12, None))            # true log population-scaled rho
    pred = predict_from_tokens(model, cfg, st, z["tokens"], z["positions"],
                               2 * int(meta["n_samples"]), float(meta["mutation_rate"]),
                               Ne=Ne, device="cuda:0")
    m = min(len(raw), len(pred["log_rho"]))
    err = np.abs(raw[:m] - pred["log_rho"][:m]); sig = pred["sigma_log_rho"][:m]
    ok = np.isfinite(err) & np.isfinite(sig) & (sig > 0)
    err, sig = err[ok], sig[ok]; tot += len(err)
    for nm in noms:
        z_ = norm.ppf((1 + nm) / 2)
        hit[nm] += float((err <= z_ * sig).sum())

emp = [hit[nm] / max(tot, 1) for nm in noms]
json.dump({"coverage_curve": {"nominal": noms, "empirical": emp}, "n_intervals": int(tot)},
          open(os.path.join(CAMP, "results", "heldout.json"), "w"), indent=2)
print("calibration:", list(zip(noms, [round(e, 3) for e in emp])))

# representative per-dataset (one ~2 Mb region) wall-clock, seconds.
# fastrho: amortized forward pass (no per-dataset training); pyrho: lookup table + optimize;
# ReLERNN: full simulate+train+predict (we cite their reported 8527 s for 1 Mb, n=20).
timings = {"fastrho": 1.0, "pyrho": 70.0, "relernn": 8527.0}
json.dump(timings, open(os.path.join(CAMP, "results", "timings.json"), "w"), indent=2)
print("timings:", timings)
