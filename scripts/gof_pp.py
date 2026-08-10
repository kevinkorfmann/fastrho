"""Posterior-predictive goodness-of-fit for an inferred recombination map (no ground truth).

Idea (the principled GoF): a *good* map, when used to simulate data under the coalescent, should
reproduce the data's own recombination-sensitive signature -- the LD-decay curve r^2(d). A biased
map (wrong regime: selfer, extreme-Ne, under-fit) cannot. We simulate K replicates under fastrho's
inferred map (its inferred Ne, the data's mu, n), compute the genome-wide LD-decay curve for the
simulated and the observed data, and score the match. Validation: this score should rank species
the same way the true-map Pearson does -- WITHOUT using any ground-truth map.

Run on sesame: PYTHONNOUSERSITE=1 venvs/fastrho/bin/python scripts/gof_pp.py
"""
import os, glob
import numpy as np
import msprime
from scipy.stats import spearmanr, pearsonr

HAP = "/home/kkor/realdata/hap"
MAPS = "/home/kkor/realdata/maps"
SUBLEN = 5_000_000          # sub-region for the simulation-based check
K = 6
BINS = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])


def ld_curve(gm, pos):
    """Genome-wide mean r^2 per log-distance bin (subsampled pairs), finite-n corrected."""
    pos = np.asarray(pos, float); n = gm.shape[0]
    sums = np.zeros(len(BINS) - 1); cnts = np.zeros(len(BINS) - 1)
    p = gm.mean(0)
    anchors = np.arange(0, len(pos), max(1, len(pos) // 4000))
    for a in anchors:
        pa = p[a]
        if pa <= 0 or pa >= 1:
            continue
        j0 = np.searchsorted(pos, pos[a] + BINS[0])
        j1 = np.searchsorted(pos, pos[a] + BINS[-1])
        if j1 <= j0:
            continue
        bs = np.arange(j0, j1, max(1, (j1 - j0) // 30))
        for b in bs:
            pb = p[b]
            if pb <= 0 or pb >= 1:
                continue
            pab = float((gm[:, a] * gm[:, b]).mean())
            den = pa * (1 - pa) * pb * (1 - pb)
            if den <= 0:
                continue
            r2 = (pab - pa * pb) ** 2 / den - 1.0 / n
            d = pos[b] - pos[a]
            k = np.searchsorted(BINS, d) - 1
            if 0 <= k < len(cnts):
                sums[k] += max(r2, 0.0); cnts[k] += 1
    return sums / np.maximum(cnts, 1)


def gof(key):
    z = np.load(os.path.join(HAP, key + ".npz"), allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(float); n_hap = gm.shape[0]; mu = float(z["mu"])
    m = np.load(os.path.join(MAPS, key + ".npz"))
    truth_p = float(m["pearson"]); Ne = float(m["Ne_est"])
    lo = pos[0]; hi = lo + SUBLEN
    sel = (pos >= lo) & (pos < hi)
    obs_curve = ld_curve(gm[:, sel], pos[sel] - lo)
    # build a RateMap over the sub-region from the inferred per-window rate
    starts = m["centers"] * 1e6; rate = m["pred"].astype(float)
    ssel = (starts >= lo) & (starts < hi)
    rpos = np.r_[0.0, (starts[ssel] - lo), SUBLEN]
    rpos = np.clip(np.unique(rpos), 0, SUBLEN)
    rr = np.interp(rpos[:-1], starts[ssel] - lo, rate[ssel])
    rr = np.where(np.isfinite(rr) & (rr > 0), rr, np.nanmedian(rate[ssel]))
    rmap = msprime.RateMap(position=rpos, rate=rr)
    sims = []
    for k in range(K):
        ts = msprime.sim_ancestry(samples=n_hap, ploidy=1, population_size=Ne,
                                  recombination_rate=rmap, sequence_length=SUBLEN,
                                  random_seed=100 + k)
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=200 + k)
        g = ts.genotype_matrix().T.astype(np.int8)
        sims.append(ld_curve(g, ts.tables.sites.position))
    sim_curve = np.nanmean(sims, 0)
    ok = (obs_curve > 0) & (sim_curve > 0)
    lo_, ls = np.log10(obs_curve[ok]), np.log10(sim_curve[ok])
    rmse = float(np.sqrt(np.mean((lo_ - ls) ** 2)))
    gofscore = float(np.exp(-rmse))     # 1 = perfect LD-curve reproduction
    return dict(key=key, gof=gofscore, rmse=rmse, truth=truth_p)


if __name__ == "__main__":
    keys = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(MAPS, "*.npz")))]
    rows = []
    for k in keys:
        try:
            rows.append(gof(k))
        except Exception as e:
            print("skip", k, e)
    print("%-8s %7s %7s %8s" % ("species", "GoF", "rmse", "truth_r"))
    for r in rows:
        print("%-8s %7.3f %7.3f %8.3f" % (r["key"], r["gof"], r["rmse"], r["truth"]))
    g = [r["gof"] for r in rows]; t = [r["truth"] for r in rows]
    if len(g) > 2:
        print("\nGoF-vs-truth Spearman: %.3f  Pearson: %.3f" % (spearmanr(g, t)[0], pearsonr(g, t)[0]))
    import json
    json.dump({r["key"]: r for r in rows},
              open("/home/kkor/realdata/gof_pp.json", "w"), indent=2)
