"""Goodness-of-fit for an inferred recombination map WITHOUT a ground-truth map.

Analog of "rescaled diversity -> TMRCA": the model-free observable for local recombination is the
decay of LD with distance (E[r^2] ~ 1/(1+rho*d), Sved). Per window we compute an LD-decay-implied
rho_LD straight from the data; the GoF is the rank-correlation between fastrho's inferred rho and
rho_LD. A good map tracks the data's own recombination signal; a wrong/flat map does not -- with no
ground truth used. We then check the GoF ranks species the same way the true-map Pearson does.

Run on sesame: PYTHONNOUSERSITE=1 venvs/fastrho/bin/python scripts/gof.py
"""
import os, glob
import numpy as np
from scipy.stats import spearmanr

HAP = "/home/kkor/realdata/hap"
MAPS = "/home/kkor/realdata/maps"
W = 100_000
# log-spaced distance bins (bp): scale-adaptive -- captures decay at 300 bp (dog) to 30 kb (human)
BINS = [(200, 500), (500, 1000), (1000, 2000), (2000, 5000), (5000, 15000), (15000, 50000)]


def ld_persistence_per_window(gm, pos):
    """Per 100kb window: 'LD persistence' = sum over log-distance bins of mean r^2.
    Scale-adaptive model-free recombination observable: high recombination -> fast LD decay
    -> low persistence in all bins. (More recombination => lower persistence.)"""
    pos = pos.astype(float); n = gm.shape[0]
    edges = np.arange(pos[0], pos[-1], W)
    starts, pers = [], []
    for e in edges:
        idx = np.where((pos >= e) & (pos < e + W))[0]
        if len(idx) > 300:
            idx = idx[np.linspace(0, len(idx) - 1, 300).astype(int)]
        if len(idx) < 40:
            starts.append(e); pers.append(np.nan); continue
        G = gm[:, idx].astype(np.float64); p = G.mean(0)
        seg = (p > 0) & (p < 1)
        G = G[:, seg]; pp = p[seg]; pi = pos[idx][seg]
        if len(pp) < 25:
            starts.append(e); pers.append(np.nan); continue
        pab = (G.T @ G) / n
        D = pab - np.outer(pp, pp)
        den = np.outer(pp * (1 - pp), pp * (1 - pp))
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = (D * D / den) - 1.0 / n      # subtract finite-sample r^2 floor (1/n)
        dist = np.abs(pi[:, None] - pi[None, :])
        vals = []
        for lo, hi in BINS:
            msk = (dist >= lo) & (dist < hi) & np.isfinite(r2)
            if msk.sum() >= 5:
                vals.append(max(float(r2[msk].mean()), 0.0))
        if len(vals) < 3:
            starts.append(e); pers.append(np.nan); continue
        pers.append(float(np.sum(vals)))     # area under the (binned) LD-decay curve
        starts.append(e)
    return np.array(starts), np.array(pers)


def gof(key):
    z = np.load(os.path.join(HAP, key + ".npz"), allow_pickle=True)
    gm = z["gm"]; pos = z["pos"]
    m = np.load(os.path.join(MAPS, key + ".npz"))
    inf_start = np.round((m["centers"] * 1e6 - W / 2) / W).astype(int)
    inf_rho = m["pred"].astype(float)
    truth_rho = m["truth"].astype(float)
    truth_p = float(m["pearson"])
    obs_start, obs_pers = ld_persistence_per_window(gm, pos)
    obs_key = {int(round(s / W)): r for s, r in zip(obs_start, obs_pers)}
    a, b, tr = [], [], []
    for s, r, t in zip(inf_start, inf_rho, truth_rho):
        o = obs_key.get(int(s))
        if o is not None and np.isfinite(o) and np.isfinite(r) and o > 0 and r > 0:
            a.append(r); b.append(o); tr.append(t)
    a = np.array(a); b = np.array(b); tr = np.array(tr)
    # more recombination -> lower LD persistence: GoF = -corr(inferred rho, persistence)
    g = -spearmanr(a, b)[0] if len(a) > 5 else np.nan
    # DIAGNOSTIC (uses truth, not part of GoF): is the observable itself a valid recomb proxy?
    obs_vs_truth = -spearmanr(b, tr)[0] if len(a) > 5 else np.nan
    return dict(key=key, gof=g, n_win=len(a), truth_pearson=truth_p, obs_vs_truth=obs_vs_truth)


if __name__ == "__main__":
    # only primary inference keys (skip derived maps like *_pyrho/*_canid/*_self2 with no hap file)
    keys = [os.path.basename(f)[:-4] for f in sorted(glob.glob(os.path.join(MAPS, "*.npz")))
            if os.path.exists(os.path.join(HAP, os.path.basename(f)[:-4] + ".npz"))]
    rows = []
    for k in keys:
        try:
            rows.append(gof(k))
        except Exception as e:
            print("skip %s: %s" % (k, e))
    print("%-8s %8s %8s %10s %6s" % ("species", "GoF", "truth_r", "obs_vs_tru", "nwin"))
    for r in rows:
        print("%-8s %8.3f %8.3f %10.3f %6d"
              % (r["key"], r["gof"], r["truth_pearson"], r["obs_vs_truth"], r["n_win"]))
    gs = [r["gof"] for r in rows if np.isfinite(r["gof"])]
    ts = [r["truth_pearson"] for r in rows if np.isfinite(r["gof"])]
    if len(gs) > 2:
        print("\nGoF-vs-truth Spearman across species: %.3f  (does GoF predict accuracy w/o truth?)"
              % spearmanr(gs, ts)[0])
    import json
    json.dump({r["key"]: r for r in rows},
              open("/home/kkor/realdata/gof_ld.json", "w"), indent=2)
