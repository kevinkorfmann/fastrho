"""Extract figure data for the linked-selection robustness figure (runs on sesame, fastrho venv).

For each regime (neutral/bgs/sweep): mean nucleotide diversity pi(x) across the 2 Mb region,
averaged over the 24 SLiM replicates -- this shows selection's spatial footprint (BGS lowers pi
uniformly; the hard sweep, always introduced at the region centre, carves a valley there).

Also extracts a representative hard-sweep region: its true recombination map and the frozen model's
inferred map, to show the landscape is recovered right through the selection valley.

Writes /home/kkor/fastrho/paper/figdata/selection_figdata.npz
"""
import os
import glob

import numpy as np
import tskit
from scipy.stats import pearsonr

import sys
sys.path.insert(0, "/home/kkor/fastrho")
from fastrho.preprocess import mean_rate_between

ROOT = "/home/kkor/fastrho_data/slim"
OUT = "/home/kkor/fastrho/paper/figdata/selection_figdata.npz"
L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2.0


def regime_pi(mode):
    pis = []
    for tp in sorted(glob.glob(os.path.join(ROOT, mode, "region_*.trees"))):
        ts = tskit.load(tp)
        pis.append(ts.diversity(windows=EDGES))   # per-bp pi per window
    return np.mean(pis, 0)


def all_windows(mode):
    """Stack (truth, fastrho) per 25 kb window across all regions of a regime."""
    pred = np.load(os.path.join(ROOT, mode, "pred_fastrho.npz"))
    T, F = [], []
    for npz in sorted(glob.glob(os.path.join(ROOT, mode, "region_*.npz"))):
        name = os.path.basename(npz)[:-4]
        if name not in pred.files:
            continue
        z = np.load(npz, allow_pickle=True)
        truth = mean_rate_between(z["map_position"], z["map_rate"], EDGES)
        fr = pred[name]
        n = min(len(truth), len(fr))
        T.append(truth[:n])
        F.append(fr[:n])
    return np.array(T), np.array(F)


def recovery_vs_distance(mode):
    """Per-window |log| agreement binned by distance from the swept site (region centre).
    Returns (distance_bin_centres_Mb, mean Pearson of log-rate within each distance bin)."""
    T, F = all_windows(mode)                       # (n_regions, n_windows)
    dist = np.abs(CENTRES - L / 2.0)               # distance of each window from centre
    bins = np.linspace(0, L / 2.0, 9)
    bc, corr = [], []
    for k in range(len(bins) - 1):
        cols = np.where((dist >= bins[k]) & (dist < bins[k + 1]))[0]
        if cols.size == 0:
            continue
        t = np.log(T[:, cols].ravel() + 1e-12)
        f = np.log(F[:, cols].ravel() + 1e-12)
        bc.append((bins[k] + bins[k + 1]) / 2.0 / 1e6)
        corr.append(pearsonr(t, f)[0])
    return np.array(bc), np.array(corr)


def hero_sweep():
    """A well-recovered hotspot sweep region (top quartile) for the illustrative map track."""
    pred = np.load(os.path.join(ROOT, "sweep", "pred_fastrho.npz"))
    cand = {}
    for npz in sorted(glob.glob(os.path.join(ROOT, "sweep", "region_*.npz"))):
        name = os.path.basename(npz)[:-4]
        if int(name.split("_")[1]) % 2 == 0 or name not in pred.files:
            continue
        z = np.load(npz, allow_pickle=True)
        truth = mean_rate_between(z["map_position"], z["map_rate"], EDGES)
        fr = pred[name]
        n = min(len(truth), len(fr))
        cand[name] = (truth[:n], fr[:n], pearsonr(truth[:n], fr[:n])[0])
    q75 = np.quantile([v[2] for v in cand.values()], 0.75)
    name = min(cand, key=lambda k: abs(cand[k][2] - q75))
    return name, cand[name]


def main():
    pi_n = regime_pi("neutral")
    pi_b = regime_pi("bgs")
    pi_s = regime_pi("sweep")
    db_n, rc_n = recovery_vs_distance("neutral")
    db_s, rc_s = recovery_vs_distance("sweep")
    name, (htruth, hfast, hpear) = hero_sweep()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, centres=CENTRES, edges=EDGES,
             pi_neutral=pi_n, pi_bgs=pi_b, pi_sweep=pi_s,
             dist_bins=db_s, recov_sweep=rc_s, recov_neutral=rc_n,
             hero_truth=htruth, hero_fastrho=hfast, hero_pearson=float(hpear),
             hero_name=name, sweep_pos=L / 2.0)
    print("hero sweep region:", name, "Pearson=%.3f" % hpear)
    print("pi neutral/bgs/sweep (genome mean): %.3e / %.3e / %.3e"
          % (pi_n.mean(), pi_b.mean(), pi_s.mean()))
    print("pi at sweep centre (neutral/sweep): %.3e / %.3e"
          % (pi_n[len(pi_n) // 2], pi_s[len(pi_s) // 2]))
    print("recovery vs distance (sweep):", np.round(rc_s, 2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
