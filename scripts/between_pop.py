"""Between-population map-difference experiment (fills pyrho's gap: no uncertainty/noise floor).

Two populations share a broad-scale (GP) recombination backbone but differ at a fraction of
hotspots (mimicking PRDM9-A vs -C placement). One amortized fastrho model infers both maps;
we measure (a) scale-resolved between-population correlation, (b) the WITHIN-population noise
floor from two non-overlapping subsamples of the same population, and (c) differential-hotspot
detection (ROC AUC). The within-vs-between gap is the biologically real divergence net of
estimation noise — quantifiable here precisely because fastrho is fast + calibrated.

Usage: python scripts/between_pop.py --checkpoint C --stats S --out OUT.json [--diff-frac 0.5]
"""
from __future__ import annotations

import json
import argparse

import numpy as np

GRID = 25_000


def _pair_maps(L, rng, diff_frac, n_shared=6, n_diff=6, res=1000):
    import msprime
    from fastrho.simulate import _ratemap_from_grid
    n = int(np.ceil(L / res))
    # broad-scale background: AR(1) log10-rate field (shared by both pops)
    phi, sigma = 0.97, 0.5
    x = np.empty(n); x[0] = rng.standard_normal()
    s = np.sqrt(1 - phi * phi)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + s * rng.standard_normal()
    bg = 10.0 ** (np.log10(1e-8) + sigma * (x - x.mean()))

    def add_hot(base, centers):
        r = base.copy()
        for c in centers:
            w = int(rng.integers(1, 4))
            r[max(0, c - w):c + w + 1] *= 10.0 ** rng.uniform(1.0, 1.6)
        return r

    shared = rng.choice(n, n_shared, replace=False)
    n_d = int(round(n_diff * diff_frac))
    a_only = rng.choice(n, n_d, replace=False)
    b_only = rng.choice(n, n_d, replace=False)
    rate_A = add_hot(bg, list(shared) + list(a_only))
    rate_B = add_hot(bg, list(shared) + list(b_only))
    return (_ratemap_from_grid(np.clip(rate_A, 1e-10, 2e-7), res, L),
            _ratemap_from_grid(np.clip(rate_B, 1e-10, 2e-7), res, L))


def _windows(rm, L):
    from fastrho.preprocess import mean_rate_between
    edges = np.append(np.arange(0, L, GRID), L)
    return mean_rate_between(rm.position, rm.rate, edges)


def _predict_windows(model, cfg, stats, gm, pos, L, device):
    from fastrho.translate import predict_map_from_genotype_matrix
    from fastrho.preprocess import mean_rate_between
    pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                            mutation_rate=1.5e-8, Ne=1e4, device=device)
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    pos2 = np.r_[0.0, bp, L] if bp[0] > 0 else np.r_[bp, L]
    r2 = np.r_[pred["r_per_bp"][0], pred["r_per_bp"], pred["r_per_bp"][-1]]
    edges = np.append(np.arange(0, L, GRID), L)
    return mean_rate_between(pos2, r2, edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--diff-frac", type=float, default=0.5)
    ap.add_argument("--regions", type=int, default=10); ap.add_argument("--n-dip", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=2_000_000)
    args = ap.parse_args()

    import msprime
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    from fastrho.translate import load_model
    model, cfg, stats = load_model(args.checkpoint, args.stats, device=args.device)
    L = args.seq_len

    tA, tB = [], []                      # true window rates
    pA, pB, pA1, pA2 = [], [], [], []    # predicted: A, B, A-subsample1, A-subsample2
    for i in range(args.regions):
        rng = np.random.default_rng(1000 + i)
        rmA, rmB = _pair_maps(L, rng, args.diff_frac)
        tA.append(_windows(rmA, L)); tB.append(_windows(rmB, L))
        out = []
        for rm in (rmA, rmB):
            ts = msprime.sim_ancestry(samples=args.n_dip, population_size=1e4,
                                      recombination_rate=rm, sequence_length=L,
                                      random_seed=2000 + i)
            ts = msprime.sim_mutations(ts, rate=1.5e-8, random_seed=3000 + i)
            gm = ts.genotype_matrix().T.astype(np.int8)
            pos = ts.tables.sites.position
            out.append((gm, pos))
        gmA, posA = out[0]
        pA.append(_predict_windows(model, cfg, stats, gmA, posA, L, args.device))
        pB.append(_predict_windows(model, cfg, stats, *out[1], L, args.device))
        # within-population noise floor: two disjoint halves of population A
        h = gmA.shape[0] // 2
        pA1.append(_predict_windows(model, cfg, stats, gmA[:h], posA, L, args.device))
        pA2.append(_predict_windows(model, cfg, stats, gmA[h:], posA, L, args.device))

    def cat(x): return np.concatenate(x)
    res = {"diff_frac": args.diff_frac, "scales": {}}
    for grid, sk in [(25000, "25kb"), (100000, "100kb"), (500000, "500kb")]:
        f = max(1, grid // GRID)
        bm = lambda a: a[:(len(a) // f) * f].reshape(-1, f).mean(1)
        A = cat([bm(x) for x in pA]); B = cat([bm(x) for x in pB])
        A1 = cat([bm(x) for x in pA1]); A2 = cat([bm(x) for x in pA2])
        TA = cat([bm(x) for x in tA]); TB = cat([bm(x) for x in tB])
        res["scales"][sk] = {
            "between_pop_pred": float(spearmanr(A, B)[0]),
            "between_pop_true": float(spearmanr(TA, TB)[0]),
            "within_pop_noise_floor": float(spearmanr(A1, A2)[0]),
        }
    # differential-hotspot detection at 25kb
    f = 1
    A = cat(pA); B = cat(pB); TA = cat(tA); TB = cat(tB)
    true_diff = (np.abs(np.log(TA + 1e-12) - np.log(TB + 1e-12)) > np.log(3)).astype(int)
    score = np.abs(np.log(A + 1e-12) - np.log(B + 1e-12))
    if true_diff.sum() > 0 and true_diff.sum() < len(true_diff):
        res["differential_hotspot_auc"] = float(roc_auc_score(true_diff, score))
    res["n_windows"] = int(len(A))
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
