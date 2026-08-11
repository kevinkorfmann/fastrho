"""Identifiability sweep: recovery power vs diversity (theta = 4 Ne mu), with a spread band.

Recombination is identifiable from LD only where enough SNPs record the breakdown of LD, so
recovery degrades CONTINUOUSLY as diversity drops. We hold the recombination-map shape, Ne and n
fixed and sweep mu, measuring fastrho's recovery Pearson vs the realized SNP density. Real species
are placed on the curve by their own diversity (Watterson theta_W).

This version additionally records the PER-REPLICATE Pearson r at each mu (``rep_r``), so Figure 5a
can draw a confidence band (rep-to-rep sampling variability of recovery), not just a single pooled
point. The pooled r over all windows is kept as the central estimate, comparable to how each real
species is measured (one pooled r).

Run on sesame (GPU):
    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 /home/kkor/venvs/fastrho/bin/python \
        scripts/identifiability_sweep.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/kkor/fastrho")
import msprime
from scipy.stats import pearsonr
from fastrho.simulate import make_recombination_map, RecombPriors
from fastrho.translate import load_model, predict_map_from_genotype_matrix
from fastrho.preprocess import mean_rate_between

CKPT = "/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt"
STATS = "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"
OUT = os.environ.get("SWEEP_OUT", "/home/kkor/realdata/identifiability_band.json")
DEV = os.environ.get("FASTRHO_DEVICE", "cuda:0")
L = 1_000_000; NE = 10_000; NDIP = 10; GRID = 25_000; REPS = int(os.environ.get("SWEEP_REPS", 20))
MUS = [5e-10, 1e-9, 2e-9, 4e-9, 8e-9, 1.5e-8, 3e-8, 5e-8]


def main():
    model, cfg, stats = load_model(CKPT, STATS, device=DEV)
    rows = []
    for mu in MUS:
        T, P, dens, rep_r = [], [], [], []                     # pool windows + keep per-rep r
        for rep in range(REPS):
            rng = np.random.default_rng(1000 + rep)            # same map shapes across mu
            kind = "hotspot" if rep % 2 else "gp"
            rm = make_recombination_map(L, rng, kind=kind, mean_rate=1e-8,
                                        priors=RecombPriors(sequence_length=L))
            ts = msprime.sim_ancestry(samples=NDIP, recombination_rate=rm, population_size=NE,
                                      sequence_length=L, random_seed=2000 + rep)
            ts = msprime.sim_mutations(ts, rate=mu, random_seed=3000 + rep + int(mu * 1e12))
            gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position.astype(float)
            dens.append(gm.shape[1] / (L / 1000.0))
            if gm.shape[1] < 20:
                continue
            pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                    mutation_rate=mu, Ne=NE, device=DEV)
            edges = np.append(np.arange(0, L, GRID), L)
            truth = mean_rate_between(rm.position, rm.rate, edges)
            bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
            pr = mean_rate_between(bp, pred["r_per_bp"], edges)
            k = min(len(truth), len(pr)); t, p = truth[:k], pr[:k]
            ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
            t, p = t[ok], p[ok]
            T.append(t); P.append(p)
            if len(t) > 10:                                    # per-rep r -> the band
                rep_r.append(float(pearsonr(t, p)[0]))
        Tc = np.concatenate(T) if T else np.array([]); Pc = np.concatenate(P) if P else np.array([])
        r = float(pearsonr(Tc, Pc)[0]) if len(Tc) > 10 else float("nan")
        # Confidence band = uncertainty of the POOLED recovery, by bootstrapping over replicates
        # (resample the maps/reps with replacement, recompute the pooled r). This reflects
        # sampling uncertainty of the diversity->recovery relationship; a raw per-rep spread would
        # instead be dominated by which map shape (gp vs hotspot) each rep drew.
        lo = hi = float("nan")
        nrep = len(T)
        if nrep >= 3:
            brng = np.random.default_rng(12345)
            boot = []
            for _ in range(1000):
                idx = brng.integers(0, nrep, nrep)
                tb = np.concatenate([T[i] for i in idx]); pb = np.concatenate([P[i] for i in idx])
                if len(tb) > 10:
                    boot.append(float(pearsonr(tb, pb)[0]))
            lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (r, r)
        rows.append(dict(mu=mu, theta=4 * NE * mu, snp_per_kb=float(np.mean(dens)),
                         pearson=r, pearson_lo=lo, pearson_hi=hi, rep_r=rep_r,
                         n_rep=len(rep_r), n_win=int(len(Tc))))
        print("mu=%.1e theta=%.4f snp/kb=%.2f  pooled=%.3f  95%% CI=[%.3f,%.3f] (%d reps)"
              % (mu, 4 * NE * mu, np.mean(dens), r, lo, hi, nrep))

    # real-species diversity (Watterson theta_W per bp) + observed Pearson, for the overlay
    real = {}
    a_n = lambda n: float(np.sum(1.0 / np.arange(1, n)))
    for key in ["human", "dmel", "athal", "wolf", "dog"]:
        try:
            z = np.load(f"/home/kkor/realdata/hap/{key}.npz", allow_pickle=True)
            gm = z["gm"]; pos = z["pos"].astype(float)
            span_kb = (pos.max() - pos.min()) / 1000.0
            m = np.load(f"/home/kkor/realdata/maps/{key}.npz")
            real[key] = dict(snp_per_kb=float(gm.shape[1] / span_kb),
                             theta_w=float(gm.shape[1] / a_n(gm.shape[0]) / (span_kb * 1000)),
                             pearson=float(m["pearson"]), n_hap=int(gm.shape[0]))
        except Exception as e:
            print("skip real", key, e)
    json.dump(dict(sweep=rows, real=real), open(OUT, "w"), indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
