"""Controlled mechanism experiment: WHY each regime breaks recombination identifiability.

Same recombination rate, different regime -> different LD (the signal inference reads).
(1) LD-decay r^2 vs distance for panmictic / bottleneck / selfing (constant r): shows the decay is
    clean (panmictic), erased by drift (bottleneck), or stretched (selfing).
(2) Inversion: a hotspot map simulated with recombination locally suppressed (a segregating
    inversion) -> fastrho's inferred map troughs across the block, vs the meiotic target.

Run on sesame (GPU): PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 venvs/fastrho/bin/python scripts/mechanism_sim.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/kkor/fastrho")
import msprime
from fastrho.simulate import make_recombination_map, RecombPriors
from fastrho.translate import load_model, predict_map_from_genotype_matrix
from fastrho.preprocess import mean_rate_between

CKPT = "/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt"
STATS = "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"
OUT = "/home/kkor/realdata/mechanism.json"
L = 2_000_000; NE = 10_000; NDIP = 12; R = 1e-8; MU = 2e-8
BINS = np.array([200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600])


def ld_accumulate(gm, pos, sums, cnts):
    pos = np.asarray(pos, float); n = gm.shape[0]; p = gm.mean(0)
    anchors = np.arange(0, len(pos), max(1, len(pos) // 6000))
    for a in anchors:
        pa = p[a]
        if pa <= 0 or pa >= 1: continue
        j0 = np.searchsorted(pos, pos[a] + BINS[0]); j1 = np.searchsorted(pos, pos[a] + BINS[-1])
        for b in range(j0, min(j1, len(pos)), max(1, (j1 - j0) // 60 or 1)):
            pb = p[b]
            if pb <= 0 or pb >= 1: continue
            d = pos[b] - pos[a]; k = np.searchsorted(BINS, d) - 1
            if not (0 <= k < len(cnts)): continue
            pab = float((gm[:, a] * gm[:, b]).mean()); den = pa * (1 - pa) * pb * (1 - pb)
            if den > 0:
                sums[k] += max((pab - pa * pb) ** 2 / den - 1.0 / n, 0.0); cnts[k] += 1


def ld_curve_pooled(simfn, reps=10):
    sums = np.zeros(len(BINS) - 1); cnts = np.zeros(len(BINS) - 1)
    for rep in range(reps):
        gm, pos = simfn(rep)
        ld_accumulate(gm, pos, sums, cnts)
    mids = np.sqrt(BINS[:-1] * BINS[1:])
    return mids, sums / np.maximum(cnts, 1)


def sim_gm(demography=None, pop_size=None, ploidy=2, rrate=R, mu=MU, seed=1):
    kw = dict(recombination_rate=rrate, sequence_length=L, random_seed=seed, ploidy=ploidy)
    if demography is not None:
        kw["demography"] = demography; kw["samples"] = NDIP if ploidy == 2 else 2 * NDIP
    else:
        kw["population_size"] = pop_size; kw["samples"] = NDIP if ploidy == 2 else 2 * NDIP
    ts = msprime.sim_ancestry(**kw)
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 7)
    return ts.genotype_matrix().T.astype(np.int8), ts.tables.sites.position.astype(float)


def main():
    out = {"ld": {}}
    # (1) LD-decay under three regimes (constant rate), POOLED over replicates for a smooth curve
    F = 0.95
    dem = msprime.Demography(); dem.add_population(name="A", initial_size=400)
    dem.add_population_parameters_change(time=400, initial_size=NE, population="A")
    regimes = {
        "panmictic": lambda rep: sim_gm(pop_size=NE, seed=100 + rep),
        "bottleneck": lambda rep: sim_gm(demography=dem, seed=200 + rep),
        "selfing": lambda rep: sim_gm(pop_size=NE / (1 + F), ploidy=1, rrate=R * (1 - F), seed=300 + rep),
    }
    for name, fn in regimes.items():
        m, r2 = ld_curve_pooled(fn, reps=10)
        out["ld"][name] = dict(d=m.tolist(), r2=r2.tolist())
        print(name, "LD r2:", np.round(r2, 3))

    # (2) inversion: hotspot map, recombination suppressed in a central block
    model, cfg, stats = load_model(CKPT, STATS, device="cuda:0")
    rng = np.random.default_rng(5)
    rm = make_recombination_map(L, rng, kind="hotspot", mean_rate=R, priors=RecombPriors(sequence_length=L))
    mp = np.asarray(rm.position, float); mr = np.asarray(rm.rate, float)
    lo, hi = 0.8e6, 1.3e6                                   # the "inversion" block
    rr_inv = mr.copy(); rr_inv[(mp[:-1] >= lo) & (mp[:-1] < hi)] = 1e-12   # suppressed
    inv_map = msprime.RateMap(position=mp, rate=rr_inv)
    ts = msprime.sim_ancestry(samples=NDIP, recombination_rate=inv_map, population_size=NE,
                              sequence_length=L, random_seed=21)
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=22)
    gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position.astype(float)
    pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats, mutation_rate=MU, Ne=NE, device="cuda:0")
    GRID = 50000; edges = np.append(np.arange(0, L, GRID), L); cen = (edges[:-1] + GRID / 2) / 1e6
    meiotic = mean_rate_between(mp, mr, edges)             # the (un-suppressed) meiotic target
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    inferred = mean_rate_between(bp, pred["r_per_bp"], edges)
    out["inversion"] = dict(centers=cen.tolist(), meiotic=meiotic.tolist(),
                            inferred=inferred.tolist(), block=[lo / 1e6, hi / 1e6])
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT, "| LD regimes:", list(out["ld"]), "| inversion windows:", len(cen))


if __name__ == "__main__":
    main()
