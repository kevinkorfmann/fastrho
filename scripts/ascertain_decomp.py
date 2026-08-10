"""Ascertainment decomposition: apply the REAL A. thaliana data artifacts to a CLEAN-sim selfer
(one at a time / cumulatively) and re-score recovery vs the exact input map. Attributes the
0.875(clean)->0.27(real) sim-to-real deficit to each mechanism, and tells us what to train against.

Operators mirror realdata_extract.py / the token-shift diagnosis (density+diversity are the OOD
features): D2 missing->reference masking (fraction of genotypes set to 0), D5 genotyping-error
bit-flips (error singletons), D6 whole-site polarization flips. Scored with the deployed self2.

Usage (sesame): CUDA_VISIBLE_DEVICES=1 python scripts/ascertain_decomp.py
"""
import os
import sys

import numpy as np
import msprime
import stdpopsim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.translate import load_model, predict_map_from_genotype_matrix
from fastrho.preprocess import mean_rate_between
from fastcxt.sfs import basic_filtering
import realdata_infer as RI
import selfer_arch as A

W = 100_000
DEV = os.environ.get("DEV", "cuda:0")
MU = 7e-9


def ascertain(gm, pos, rng, p_miss=0.0, p_err=0.0, p_flip=0.0):
    """Apply real-data artifacts to a clean 0/1 haploid matrix, then re-filter monomorphic."""
    gm = gm.copy()
    if p_miss > 0:                       # D2: missing -> reference (real extractor: missing => 0)
        gm[rng.random(gm.shape) < p_miss] = 0
    if p_err > 0:                        # D5: genotyping error bit-flip
        e = rng.random(gm.shape) < p_err
        gm[e] = 1 - gm[e]
    if p_flip > 0:                       # D6: whole-site polarization flip (ref != ancestral)
        f = rng.random(gm.shape[1]) < p_flip
        gm[:, f] = 1 - gm[:, f]
    return basic_filtering(gm.astype(np.int8), pos)


def main():
    sp = stdpopsim.get_species("AraTha")
    rm = sp.get_genetic_map("SalomeAveraged_TAIR10").get_chromosome_map("1")
    Lchr = np.asarray(rm.position, float)[-1]
    K = int(os.environ.get("K", 10)); SL = int(os.environ.get("SL", 2_000_000))
    SELF_NE = 2e5; s = 0.98; F = A.selfing_F(s)
    self_m = load_model(*RI.get_ck("self2"), device=DEV)

    settings = [("clean", dict()),
                ("miss5%", dict(p_miss=0.05)), ("miss10%", dict(p_miss=0.10)),
                ("miss20%", dict(p_miss=0.20)),
                ("err0.2%", dict(p_err=0.002)), ("flip10%", dict(p_flip=0.10)),
                ("miss10+err0.2", dict(p_miss=0.10, p_err=0.002))]
    acc = {name: ([], []) for name, _ in settings}

    for k in range(K):
        lo = (k + 0.5) * (Lchr - SL) / K
        sub = rm.slice(left=lo, right=lo + SL, trim=True)
        pos = np.asarray(sub.position, float).copy(); pos[-1] = float(int(pos[-1]))
        rate = np.where(np.isfinite(sub.rate), sub.rate, 0.0); L = pos[-1]
        if pos[-1] <= pos[-2] or np.average(rate, weights=np.diff(pos)) <= 1e-10:
            continue
        eff = msprime.RateMap(position=pos, rate=rate * (1 - F))
        tc = msprime.sim_ancestry(samples=156, ploidy=2, population_size=SELF_NE / (1 + F),
                                  recombination_rate=eff, sequence_length=L, random_seed=3000 + k)
        ts = msprime.sim_mutations(tc.simplify([int(tc.individual(j).nodes[0]) for j in range(156)]),
                                   rate=MU, random_seed=4000 + k)
        gm0 = ts.genotype_matrix().T.astype(np.int8); gpos0 = ts.tables.sites.position.astype(float)
        edges = np.append(np.arange(0, L, W), L)
        tr = mean_rate_between(pos, rate, edges)
        for name, kw in settings:
            rng = np.random.default_rng(9000 + k)
            gm, gpos = ascertain(gm0, gpos0, rng, **kw)
            if gm.shape[1] < 20:
                continue
            p = predict_map_from_genotype_matrix(gm, gpos, self_m[0], self_m[1], self_m[2],
                                                 mutation_rate=MU, Ne=None, device=DEV)
            bp = np.r_[p["pos_left"][0], p["pos_right"]]
            pr = mean_rate_between(bp, p["r_per_bp"], edges)
            ok = np.isfinite(pr) & np.isfinite(tr) & (pr > 0) & (tr > 0)
            acc[name][0].append(pr[ok] - pr[ok].mean()); acc[name][1].append(tr[ok] - tr[ok].mean())
        print("slice %d done" % k, flush=True)

    print("\n=== recovery vs input map (self2, clean-sim + real-data artifacts; %d x %.1fMb) ===" % (K, SL / 1e6))
    for name, _ in settings:
        if not acc[name][0]:
            continue
        p = np.concatenate(acc[name][0]); t = np.concatenate(acc[name][1])
        print("  %-16s recovery = %.3f" % (name, float(np.corrcoef(p, t)[0, 1])))
    print("  [real self2 vs Salome = 0.34 (chr1); clean-sim = 0.875]")


if __name__ == "__main__":
    main()
