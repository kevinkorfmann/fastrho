"""Ceiling test: simulate under the REAL Salome/TAIR10 map so the truth is EXACTLY the input map
(no truth-noise, no real-data artifacts) and measure 100kb map-shape recovery for:
  OUTCROSSER (base model)  -> how recoverable the Salome map is at all  (map-quality/info ceiling)
  SELFER     (self2 model)  -> the same map when selfing has erased ~95% of the recombination signal
The outcrosser number is the key: if even a good model on clean outcrosser data can't recover the
Salome map at 100kb, the MAP/scoring is the ceiling; if it recovers it well, SELFING is the ceiling.
Compare both to the real-data 0.34 (which adds truth-noise + real-data realism on top).

Within-slice mean-centred + pooled, so it measures WITHIN-landscape recovery (like a real chromosome),
not the easy between-region level variance. Run on sesame: CUDA_VISIBLE_DEVICES=1 python scripts/selfer_ceiling_test.py
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
import realdata_infer as RI

W = 100_000
DEV = os.environ.get("DEV", "cuda:0")
MU = 7e-9


def _infer(gm, pos, m):
    p = predict_map_from_genotype_matrix(gm, pos, m[0], m[1], m[2], mutation_rate=MU, Ne=None, device=DEV)
    return np.r_[p["pos_left"][0], p["pos_right"]], p["r_per_bp"]


def _binned(bp, r, pos, rate, L):
    edges = np.append(np.arange(0, L, W), L)
    pr = mean_rate_between(bp, r, edges)
    tr = mean_rate_between(pos, rate, edges)
    ok = np.isfinite(pr) & np.isfinite(tr) & (pr > 0) & (tr > 0)
    return pr[ok] - np.mean(pr[ok]), tr[ok] - np.mean(tr[ok])   # within-slice mean-centred


def main():
    sp = stdpopsim.get_species("AraTha")
    rm = sp.get_genetic_map("SalomeAveraged_TAIR10").get_chromosome_map(os.environ.get("CHR", "1"))
    fullpos = np.asarray(rm.position, float); Lchr = fullpos[-1]
    K = int(os.environ.get("K", 12)); SL = int(os.environ.get("SL", 2_000_000))
    OUT_NE = float(os.environ.get("OUT_NE", 2e4)); SELF_NE = float(os.environ.get("SELF_NE", 2e5))
    s = 0.98; F = s / (2 - s)
    base_m = load_model(*RI.get_ck("base"), device=DEV)
    self_m = load_model(*RI.get_ck("self2"), device=DEV)

    acc = {"out": ([], []), "self": ([], [])}
    for k in range(K):
        lo = (k + 0.5) * (Lchr - SL) / K
        sub = rm.slice(left=lo, right=lo + SL, trim=True)
        pos = np.asarray(sub.position, float).copy(); pos[-1] = float(int(pos[-1]))
        rate = np.where(np.isfinite(sub.rate), sub.rate, 0.0); L = pos[-1]
        if pos[-1] <= pos[-2] or np.average(rate, weights=np.diff(pos)) <= 1e-10:
            continue
        # OUTCROSSER: full recombination, base model
        mp = msprime.RateMap(position=pos, rate=rate)
        ts = msprime.sim_ancestry(samples=50, ploidy=2, population_size=OUT_NE,
                                  recombination_rate=mp, sequence_length=L, random_seed=1000 + k)
        ts = msprime.sim_mutations(ts, rate=MU, random_seed=2000 + k)
        po, to = _binned(*_infer(ts.genotype_matrix().T.astype(np.int8),
                                 ts.tables.sites.position.astype(float), base_m), pos, rate, L)
        acc["out"][0].append(po); acc["out"][1].append(to)
        # SELFER: effective recombination r*(1-F), diploid at Ne_eff, one hap/individual, self2
        effmp = msprime.RateMap(position=pos, rate=rate * (1 - F))
        tc = msprime.sim_ancestry(samples=100, ploidy=2, population_size=SELF_NE / (1 + F),
                                  recombination_rate=effmp, sequence_length=L, random_seed=3000 + k)
        ts2 = msprime.sim_mutations(tc.simplify([int(tc.individual(j).nodes[0]) for j in range(100)]),
                                    rate=MU, random_seed=4000 + k)
        ps, tsr = _binned(*_infer(ts2.genotype_matrix().T.astype(np.int8),
                                  ts2.tables.sites.position.astype(float), self_m), pos, rate, L)
        acc["self"][0].append(ps); acc["self"][1].append(tsr)
        print("slice %2d: outcross %d SNP, selfer %d SNP" %
              (k, ts.num_sites, ts2.num_sites), flush=True)

    print("\n=== 100kb map-recovery under the REAL Salome map (truth = exact input; %d x %.1fMb slices) ==="
          % (K, SL / 1e6))
    for name, lab in (("out", "OUTCROSSER (base) "), ("self", "SELFER     (self2)")):
        p = np.concatenate(acc[name][0]); t = np.concatenate(acc[name][1])
        r = float(np.corrcoef(p, t)[0, 1])
        print("  %s recovery = %.3f   (n=%d windows)" % (lab, r, len(p)))
    print("  [real A. thaliana self2 vs Salome = ~0.34 for reference: that = this selfer ceiling + truth-noise + real-data realism]")


if __name__ == "__main__":
    main()
