"""Seam validator (NON-NEGOTIABLE gate before any BGS run).

The forward SLiM phase self; the panmictic recapitation cannot. selfing_slim_gen compensates by
feeding the recap EFFECTIVE parameters (rate*Q*(1-F), ancestral Ne = N'/(1+F)). If that (1-F)/(1+F)/Q
scaling is wrong, a NEUTRAL selfing SLiM region will NOT match a NEUTRAL selfing coalescent region
at matched (Ne_eff, map, s, n, mu). This script builds matched pairs and compares diversity, tree
density (∝ effective recombination), and the LD-decay r^2 bands (the exact features the model reads).
Mirrors slim_gen's "neutral == msprime" control.

Run on sesame (needs SLiM):
  SLIM_BIN=/home/kkor/.local/bin/slim python scripts/selfing_seam_check.py
"""
import os
import sys

import numpy as np
import msprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selfer_arch as A
import selfing_slim_gen as S
from fastrho.simulate import make_recombination_map, RecombPriors
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig, mean_r2_slice

_FEAT = GTTokenFeaturizer(config=FeatureConfig(), fold=True)
_SL = mean_r2_slice(FeatureConfig())


def _metrics(ts):
    gm = ts.genotype_matrix().T.astype(np.int8)
    pos = ts.tables.sites.position.astype(np.float64)
    tok = _FEAT(gm, pos, {"sequence_length": float(ts.sequence_length)})["tokens"]
    mr = tok[:, _SL]
    ld = np.nanmean(np.where(mr > 0, mr, np.nan), axis=0)     # mean r^2 per LD band
    return dict(pi=float(ts.diversity(span_normalise=True)),
                nsites=int(ts.num_sites), ntrees=int(ts.num_trees), ld=ld)


def main():
    Q = S.Q
    Nprime = int(os.environ.get("SEAM_NPRIME", 1000))
    npair = int(os.environ.get("SEAM_NPAIR", 8))
    L = int(os.environ.get("SEAM_L", 500_000))
    mu = A.ATHAL["mu"]
    dfe = (-0.025, 0.20, 1.0)     # unused in neutral mode
    S.NPRIME = Nprime             # small census so the seam check is quick

    slim_m, coal_m = [], []
    for i in range(npair):
        rng = np.random.default_rng(4000 + i)
        s = float(rng.uniform(0.95, 0.99)); F = A.selfing_F(s)
        n = 100
        rm = make_recombination_map(L, rng, kind="gp", mean_rate=10.0 ** rng.uniform(-8.0, -7.4),
                                    priors=RecombPriors(sequence_length=L))
        pos = np.asarray(rm.position, float); rate = np.asarray(rm.rate, float)
        Ne_eff_real = Nprime * Q / (1.0 + F)

        ts_s = S.slim_selfing_ts(pos, rate, s, n, "neutral", np.zeros((0, 2), int),
                                 dfe, Nprime / (1.0 + F), seed=4000 + i)
        # analytic selfing coalescent (reference): a genuine DIPLOID coalescent at Ne_eff with
        # effective recombination r*(1-F), sampled ONE genome per individual -- mirrors what the
        # SLiM path does (forward diploid selfing -> one node per individual). ploidy=2 gives the
        # correct diploid theta=4*Ne_eff*mu and rho=4*Ne_eff*r_eff (ploidy=1 would halve both).
        eff = msprime.RateMap(position=pos, rate=rate * (1.0 - F))
        tc0 = msprime.sim_ancestry(samples=n, ploidy=2, population_size=Ne_eff_real,
                                   recombination_rate=eff, sequence_length=pos[-1],
                                   random_seed=4000 + i)
        nodes = [int(tc0.individual(j).nodes[0]) for j in range(n)]   # one haplotype / individual
        ts_c = msprime.sim_mutations(tc0.simplify(nodes), rate=mu, random_seed=4007 + i)
        ms, mc = _metrics(ts_s), _metrics(ts_c)
        slim_m.append(ms); coal_m.append(mc)
        print("pair %d s=%.3f: pi slim/coal=%.2e/%.2e  ntrees=%d/%d  nsites=%d/%d"
              % (i, s, ms["pi"], mc["pi"], ms["ntrees"], mc["ntrees"], ms["nsites"], mc["nsites"]),
              flush=True)

    def _mean(ms, k): return float(np.mean([m[k] for m in ms]))
    def _reldiff(a, b): return abs(a - b) / max(abs(b), 1e-30)
    pi_d = _reldiff(_mean(slim_m, "pi"), _mean(coal_m, "pi"))
    tr_d = _reldiff(_mean(slim_m, "ntrees"), _mean(coal_m, "ntrees"))
    ld_s = np.nanmean(np.vstack([m["ld"] for m in slim_m]), axis=0)
    ld_c = np.nanmean(np.vstack([m["ld"] for m in coal_m]), axis=0)
    ld_d = float(np.nanmax(np.abs(ld_s - ld_c) / np.maximum(ld_c, 1e-6)))

    tol = float(os.environ.get("SEAM_TOL", 0.25))
    print("\n=== SEAM SUMMARY (%d pairs, tol=%.2f) ===" % (npair, tol))
    print("  pi rel-diff      = %.3f   (PASS<%.2f)" % (pi_d, tol))
    print("  LD mean rel-diff = %.3f   (PASS<%.2f)" % (np.nanmean(np.abs(ld_s - ld_c) / np.maximum(ld_c, 1e-6)), tol))
    print("  LD max rel-diff  = %.3f   (bands slim=%s coal=%s)"
          % (ld_d, np.round(ld_s, 3), np.round(ld_c, 3)))
    # ntrees is REPORTED but NOT a pass criterion: under selfing SLiM records "invisible"
    # recombination breakpoints between identical haplotypes that inflate num_trees without
    # changing the genealogy/LD (they survive simplify as topologically identical adjacent trees).
    print("  ntrees rel-diff  = %.3f   (diagnostic only -- inflated by selfing-invisible breakpoints)" % tr_d)
    ld_mean_d = float(np.nanmean(np.abs(ld_s - ld_c) / np.maximum(ld_c, 1e-6)))
    ok = (pi_d < tol) and (ld_mean_d < tol)
    print("  SEAM:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
