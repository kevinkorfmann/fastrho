"""Demography/structure decomposition: the ascertainment operators didn't reproduce the sim-to-real
gap (missing->ref/error/polarization cost only ~0.06). The token-shift said real athal has MORE unique
haplotypes + denser SNPs than a panmictic selfer sim -- the signature of POPULATION STRUCTURE. This
simulates a selfer under the real Salome chr1 map with different demographies/structures and scores
recovery with the deployed self2, to test whether structure collapses recovery toward the real 0.34.

Settings: panmictic constant; SouthMiddleAtlas expansion (single pop); 2-deme island at a range of
migration rates (sampling both demes = Wahlund/admixture LD). All with the selfing effective scaling.

Usage (sesame): CUDA_VISIBLE_DEVICES=1 python scripts/demog_decomp.py
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
SELF_NE = 2e5
s = 0.98
F = A.selfing_F(s)
NE_EFF = SELF_NE / (1 + F)


def _split(T_coal, size):
    """2 demes that diverged T_coal * 2*Ne generations ago (no migration): clean Wahlund structure,
    Fst ~ 1-exp(-T_coal). Fast to simulate (no rare-migration waiting)."""
    d = msprime.Demography()
    d.add_population(name="anc", initial_size=size)
    d.add_population(name="p0", initial_size=size)
    d.add_population(name="p1", initial_size=size)
    d.add_population_split(time=T_coal * 2 * size, derived=["p0", "p1"], ancestral="anc")
    return d


def sim_selfer(eff_map, L, setting, seed, sp):
    """Return n=156 haploid samples (one/individual) under `setting`, selfing-scaled."""
    if setting == "panmictic":
        tc = msprime.sim_ancestry(samples=156, ploidy=2, population_size=NE_EFF,
                                  recombination_rate=eff_map, sequence_length=L, random_seed=seed)
    elif setting == "SMA_expansion":
        demog = A.build_demography(sp, "SouthMiddleAtlas_1D17", F=F, Q=1.0)
        tc = msprime.sim_ancestry(samples=156, ploidy=2, demography=demog,
                                  recombination_rate=eff_map, sequence_length=L, random_seed=seed)
    else:  # split_T<x>: two diverged demes (Wahlund), 78 individuals each
        T = float(setting.split("_T")[1])
        demog = _split(T, NE_EFF)
        tc = msprime.sim_ancestry(samples={"p0": 78, "p1": 78}, ploidy=2, demography=demog,
                                  recombination_rate=eff_map, sequence_length=L, random_seed=seed)
    nodes = [int(tc.individual(j).nodes[0]) for j in range(156)]
    return msprime.sim_mutations(tc.simplify(nodes), rate=MU, random_seed=seed + 1)


def main():
    sp = stdpopsim.get_species("AraTha")
    rm = sp.get_genetic_map("SalomeAveraged_TAIR10").get_chromosome_map("1")
    Lchr = np.asarray(rm.position, float)[-1]
    K = int(os.environ.get("K", 8)); SL = int(os.environ.get("SL", 2_000_000))
    self_m = load_model(*RI.get_ck("self2"), device=DEV)
    settings = ["panmictic", "SMA_expansion",
                "split_T0.1", "split_T0.2", "split_T0.35"]   # Fst ~ 0.10 / 0.18 / 0.30 (real athal range)
    acc = {s0: ([], [], []) for s0 in settings}   # (pred-centred, truth-centred, n_hap_frac proxy=nsnp)

    for k in range(K):
        lo = (k + 0.5) * (Lchr - SL) / K
        sub = rm.slice(left=lo, right=lo + SL, trim=True)
        pos = np.asarray(sub.position, float).copy(); pos[-1] = float(int(pos[-1]))
        rate = np.where(np.isfinite(sub.rate), sub.rate, 0.0); L = pos[-1]
        if pos[-1] <= pos[-2] or np.average(rate, weights=np.diff(pos)) <= 1e-10:
            continue
        eff = msprime.RateMap(position=pos, rate=rate * (1 - F))
        edges = np.append(np.arange(0, L, W), L)
        tr = mean_rate_between(pos, rate, edges)
        for setting in settings:
            ts = sim_selfer(eff, L, setting, 3000 + k, sp)
            gm = ts.genotype_matrix().T.astype(np.int8); gpos = ts.tables.sites.position.astype(float)
            gm, gpos = basic_filtering(gm, gpos)
            if gm.shape[1] < 20:
                continue
            p = predict_map_from_genotype_matrix(gm, gpos, self_m[0], self_m[1], self_m[2],
                                                 mutation_rate=MU, Ne=None, device=DEV)
            bp = np.r_[p["pos_left"][0], p["pos_right"]]
            pr = mean_rate_between(bp, p["r_per_bp"], edges)
            ok = np.isfinite(pr) & np.isfinite(tr) & (pr > 0) & (tr > 0)
            acc[setting][0].append(pr[ok] - pr[ok].mean()); acc[setting][1].append(tr[ok] - tr[ok].mean())
            acc[setting][2].append(gm.shape[1])
        # running estimate after each slice (so a partial/timed-out run still yields numbers)
        run = []
        for s0 in settings:
            if acc[s0][0]:
                p = np.concatenate(acc[s0][0]); t = np.concatenate(acc[s0][1])
                run.append("%s=%.3f" % (s0, float(np.corrcoef(p, t)[0, 1])))
        print("slice %d | running: %s" % (k, "  ".join(run)), flush=True)

    print("\n=== recovery vs input map (self2) under demography/structure (%d x %.1fMb) ===" % (K, SL / 1e6))
    for setting in settings:
        if not acc[setting][0]:
            continue
        p = np.concatenate(acc[setting][0]); t = np.concatenate(acc[setting][1])
        nsnp = int(np.mean(acc[setting][2]))
        print("  %-16s recovery = %.3f   (mean %d SNP/2Mb)" % (setting, float(np.corrcoef(p, t)[0, 1]), nsnp))
    print("  [real self2 vs Salome = 0.34 (chr1)]")


if __name__ == "__main__":
    main()
