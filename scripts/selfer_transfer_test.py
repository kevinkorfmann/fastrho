"""Cross-species generality: does the A. thaliana-trained selfing model (self2, FROZEN, no retraining)
recover a DIFFERENT selfer's recombination landscape? Simulate a selfer under the C. elegans
RockmanRIAIL_ce11 map (the distinctive arm-high / centre-low "domain" structure) and score self2 vs
pyrho-style truth. The RockmanRIAIL map is very coarse (~1 pt / 2 Mb), so the recoverable signal is
the broad domain structure -- score at 100 kb AND 500 kb; the coarse number reflects the true
transferable landscape, the fine number is truth-resolution-limited (as for A. thaliana).

Usage (sesame): CUDA_VISIBLE_DEVICES=1 SPECIES=CaeEle MAP=RockmanRIAIL_ce11 python scripts/selfer_transfer_test.py
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

DEV = os.environ.get("DEV", "cuda:0")
MU = float(os.environ.get("MU", 2.7e-9))     # C. elegans ~2.7e-9
NE = float(os.environ.get("NE", 1e5))
S = float(os.environ.get("S", 0.99))         # C. elegans is a near-obligate selfer (top of the prior)


def score_at(bp, r, pos, rate, L, W):
    edges = np.append(np.arange(0, L, W), L)
    pr = mean_rate_between(bp, r, edges); tr = mean_rate_between(pos, rate, edges)
    ok = np.isfinite(pr) & np.isfinite(tr) & (pr > 0) & (tr > 0)
    return pr[ok] - pr[ok].mean(), tr[ok] - tr[ok].mean()


def main():
    species = os.environ.get("SPECIES", "CaeEle"); mapid = os.environ.get("MAP", "RockmanRIAIL_ce11")
    sp = stdpopsim.get_species(species); gm_ = sp.get_genetic_map(mapid)
    chroms = os.environ.get("CHROMS", "I,II,III,IV,V").split(",")
    F = A.selfing_F(S)
    self_m = load_model(*RI.get_ck("self2"), device=DEV)
    acc = {100_000: ([], []), 500_000: ([], [])}
    print(f"{species}/{mapid}, frozen A.thaliana self2, s={S}, Ne={NE:.0e}, mu={MU:.1e}")
    for ch in chroms:
        rm = gm_.get_chromosome_map(ch)
        pos = np.asarray(rm.position, float).copy(); pos[-1] = float(int(pos[-1]))
        rate = np.where(np.isfinite(rm.rate), rm.rate, 0.0); L = pos[-1]
        if np.average(rate, weights=np.diff(pos)) <= 1e-10:
            continue
        eff = msprime.RateMap(position=pos, rate=rate * (1 - F))
        tc = msprime.sim_ancestry(samples=100, ploidy=2, population_size=NE / (1 + F),
                                  recombination_rate=eff, sequence_length=L, random_seed=hash(ch) % 9999 + 1)
        ts = msprime.sim_mutations(tc.simplify([int(tc.individual(j).nodes[0]) for j in range(100)]),
                                   rate=MU, random_seed=7)
        g = ts.genotype_matrix().T.astype(np.int8); gpos = ts.tables.sites.position.astype(float)
        g, gpos = basic_filtering(g, gpos)
        p = predict_map_from_genotype_matrix(g, gpos, self_m[0], self_m[1], self_m[2],
                                             mutation_rate=MU, Ne=None, device=DEV)
        bp = np.r_[p["pos_left"][0], p["pos_right"]]
        r100 = np.corrcoef(*[np.concatenate([x]) for x in score_at(bp, p["r_per_bp"], pos, rate, L, 100_000)])[0, 1]
        r500 = np.corrcoef(*[np.concatenate([x]) for x in score_at(bp, p["r_per_bp"], pos, rate, L, 500_000)])[0, 1]
        for W, rr in ((100_000, score_at(bp, p["r_per_bp"], pos, rate, L, 100_000)),
                      (500_000, score_at(bp, p["r_per_bp"], pos, rate, L, 500_000))):
            acc[W][0].append(rr[0]); acc[W][1].append(rr[1])
        print("  chr%-3s %d SNP  r@100kb=%+.3f  r@500kb=%+.3f" % (ch, g.shape[1], r100, r500), flush=True)
    print("--- pooled (within-chrom mean-centred) ---")
    for W in (100_000, 500_000):
        p = np.concatenate(acc[W][0]); t = np.concatenate(acc[W][1])
        print("  transfer recovery @%dkb = %+.3f  (n=%d windows)" % (W // 1000, float(np.corrcoef(p, t)[0, 1]), len(p)))


if __name__ == "__main__":
    main()
