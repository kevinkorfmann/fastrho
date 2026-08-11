"""Token-shift diagnostic: which of the 17 features are OUT OF DISTRIBUTION on real A. thaliana vs
clean-sim selfers? Localizes the sim-to-real gap (the 0.55 truth-ceiling -> 0.27 real deficit) to a
mechanism BEFORE retraining. Compares the committed real athal token cache (self_featurize_cache.py:
raw SNPTokenFeaturizer tokens) against clean-sim selfer tokens from the SAME featurizer, per column.

Big shifts in `derived_af`/`cfg_*` => polarization (real 0=TAIR10 ref, sim=ancestral).
Big shifts in `local_theta_pi`/`mean_r2`/`log_npairs` => SFS/density/missingness ascertainment.

Usage (sesame): CHR=1 python scripts/token_shift_diag.py
"""
import os
import sys

import numpy as np
import msprime
import stdpopsim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.features import SNPTokenFeaturizer, feature_names, FeatureConfig
from fastcxt.sfs import basic_filtering
import selfer_arch as A

CACHE = "/home/kkor/fastrho_data/campaign_self/athal_token_cache"


def sim_selfer_tokens(chrom, n, mu, seed=1, ne=2e5, s=0.98):
    """Clean-sim selfer under a real chr-`chrom` map slice; same featurizer as the real cache."""
    sp = stdpopsim.get_species("AraTha")
    rng = np.random.default_rng(seed)
    F = A.selfing_F(s)
    pos, rate, ch, lo = A.load_real_map_slice(sp, "SalomeAveraged_TAIR10", 4_000_000, rng, peri_bias=0.5)
    eff = msprime.RateMap(position=pos, rate=rate * (1 - F))
    tc = msprime.sim_ancestry(samples=n, ploidy=2, population_size=ne / (1 + F),
                              recombination_rate=eff, sequence_length=pos[-1], random_seed=seed)
    ts = msprime.sim_mutations(tc.simplify([int(tc.individual(j).nodes[0]) for j in range(n)]),
                               rate=mu, random_seed=seed + 1)
    gm = ts.genotype_matrix().T.astype(np.int8); gpos = ts.tables.sites.position.astype(float)
    gm, gpos = basic_filtering(gm, gpos)
    return SNPTokenFeaturizer()(gm, gpos, {"sequence_length": float(gpos[-1] + 1)})["tokens"]


def main():
    chrom = os.environ.get("CHR", "1")
    z = np.load(f"{CACHE}/athal_c{chrom}_tokens.npz", allow_pickle=True)
    real = np.asarray(z["tokens"], float)
    n = int(z["n_hap"]); mu = float(z["mu"])
    names = feature_names(FeatureConfig())
    print(f"real athal chr{chrom}: {real.shape[0]} SNP-tokens x {real.shape[1]} feat, n_hap={n}, mu={mu:.1e}")

    sims = [sim_selfer_tokens(chrom, n, mu, seed=s) for s in (1, 2, 3)]
    sim = np.concatenate(sims, 0)
    print(f"clean-sim selfer: {sim.shape[0]} tokens (3 slices)\n")

    print(f"{'feature':>16} {'real_mean':>10} {'sim_mean':>10} {'real_std':>9} {'sim_std':>9} {'|Δ|/pooled_sd':>13}")
    rows = []
    for j, nm in enumerate(names):
        rm, sm = np.nanmean(real[:, j]), np.nanmean(sim[:, j])
        rs, ss = np.nanstd(real[:, j]), np.nanstd(sim[:, j])
        psd = np.sqrt(0.5 * (rs**2 + ss**2)) + 1e-9
        d = abs(rm - sm) / psd
        rows.append((d, nm, rm, sm, rs, ss))
    for d, nm, rm, sm, rs, ss in rows:
        print(f"{nm:>16} {rm:>10.3f} {sm:>10.3f} {rs:>9.3f} {ss:>9.3f} {d:>13.2f}")
    print("\nMost out-of-distribution (by |Δmean|/pooled_sd):")
    for d, nm, *_ in sorted(rows, reverse=True)[:6]:
        print(f"  {nm:>16}  shift={d:.2f} sd")


if __name__ == "__main__":
    main()
