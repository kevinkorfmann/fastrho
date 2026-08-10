"""Coalescent selfer generator + BLEND dispatcher for the realism campaign.

Two jobs:
  1. gen_one_coalescent -- a fast msprime selfer region (re-centred selfing prior, real AraTha
     Ne(t) or constant, real-map slice or synthetic gp/hotspot/pericentromere map), emitting the
     same ts_* training contract as selfing_gen.py / selfing_slim_gen.py. This is the cheap
     "volume + prior breadth" fraction that dilutes memorisation of the one real map.
  2. main() -- dispatch a total of N regions into ONE dir: indices [0, n_slim) forward-simulated
     with BGS+selfing (scripts/selfing_slim_gen.gen_one), [n_slim, N) coalescent. All write ts_%08d
     so fastrho/preprocess.py picks up the whole blend.

Ne convention matches selfing_slim_gen: meta["Ne"] = diversity/(4 mu) (the LD-effective size).
Selfing suppresses effective recombination (rate*(1-F)), so the coalescent-with-recombination is
cheap even at A. thaliana Ne. LOCO: SELF_EXCLUDE_CHROM holds a chromosome's real map out of training.

Usage:
  python scripts/selfing_blend_gen.py <outdir> <total_N> [frac_slim=0.33] [nproc=20] [offset=0]
"""
import os
import sys
import json

import numpy as np
import msprime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selfer_arch as A
from fastrho.simulate import make_recombination_map, RecombPriors

L_COAL = int(os.environ.get("SELF_LCOAL", 1_000_000))
CFG = dict(A.ATHAL)
_DEMOGS = ["SouthMiddleAtlas_1D17", "African2Epoch_1H18", "African3Epoch_1H18"]


def gen_one_coalescent(i, outdir, exclude_chrom=None):
    try:
        _gen_one_coalescent(i, outdir, exclude_chrom)
    except Exception as e:      # never let one region kill the pool
        print("ts_%08d FAILED (coal): %s" % (i, e), flush=True)


def _gen_one_coalescent(i, outdir, exclude_chrom=None):
    base = os.path.join(outdir, "ts_%08d" % i)
    if os.path.exists(base + ".trees"):
        return
    import stdpopsim
    sp = stdpopsim.get_species(CFG["species"])
    seed = 800_000 + i
    rng = np.random.default_rng(seed)
    s = A.draw_selfing(rng); F = A.selfing_F(s)
    n = int(rng.choice([50, 80, 120, 156, 200]))
    mu = 10.0 ** rng.uniform(-8.3, -8.0)          # ~5e-9..1e-8 (Arabidopsis ~7e-9)

    # map: 60% real AraTha slice, 40% synthetic (breadth + pericentromere shape)
    if rng.random() < 0.6:
        pos, rate, chrom, lo = A.load_real_map_slice(sp, CFG["map_id"], L_COAL, rng,
                                                     peri_bias=0.4, exclude_chrom=exclude_chrom)
        src = "real:%s" % chrom
    else:
        mean_r = 10.0 ** rng.uniform(-8.0, -7.4)
        kind = str(rng.choice(["gp", "hotspot", "pericentromere"], p=[0.4, 0.3, 0.3]))
        rm = make_recombination_map(L_COAL, rng, kind=kind, mean_rate=mean_r,
                                    priors=RecombPriors(sequence_length=L_COAL))
        pos = np.asarray(rm.position, float); rate = np.asarray(rm.rate, float); src = "synth:%s" % kind

    # DIPLOID selfing coalescent (ploidy=2 at Ne_eff = Ne/(1+F), effective recomb r*(1-F)), sampled
    # ONE genome per individual -- matches the SLiM ground truth and the validated seam check.
    # (The incumbent selfing_gen.py used ploidy=1/Ne_eff, which halves theta and rho; that would
    # make the SLiM and coalescent halves of the blend inconsistent. See selfing_seam_check.py.)
    eff_map = msprime.RateMap(position=pos, rate=rate * (1.0 - F))
    for attempt in range(6):
        try:
            if rng.random() < 0.7:               # 70% real AraTha Ne(t)
                demog = A.build_demography(sp, str(rng.choice(_DEMOGS)), F=F, Q=1.0)
                tc = msprime.sim_ancestry(samples=n, ploidy=2, demography=demog,
                                          recombination_rate=eff_map, sequence_length=pos[-1],
                                          random_seed=seed + attempt + 1)
            else:                                # 30% constant, for prior breadth
                Ne = 10.0 ** rng.uniform(5.0, 5.6)
                tc = msprime.sim_ancestry(samples=n, ploidy=2, population_size=Ne / (1.0 + F),
                                          recombination_rate=eff_map, sequence_length=pos[-1],
                                          random_seed=seed + attempt + 1)
            break
        except Exception:
            if attempt == 5:
                raise
    nodes = [int(tc.individual(j).nodes[0]) for j in range(n)]   # one haplotype / individual
    ts = msprime.sim_mutations(tc.simplify(nodes), rate=mu, random_seed=seed + 100)

    Ne_eff = max(1.0, float(ts.diversity(span_normalise=True)) / (4.0 * mu))
    meta = dict(seed=int(seed), n_samples=int(ts.num_samples), mutation_rate=float(mu),
                Ne=float(Ne_eff), selfing=float(s), sequence_length=float(pos[-1]),
                window_size=2000, num_sites=int(ts.num_sites),
                mode="coalescent_selfing", source=src)
    ts.dump(base + ".trees")
    np.savez(base + ".npz", map_position=pos, map_rate=rate * (1.0 - F), meta=json.dumps(meta))
    print("ts_%08d [coal %s] s=%.3f %dhap %dsites Ne=%.0f"
          % (i, src, s, ts.num_samples, ts.num_sites, Ne_eff), flush=True)


def main():
    outdir = sys.argv[1]; total = int(sys.argv[2])
    frac_slim = float(sys.argv[3]) if len(sys.argv) > 3 else 0.33
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    off = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    exclude = os.environ.get("SELF_EXCLUDE_CHROM")
    os.makedirs(outdir, exist_ok=True)
    n_slim = int(round(total * frac_slim))
    slim_idx = list(range(off, off + n_slim))
    coal_idx = list(range(off + n_slim, off + total))
    print("blend: %d SLiM + %d coalescent (frac_slim=%.2f) exclude=%s"
          % (len(slim_idx), len(coal_idx), frac_slim, exclude), flush=True)

    # coalescent fraction (fast, de-risks the pipeline first)
    from multiprocessing import get_context
    import selfing_slim_gen as S
    if coal_idx:
        with get_context("fork").Pool(nproc) as pool:
            pool.starmap(gen_one_coalescent, [(i, outdir, exclude) for i in coal_idx])
    # SLiM fraction (slow) -- reuse selfing_slim_gen.gen_one (bgs) in the SAME dir
    if slim_idx:
        with get_context("fork").Pool(min(nproc, 40)) as pool:
            pool.starmap(S.gen_one, [(i, outdir, "bgs", exclude) for i in slim_idx])
    print("done blend", outdir, total)


if __name__ == "__main__":
    main()
