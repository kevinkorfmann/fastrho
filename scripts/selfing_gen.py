"""Generate selfing-population training regions for a selfer-aware fastrho model.

A partially-selfing population (selfing rate s, F=s/(2-s)) is modelled in the coalescent by
scaling: effective recombination r_eff = r*(1-F), effective Ne_eff = Ne/(1+F), and each
(near-homozygous) accession contributes ONE haplotype (ploidy=1 sampling). The generative
*effective* map is saved as the target, so the model learns the selfer LD-decay -> rate mapping;
its relative landscape matches the meiotic map (Pearson/Spearman are scale-invariant).

Writes ts_{i}.trees + ts_{i}.npz (map_position, map_rate=effective, meta) like the base pipeline.
"""
import os, sys, json
import numpy as np
import msprime
sys.path.insert(0, "/home/kkor/fastrho")
from fastrho.simulate import make_recombination_map, RecombPriors

L = 1_000_000


def gen_one(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([50, 80, 120, 156, 200]))           # haploid lines
    mu = 10.0 ** rng.uniform(-8.3, -8.0)                    # ~5e-9..1e-8 (Arabidopsis ~7e-9)
    Ne = 10.0 ** rng.uniform(5.0, 5.6)                      # effective-ish 1e5..4e5
    mean_r = 10.0 ** rng.uniform(-8.0, -7.4)               # meiotic 1e-8..4e-8
    s = rng.uniform(0.90, 0.99)                             # selfing rate
    F = s / (2.0 - s)
    Ne_eff = Ne / (1.0 + F)
    priors = RecombPriors(sequence_length=L)
    kind = "hotspot" if rng.random() < 0.5 else "gp"
    rm = make_recombination_map(L, rng, kind=kind, mean_rate=mean_r, priors=priors)
    eff_rate = np.asarray(rm.rate, float) * (1.0 - F)       # effective recombination
    eff_map = msprime.RateMap(position=np.asarray(rm.position, float), rate=eff_rate)
    ts = msprime.sim_ancestry(samples=n, ploidy=1, population_size=Ne_eff,
                              recombination_rate=eff_map, sequence_length=L,
                              random_seed=int(rng.integers(1, 2**31)))
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))
    meta = dict(seed=int(seed), n_samples=n, mutation_rate=float(mu),
                mean_rate=float(np.average(eff_rate, weights=np.diff(rm.position))),
                Ne=float(Ne_eff), selfing=float(s), sequence_length=float(L),
                window_size=2000, num_sites=int(ts.num_sites))
    return ts, eff_map, meta


def dump(i, outdir):
    base = os.path.join(outdir, "ts_%08d" % i)
    if os.path.exists(base + ".trees"):
        return
    ts, rm, meta = gen_one(1_000 + i)
    ts.dump(base + ".trees")
    np.savez(base + ".npz", map_position=np.asarray(rm.position), map_rate=np.asarray(rm.rate),
             meta=json.dumps(meta))


def main():
    outdir = sys.argv[1]; n = int(sys.argv[2]); off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    os.makedirs(outdir, exist_ok=True)
    from multiprocessing import get_context
    with get_context("fork").Pool(40) as pool:
        pool.starmap(dump, [(off + i, outdir) for i in range(n)])
    print("done", outdir, n)


if __name__ == "__main__":
    main()
