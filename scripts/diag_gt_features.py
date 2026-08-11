"""E0 diagnostic: validate phase-invariant GT features against haplotype features."""
import numpy as np, msprime
from fastrho.features import SNPTokenFeaturizer, feature_names
from fastrho.gt_features import GTTokenFeaturizer


def unphase(gm, rng):
    g = gm.copy()
    for k in range(g.shape[0] // 2):
        a, b = g[2 * k].copy(), g[2 * k + 1].copy()
        s = rng.random(g.shape[1]) < 0.5
        g[2 * k] = np.where(s, b, a); g[2 * k + 1] = np.where(s, a, b)
    return g


def cor(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
        return float("nan")
    return np.corrcoef(x[m], y[m])[0, 1]


rng = np.random.default_rng(0)
ts = msprime.sim_ancestry(samples=50, recombination_rate=1e-8, sequence_length=2_000_000,
                          population_size=10000, random_seed=3)
ts = msprime.sim_mutations(ts, rate=1.5e-8, random_seed=4)
gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position
print("n_hap", gm.shape[0], "S", gm.shape[1])
meta = {"sequence_length": 2_000_000.0}
hap = SNPTokenFeaturizer()(gm, pos, meta)["tokens"]
gtp = GTTokenFeaturizer()(gm, pos, meta)["tokens"]
gms = unphase(gm, rng)
gtu = GTTokenFeaturizer()(gms, pos, meta)["tokens"]
haps = SNPTokenFeaturizer()(gms, pos, meta)["tokens"]
names = feature_names()
print("\nGT phased vs GT unphased: max|diff| = %.2e (should be ~0)" % np.max(np.abs(gtp - gtu)))
print("\nper-feature Pearson r vs HAP(phased=training dist):")
print("%-16s %11s %12s %20s" % ("feature", "GT(phased)", "GT(unphased)", "HAP(unphased=broken)"))
for c, nm in enumerate(names):
    print("%-16s %11.3f %12.3f %20.3f" %
          (nm, cor(hap[:, c], gtp[:, c]), cor(hap[:, c], gtu[:, c]), cor(hap[:, c], haps[:, c])))
