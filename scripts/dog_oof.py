"""Out-of-family demography stress test for the dogbn model (addresses the 'validated only
on its own generator' critique). Eval the EXISTING model (no retraining) on demographies
whose SHAPE differs from the piecewise-constant training family:
  * village_exp : smooth EXPONENTIAL decline (present 30-50k -> ancestral ~10k), not steps
  * breed_grad  : GRADUAL multi-generation crash (15k -> 150 ramped over ~150 gen), not a step
If village shape recovery holds (~0.7+), the model generalizes beyond its training family.

Usage: python scripts/dog_oof.py <ckpt.txt> <feat_stats.npz> [n]
"""
import sys
import numpy as np
import msprime
from scipy.stats import pearsonr

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fastrho.translate import load_model, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from fastcxt.sfs import basic_filtering
from scripts.dog_gen import make_recombination_map, RecombPriors, L, MAX_RHO

ckpt = open(sys.argv[1]).read().strip()
stats_p = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 80
DEV = "cuda:0"
model, cfg, stats = load_model(ckpt, stats_p, device=DEV)


def build_fc(s):
    kw = {}
    if "ld_radii" in s: kw["ld_radii"] = tuple(int(x) for x in np.asarray(s["ld_radii"]).ravel())
    if "disjoint_bands" in s: kw["disjoint_bands"] = bool(int(s["disjoint_bands"]))
    if "stride_after" in s: kw["stride_after"] = int(s["stride_after"])
    if "max_neighbors" in s: kw["max_neighbors"] = int(s["max_neighbors"])
    return FeatureConfig(**kw)


FEAT = GTTokenFeaturizer(config=build_fc(stats), fold=True)
EDGES = np.append(np.arange(0, L, 100000), L)


def demo_village_exp(rng):
    """Smooth exponential decline via fine piecewise steps (OUT of the step-family)."""
    ne0 = 10 ** rng.uniform(np.log10(3e4), np.log10(5e4))
    ne_anc = 10 ** rng.uniform(np.log10(8e3), np.log10(1.5e4))
    ts = np.linspace(0, 20000, 20)
    sizes = ne0 * (ne_anc / ne0) ** (ts / 20000.0)
    d = msprime.Demography(); d.add_population(initial_size=sizes[0])
    for t, s in zip(ts[1:], sizes[1:]):
        d.add_population_parameters_change(time=float(t), initial_size=float(s))
    return d, float(ne0)


def demo_breed_grad(rng):
    """Gradual multi-generation breed crash (ramped), not a single step."""
    ne_now = 10 ** rng.uniform(np.log10(80), np.log10(400))
    ne_pre = 10 ** rng.uniform(np.log10(2e4), np.log10(5e4))
    t_end = rng.uniform(120, 200)
    ts = np.linspace(0, t_end, 12)
    sizes = ne_now * (ne_pre / ne_now) ** (ts / t_end)
    d = msprime.Demography(); d.add_population(initial_size=sizes[0])
    for t, s in zip(ts[1:], sizes[1:]):
        d.add_population_parameters_change(time=float(t), initial_size=float(s))
    d.add_population_parameters_change(time=float(t_end + 1), initial_size=float(ne_pre))
    return d, float(ne_now)


def gen(seed, which):
    rng = np.random.default_rng(seed)
    n_dip = int(rng.choice([30, 50, 67]))
    mu = 10 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))
    mean_r = 10 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))
    demo, ne_pres = (demo_village_exp(rng) if which == "village" else demo_breed_grad(rng))
    ne_max = max(p.initial_size for p in demo.populations) if False else 5e4
    rt = 4.0 * ne_max * mean_r * L
    if rt > MAX_RHO: mean_r *= MAX_RHO / rt
    rm = make_recombination_map(L, rng, kind=("hotspot" if rng.random() < 0.5 else "gp"),
                                mean_rate=mean_r, priors=RecombPriors(sequence_length=L))
    ts = msprime.sim_ancestry(samples=n_dip, demography=demo, recombination_rate=rm,
                              sequence_length=L, random_seed=int(rng.integers(1, 2**31)))
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))
    return rm, ts, mu


def lp(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return pearsonr(np.log(a[ok]), np.log(b[ok]))[0] if ok.sum() >= 8 else np.nan


for which in ["village", "breed"]:
    scores = []
    for i in range(N):
        try:
            rm, ts, mu = gen(20_000 + i if which == "village" else 40_000 + i, which)
            gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position.astype(np.float64)
            gmf, posf = basic_filtering(gm, pos)
            if gmf.shape[1] < 12: continue
            pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=DEV, featurizer=FEAT)
            bp = np.concatenate([[pred["pos_left"][0]], pred["pos_right"]])
            pw = mean_rate_between(bp, pred["r_per_bp"], EDGES)
            tw = mean_rate_between(np.asarray(rm.position), np.asarray(rm.rate), EDGES)
            scores.append(lp(pw, tw))
        except Exception:
            continue
    sc = np.array(scores)
    print(f"OUT-OF-FAMILY {which:8s} (n={np.isfinite(sc).sum():3d}): 100kb logPearson median={np.nanmedian(sc):.3f}")
print("(in-family reference: village 0.78, breed-own ~0.2-0.35)")
