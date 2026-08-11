"""P2: village->breed recombination-map TRANSFER demonstration (no retraining).

Dogs carry a PRDM9 pseudogene, so the fine-scale recombination landscape is promoter/CpG-
anchored and CONSERVED across canid populations. Therefore a breed's map -- unrecoverable
from the breed's own saturated-LD data -- can be obtained by inferring it from a CO-LOCATED
high-diversity panel (village dog / wolf) that shares the same locus, then re-anchoring scale.

This script proves the inference half: simulate PAIRED populations (ONE shared recombination
map; a village demography AND a breed demography), run the existing dogbn model on each, and
compare to the shared true map. Expectation: village-inferred ~0.78, breed-inferred ~0.2 ->
transfer recovers the breed's true map at the village level.

Usage: python scripts/dog_transfer.py <ckpt.txt> <feat_stats.npz> [n_regions] [out.pdf]
"""
import sys, json
import numpy as np
import msprime
from scipy.stats import pearsonr

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fastrho.translate import load_model, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from fastcxt.sfs import basic_filtering
from scripts.dog_gen import (_village_traj, _breed_traj, _demography_from_traj,
                             make_recombination_map, RecombPriors, L, MAX_RHO)

ckpt = open(sys.argv[1]).read().strip()
stats_p = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
out_pdf = sys.argv[4] if len(sys.argv) > 4 else "/home/kkor/fastrho_data/campaign_dog_bottleneck/dog_transfer.pdf"
DEV = "cuda:0"
model, cfg, stats = load_model(ckpt, stats_p, device=DEV)


def build_fc(stats):
    kw = {}
    if "ld_radii" in stats:
        kw["ld_radii"] = tuple(int(x) for x in np.asarray(stats["ld_radii"]).ravel())
    if "disjoint_bands" in stats:
        kw["disjoint_bands"] = bool(int(stats["disjoint_bands"]))
    if "stride_after" in stats:
        kw["stride_after"] = int(stats["stride_after"])
    if "max_neighbors" in stats:
        kw["max_neighbors"] = int(stats["max_neighbors"])
    return FeatureConfig(**kw)


FC = build_fc(stats)
FEAT = GTTokenFeaturizer(config=FC, fold=True)
EDGES = np.append(np.arange(0, L, 100000), L)


def gen_paired(seed):
    """One shared map; a village ts and a breed ts under it."""
    rng = np.random.default_rng(seed)
    n_dip = int(rng.choice([30, 50, 67]))
    mu = 10.0 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))
    mean_r = 10.0 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))
    vtraj = _village_traj(rng); btraj = _breed_traj(rng)
    ne_max = max(max(s for _, s in vtraj), max(s for _, s in btraj))
    rt = 4.0 * ne_max * mean_r * L
    if rt > MAX_RHO:
        mean_r *= MAX_RHO / rt
    rm = make_recombination_map(L, rng, kind=("hotspot" if rng.random() < 0.5 else "gp"),
                                mean_rate=mean_r, priors=RecombPriors(sequence_length=L))

    def sim(traj):
        ts = msprime.sim_ancestry(samples=n_dip, demography=_demography_from_traj(traj),
                                  recombination_rate=rm, sequence_length=L,
                                  random_seed=int(rng.integers(1, 2**31)))
        return msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))
    return rm, sim(vtraj), sim(btraj), mu


def infer_windowed(ts, mu):
    gm = ts.genotype_matrix().T.astype(np.int8)
    pos = ts.tables.sites.position.astype(np.float64)
    gmf, posf = basic_filtering(gm, pos)
    if gmf.shape[1] < 12:
        return None
    pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=DEV, featurizer=FEAT)
    bp = np.concatenate([[pred["pos_left"][0]], pred["pos_right"]])
    return mean_rate_between(bp, pred["r_per_bp"], EDGES)


def lp(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return pearsonr(np.log(a[ok]), np.log(b[ok]))[0] if ok.sum() >= 8 else np.nan


vil_self, brd_self, transfer = [], [], []
for i in range(N):
    try:
        rm, vts, bts, mu = gen_paired(10_000 + i)
    except Exception:
        continue
    true_w = mean_rate_between(np.asarray(rm.position), np.asarray(rm.rate), EDGES)
    vw = infer_windowed(vts, mu)
    bw = infer_windowed(bts, mu)
    if vw is None or bw is None:
        continue
    vil_self.append(lp(vw, true_w))       # village inference vs shared true map
    brd_self.append(lp(bw, true_w))       # breed inference vs SAME true map (its own map)
    transfer.append(lp(vw, true_w))       # transfer = village-inferred used as the breed map

vil_self = np.array(vil_self); brd_self = np.array(brd_self)
print(f"PAIRED regions (shared map): n={len(vil_self)}")
print(f"  breed map from BREED's own data   : 100kb logPearson = {np.nanmedian(brd_self):.3f}")
print(f"  breed map by TRANSFER from village : 100kb logPearson = {np.nanmedian(vil_self):.3f}")
print(f"  -> transfer lifts the breed map from {np.nanmedian(brd_self):.2f} to "
      f"{np.nanmedian(vil_self):.2f} (PRDM9-loss conserved landscape)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.boxplot([brd_self[np.isfinite(brd_self)], vil_self[np.isfinite(vil_self)]],
           tick_labels=["breed\n(own data)", "breed\n(village transfer)"], showfliers=False)
ax.set_ylabel("breed-map 100 kb logPearson"); ax.set_ylim(-0.2, 1)
ax.axhline(0, color="grey", lw=0.6)
ax.set_title("village->breed transfer recovers the conserved map")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(out_pdf)
print("wrote", out_pdf)
