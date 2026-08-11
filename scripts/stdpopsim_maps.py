"""Recover every stdpopsim genetic map with one frozen fastrho model.

For each species that ships an empirical recombination map in stdpopsim, slice real
chromosome regions, simulate them under a common in-prior constant-Ne coalescent (so the
*map shape* is real but demography is held fixed and in-distribution), and infer the map
with the single pretrained base model. Saves a per-species track + accuracy to
results/stdpopsim_<mode>.json.

Modes:
  phased    -- haplotypes as simulated.
  unphased  -- phase scrambled within each diploid (random per-site allele assignment),
               the worst case when haplotype phase is unknown.

Run on sesame:
  PYTHONNOUSERSITE=1 venvs/fastrho/bin/python scripts/stdpopsim_maps.py phased
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
import bench
from bench import Config, _sim_realmap, _resample, GRID
from fastrho.translate import (load_model, predict_map_from_genotype_matrix,
                               predict_intervals, dr_variant_stats)
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig

CKPT = "/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt"
STATS = "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"
OUTDIR = "/home/kkor/fastrho_data/campaign/results"
WORK = "/home/kkor/fastrho_data/stdpopsim_work"
DEVICE = "cuda:0"

# (key, species_id, map_id, common_name, latin, Ne, seq_len, n_dip, n_regions)
SPECIES = [
    ("human",      "HomSap", "HapMapII_GRCh38",        "Human",          "Homo sapiens",            10_000, 2_000_000, 50, 5),
    ("orangutan",  "PonAbe", "NaterPA_PonAbe3",        "Orangutan",      "Pongo abelii",            10_000, 2_000_000, 50, 5),
    ("baboon",     "PapAnu", "Pyrho_PAnubis1_0",       "Olive baboon",   "Papio anubis",            10_000, 2_000_000, 50, 5),
    ("dog",        "CanFam", "Campbell2016_CanFam3_1", "Dog",            "Canis familiaris",        10_000, 2_000_000, 50, 5),
    ("fly",        "DroMel", "ComeronCrossover_dm6",   "Fruit fly",      "Drosophila melanogaster", 10_000, 1_000_000, 50, 5),
    ("worm",       "CaeEle", "RockmanRIAIL_ce11",      "Nematode",       "Caenorhabditis elegans",  10_000, 2_000_000, 50, 5),
    ("arabidopsis","AraTha", "SalomeAveraged_TAIR10",  "Thale cress",    "Arabidopsis thaliana",    10_000, 2_000_000, 50, 5),
]


def unphase(gm, rng):
    """Scramble phase within each diploid: random per-site allele assignment between
    the two haplotypes of every individual. Preserves genotype dosage, destroys phase."""
    g = gm.copy()
    nh, ns = g.shape
    for k in range(nh // 2):
        a, b = g[2 * k].copy(), g[2 * k + 1].copy()
        swap = rng.random(ns) < 0.5
        g[2 * k] = np.where(swap, b, a)
        g[2 * k + 1] = np.where(swap, a, b)
    return g


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["phased", "unphased", "unphased_gt", "unphased_unpol_gt"])
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--stats", default=STATS)
    ap.add_argument("--tag", default="")          # output suffix to avoid clobbering
    ap.add_argument("--device", default=DEVICE)
    # Domain-randomized model: pass its feat_stats_dr.npz here (and the DR ckpt via --ckpt).
    # The featurizer becomes 18-dim (sfs_shape + r2_debias) and the (is_gt,is_folded) view
    # bits are appended to the conditioning. Only meaningful for the unphased*_gt modes.
    ap.add_argument("--dr-stats", default=None)
    a = ap.parse_args()
    mode = a.mode
    dr = a.dr_stats is not None
    if dr:
        a.stats = a.dr_stats
    # phased             : haplotype features, phased data
    # unphased           : haplotype features, phase scrambled (broken baseline)
    # unphased_gt        : phase-invariant composite-LD features, phase scrambled (rescue)
    # unphased_unpol_gt  : phase- AND polarization-invariant (composite-LD + folded)
    scramble = mode.startswith("unphased")
    use_gt = mode.endswith("_gt")
    fold = "unpol" in mode
    # DR model: 18-dim featurizer + view-bit conditioning; pick this view's stats.
    if use_gt and dr:
        gt_feat = GTTokenFeaturizer(
            config=FeatureConfig(sfs_shape=True, r2_debias=True), fold=fold)
    elif use_gt:
        gt_feat = GTTokenFeaturizer(fold=fold)
    else:
        gt_feat = None
    os.makedirs(OUTDIR, exist_ok=True)
    device = a.device
    model, mcfg, stats = load_model(a.ckpt, a.stats, device=device)
    if dr:
        variant = "gtf" if fold else "gt"
        stats = dr_variant_stats(stats, variant)
        cond_extra = [1.0, 1.0 if fold else 0.0]
        print(f"[DR] view={variant} cond_extra={cond_extra} cond_dim={mcfg.cond_dim} "
              f"n_features={mcfg.n_features}")
    else:
        cond_extra = None
    # same seed for unphased / unphased_gt -> same regions + same scrambling, comparable
    rng = np.random.default_rng(20260628 + (7 if scramble else 0))
    results = {}
    for key, spid, mapid, common, latin, Ne, seqlen, ndip, nreg in SPECIES:
        cfg = Config(name=key, demography="realmap", n_dip=ndip, mu=1.5e-8, Ne=float(Ne),
                     seq_len=seqlen, n_regions=nreg, genetic_map=mapid, species=spid)
        regions = []
        pool_t, pool_p = [], []   # all windows across regions -> genome-wide correlation
        for i in range(nreg):
            seed = int(rng.integers(1, 2**31 - 1))
            try:
                ts, rm = _sim_realmap(cfg, seed, i)
            except Exception as e:
                print(f"  [{key}] region {i} sim failed: {e}"); continue
            gm = ts.genotype_matrix().T.astype(np.int8)
            pos = ts.tables.sites.position.astype(np.float64)
            L = float(rm.position[-1])
            if scramble:
                gm = unphase(gm, rng)
            try:
                if use_gt:
                    from fastcxt.sfs import basic_filtering
                    gmf, posf = basic_filtering(gm.astype(np.int8), pos.astype(np.float64))
                    pred = predict_intervals(model, mcfg, stats, gmf, posf, cfg.mu,
                                             Ne=cfg.Ne, device=device, featurizer=gt_feat,
                                             cond_extra=cond_extra)
                else:
                    pred = predict_map_from_genotype_matrix(
                        gm, pos, model, mcfg, stats, mutation_rate=cfg.mu, Ne=cfg.Ne, device=device)
            except Exception as e:
                print(f"  [{key}] region {i} predict failed: {e}"); continue
            truth = _resample(rm.position, rm.rate, L)
            bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
            predr = _resample(bp, pred["r_per_bp"], L)
            k = min(len(truth), len(predr))
            t, p = truth[:k], predr[:k]
            ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
            if ok.sum() < 5:
                continue
            r = pearsonr(t[ok], p[ok])[0]
            lr = pearsonr(np.log(t[ok]), np.log(p[ok]))[0]
            pool_t.append(t[ok]); pool_p.append(p[ok])
            centers = (np.arange(k) * GRID + GRID / 2) / 1e6
            regions.append(dict(pearson=r, log_pearson=lr, L=L,
                                centers=centers[:k].tolist(),
                                truth=t.tolist(), pred=p.tolist(),
                                var=float(np.nanstd(np.log(t[ok])))))
            print(f"  [{key}] region {i}: r={r:.3f} logr={lr:.3f} ({ok.sum()} windows)")
        if not regions:
            print(f"  [{key}] NO valid regions"); continue
        best = max(regions, key=lambda d: d["var"])     # most structured -> nicest track
        T = np.concatenate(pool_t); P = np.concatenate(pool_p)   # genome-wide pooling
        pooled = float(pearsonr(T, P)[0])
        pooled_log = float(pearsonr(np.log(T), np.log(P))[0])
        results[key] = dict(
            species=spid, map=mapid, common=common, latin=latin, Ne=Ne,
            seq_len=seqlen, n_hap=2 * ndip,
            pearson=pooled, log_pearson=pooled_log,
            pearson_mean_region=float(np.mean([d["pearson"] for d in regions])),
            n_regions=len(regions), n_windows=int(T.size),
            track=dict(centers=best["centers"], truth=best["truth"], pred=best["pred"]),
        )
        print(f"== {key}: pooled r={pooled:.3f} (log {pooled_log:.3f}; "
              f"per-region mean {results[key]['pearson_mean_region']:.3f}; "
              f"n={results[key]['n_regions']} regions, {T.size} windows) ==")
    out = os.path.join(OUTDIR, f"stdpopsim_{mode}{a.tag}.json")
    with open(out, "w") as fh:
        json.dump(results, fh)
    print("wrote", out, "->", list(results))


if __name__ == "__main__":
    main()
