"""Illustrative 10 Mb display tracks (true vs DR-predicted) WITH 95% CI bands for the
three panel-c species of Figure 5 (dog / nematode / thale cress).

This is a *display-only* track: it does NOT recompute or touch the genome-wide accuracy
numbers in _stdpopsim_*.json -- those stay locked to the paper / claims.py. It only
produces a longer (10 Mb), uncertainty-aware track for the figure, under the exact same
DR-folded view fig5 panel (c) already uses (unphased + unpolarized, composite-LD, folded).

For each species it simulates a few candidate 10 Mb real-map regions and keeps the most
structured one (largest log-rate std), mirroring stdpopsim_maps.py's "nicest track" pick.

Run from the isolated DR repo on sesame:
  cd /home/kkor/fastrho_dr
  PYTHONNOUSERSITE=1 PYTHONPATH=/home/kkor/fastrho_dr \
      /home/kkor/venvs/fastrho/bin/python scripts/gen_track10.py
"""
from __future__ import annotations
import os, sys, json
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(__file__))
import bench
from bench import Config, _sim_realmap, _resample, GRID
from fastrho.translate import load_model, predict_intervals, dr_variant_stats
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig

CKPT = ("/home/kkor/fastrho_data/campaign/train_dr/fastrho/version_0/"
        "checkpoints/epoch=50-val_pearson=0.846.ckpt")
DRSTATS = "/home/kkor/fastrho_data/campaign/shards_dr/feat_stats_dr.npz"
DEVICE = "cuda:0"
OUT = "/home/kkor/fastrho_dr/paper/figures/_stdpopsim_panelc_track10.json"

SEQLEN = 10_000_000
NDIP = 50
NCAND = 12         # candidate 10 Mb regions per species; keep the most identifiable-structured

# (key, species_id, map_id, common, latin, Ne)
SPECIES = [
    ("human",       "HomSap", "HapMapII_GRCh38",        "Human",       "Homo sapiens",            10_000),
    ("dog",         "CanFam", "Campbell2016_CanFam3_1", "Dog",         "Canis familiaris",        10_000),
    ("fly",         "DroMel", "ComeronCrossover_dm6",   "Fruit fly",   "Drosophila melanogaster", 10_000),
    ("baboon",      "PapAnu", "Pyrho_PAnubis1_0",       "Olive baboon", "Papio anubis",           10_000),
    ("worm",        "CaeEle", "RockmanRIAIL_ce11",      "Nematode",    "Caenorhabditis elegans",  10_000),
    ("arabidopsis", "AraTha", "SalomeAveraged_TAIR10",  "Thale cress", "Arabidopsis thaliana",    10_000),
    ("orangutan",   "PonAbe", "NaterPA_PonAbe3",        "Orangutan",   "Pongo abelii",            10_000),
]


def unphase(gm, rng):
    """Scramble phase within each diploid (matches stdpopsim_maps.unphase)."""
    g = gm.copy(); nh, ns = g.shape
    for k in range(nh // 2):
        a, b = g[2 * k].copy(), g[2 * k + 1].copy()
        swap = rng.random(ns) < 0.5
        g[2 * k] = np.where(swap, b, a)
        g[2 * k + 1] = np.where(swap, a, b)
    return g


def main():
    from fastcxt.sfs import basic_filtering
    model, mcfg, stats = load_model(CKPT, DRSTATS, device=DEVICE)
    stats = dr_variant_stats(stats, "gtf")              # folded (unpolarized) view
    cond_extra = [1.0, 1.0]                             # is_gt=1, is_folded=1
    gt_feat = GTTokenFeaturizer(
        config=FeatureConfig(sfs_shape=True, r2_debias=True), fold=True)
    print(f"[DR] view=gtf cond_extra={cond_extra} cond_dim={mcfg.cond_dim} "
          f"n_features={mcfg.n_features}")

    rng = np.random.default_rng(20260630)
    out = {}
    for key, spid, mapid, common, latin, Ne in SPECIES:
        cfg = Config(name=key, demography="realmap", n_dip=NDIP, mu=1.5e-8, Ne=float(Ne),
                     seq_len=SEQLEN, n_regions=1, genetic_map=mapid, species=spid)
        cands = []
        for j in range(NCAND):
            seed = int(rng.integers(1, 2**31 - 1))
            try:
                ts, rm = _sim_realmap(cfg, seed, j)
            except Exception as e:
                print(f"  [{key}] cand {j} sim fail: {e}"); continue
            gm = ts.genotype_matrix().T.astype(np.int8)
            pos = ts.tables.sites.position.astype(np.float64)
            L = float(rm.position[-1])
            gm = unphase(gm, rng)
            gmf, posf = basic_filtering(gm.astype(np.int8), pos.astype(np.float64))
            pred = predict_intervals(model, mcfg, stats, gmf, posf, cfg.mu,
                                     Ne=cfg.Ne, device=DEVICE, featurizer=gt_feat,
                                     cond_extra=cond_extra)
            truth = _resample(rm.position, rm.rate, L)
            bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
            predr = _resample(bp, pred["r_per_bp"], L)
            lo = _resample(bp, pred["r_ci_lo"], L)
            hi = _resample(bp, pred["r_ci_hi"], L)
            k = min(len(truth), len(predr), len(lo), len(hi))
            t, p, lo, hi = truth[:k], predr[:k], lo[:k], hi[:k]
            ok = np.isfinite(t) & np.isfinite(p) & (t > 0) & (p > 0)
            if ok.sum() < 50:
                print(f"  [{key}] cand {j}: too few valid windows ({ok.sum()})"); continue
            r = float(pearsonr(t[ok], p[ok])[0])
            var = float(np.nanstd(np.log(t[ok])))
            # Rates below FLOOR_ID (~4 Ne r < 1e-4) are essentially unidentifiable from LD (linkage
            # equilibrium; any LD method floors there). Score a candidate by its structure in the
            # IDENTIFIABLE range, and track the desert fraction so we don't display a region that is
            # dominated by unrecoverable near-zero valleys (which unfairly showcases the LD floor).
            FLOOR_ID = 1e-9
            score_id = float(np.nanstd(np.log(np.clip(t[ok], FLOOR_ID, None))))
            # desert fraction over ALL windows, including true zeros / map-gap windows (t==0),
            # which the t>0 mask would otherwise hide -- those ARE unidentifiable-desert windows.
            frac_desert = float(np.mean(np.nan_to_num(t, nan=0.0) < FLOOR_ID))
            centers = ((np.arange(k) * GRID + GRID / 2) / 1e6)
            cands.append(dict(r=r, var=var, score_id=score_id, frac_desert=frac_desert, L=L,
                              nsite=int(posf.size), centers=centers.tolist(), truth=t.tolist(),
                              pred=p.tolist(), ci_lo=lo.tolist(), ci_hi=hi.tolist()))
            print(f"  [{key}] cand {j}: r={r:.3f} score_id={score_id:.3f} desert={frac_desert:.2f} "
                  f"({ok.sum()} win, {posf.size} sites)")
        if not cands:
            print(f"  [{key}] NO valid candidates"); continue
        # Honest selection: drop only the desert-DOMINATED regions (near-zero rates are
        # unidentifiable from LD -- no method recovers them), then show the MEDIAN-recovery region
        # among what remains -- a REPRESENTATIVE region, not the best-structured / best-recovered
        # one. (We do not select on how well the model does beyond excluding unidentifiable deserts.)
        clean = [c for c in cands if c["frac_desert"] < 0.30]
        if not clean:                                              # no non-desert region exists
            best = min(cands, key=lambda d: d["frac_desert"])      # -> show the least-desert one
        else:
            elig = [c for c in clean if c["score_id"] > 0.6] or clean   # keep some real structure
            elig = sorted(elig, key=lambda d: d["r"])
            best = elig[len(elig) // 2]                            # median recovery = representative
        out[key] = dict(common=common, latin=latin, pearson_track=best["r"],
                        seq_len=SEQLEN, n_hap=2 * NDIP, L=best["L"],
                        centers=best["centers"], truth=best["truth"], pred=best["pred"],
                        ci_lo=best["ci_lo"], ci_hi=best["ci_hi"])
        print(f"== {key}: track r={best['r']:.3f} score_id={best['score_id']:.3f} desert={best['frac_desert']:.2f} L={best['L']/1e6:.2f}Mb ==")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(out, fh)
    print("wrote", OUT, "->", list(out))


if __name__ == "__main__":
    main()
