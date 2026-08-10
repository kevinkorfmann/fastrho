"""Infer redpoll chromosome-1 recombination maps within inversion arrangements.

The pooled redpoll panel contains two inversion homokaryotypes (37 and 28 birds)
and seven heterokaryotypes.  A pooled LD map cannot distinguish meiotic crossover
suppression from long-range LD between the two arrangements.  This analysis uses
the inversion-genotype PCA labels saved by ``fieldguide_run.py`` to infer a map
within each homokaryotype class, where homologs are collinear and crossovers inside
the inversion should be viable.

Run on sesame::

    PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 ~/venvs/fastrho/bin/python \
      scripts/redpoll_karyotype_maps.py \
      --hap /home/kkor/realdata/hap/redpoll_chr1.npz \
      --fieldguide /home/kkor/realdata/fieldguide/redpoll_chr1.npz \
      --out /home/kkor/realdata/fieldguide/redpoll_karyotype_maps.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastcxt.sfs import basic_filtering
from fastrho.features import FeatureConfig
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.preprocess import mean_rate_between
from fastrho.translate import load_model, predict_intervals
from realdata_infer import get_ck


DEV = "cuda:0"
MU = 4.6e-9
WINDOW = 500_000


def infer(gm, pos, model, cfg, stats):
    gmf, posf = basic_filtering(gm.astype(np.int8), pos)
    return predict_intervals(
        model,
        cfg,
        stats,
        gmf,
        posf,
        MU,
        Ne=None,
        device=DEV,
        featurizer=GTTokenFeaturizer(config=FeatureConfig(), fold=True),
    )


def subset_diploids(gm, individuals):
    rows = np.ravel(np.column_stack((2 * individuals, 2 * individuals + 1)))
    return gm[rows]


def window_map(pred, edges):
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    return mean_rate_between(bp, pred["rho_per_bp"], edges)


def safe_corr(a, b, rank=False):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if ok.sum() < 3:
        return np.nan
    if rank:
        return float(spearmanr(a[ok], b[ok]).statistic)
    return float(pearsonr(np.log(a[ok]), np.log(b[ok])).statistic)


def map_stats(rate, centers, inv_start, inv_end):
    inside = (centers >= inv_start) & (centers < inv_end)
    flank = ~inside
    # A conservative core omits 2 Mb at each breakpoint, where mapping and
    # breakpoint-associated sequence properties can dominate a window.
    core = (centers >= inv_start + 2_000_000) & (centers < inv_end - 2_000_000)
    finite = np.isfinite(rate) & (rate > 0)

    def med(mask):
        x = rate[mask & finite]
        return float(np.median(x)) if x.size else np.nan

    inside_median = med(inside)
    core_median = med(core)
    flank_median = med(flank)
    return {
        "inside_median": inside_median,
        "core_median": core_median,
        "flank_median": flank_median,
        "inside_flank_ratio": inside_median / flank_median,
        "core_flank_ratio": core_median / flank_median,
        "n_inside": int((inside & finite).sum()),
        "n_core": int((core & finite).sum()),
        "n_flank": int((flank & finite).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", required=True)
    ap.add_argument("--fieldguide", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = np.load(args.hap, allow_pickle=True)
    fg = np.load(args.fieldguide, allow_pickle=True)
    gm = raw["gm"]
    pos = raw["pos"].astype(float)
    labels = fg["pca_labels"].astype(int)
    inv_start = int(fg["inv_start"])
    inv_end = int(fg["inv_end"])
    sizes = np.bincount(labels, minlength=3)
    if gm.shape[0] != 2 * len(labels):
        raise ValueError("PCA labels and pseudo-haplotype rows do not align")
    if sorted(sizes.tolist()) != [7, 28, 37]:
        raise ValueError(f"Unexpected karyotype sizes: {sizes.tolist()}")

    homo_labels = [int(x) for x in np.argsort(sizes)[-2:]]
    homo_labels.sort(key=lambda x: float(np.mean(fg["pca_pc1"][labels == x])))
    names = ["arrangement_A", "arrangement_B"]
    groups = {name: np.flatnonzero(labels == lab) for name, lab in zip(names, homo_labels)}

    lo = int(pos[0])
    hi = int(pos[-1])
    edges = np.append(np.arange(lo, hi, WINDOW), hi)
    centers = (edges[:-1] + edges[1:]) / 2

    ckpt, stats_path = get_ck("gt")
    model, cfg, stats = load_model(ckpt, stats_path, device=DEV)

    bundle = {
        "window_bp": np.array(WINDOW),
        "edges": edges,
        "centers": centers,
        "inv_start": np.array(inv_start),
        "inv_end": np.array(inv_end),
        "karyotype_sizes": sizes,
        "pca_labels": labels,
    }
    summary = {"window_bp": WINDOW, "inv_start": inv_start, "inv_end": inv_end, "groups": {}}

    full_maps = {}
    half_maps = {}
    for seed_offset, (name, inds) in enumerate(groups.items()):
        pred = infer(subset_diploids(gm, inds), pos, model, cfg, stats)
        rate = window_map(pred, edges)
        full_maps[name] = rate
        bundle[f"{name}_rate"] = rate
        bundle[f"{name}_individuals"] = inds

        rng = np.random.default_rng(7300 + seed_offset)
        shuffled = rng.permutation(inds)
        split = len(shuffled) // 2
        halves = [shuffled[:split], shuffled[split:]]
        hmaps = []
        for h, half in enumerate(halves, start=1):
            hp = infer(subset_diploids(gm, half), pos, model, cfg, stats)
            hm = window_map(hp, edges)
            hmaps.append(hm)
            bundle[f"{name}_half{h}_rate"] = hm
        half_maps[name] = hmaps

        ms = map_stats(rate, centers, inv_start, inv_end)
        ms.update(
            n_individuals=int(len(inds)),
            Ne_estimated=float(pred["Ne_estimated"]),
            half_repro_log_pearson=safe_corr(hmaps[0], hmaps[1]),
            half_repro_spearman=safe_corr(hmaps[0], hmaps[1], rank=True),
        )
        summary["groups"][name] = ms
        print(name, json.dumps(ms, indent=2))

    inside = (centers >= inv_start) & (centers < inv_end)
    flank = ~inside
    a, b = full_maps[names[0]], full_maps[names[1]]
    comparisons = {
        "between_arrangements_log_pearson_all": safe_corr(a, b),
        "between_arrangements_log_pearson_inside": safe_corr(a[inside], b[inside]),
        "between_arrangements_log_pearson_flanks": safe_corr(a[flank], b[flank]),
        "between_arrangements_spearman_all": safe_corr(a, b, rank=True),
        "between_arrangements_spearman_inside": safe_corr(a[inside], b[inside], rank=True),
        "between_arrangements_spearman_flanks": safe_corr(a[flank], b[flank], rank=True),
        "within_arrangement_floor_mean": float(
            np.nanmean([safe_corr(*half_maps[name]) for name in names])
        ),
    }
    comparisons["excess_arrangement_divergence"] = (
        comparisons["within_arrangement_floor_mean"]
        - comparisons["between_arrangements_log_pearson_all"]
    )
    summary["comparisons"] = comparisons
    print("comparisons", json.dumps(comparisons, indent=2))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **bundle)
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
