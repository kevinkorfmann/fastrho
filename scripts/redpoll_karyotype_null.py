"""Sample-size null for the redpoll arrangement-specific recombination maps.

Random subsets of the pooled 72-bird panel are drawn at the sizes of the two
homokaryotype groups.  If the within-arrangement rebound is biological rather
than a small-sample artifact, these mixed-karyotype subsets should retain the
deep pooled inversion trough.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from redpoll_karyotype_maps import (
    DEV,
    WINDOW,
    get_ck,
    infer,
    load_model,
    map_stats,
    subset_diploids,
    window_map,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", required=True)
    ap.add_argument("--fieldguide", required=True)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = np.load(args.hap, allow_pickle=True)
    fg = np.load(args.fieldguide, allow_pickle=True)
    gm = raw["gm"]
    pos = raw["pos"].astype(float)
    labels = fg["pca_labels"].astype(int)
    inv_start = int(fg["inv_start"])
    inv_end = int(fg["inv_end"])
    sizes = sorted(np.bincount(labels, minlength=3).tolist(), reverse=True)[:2]

    lo, hi = int(pos[0]), int(pos[-1])
    edges = np.append(np.arange(lo, hi, WINDOW), hi)
    centers = (edges[:-1] + edges[1:]) / 2
    ckpt, stats_path = get_ck("gt")
    model, cfg, stats = load_model(ckpt, stats_path, device=DEV)

    rng = np.random.default_rng(88117)
    all_individuals = np.arange(len(labels))
    records = []
    bundle = {"edges": edges, "centers": centers, "sizes": np.asarray(sizes)}
    for size in sizes:
        for rep in range(args.reps):
            inds = np.sort(rng.choice(all_individuals, size=size, replace=False))
            pred = infer(subset_diploids(gm, inds), pos, model, cfg, stats)
            rate = window_map(pred, edges)
            ms = map_stats(rate, centers, inv_start, inv_end)
            composition = np.bincount(labels[inds], minlength=3).tolist()
            rec = {
                "size": int(size),
                "rep": int(rep),
                "karyotype_composition": composition,
                **ms,
            }
            records.append(rec)
            bundle[f"n{size}_rep{rep}_rate"] = rate
            print(json.dumps(rec))

    summary = {"reps": args.reps, "records": records}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **bundle)
    with open(args.out.replace(".npz", ".json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
