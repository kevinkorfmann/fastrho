"""Shared head-to-head benchmark: fastrho vs pyrho vs ReLERNN on identical regions.

Design for fairness:
  * constant Ne, fixed sample size & mutation rate (pyrho needs a matched lookup table;
    ReLERNN trains under equilibrium) -- only the recombination map varies.
  * every method's output is resampled onto a common bp-window grid and scored vs the
    known generative map with identical metrics.
  * primary metrics are scale-invariant correlations (Pearson/Spearman/log-Pearson), so
    cross-tool unit conventions don't bias the comparison; absolute bias is reported in
    per-bp r units (Ne known).

Subcommands:
  gen      -- simulate regions; dump .trees, true-map .npz, per-contig VCF, genome.bed
  fastrho  -- run a trained fastrho model; write per-contig (pos, r) predictions
  score    -- given truth + one or more method prediction files, print the comparison
"""

from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np

NE = 10_000.0
MU = 1.5e-8
N_DIP = 10                 # 20 haploids: pyrho exact table is cheap
SEQLEN = 2_000_000
GRID = 25_000             # scoring window (bp)


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------

def cmd_gen(args):
    import msprime
    from fastrho.simulate import make_recombination_map, RecombPriors
    os.makedirs(args.out, exist_ok=True)
    priors = RecombPriors(sequence_length=SEQLEN)
    rng = np.random.default_rng(args.seed)
    for i in range(args.n):
        kind = "hotspot" if i % 2 else "gp"
        rm = make_recombination_map(SEQLEN, np.random.default_rng(args.seed + i),
                                    kind=kind, mean_rate=1e-8, priors=priors)
        ts = msprime.sim_ancestry(samples=N_DIP, population_size=NE,
                                  recombination_rate=rm, sequence_length=SEQLEN,
                                  random_seed=args.seed + i + 1)
        ts = msprime.sim_mutations(ts, rate=MU, random_seed=args.seed + i + 1000)
        base = os.path.join(args.out, f"region_{i:03d}")
        ts.dump(base + ".trees")
        np.savez(base + ".npz", map_position=rm.position, map_rate=rm.rate,
                 meta=json.dumps(dict(Ne=NE, mutation_rate=MU, n_samples=N_DIP,
                                      sequence_length=SEQLEN, window_size=2000,
                                      contig=f"chr{i+1}")))
        with open(base + ".vcf", "w") as fh:
            ts.write_vcf(fh, contig_id=f"chr{i+1}")
        print(f"region {i}: {ts.num_sites} SNPs, kind={kind}")
    with open(os.path.join(args.out, "genome.bed"), "w") as fh:
        for i in range(args.n):
            fh.write(f"chr{i+1}\t0\t{SEQLEN}\n")
    print(f"wrote {args.n} regions to {args.out}")


# ---------------------------------------------------------------------------
# common scoring grid
# ---------------------------------------------------------------------------

def truth_windows(npz_path):
    from fastrho.preprocess import mean_rate_between
    z = np.load(npz_path, allow_pickle=True)
    edges = np.append(np.arange(0, SEQLEN, GRID), SEQLEN)
    true_r = mean_rate_between(z["map_position"], z["map_rate"], edges)
    return edges[:-1], true_r


def resample_to_grid(positions, rates):
    """Map a step function onto the GRID windows.

    `positions` are segment breakpoints (len == len(rates)+1); the function is extended
    with its edge rates to cover [0, SEQLEN].
    """
    from fastrho.preprocess import mean_rate_between
    pos = np.asarray(positions, float)
    rr = np.asarray(rates, float)
    if pos[0] > 0:
        pos = np.concatenate([[0.0], pos]); rr = np.concatenate([[rr[0]], rr])
    if pos[-1] < SEQLEN:
        pos = np.concatenate([pos, [SEQLEN]]); rr = np.concatenate([rr, [rr[-1]]])
    edges = np.append(np.arange(0, SEQLEN, GRID), SEQLEN)
    return mean_rate_between(pos, rr, edges)


# ---------------------------------------------------------------------------
# fastrho
# ---------------------------------------------------------------------------

def cmd_fastrho(args):
    import tskit
    from fastrho.translate import load_model, predict_map_from_ts
    model, cfg, stats = load_model(args.checkpoint, args.stats, device=args.device)
    out = {}
    for tp in sorted(glob.glob(os.path.join(args.data, "region_*.trees"))):
        base = tp[:-6]
        ts = tskit.load(tp)
        pred = predict_map_from_ts(ts, model, cfg, stats, mutation_rate=MU,
                                   Ne=NE, device=args.device)
        # per-interval r: breakpoints = [left[0], right...] (len S), rates (len S-1)
        bp = np.concatenate([[pred["pos_left"][0]], pred["pos_right"]])
        binned = resample_to_grid(bp, pred["r_per_bp"])
        out[os.path.basename(base)] = binned
    np.savez(args.save, **out)
    print(f"fastrho predictions -> {args.save} ({len(out)} regions)")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def cmd_score(args):
    from fastrho.evaluate import score_rates
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        average_precision_score = None

    regions = sorted(glob.glob(os.path.join(args.data, "region_*.npz")))
    truth = {}
    for npz in regions:
        name = os.path.basename(npz)[:-4]
        _, tr = truth_windows(npz)
        truth[name] = tr

    # restrict to regions solved by ALL methods (apples-to-apples)
    common = set(truth)
    for mp in args.methods:
        common &= set(np.load(mp, allow_pickle=True).files)
    common = sorted(common)
    truth = {k: truth[k] for k in common}

    f = max(1, args.grid // GRID)            # block-average 25kb base -> args.grid
    print(f"\n=== head-to-head on {len(truth)} common regions, {args.grid//1000}kb windows ===")
    print(f"{'method':<10} {'pearson':>8} {'spearman':>8} {'logP':>7} {'logL2':>7} "
          f"{'bias':>6} {'hotAUPRC':>9}")
    for mp in args.methods:
        method = os.path.basename(mp).replace("pred_", "").replace(".npz", "")
        preds = np.load(mp, allow_pickle=True)
        P, T = [], []
        ht, hs = [], []
        for name, tr in truth.items():
            if name not in preds.files:
                continue
            m = min(len(preds[name]), len(tr))
            pr = _block_mean(preds[name][:m], f)
            trb = _block_mean(tr[:m], f)
            P.append(pr); T.append(trb)
            ht.append((trb > 2 * np.median(trb)).astype(int)); hs.append(pr)
        P = np.concatenate(P); T = np.concatenate(T)
        s = score_rates(P, T)
        au = ""
        if average_precision_score is not None:
            yt = np.concatenate(ht)
            if yt.sum() > 0:
                au = f"{average_precision_score(yt, np.concatenate(hs)):.3f}"
        print(f"{method:<10} {s.get('pearson',float('nan')):>8.3f} "
              f"{s.get('spearman',float('nan')):>8.3f} {s.get('log_pearson',float('nan')):>7.3f} "
              f"{s.get('log_l2',float('nan')):>7.3f} {s.get('bias_ratio',float('nan')):>6.2f} {au:>9}")


def _relernn_windows_by_region(predict_path):
    """Parse a ReLERNN PREDICT.txt -> {region_XXX: (starts, ends, rates)}."""
    import csv, re
    by = {}
    with open(predict_path) as fh:
        rdr = csv.reader(fh, delimiter="\t")
        next(rdr)
        for row in rdr:
            if not row:
                continue
            chrom, s, e, r = row[0], float(row[1]), float(row[2]), float(row[-1])
            i = int(re.search(r"chr(\d+)", chrom).group(1)) - 1
            by.setdefault(f"region_{i:03d}", []).append((s, e, r))
    out = {}
    for name, recs in by.items():
        recs.sort()
        out[name] = (np.array([x[0] for x in recs]), np.array([x[1] for x in recs]),
                     np.array([x[2] for x in recs]))
    return out


def _stepfn_mean(pos, rate, a, b):
    from fastrho.preprocess import mean_rate_between
    return float(mean_rate_between(np.asarray(pos, float), np.asarray(rate, float),
                                   np.array([a, b], float))[0])


def cmd_score_native(args):
    """Score every method at ReLERNN's OWN data-driven window edges.

    This is the fair 'credit ReLERNN for the window-mean' comparison: instead of
    scoring on the fine 25-kb grid (below ReLERNN's resolution), we re-bin the true
    map -- and, if supplied, fastrho/pyrho -- onto each ReLERNN window and correlate
    there.  Reveals whether ReLERNN recovers the window mean (its documented strength)
    and how the methods rank once everything is at ReLERNN's native resolution.
    """
    from fastrho.evaluate import score_rates

    rel = _relernn_windows_by_region(args.relernn)
    # optional fine-grid predictions for the other methods (25-kb npz from cmd_fastrho/ingest)
    others = {}
    for tag, path in [("fastrho", args.fastrho), ("pyrho", args.pyrho)]:
        if path:
            others[tag] = np.load(path, allow_pickle=True)
    grid_pos = np.append(np.arange(0, SEQLEN, GRID), SEQLEN)

    acc = {"relernn": ([], [])}
    for t in others:
        acc[t] = ([], [])
    widths = []
    regions = sorted(glob.glob(os.path.join(args.data, "region_*.npz")))
    # restrict to regions present for ALL methods so per-method n matches (apples-to-apples)
    truth_names = {os.path.basename(p)[:-4] for p in regions}
    common = truth_names & set(rel)
    for preds in others.values():
        common &= {f for f in preds.files if not f.startswith("_")}
    n_reg = 0
    for npz in regions:
        name = os.path.basename(npz)[:-4]
        if name not in common:
            continue
        n_reg += 1
        z = np.load(npz, allow_pickle=True)
        mpos, mrate = z["map_position"], z["map_rate"]
        s, e, rr = rel[name]
        for a, b, rate in zip(s, e, rr):
            if b <= a:
                continue
            widths.append(b - a)
            true_m = _stepfn_mean(mpos, mrate, a, b)
            acc["relernn"][0].append(rate); acc["relernn"][1].append(true_m)
            for t, preds in others.items():
                if name in preds.files:
                    acc[t][0].append(_stepfn_mean(grid_pos, preds[name], a, b))
                    acc[t][1].append(true_m)

    w = np.array(widths)
    print(f"\n=== native-window scoring on {n_reg} regions, "
          f"{len(w)} ReLERNN windows (median {np.median(w)/1000:.0f} kb) ===")
    print(f"{'method':<10} {'pearson':>8} {'spearman':>8} {'logP':>7} {'bias':>6} {'n':>5}")
    out = {"config": os.path.basename(os.path.normpath(args.data)),
           "n_regions": n_reg, "n_windows": len(w),
           "median_window_kb": float(np.median(w) / 1000), "methods": {}}
    for method, (P, T) in acc.items():
        if not P:
            continue
        P = np.array(P); T = np.array(T)
        sm = score_rates(P, T)
        out["methods"][method] = sm
        print(f"{method:<10} {sm.get('pearson',float('nan')):>8.3f} "
              f"{sm.get('spearman',float('nan')):>8.3f} {sm.get('log_pearson',float('nan')):>7.3f} "
              f"{sm.get('bias_ratio',float('nan')):>6.2f} {len(P):>5}")
    if getattr(args, "out", None):
        with open(args.out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"native-window metrics -> {args.out}")


def cmd_ingest(args):
    """Fold an external method's per-region output onto the shared grid.

    pyrho:   <dir>/region_XXX.rmap  (cols: start end rho_per_bp)  -> r = rho/(4 Ne)
    relernn: a single PREDICT.txt (cols: chrom start end nSites recombRate=r_per_bp)
    """
    out = {}
    if args.kind == "pyrho":
        for rmap in sorted(glob.glob(os.path.join(args.dir, "region_*.rmap"))):
            name = os.path.basename(rmap)[:-5]
            rows = np.loadtxt(rmap, ndmin=2)
            if rows.size == 0:
                continue
            start, end, rho = rows[:, -3], rows[:, -2], rows[:, -1]
            bp = np.concatenate([[start[0]], end])
            # pyrho's per-bp rate is already in ~r units here; correlations are
            # scale-invariant regardless, so ingest as-is.
            out[name] = resample_to_grid(bp, rho)
    elif args.kind == "relernn":
        import csv, re
        by = {}
        with open(args.predict) as fh:
            rdr = csv.reader(fh, delimiter="\t")
            next(rdr)
            for row in rdr:
                if not row:
                    continue
                chrom, s, e, r = row[0], float(row[1]), float(row[2]), float(row[-1])
                by.setdefault(chrom, []).append((s, e, r))
        for chrom, recs in by.items():
            i = int(re.search(r"chr(\d+)", chrom).group(1)) - 1   # handles b'chr1'
            recs.sort()
            starts = np.array([x[0] for x in recs]); ends = np.array([x[1] for x in recs])
            rate = np.array([x[2] for x in recs])
            bp = np.concatenate([[starts[0]], ends])
            out[f"region_{i:03d}"] = resample_to_grid(bp, rate)
    np.savez(args.save, **out)
    print(f"ingested {args.kind}: {len(out)} regions -> {args.save}")


def main():
    ap = argparse.ArgumentParser(description="fastrho vs pyrho vs ReLERNN head-to-head")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--out", required=True)
    g.add_argument("--n", type=int, default=10); g.add_argument("--seed", type=int, default=7)
    f = sub.add_parser("fastrho")
    f.add_argument("--data", required=True); f.add_argument("--checkpoint", required=True)
    f.add_argument("--stats", required=True); f.add_argument("--save", required=True)
    f.add_argument("--device", default="cuda:0")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--methods", nargs="+", required=True, help="pred_*.npz files")
    s.add_argument("--grid", type=int, default=GRID, help="scoring window (bp)")
    ig = sub.add_parser("ingest")
    ig.add_argument("--kind", required=True, choices=["pyrho", "relernn"])
    ig.add_argument("--dir", default=None, help="pyrho: dir of region_*.rmap")
    ig.add_argument("--predict", default=None, help="relernn: PREDICT.txt")
    ig.add_argument("--save", required=True)
    sn = sub.add_parser("score-native", help="score all methods at ReLERNN's native windows")
    sn.add_argument("--data", required=True)
    sn.add_argument("--relernn", required=True, help="ReLERNN PREDICT.txt")
    sn.add_argument("--fastrho", default=None, help="pred_fastrho.npz (25kb grid)")
    sn.add_argument("--pyrho", default=None, help="pred_pyrho.npz (25kb grid)")
    sn.add_argument("--out", default=None, help="write native-window metrics JSON here")
    args = ap.parse_args()
    {"gen": cmd_gen, "fastrho": cmd_fastrho, "score": cmd_score,
     "ingest": cmd_ingest, "score-native": cmd_score_native}[args.cmd](args)


if __name__ == "__main__":
    main()
