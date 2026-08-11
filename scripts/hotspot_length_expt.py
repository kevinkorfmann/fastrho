"""Beat ReLERNN on its own diagnostic (their SI Fig. S13): detectability of a single hotspot
as a function of hotspot length. ReLERNN (one rate per ~100 kb window) can only see the longest
hotspots; fastrho (per SNP interval) resolves short ones.

Setup mirrors S13: a 250 kb region with a single centered hotspot of length L_h in
{1,2,4,6,8,10} kb, background r=2.5e-9, hotspot r=1.25e-7 (50x), n=20 haplotypes; many
replicates per length. Detectability = predicted fold-enrichment at the true hotspot interval
relative to the flanking background (true value = 50).

Subcommands (run on sesame):
  gen    -- simulate regions (msprime) -> region_*.{trees,vcf,npz}, genome.bed, config.json
  score  -- fastrho per-interval predict + parse ReLERNN PREDICT.txt -> enrichment vs length,
            write hotspot_length.json and fig_hotspot_length.pdf
"""
from __future__ import annotations
import os, csv, re, json, glob, argparse
import numpy as np

L_REGION = 250_000
BG = 2.5e-9
HS = 1.25e-7
LENGTHS_KB = [1, 2, 4, 6, 8, 10]
NREP = 25
NDIP = 10
MU = 1.5e-8
NE = 1e4
FAST = "#1f78b4"; REL = "#949494"


def cmd_gen(args):
    import msprime
    os.makedirs(args.out, exist_ok=True)
    idx = 0; index = []
    for Lh_kb in LENGTHS_KB:
        Lh = Lh_kb * 1000
        for rep in range(NREP):
            seed = args.seed + idx
            hs0 = (L_REGION - Lh) // 2; hs1 = hs0 + Lh
            pos = np.array([0, hs0, hs1, L_REGION], float)
            rate = np.array([BG, HS, BG], float)
            rm = msprime.RateMap(position=pos, rate=rate)
            ts = msprime.sim_ancestry(samples=NDIP, ploidy=2, population_size=NE,
                                      recombination_rate=rm, sequence_length=L_REGION,
                                      random_seed=seed)
            ts = msprime.sim_mutations(ts, rate=MU, model=msprime.BinaryMutationModel(),
                                       random_seed=seed + 1)
            base = os.path.join(args.out, f"region_{idx:03d}")
            ts.dump(base + ".trees")
            np.savez(base + ".npz", map_position=pos, map_rate=rate,
                     meta=json.dumps(dict(Ne=NE, mutation_rate=MU, n_samples=NDIP,
                                          sequence_length=L_REGION, contig=f"chr{idx+1}",
                                          hotspot_len=Lh, hs_start=hs0, hs_end=hs1)))
            with open(base + ".vcf", "w") as fh:
                ts.write_vcf(fh, contig_id=f"chr{idx+1}",
                             position_transform=lambda x: np.fmax(1, x))
            index.append(dict(region=idx, hotspot_len_kb=Lh_kb, hs0=hs0, hs1=hs1,
                              nsites=int(ts.num_sites)))
            idx += 1
        print(f"length {Lh_kb}kb: {NREP} reps done", flush=True)
    with open(os.path.join(args.out, "genome.bed"), "w") as fh:
        for i in range(idx):
            fh.write(f"chr{i+1}\t0\t{L_REGION}\n")
    json.dump(dict(mu=MU, Ne=NE, relernn_full=True, seq_len=L_REGION, n_dip=NDIP),
              open(os.path.join(args.out, "config.json"), "w"))
    json.dump(index, open(os.path.join(args.out, "hotspot_index.json"), "w"))
    print(f"wrote {idx} regions to {args.out}")


def _enrichment(edges, rates, hs0, hs1):
    """Mean predicted rate inside [hs0,hs1] / mean outside, given a step function."""
    edges = np.asarray(edges, float); rates = np.asarray(rates, float)
    inside = outside = win = wout = 0.0
    for a, b, r in zip(edges[:-1], edges[1:], rates):
        if not np.isfinite(r):
            continue
        ov = max(0.0, min(b, hs1) - max(a, hs0))   # overlap with hotspot
        out = (b - a) - ov
        inside += r * ov; win += ov
        outside += r * out; wout += out
    mi = inside / win if win > 0 else np.nan
    mo = outside / wout if wout > 0 else np.nan
    return mi / mo if (mo and mo > 0) else np.nan


def _relernn_by_contig(predict_path):
    by = {}
    with open(predict_path) as fh:
        rdr = csv.reader(fh, delimiter="\t"); next(rdr)
        for row in rdr:
            if not row:
                continue
            m = re.search(r"chr(\d+)", row[0])
            if not m:
                continue
            by.setdefault(int(m.group(1)) - 1, []).append(
                (float(row[1]), float(row[2]), float(row[-1])))
    out = {}
    for i, recs in by.items():
        recs.sort()
        s = np.array([r[0] for r in recs]); e = np.array([r[1] for r in recs])
        out[i] = (np.r_[s, e[-1]], np.array([r[2] for r in recs]))
    return out


def cmd_score(args):
    import tskit
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from fastrho.translate import load_model, predict_map_from_ts
    model, mcfg, stats = load_model(args.ckpt, args.stats, device=args.device)
    index = {d["region"]: d for d in json.load(open(os.path.join(args.data, "hotspot_index.json")))}
    rel = _relernn_by_contig(args.relernn) if args.relernn and os.path.exists(args.relernn) else {}

    rows = {}  # length_kb -> {"fastrho":[], "relernn":[]}
    for tp in sorted(glob.glob(os.path.join(args.data, "region_*.trees"))):
        i = int(re.search(r"region_(\d+)", tp).group(1))
        meta = index[i]; Lkb = meta["hotspot_len_kb"]; hs0, hs1 = meta["hs0"], meta["hs1"]
        ts = tskit.load(tp)
        pred = predict_map_from_ts(ts, model, mcfg, stats, mutation_rate=MU, Ne=NE,
                                   device=args.device)
        fx = np.r_[pred["pos_left"][0], pred["pos_right"]]
        fr = np.asarray(pred["r_per_bp"], float)
        rows.setdefault(Lkb, {"fastrho": [], "relernn": []})
        rows[Lkb]["fastrho"].append(_enrichment(fx, fr, hs0, hs1))
        if i in rel:
            rows[Lkb]["relernn"].append(_enrichment(rel[i][0], rel[i][1], hs0, hs1))

    summ = {}
    for Lkb in sorted(rows):
        d = rows[Lkb]
        summ[Lkb] = {m: {"mean": float(np.nanmean(v)), "se": float(np.nanstd(v) / max(1, np.sqrt(len(v)))),
                         "n": int(np.sum(np.isfinite(v)))}
                     for m, v in d.items() if v}
    json.dump(summ, open(os.path.join(args.out, "hotspot_length.json"), "w"), indent=2)
    print(json.dumps(summ, indent=2))

    xs = sorted(summ)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(50, ls=":", color="0.5", lw=1.2, label="true enrichment (50$\\times$)")
    ax.axhline(1, ls="--", color="0.7", lw=1, label="no detection (1$\\times$)")
    for m, col, lab in [("fastrho", FAST, "fastrho (per SNP interval)"),
                        ("relernn", REL, "ReLERNN (1 rate / window)")]:
        ys = [summ[x][m]["mean"] for x in xs if m in summ[x]]
        es = [summ[x][m]["se"] for x in xs if m in summ[x]]
        xx = [x for x in xs if m in summ[x]]
        if xx:
            ax.errorbar(xx, ys, yerr=es, marker="o", color=col, lw=2.2, capsize=3, label=lab)
    ax.set_xlabel("hotspot length (kb)"); ax.set_ylabel("predicted hotspot enrichment ($\\times$ background)")
    ax.set_title("Detecting short hotspots: fastrho resolves what ReLERNN cannot\n"
                 "(ReLERNN SI Fig. S13 setup: 250 kb region, 50$\\times$ hotspot, $n=20$)",
                 fontsize=11, loc="left")
    ax.set_xticks(xs); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False); ax.grid(ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_hotspot_length.pdf")
    fig.savefig(p, dpi=200); fig.savefig(p[:-4] + ".png", dpi=160); plt.close()
    print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--out", required=True); g.add_argument("--seed", type=int, default=9000)
    s = sub.add_parser("score")
    s.add_argument("--data", required=True); s.add_argument("--ckpt", required=True)
    s.add_argument("--stats", required=True); s.add_argument("--relernn", default=None)
    s.add_argument("--out", required=True); s.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    {"gen": cmd_gen, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
