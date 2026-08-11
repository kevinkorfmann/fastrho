"""Reproduce a ReLERNN main-text figure (Fig 2A) and run the full three-method showdown on it.

Paper: Adrion, Galloway & Kern 2020, Fig 2A -- a simulated *D. melanogaster* chr 2L
recombination landscape (Comeron et al. 2012 crossover map) inferred by ReLERNN at n=20,
reported R^2 = 0.931 (MAE 3.72e-8) on 100-kb windows.

We reproduce that result to confirm we drive ReLERNN as intended, and -- because this is
ReLERNN's own headline dataset -- we additionally run fastrho and pyrho on the *identical*
simulated genotypes so the figure shows how all three methods recover the Comeron 2L map.

Reproduction parameters (from their Methods, "Testing the Accuracy..."):
  real Comeron 2L map (stdpopsim ComeronCrossover_dm6), Ne = 2.5e5, mu_bar = 2.8e-9,
  n = 20 chromosomes, rho_max ~ 1.2e-7 c/bp.  ReLERNN's per-base rate prior is
  r ~ U(0, mu * upperRhoThetaRatio)  (ReLERNN_SIMULATE: priorHighsRho = assumedMu*upRTR),
  so to cover the map peak we set upRTR = ceil(SAFETY * map_peak / mu).

Stages (driven by repro_relernn.sh; each runs in its own venv):
  data     -- build Comeron 2L VCF + true-map npz + genome.bed + repro_meta.json + config.json
              (config.json also carries the pyrho size-history fields so run_pyrho_config works)
  fastrho  -- single forward pass of the frozen model on region_000.vcf -> fastrho_pred.npz
  (pyrho)  -- run_pyrho_config.py builds an Ne=2.5e5 ldpop table + optimize -> region_000.rmap
  (ReLERNN)-- run_relernn_config.py SIMULATE->TRAIN->PREDICT -> relernn_proj/*.PREDICT.txt
  score    -- rebin truth + every available method to 100-kb windows (paper metric), compute
              per-method Pearson/R^2/MAE/bias, and write repro_showdown.npz + repro_metrics.json
  plot     -- render the stacked landscape figure from repro_showdown.npz (paper_style)
"""
from __future__ import annotations

import os
import json
import math
import argparse

import numpy as np

# Paper Fig 2A parameters
NE = 2.5e5
MU = 2.8e-9
N_DIP = 10          # 20 haploid chromosomes
SAFETY = 1.15       # headroom above the map peak for the rate prior ceiling
REPORT_WIN = 100_000  # paper reports on 100-kb windows
HI_WIN = 25_000       # fine "high-res" scale for the multires figure (the paper's 25-kb eval scale)

# fastrho checkpoints (sesame paths). The "base" model's training prior caps at
# log10_Ne=4.8 (Ne~63k), so Drosophila's Ne=2.5e5 is 4x out-of-distribution for it; the
# "highne" model (campaign2) is trained with the broadened dipteran/mosquito prior
# (--log10-ne-max ~6.3, Ne up to ~2e6) and is the correct model for this regime -- it is the
# same model used for the Anopheles/Ag1000G analyses (infer_agam.py) and real_drosophila.
MODELS = {
    "hidip":  ("/home/kkor/fastrho_data/campaign_hidip/train15k/fastrho/version_0/"
               "checkpoints/epoch=37-val_loss=-0.178.ckpt",
               "/home/kkor/fastrho_data/campaign_hidip/shards15k/feat_stats.npz"),
    "highne": ("/home/kkor/fastrho_data/campaign2/train/fastrho/version_0/"
               "checkpoints/epoch=49-val_loss=-0.151.ckpt",
               "/home/kkor/fastrho_data/campaign2/shards/feat_stats.npz"),
    "base":   ("/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/"
               "checkpoints/epoch=45-val_loss=-0.019.ckpt",
               "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"),
}

# canonical method order + display labels for the figure / metrics
METHODS = ("fastrho", "pyrho", "relernn")
LABELS = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def cmd_data(args):
    import stdpopsim
    import msprime

    sp = stdpopsim.get_species("DroMel")
    contig = sp.get_contig("2L", genetic_map="ComeronCrossover_dm6", mutation_rate=MU)
    rmap = contig.recombination_map           # msprime.RateMap from the Comeron map
    L = int(rmap.sequence_length)
    peak = float(np.nanmax(rmap.rate))
    mean_r = float(np.nansum(rmap.rate * np.diff(rmap.position)) / L)
    up_rtr = int(math.ceil(SAFETY * peak / MU))

    os.makedirs(args.out, exist_ok=True)
    print(f"Comeron 2L: length={L:,} bp  peak_r={peak:.3e}  mean_r={mean_r:.3e}")
    print(f"recommended upperRhoThetaRatio = ceil({SAFETY}*{peak:.3e}/{MU:.1e}) = {up_rtr}")

    ts = msprime.sim_ancestry(
        samples=N_DIP, ploidy=2, population_size=NE,
        recombination_rate=rmap, sequence_length=L, random_seed=args.seed)
    # binary model guarantees biallelic sites (no multiallelic at this high theta)
    ts = msprime.sim_mutations(ts, rate=MU, model=msprime.BinaryMutationModel(),
                               random_seed=args.seed + 1)
    print(f"simulated {ts.num_sites:,} segregating sites, {ts.num_samples} haplotypes")

    base = os.path.join(args.out, "region_000")          # single contig, named so run_relernn_config can reuse
    np.savez(base + ".npz", map_position=rmap.position, map_rate=rmap.rate,
             meta=json.dumps(dict(Ne=NE, mutation_rate=MU, n_samples=N_DIP,
                                  sequence_length=L, contig="2L",
                                  source="ComeronCrossover_dm6")))
    with open(base + ".vcf", "w") as fh:
        ts.write_vcf(fh, contig_id="2L")
    with open(os.path.join(args.out, "genome.bed"), "w") as fh:
        fh.write(f"2L\t0\t{L}\n")
    # config.json carries both the ReLERNN flags and the pyrho size-history fields
    # (n_dip/popsizes/epochtimes) so run_pyrho_config.py can build a matched ldpop table.
    with open(os.path.join(args.out, "config.json"), "w") as fh:
        json.dump(dict(mu=MU, relernn_full=True, up_rtr=up_rtr, Ne=NE,
                       contig="2L", sequence_length=L,
                       n_dip=N_DIP, popsizes=[NE], epochtimes=[]), fh)
    with open(os.path.join(args.out, "repro_meta.json"), "w") as fh:
        json.dump(dict(length=L, peak_rate=peak, mean_rate=mean_r,
                       recommended_upRTR=up_rtr, assumedMu=MU, Ne=NE,
                       n_chromosomes=2 * N_DIP, num_sites=int(ts.num_sites)), fh, indent=2)
    print(f"wrote dataset to {args.out} (use upRTR={up_rtr}, assumedMu={MU})")


# ---------------------------------------------------------------------------
# fastrho: single frozen-model forward pass on the identical region_000.vcf
# ---------------------------------------------------------------------------

def cmd_fastrho(args):
    # predict_map_from_vcf -> fastrho.io.read_vcf handles tskit's numeric 0/1 alleles, so the
    # frozen model runs on the same region_000.vcf ReLERNN/pyrho use (each `a|b` -> 2 haplotypes).
    from fastrho.translate import load_model, predict_map_from_vcf

    ckpt = args.checkpoint or MODELS[args.model][0]
    stats_p = args.stats or MODELS[args.model][1]
    print(f"fastrho model='{args.model}'  ckpt={os.path.basename(ckpt)}")
    model, cfg, stats = load_model(ckpt, stats_p, device=args.device)
    vcf = os.path.join(args.data, "region_000.vcf")
    # known Ne (2.5e5) supplied for the absolute-rate conversion, exactly as the benchmark
    # engine does; the headline correlation is invariant to this scalar.
    pred = predict_map_from_vcf(vcf, model, cfg, stats, contig="2L",
                                mutation_rate=MU, Ne=NE, device=args.device)
    out = os.path.join(args.data, "fastrho_pred.npz")
    np.savez(out,
             pos_left=pred["pos_left"], pos_right=pred["pos_right"],
             r_per_bp=pred["r_per_bp"], r_ci_lo=pred["r_ci_lo"], r_ci_hi=pred["r_ci_hi"],
             Ne_used=pred["Ne_used"], Ne_estimated=pred["Ne_estimated"])
    print(f"fastrho: {len(pred['r_per_bp'])} SNP intervals, "
          f"Ne_used={pred['Ne_used']:.3e} (est {pred['Ne_estimated']:.3e}) -> {out}")


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def _read_predict(predict_path, contig="2L"):
    """ReLERNN PREDICT.txt -> sorted (starts, ends, window_rates) for one contig."""
    import csv
    s, e, r = [], [], []
    with open(predict_path) as fh:
        rdr = csv.reader(fh, delimiter="\t")
        next(rdr)
        for row in rdr:
            # ReLERNN writes the contig as a bytes-repr, e.g. b'2L'; match by substring
            if not row or contig not in row[0]:
                continue
            s.append(float(row[1])); e.append(float(row[2])); r.append(float(row[-1]))
    order = np.argsort(s)
    return np.array(s)[order], np.array(e)[order], np.array(r)[order]


def _read_pyrho_rmap(rmap_path):
    """pyrho optimize .rmap -> (starts, ends, rates). Last column is the per-bp rate r
    directly (NOT rho=4Ne r): verified against known truth on a benchmark region."""
    rows = np.loadtxt(rmap_path, ndmin=2)
    if rows.size == 0:
        return None
    return rows[:, -3], rows[:, -2], rows[:, -1]


def _step_rebin(starts, ends, rates, edges):
    """Span-weighted rebin of a contiguous step function onto `edges` (len(edges)-1 values).
    Pads the gap before the first / after the last interval with the edge rate so partial
    boundary windows are filled rather than dropped (matches the original ReLERNN rebinning)."""
    starts = np.asarray(starts, float); ends = np.asarray(ends, float)
    rates = np.asarray(rates, float)
    bp = np.concatenate([[starts[0]], ends])          # breakpoints, len = len(rates)+1
    rr = rates
    if bp[0] > edges[0]:
        bp = np.concatenate([[edges[0]], bp]); rr = np.concatenate([[rates[0]], rr])
    if bp[-1] < edges[-1]:
        bp = np.concatenate([bp, [edges[-1]]]); rr = np.concatenate([rr, [rates[-1]]])
    from fastrho.preprocess import mean_rate_between
    return mean_rate_between(bp, rr, edges)


def _metrics(pred, true):
    pred = np.asarray(pred, float); true = np.asarray(true, float)
    # drop masked/zero windows (methods emit 0 where they have no signal; true=0 in masked intervals)
    ok = np.isfinite(pred) & np.isfinite(true) & (pred > 0) & (true > 0)
    pred, true = pred[ok], true[ok]
    if len(pred) < 3:
        return dict(pearson=float("nan"), r2=float("nan"), mae=float("nan"),
                    bias=float("nan"), n=int(len(pred)))
    r = float(np.corrcoef(pred, true)[0, 1])
    return dict(pearson=r, r2=r * r, mae=float(np.mean(np.abs(pred - true))),
                bias=float(np.median(pred / true)), n=int(len(pred)))


def cmd_score(args):
    z = np.load(os.path.join(args.data, "region_000.npz"), allow_pickle=True)
    mpos, mrate = z["map_position"], z["map_rate"]
    mrate = np.where(np.isfinite(mrate), mrate, 0.0)   # Comeron map has NaN in masked intervals
    L = int(mpos[-1])
    edges = np.append(np.arange(0, L, REPORT_WIN), L)
    centers = (edges[:-1] + edges[1:]) / 2.0
    from fastrho.preprocess import mean_rate_between
    true_100 = mean_rate_between(mpos, mrate, edges)

    # --- gather whatever methods are available on disk ---
    native = {}    # method -> (starts, ends, rates)  (for the record / future fine plots)
    rebinned = {}  # method -> 100-kb array aligned to `edges`

    if args.predict and os.path.exists(args.predict):
        s, e, rel = _read_predict(args.predict)
        if len(rel):
            native["relernn"] = (s, e, rel)
            rebinned["relernn"] = _step_rebin(s, e, rel, edges)
            print(f"ReLERNN: {len(rel)} native windows (~{np.median(e - s) / 1000:.0f} kb)")

    fpath = os.path.join(args.data, "fastrho_pred.npz")
    if os.path.exists(fpath):
        fz = np.load(fpath, allow_pickle=True)
        starts, ends, rr = fz["pos_left"], fz["pos_right"], fz["r_per_bp"]
        native["fastrho"] = (starts, ends, rr)
        rebinned["fastrho"] = _step_rebin(starts, ends, rr, edges)
        print(f"fastrho: {len(rr)} SNP intervals")

    rpath = os.path.join(args.data, "region_000.rmap")
    if os.path.exists(rpath):
        pr = _read_pyrho_rmap(rpath)
        if pr is not None:
            native["pyrho"] = pr
            rebinned["pyrho"] = _step_rebin(pr[0], pr[1], pr[2], edges)
            print(f"pyrho: {len(pr[2])} intervals")

    if not rebinned:
        raise SystemExit("no method outputs found (need --predict, fastrho_pred.npz, or region_000.rmap)")

    # --- per-method 100-kb metrics (paper scale) ---
    metrics = {}
    for k in METHODS:
        if k in rebinned:
            m = min(len(rebinned[k]), len(true_100))
            metrics[k] = _metrics(rebinned[k][:m], true_100[:m])
            mm = metrics[k]
            print(f"{LABELS[k]:>8} 100kb: Pearson={mm['pearson']:.3f} R^2={mm['r2']:.3f} "
                  f"MAE={mm['mae']:.2e} bias={mm['bias']:.2f}  (n={mm['n']})")
    print("  [paper Fig 2A: ReLERNN R^2=0.931, MAE=3.72e-8]")

    # --- fine ("high-res") scale: the SAME maps + method outputs rebinned to HI_WIN (25 kb),
    #     the paper's fine evaluation scale. fastrho/pyrho stay accurate here while ReLERNN,
    #     whose native windows are ~100 kb, degrades to a staircase -- the multires figure. ---
    edges_hi = np.append(np.arange(0, L, HI_WIN), L)
    centers_hi = (edges_hi[:-1] + edges_hi[1:]) / 2.0
    true_hi = mean_rate_between(mpos, mrate, edges_hi)
    rebinned_hi, metrics_hi = {}, {}
    for k in METHODS:
        if k in native:
            s0, e0, r0 = native[k]
            rebinned_hi[k] = _step_rebin(s0, e0, r0, edges_hi)
            m = min(len(rebinned_hi[k]), len(true_hi))
            metrics_hi[k] = _metrics(rebinned_hi[k][:m], true_hi[:m])
            mm = metrics_hi[k]
            print(f"{LABELS[k]:>8} {HI_WIN // 1000}kb: Pearson={mm['pearson']:.3f} "
                  f"R^2={mm['r2']:.3f} bias={mm['bias']:.2f}  (n={mm['n']})")

    meta = dict(figure="Fig 2A repro (Comeron 2L, n=20)", Ne=NE, mu=MU,
                n_hap=2 * N_DIP, length=L, report_win=REPORT_WIN, report_win_hi=HI_WIN,
                paper_reference=dict(method="relernn", r2=0.931, mae=3.72e-8),
                metrics=metrics, metrics_hi=metrics_hi)

    # --- committed figdata bundle: both the 100-kb (low-res) and 25-kb (high-res) figures
    #     regenerate locally from this npz (matplotlib only, no torch/pyrho needed). ---
    showdown = dict(centers_mb=centers / 1e6, edges=edges, truth_100=true_100,
                    centers_hi_mb=centers_hi / 1e6, truth_hi=true_hi,
                    meta=json.dumps(meta))
    for k in METHODS:
        if k in rebinned:
            showdown[k + "_100"] = rebinned[k]
        if k in rebinned_hi:
            showdown[k + "_hi"] = rebinned_hi[k]
    np.savez(args.showdown, **showdown)
    print(f"figdata -> {args.showdown}")

    with open(args.metrics, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"metrics -> {args.metrics}")

    if args.png:
        _plot_from_arrays(centers / 1e6, true_100, rebinned, metrics, args.png)


# ---------------------------------------------------------------------------
# plot  (from the committed figdata npz; runnable anywhere with matplotlib)
# ---------------------------------------------------------------------------

def _plot_from_arrays(centers_mb, true_100, rebinned, metrics, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import paper_style as ps
        ps.style()
        COL = dict(ps.C)
    except Exception:                                   # fallback palette if paper_style absent
        COL = {"fastrho": "#1f78b4", "pyrho": "#33a02c",
               "relernn": "#949494", "truth": "#111111"}
        ps = None

    present = [k for k in METHODS if k in rebinned]
    scale = 1e8                                         # plot in units of 1e-8 c/bp
    fig, axes = plt.subplots(len(present), 1, sharex=True, sharey=True,
                             figsize=(8.4, 1.5 * len(present) + 0.5))
    if len(present) == 1:
        axes = [axes]

    x = np.asarray(centers_mb, float)
    t = np.asarray(true_100, float) * scale
    for ax, k in zip(axes, present):
        ax.fill_between(x, 0, t, color=COL["truth"], alpha=0.10, lw=0, zorder=1)
        ax.plot(x, t, color=COL["truth"], lw=1.0, zorder=2,
                label="Comeron 2L (true map)")          # only the truth line goes in the legend
        ax.plot(x, np.asarray(rebinned[k], float) * scale, color=COL[k], lw=1.7,
                zorder=3)                                # method identified by the panel label
        mm = metrics.get(k, {})
        ax.text(0.992, 0.94,
                f"{LABELS[k]}", transform=ax.transAxes, ha="right", va="top",
                fontsize=11, color=COL[k], fontweight="bold")
        ax.text(0.992, 0.78,
                f"Pearson $r$={mm.get('pearson', float('nan')):.2f}   "
                f"$R^2$={mm.get('r2', float('nan')):.2f}   "
                f"bias={mm.get('bias', float('nan')):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color="0.30")
        ax.set_ylim(bottom=0)
        ax.set_xlim(x.min(), x.max())
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=9, frameon=False, handlelength=1.4)
    axes[-1].set_xlabel("Chromosome position (Mb)")
    fig.supylabel(r"Recombination rate ($\times10^{-8}$ c/bp)", fontsize=10.5)
    fig.tight_layout()

    base, ext = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {base}.pdf (+ .png)")


def cmd_plot(args):
    z = np.load(args.showdown, allow_pickle=True)
    meta = json.loads(str(z["meta"]))
    metrics = meta["metrics"]
    rebinned = {k: z[k + "_100"] for k in METHODS if (k + "_100") in z.files}
    _plot_from_arrays(z["centers_mb"], z["truth_100"], rebinned, metrics, args.png)


# ---------------------------------------------------------------------------
# multires plot: 6 stacked panels (each tool at low-res 100 kb then high-res 25 kb)
# ---------------------------------------------------------------------------

def _plot_multires_from_arrays(z, out_path):
    """Six stacked panels grouped by tool: for each of fastrho, pyrho, ReLERNN a
    low-res (100 kb) panel then a high-res (25 kb) panel, over the same Comeron 2L
    truth. fastrho/pyrho stay on the map at both scales; ReLERNN (native ~100 kb
    windows) is fine at 100 kb but a staircase at 25 kb -- the coarse-by-design point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        import paper_style as ps
        ps.style()
        COL = dict(ps.C)
        LABm = dict(ps.LAB)
    except Exception:                                   # fallback palette if paper_style absent
        COL = {"fastrho": "#1f78b4", "pyrho": "#33a02c",
               "relernn": "#949494", "truth": "#111111"}
        LABm = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}

    meta = json.loads(str(z["meta"]))
    met_lo = meta.get("metrics", {})
    met_hi = meta.get("metrics_hi", {})
    win_lo = int(meta.get("report_win", 100_000)) // 1000
    win_hi = int(meta.get("report_win_hi", 25_000)) // 1000
    scale = 1e8                                         # plot in units of 1e-8 c/bp

    x_lo = np.asarray(z["centers_mb"], float)
    t_lo = np.asarray(z["truth_100"], float) * scale
    x_hi = np.asarray(z["centers_hi_mb"], float)
    t_hi = np.asarray(z["truth_hi"], float) * scale

    # by-tool order: (fastrho lo, fastrho hi, pyrho lo, pyrho hi, ReLERNN lo, ReLERNN hi)
    present = [k for k in METHODS if (k + "_100") in z.files and (k + "_hi") in z.files]
    panels = [(k, res) for k in present for res in ("lo", "hi")]
    n = len(panels)
    fig, axes = plt.subplots(n, 1, sharex=True, sharey=True,
                             figsize=(8.4, 1.15 * n + 0.4))
    if n == 1:
        axes = [axes]

    for ax, (k, res) in zip(axes, panels):
        if res == "lo":
            x, t, arr, mm, win = x_lo, t_lo, np.asarray(z[k + "_100"], float) * scale, met_lo.get(k, {}), win_lo
        else:
            x, t, arr, mm, win = x_hi, t_hi, np.asarray(z[k + "_hi"], float) * scale, met_hi.get(k, {}), win_hi
        ax.fill_between(x, 0, t, color=COL["truth"], alpha=0.10, lw=0, zorder=1)
        ax.plot(x, t, color=COL["truth"], lw=0.9, zorder=2, label="Comeron 2L (true map)")
        ax.plot(x, arr, color=COL[k], lw=1.5, zorder=3)
        # annotation stack, top-right (keeps the top-left legend on panel 0 clear)
        ax.text(0.992, 0.90, f"{LABm.get(k, k)}", transform=ax.transAxes, ha="right",
                va="top", fontsize=10.5, color=COL[k], fontweight="bold")
        ax.text(0.992, 0.66, f"{win} kb", transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color="0.35")
        ax.text(0.992, 0.44,
                f"$r$={mm.get('pearson', float('nan')):.2f}   "
                f"$R^2$={mm.get('r2', float('nan')):.2f}   "
                f"bias={mm.get('bias', float('nan')):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8, color="0.30")
        ax.set_ylim(bottom=0)
        ax.set_xlim(x.min(), x.max())
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].legend(loc="upper left", fontsize=8.5, frameon=False, handlelength=1.4)
    axes[-1].set_xlabel("Chromosome position (Mb)")
    fig.supylabel(r"Recombination rate ($\times10^{-8}$ c/bp)", fontsize=10.5)
    fig.tight_layout()

    base, _ = os.path.splitext(out_path)
    fig.savefig(base + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"multires figure -> {base}.pdf (+ .png)")


def cmd_multires(args):
    z = np.load(args.showdown, allow_pickle=True)
    need = [k + s for k in METHODS for s in ("_100", "_hi")] + ["centers_hi_mb", "truth_hi"]
    missing = [q for q in need if q not in z.files]
    if missing:
        raise SystemExit(
            f"showdown npz is missing high-res arrays {missing}.\n"
            "Re-run the `score` stage with the updated repro_relernn_fig2.py on sesame "
            "(it now also stores the 25-kb high-res arrays), then copy the refreshed "
            "repro_showdown.npz back to paper/figdata/.")
    _plot_multires_from_arrays(z, args.png)


def main():
    ap = argparse.ArgumentParser(description="Reproduce ReLERNN Fig 2A + 3-method showdown (Comeron 2L)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data"); d.add_argument("--out", required=True)
    d.add_argument("--seed", type=int, default=42)

    f = sub.add_parser("fastrho")
    f.add_argument("--data", required=True)
    f.add_argument("--model", choices=list(MODELS), default="hidip",
                   help="which trained model (default highne: correct for Drosophila Ne=2.5e5)")
    f.add_argument("--checkpoint", default=None, help="override the --model checkpoint")
    f.add_argument("--stats", default=None, help="override the --model feat_stats")
    f.add_argument("--device", default="cuda:0")

    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--predict", default=None, help="ReLERNN *.PREDICT.txt")
    s.add_argument("--metrics", required=True)
    s.add_argument("--showdown", required=True, help="output figdata npz")
    s.add_argument("--png", default=None)

    p = sub.add_parser("plot")
    p.add_argument("--showdown", required=True)
    p.add_argument("--png", required=True)

    mr = sub.add_parser("multires", help="6-panel low-res/high-res figure (per tool: 100 kb then 25 kb)")
    mr.add_argument("--showdown", required=True)
    mr.add_argument("--png", required=True)

    args = ap.parse_args()
    {"data": cmd_data, "fastrho": cmd_fastrho, "score": cmd_score, "plot": cmd_plot,
     "multires": cmd_multires}[args.cmd](args)


if __name__ == "__main__":
    main()
