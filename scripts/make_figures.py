"""Generate paper figures + the main results LaTeX table from campaign outputs.

Reads <campaign>/results/summary.json (per-config metrics) and the per-config
pred_*.npz (for the example-track figure). Optional: <campaign>/results/heldout.json
(coverage curve) and timings.json (wall-clock). Writes PDFs to --out and
main_results.tex to --tables. Run in an env with matplotlib.
"""

from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GRID = 25_000
METHODS = ["fastrho", "pyrho", "relernn"]
# Palette is shared with the hand-made strength figures: fastrho=blue, pyrho=green,
# ReLERNN=red. Keep these in sync so a colour means the same method everywhere.
COLORS = {"fastrho": "#1f78b4", "pyrho": "#33a02c", "relernn": "#949494",
          "gruseq2seq": "#6a3d9a"}
LABELS = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN",
          "gruseq2seq": "ReLERNN-seq2seq"}
NICE = {"const_n20": "constant\n$n{=}20$", "const_n40": "constant\n$n{=}40$",
        "const_n100": "constant\n$n{=}100$", "real_hapmap": "HapMap II",
        "real_decode": "deCODE", "bottleneck_n20": "bottleneck\n$n{=}20$",
        "expansion_n20": "expansion\n$n{=}20$", "real_dog": "dog",
        "real_drosophila": "Drosophila", "anopheles_synth": "Anopheles"}


def nice(c):
    return NICE.get(c, c.replace("_", " "))


def _set_style():
    matplotlib.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "savefig.bbox": "tight",
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "normal",
        "axes.labelsize": 11, "axes.labelweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#555555", "axes.linewidth": 1.0,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": "#b0b0b0", "grid.alpha": 0.28, "grid.linewidth": 0.6,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "xtick.color": "#555555", "ytick.color": "#555555",
        "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 2.4, "lines.markersize": 6.5,
    })


def _bar_labels(ax, fmt="%.2f", fontsize=8, dy=1.5):
    for p in ax.patches:
        h = p.get_height()
        if h and h == h and h > 0.001:
            ax.annotate(fmt % h, (p.get_x() + p.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=fontsize, color="#3a3a3a",
                        xytext=(0, dy), textcoords="offset points")


def _legend_above(ax, title, ncol, pad=24, y=1.0):
    """Legend in the margin just above the axes, title above that -- nothing collides
    with tall bars or value labels."""
    ax.legend(ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, y),
              columnspacing=1.5, handletextpad=0.5, borderaxespad=0.0, handlelength=1.4)
    if title:
        ax.set_title(title, pad=pad)


def _acc_axis(ax, ylabel):
    ax.set_ylim(0, 1.06)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(ylabel)
    ax.axhline(1.0, color="#cfcfcf", lw=0.8, zorder=0)


def _grouped_bars(ax, present, methods, valfn, w=0.26):
    """Grouped accuracy bars over configs x methods, fastrho outlined for emphasis."""
    x = np.arange(len(present))
    for i, m in enumerate(methods):
        vals = [valfn(c, m) for c in present]
        if np.all([v != v for v in vals]):
            continue
        kw = dict(color=COLORS[m], label=LABELS[m], zorder=3)
        if m == "fastrho":
            kw.update(edgecolor="#0d3b66", linewidth=0.8)
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, np.nan_to_num(vals), w, **kw)
    ax.set_xticks(x)
    ax.set_xticklabels([nice(c) for c in present], fontsize=9)
    return x


def _pear(summary, cfg, scale, method):
    try:
        return summary[cfg]["scales"][scale][method].get("pearson", np.nan)
    except KeyError:
        return np.nan


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_headtohead(summary, out, configs, scale="100kb"):
    present = [c for c in configs if c in summary]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _grouped_bars(ax, present, METHODS, lambda c, m: _pear(summary, c, scale, m))
    _acc_axis(ax, f"Pearson $r$ ({scale})")
    _bar_labels(ax, fontsize=7.6)
    _legend_above(ax, "Head-to-head accuracy on the headline configurations", ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_headtohead.pdf")); plt.close(fig)


def fig_scale(summary, out, configs):
    scales = ["25kb", "100kb", "500kb"]
    present = [c for c in configs if c in summary]
    xs = np.arange(len(scales))
    fig, ax = plt.subplots(figsize=(5.1, 3.8))
    ax.axvspan(-0.25, 0.5, color="#f3f3f3", zorder=0)
    ax.text(0.0, 0.06, "fine\nscale", ha="center", va="bottom", fontsize=8,
            color="#888888", style="italic")
    ys = {}
    for m in METHODS:
        ys[m] = [np.nanmean([_pear(summary, c, s, m) for c in present]) for s in scales]
        ax.plot(xs, ys[m], "-o", label=LABELS[m], color=COLORS[m],
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    # annotate the fine-scale gap fastrho vs ReLERNN
    ax.annotate("", xy=(0, ys["fastrho"][0]), xytext=(0, ys["relernn"][0]),
                arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.1))
    ax.text(0.08, (ys["fastrho"][0] + ys["relernn"][0]) / 2,
            f"+{ys['fastrho'][0]-ys['relernn'][0]:.2f}", fontsize=8.5, color="#555555",
            va="center")
    ax.set_xticks(xs); ax.set_xticklabels(scales)
    ax.set_xlim(-0.3, len(scales) - 0.7)
    _acc_axis(ax, "Pearson $r$ (mean over configs)")
    ax.set_xlabel("scoring resolution")
    ax.legend(loc="lower right", title="method")
    ax.set_title("Accuracy is scale-robust for fastrho/pyrho", pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_scale.pdf")); plt.close(fig)


def fig_group(summary, out, configs, fname, title, scale="100kb"):
    present = [c for c in configs if c in summary]
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    _grouped_bars(ax, present, METHODS, lambda c, m: _pear(summary, c, scale, m))
    _acc_axis(ax, f"Pearson $r$ ({scale})")
    _bar_labels(ax, fontsize=7.6)
    _legend_above(ax, title, ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)


def fig_example_track(campaign, out, config="real_hapmap", region="region_000"):
    cdir = os.path.join(campaign, "configs", config)
    npz = os.path.join(cdir, region + ".npz")
    if not os.path.exists(npz):
        return
    from sys import path
    path.insert(0, os.path.join(campaign, "..", "..", "fastrho"))
    z = np.load(npz, allow_pickle=True)
    meta = json.loads(str(z["meta"])); L = meta["sequence_length"]
    edges = np.append(np.arange(0, L, GRID), L); centers = edges[:-1] / 1e6
    from fastrho.preprocess import mean_rate_between
    truth = mean_rate_between(z["map_position"], z["map_rate"], edges)
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    # highlight the strongest hotspot in the true map
    hp = int(np.nanargmax(truth))
    ax.axvspan(centers[max(0, hp - 1)], centers[min(len(centers) - 1, hp + 1)],
               color="#fff2cc", zorder=0)
    ax.plot(centers, truth, "-", lw=2.6, color="#111111", label="true map", zorder=6)
    for m in METHODS:
        p = os.path.join(cdir, f"pred_{m}.npz")
        if not os.path.exists(p):
            continue
        pr = np.load(p, allow_pickle=True)
        if region in pr.files:
            y = pr[region][:len(centers)]
            ax.plot(centers[:len(y)], y, "-", lw=1.5, alpha=0.9,
                    label=LABELS[m], color=COLORS[m])
    ax.text(centers[hp], ax.get_ylim()[1], "hotspot", ha="center", va="bottom",
            fontsize=8, color="#b08900")
    ax.set_yscale("log"); ax.set_xlabel("position (Mb)")
    ax.set_ylabel("recombination rate (/bp)")
    ax.margins(x=0.01)
    _legend_above(ax, "Example inferred map (real HapMap region): fastrho tracks the truth"
                  " at fine scale", ncol=4)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_example_track.pdf")); plt.close(fig)


def fig_runtime(out, timings):
    if not timings:
        return
    ms = [m for m in METHODS if m in timings][::-1]   # fastrho at top
    vals = [timings[m] for m in ms]
    base = min(timings.values())
    fig, ax = plt.subplots(figsize=(5.1, 3.4))
    y = np.arange(len(ms))
    ax.barh(y, vals, color=[COLORS[m] for m in ms], height=0.62, zorder=3,
            edgecolor="white", linewidth=0.6)
    ax.set_xscale("log")
    for yi, m, v in zip(y, ms, vals):
        lab = f"{v:.0f}$\\times$"
        if m == "fastrho":
            lab += "  (≈10–44 s, one forward pass)"
        ax.text(v * 1.3, yi, lab, va="center", ha="left", fontsize=8.8, color="#3a3a3a")
    ax.set_yticks(y); ax.set_yticklabels([LABELS[m] for m in ms])
    ax.set_xlim(base * 0.5, max(vals) * 30)
    ax.set_xlabel("relative end-to-end cost per dataset (log scale)")
    ax.grid(axis="y", alpha=0)
    ax.set_title("Cost to estimate one map", pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_runtime.pdf")); plt.close(fig)


def fig_calibration(out, heldout):
    if not heldout or "coverage_curve" not in heldout:
        return
    nom = np.asarray(heldout["coverage_curve"]["nominal"], float)
    emp = np.asarray(heldout["coverage_curve"]["empirical"], float)
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="#dfe9f3", alpha=0.7,
                    zorder=0, label="$\\pm$5% band")
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1.2, zorder=1, label="perfect")
    ax.plot(nom, emp, "-o", color=COLORS["fastrho"], markeredgecolor="white",
            markeredgewidth=1.0, zorder=3, label="fastrho")
    # callout at the 95% level
    if np.any(nom >= 0.94):
        j = int(np.argmin(np.abs(nom - 0.95)))
        ax.annotate(f"{emp[j]*100:.1f}% at 95%", (nom[j], emp[j]),
                    xytext=(0.40, 0.86), fontsize=8.5, color="#1f77b4",
                    arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.0))
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_title("Single-pass uncertainty is calibrated", pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_calibration.pdf")); plt.close(fig)


def fig_crossspecies(summary, out, configs):
    present = [c for c in configs if c in summary]
    if not present:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    _grouped_bars(ax, present, ["fastrho", "pyrho"],
                  lambda c, m: _pear(summary, c, "100kb", m), w=0.36)
    ax.set_xticklabels([nice(c).replace("real ", "") for c in present], fontsize=9)
    _acc_axis(ax, "Pearson $r$ (100kb)")
    _bar_labels(ax)
    _legend_above(ax, "One model recovers real maps across the PRDM9 gradient", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_crossspecies.pdf")); plt.close(fig)


def fig_between_pop(out, bp):
    if not bp:
        return
    scales = ["25kb", "100kb", "500kb"]; xs = np.arange(len(scales))
    # between_pop_true = corr of the two populations' TRUE maps (real similarity ceiling);
    # between_pop_pred = corr of fastrho's inferred maps across pops;
    # within_pop_noise_floor = reproducibility across disjoint subsamples (the noise floor).
    series = [("between_pop_true", "between-pop, true maps (ceiling)", "#111111", "--", "s"),
              ("within_pop_noise_floor", "within-pop reproducibility (noise floor)",
               "#888888", "-", "o"),
              ("between_pop_pred", "between-pop, fastrho", COLORS["fastrho"], "-", "o")]
    y = {k: np.asarray([bp["scales"][s][k] for s in scales], float) for k, *_ in series}
    fig, ax = plt.subplots(figsize=(5.9, 3.8))
    # at fine scale the fastrho between-pop estimate sits ~at the noise floor: the residual
    # below the true-map ceiling is what is genuinely attributable to divergence + noise
    ax.annotate("", xy=(0, y["between_pop_true"][0]), xytext=(0, y["between_pop_pred"][0]),
                arrowprops=dict(arrowstyle="<->", color="#9a9a9a", lw=1.1))
    ax.text(0.07, (y["between_pop_true"][0] + y["between_pop_pred"][0]) / 2,
            "noise +\nresolution", fontsize=7.6, color="#777777", va="center")
    for k, lab, c, ls, mk in series:
        ax.plot(xs, y[k], ls, marker=mk, label=lab, color=c,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(scales)
    ax.set_xlim(-0.3, len(scales) - 0.7)
    ax.set_ylim(0, 1.06); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("scoring resolution"); ax.set_ylabel("Spearman correlation")
    ax.legend(loc="lower left", fontsize=8.0)
    auc = bp.get("differential_hotspot_auc")
    t = "Between-population divergence vs. estimation noise"
    if auc:
        t += f"\n(detects truly-divergent hotspots: AUROC $=$ {auc:.2f})"
    ax.set_title(t, pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_between_pop.pdf")); plt.close(fig)


def fig_misspecification(summary, out):
    if "bottleneck_n20" not in summary or "bottleneck_n20_wd" not in summary:
        return
    fr = _pear(summary, "bottleneck_n20", "100kb", "fastrho")
    pc = _pear(summary, "bottleneck_n20", "100kb", "pyrho")
    pw = _pear(summary, "bottleneck_n20_wd", "100kb", "pyrho")
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    labels = ["fastrho\n(no demography)", "pyrho\n(correct demo.)", "pyrho\n(wrong demo.)"]
    bars = ax.bar(range(3), [fr, pc, pw], width=0.6, zorder=3,
                  color=[COLORS["fastrho"], COLORS["pyrho"], "#9ed99a"])
    bars[0].set_edgecolor("#0d3b66"); bars[0].set_linewidth(0.8)
    bars[2].set_hatch("////"); bars[2].set_edgecolor("white")
    ax.annotate("", xy=(2, pw), xytext=(1, pc),
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.1,
                                connectionstyle="arc3,rad=-0.25"))
    ax.text(1.5, max(pc, pw) + 0.07, "pyrho needs\nthe demography", ha="center",
            fontsize=8, color="#666666")
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
    _acc_axis(ax, "Pearson $r$ (100kb)")
    _bar_labels(ax)
    ax.set_title("fastrho is demography-free; pyrho is not", pad=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_misspecification.pdf")); plt.close(fig)


def fig_dipteran(summary, out):
    cands = [("real_drosophila", "Drosophila\n(real Comeron map)"),
             ("anopheles_synth", "Anopheles\n(synthetic)")]
    present = [(c, l) for c, l in cands if c in summary]
    if not present:
        return
    x = np.arange(len(present)); w = 0.36
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    p25 = [_pear(summary, c, "25kb", "fastrho") for c, _ in present]
    p100 = [_pear(summary, c, "100kb", "fastrho") for c, _ in present]
    ax.bar(x - w / 2, np.nan_to_num(p25), w, label="25 kb", color=COLORS["fastrho"],
           edgecolor="#0d3b66", linewidth=0.8, zorder=3)
    ax.bar(x + w / 2, np.nan_to_num(p100), w, label="100 kb", color="#9ecae1", zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in present], fontsize=9)
    _acc_axis(ax, "fastrho Pearson $r$")
    _bar_labels(ax)
    _legend_above(ax, "High-$N_e$ dipteran regime (broadened model)", ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_dipteran.pdf")); plt.close(fig)


def fig_steelman(summary, out):
    cfgs = ["const_n20", "const_n40", "real_decode", "real_hapmap"]
    present = [c for c in cfgs
              if c in summary and summary[c]["scales"].get("25kb", {}).get("gruseq2seq")]
    if not present:
        return
    methods = ["fastrho", "gruseq2seq", "relernn"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    _grouped_bars(ax, present, methods, lambda c, m: _pear(summary, c, "25kb", m))
    _acc_axis(ax, "Pearson $r$ (25 kb)")
    _bar_labels(ax, fontsize=7.6)
    _legend_above(ax, "Steelman: per-SNP GRU on raw genotypes", ncol=3)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_steelman.pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--tables", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _set_style()
    summary = json.load(open(os.path.join(args.campaign, "results", "summary.json")))
    heldout = {}
    hp = os.path.join(args.campaign, "results", "heldout.json")
    if os.path.exists(hp):
        heldout = json.load(open(hp))
    timings = {}
    tp = os.path.join(args.campaign, "results", "timings.json")
    if os.path.exists(tp):
        timings = json.load(open(tp))

    headline = ["const_n20", "const_n40", "real_hapmap", "real_decode"]
    fig_headtohead(summary, args.out, headline)
    fig_scale(summary, args.out, headline)
    fig_group(summary, args.out, ["const_n20", "bottleneck_n20", "expansion_n20"],
              "fig_demography.pdf", "Robustness across demographies")
    fig_group(summary, args.out, ["const_n20", "const_n40", "const_n100"],
              "fig_samplesize.pdf", "Robustness across sample size")
    fig_group(summary, args.out, ["real_hapmap", "real_decode"],
              "fig_realmap.pdf", "Real genetic-map recovery")
    fig_example_track(args.campaign, args.out)
    fig_runtime(args.out, timings)
    fig_calibration(args.out, heldout)
    fig_crossspecies(summary, args.out, ["real_hapmap", "real_decode", "real_dog"])
    fig_misspecification(summary, args.out)
    bp = {}
    bpp = os.path.join(args.campaign, "results", "between_pop_d50.json")
    if os.path.exists(bpp):
        bp = json.load(open(bpp))
    fig_between_pop(args.out, bp)
    fig_dipteran(summary, args.out)
    # fig_steelman(summary, args.out)  # RETIRED: superseded by scripts/fig_steelman.py
    #   (dumbbell dot plot). Do not regenerate the old grouped-bar version here.
    # The active benchmark table is generated by build_manuscript_derived.py so
    # its bottleneck/expansion cells follow the registered matched-arm policy.
    print("legacy benchmark figures written to", args.out)


if __name__ == "__main__":
    main()
