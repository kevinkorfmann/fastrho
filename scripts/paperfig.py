"""Unified main-text figure suite for the fastrho paper (6 figures, one design system).

Builds every main figure from data on sesame so they share one visual language. Run:
  PYTHONNOUSERSITE=1 venvs/fastrho/bin/python scripts/paperfig.py [fig2 fig3 fig5 fig6 ...]
-> paper/figures/figN_*.pdf
"""
from __future__ import annotations
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

CAMP = "/home/kkor/fastrho_data/campaign"
OUT = "/home/kkor/fastrho/paper/figures"
GRID = 25_000

# ---- locked palette + publication style: single source of truth in paper_style ----
from paper_style import C, LAB, style, panel, barlabels  # noqa: E402

NICE = {"const_n20": "constant\n$n{=}20$", "const_n40": "constant\n$n{=}40$",
        "const_n100": "constant\n$n{=}100$", "real_hapmap": "HapMap II",
        "real_decode": "deCODE", "bottleneck_n20": "bottleneck", "expansion_n20": "expansion"}


def summ():
    return json.load(open(os.path.join(CAMP, "results", "summary.json")))


def pear(s, cfg, scale, m):
    try:
        return s[cfg]["scales"][scale][m].get("pearson", np.nan)
    except KeyError:
        return np.nan


# ---------------------------------------------------------------------------
# resolution sweep (pooled over headline configs) -- inline compute
# ---------------------------------------------------------------------------
def _block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def resolution_curve(configs=("const_n20", "const_n40", "real_decode", "real_hapmap")):
    from fastrho.preprocess import mean_rate_between
    grids = [25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
    methods = ["fastrho", "pyrho", "relernn"]
    curve = {m: [] for m in methods}
    for grid in grids:
        f = max(1, grid // GRID)
        pool = {m: ([], []) for m in methods}
        for name in configs:
            cd = os.path.join(CAMP, "configs", name)
            L = json.load(open(os.path.join(cd, "config.json")))["seq_len"]
            truth = {}
            for r in glob.glob(os.path.join(cd, "region_*.npz")):
                z = np.load(r, allow_pickle=True)
                edges = np.append(np.arange(0, L, GRID), L)
                truth[os.path.basename(r)[:-4]] = mean_rate_between(z["map_position"], z["map_rate"], edges)
            for m in methods:
                p = os.path.join(cd, f"pred_{m}.npz")
                if not os.path.exists(p):
                    continue
                pr = np.load(p, allow_pickle=True)
                for rn, tr in truth.items():
                    if rn not in pr.files:
                        continue
                    k = min(len(pr[rn]), len(tr))
                    pool[m][0].append(_block_mean(pr[rn][:k], f))
                    pool[m][1].append(_block_mean(tr[:k], f))
        for m in methods:
            if pool[m][0]:
                P = np.concatenate(pool[m][0]); T = np.concatenate(pool[m][1])
                ok = np.isfinite(P) & np.isfinite(T) & (P > 0) & (T > 0)
                curve[m].append(pearsonr(P[ok], T[ok])[0] if ok.sum() > 3 else np.nan)
            else:
                curve[m].append(np.nan)
    return np.array([g / 1000 for g in grids], float), curve


def _track(ax, config, region):
    """Panel: truth vs fastrho/pyrho/ReLERNN over a real map region (log rate).

    ReLERNN predicts an exact 0 in many windows here; on a log axis those would vanish, so
    we pin every trace to a visible floor and label ReLERNN's zero stretch explicitly.
    """
    from fastrho.preprocess import mean_rate_between
    cd = os.path.join(CAMP, "configs", config)
    z = np.load(os.path.join(cd, region + ".npz"), allow_pickle=True)
    meta = json.loads(str(z["meta"])); L = meta["sequence_length"]
    edges = np.append(np.arange(0, L, GRID), L); cen = edges[:-1] / 1e6
    truth = mean_rate_between(z["map_position"], z["map_rate"], edges)
    hp = int(np.nanargmax(truth))
    floor = 1.3e-10
    ax.axvspan(cen[max(0, hp - 1)], cen[min(len(cen) - 1, hp + 1)], color="#fff2cc", zorder=0)
    ax.plot(cen, np.clip(truth, floor, None), "-", color=C["truth"], lw=2.2, label="true map", zorder=6)
    rel_raw = None
    for m in ["fastrho", "pyrho", "relernn"]:
        p = os.path.join(cd, f"pred_{m}.npz")
        if os.path.exists(p):
            pr = np.load(p, allow_pickle=True)
            if region in pr.files:
                raw = np.asarray(pr[region][:len(cen)], float)
                if m == "relernn":
                    rel_raw = raw
                y = np.clip(raw, floor, None)  # keep sub-floor / zero values on-frame
                ds = "steps-mid" if m == "relernn" else "default"
                lw = 1.7 if m == "relernn" else 2.0
                al = 0.62 if m == "relernn" else 0.95
                zo = 2 if m == "relernn" else (4 if m == "fastrho" else 3)
                ax.plot(cen[:len(y)], y, "-", color=C[m], lw=lw, alpha=al, label=LAB[m],
                        drawstyle=ds, zorder=zo)
    ax.set_yscale("log"); ax.set_ylim(1.0e-10, 1.8e-7)
    ax.text(cen[hp], 1.5e-7, "hotspot", ha="center", va="top", fontsize=8.5, color="#b08900")
    # call out ReLERNN's exact-zero windows (they ride the floor) at their longest run
    if rel_raw is not None and (rel_raw == 0).any():
        zr = rel_raw == 0
        best = run = s0 = bs = 0
        for i, v in enumerate(zr):
            run = run + 1 if v else 0
            if run == 1:
                s0 = i
            if run > best:
                best, bs = run, s0
        xc = cen[bs + best // 2]
        ax.annotate("ReLERNN = 0", (xc, floor), xytext=(0, 15), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color="#777",
                    arrowprops=dict(arrowstyle="-", color="#c2c2c2", lw=0.8))
    ax.set_xlabel("position (Mb)"); ax.set_ylabel("recombination rate (bp$^{-1}$)")
    ax.margins(x=0.01)


def fig2():
    s = summ()
    cfgs = ["const_n20", "const_n40", "real_hapmap", "real_decode"]
    methods = ["fastrho", "pyrho", "relernn"]
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(11.5, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.42, wspace=0.26,
                          top=0.895, bottom=0.075, left=0.075, right=0.975)

    # (a) head-to-head at 25 kb -- Cleveland dot plot (no bars)
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(cfgs))[::-1]
    for yi, c in zip(y, cfgs):
        vals = [pear(s, c, "25kb", m) for m in methods]
        good = [v for v in vals if v == v]
        if good:
            ax.plot([min(good), max(good)], [yi, yi], color="#dcdcdc", lw=2.2, zorder=1, solid_capstyle="round")
        for m, v in zip(methods, vals):
            if v == v:
                ax.scatter(v, yi, s=95, color=C[m], edgecolor="white", linewidth=1.1, zorder=3)
                col = C[m] if m != "relernn" else "#949494"
                dy = 9 if m == "fastrho" else -13
                if v < 0.08:  # keep low values off the left spine
                    ax.annotate("%.2f" % v, (v, yi), xytext=(7, dy), textcoords="offset points",
                                ha="left", fontsize=8.0, color=col)
                else:
                    ax.annotate("%.2f" % v, (v, yi), xytext=(0, dy), textcoords="offset points",
                                ha="center", fontsize=8.0, color=col)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(c, c).replace("\n", " ") for c in cfgs], fontsize=9)
    ax.set_xlim(0, 1.04); ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Pearson $r$ vs. true map  (25 kb, fine scale)")
    ax.set_ylim(-0.6, len(cfgs) - 0.4)
    ax.grid(axis="y", visible=False)
    panel(ax, "a")

    # (b) resolution sweep (line)
    ax = fig.add_subplot(gs[0, 1]); xs, curve = resolution_curve()
    ax.axvspan(min(xs), 110, color="#f0f0f0", alpha=0.9, zorder=0)
    ax.text(45, 0.07, "fine scale\n(hotspots)", ha="center", va="bottom", fontsize=8,
            color="#777", style="italic")
    mzo = {"fastrho": 5, "pyrho": 4, "relernn": 3}  # hero markers on top where methods tie
    for m in methods:
        ax.plot(xs, curve[m], "-o", color=C[m], markeredgecolor="white", markeredgewidth=1.0,
                label=LAB[m], zorder=mzo[m])
    g = curve["fastrho"][0] - curve["relernn"][0]
    ax.annotate("", xy=(xs[0], curve["fastrho"][0]), xytext=(xs[0], curve["relernn"][0]),
                arrowprops=dict(arrowstyle="<->", color="#888", lw=1.1))
    ax.text(xs[0] * 1.85, (curve["fastrho"][0] + curve["relernn"][0]) / 2,
            f"$\\Delta r = {g:.2f}$\nat finest scale", ha="center", va="center",
            fontsize=8.6, color="#555", linespacing=1.25)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xticks(xs); ax.set_xticklabels([f"{int(v)}" for v in xs], fontsize=8.5)
    ax.set_xlabel("$\\leftarrow$ coarser     window size (kb)     finer $\\rightarrow$")
    ax.set_ylabel("Pearson $r$ vs. true map"); ax.set_ylim(0, 1.0)
    panel(ax, "b")

    # (c) full-width example map track on a real HapMap hotspot region
    ax = fig.add_subplot(gs[1, :]); _track(ax, "real_hapmap", "region_000")
    ax.set_title("Real HapMap region — fastrho and pyrho recover the fine-scale map; "
                 "ReLERNN over-smooths to blocks and zeros", fontsize=9.5, color="#333", pad=8)
    panel(ax, "c", x=-0.055)

    # one shared figure-level key (consistent glyph) -- removes the 3 redundant panel legends
    handles = [Line2D([0], [0], color=C[m], lw=2.4, marker="o", mfc=C[m], mec="white",
                      ms=7.5, label=LAB[m]) for m in methods]
    handles.append(Line2D([0], [0], color=C["truth"], lw=2.6, label="true map"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975), ncol=4,
               frameon=False, fontsize=11, columnspacing=2.4, handletextpad=0.6, handlelength=1.9)
    fig.savefig(os.path.join(OUT, "fig2_accuracy.pdf"))
    plt.close(fig); print("wrote fig2_accuracy.pdf")


def fig3():
    s = summ()
    heldout = json.load(open(os.path.join(CAMP, "results", "heldout.json")))
    timings = json.load(open(os.path.join(CAMP, "results", "timings.json")))
    head = ["const_n20", "const_n40", "real_hapmap", "real_decode"]
    fig = plt.figure(figsize=(13.5, 4.3))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    # (a) calibration
    ax = fig.add_subplot(gs[0])
    nom = np.array(heldout["coverage_curve"]["nominal"], float)
    emp = np.array(heldout["coverage_curve"]["empirical"], float)
    ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color="#dfe9f3", alpha=0.7, zorder=0,
                    label="$\\pm$5% band")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1.2, label="perfect", zorder=1)
    ax.plot(nom, emp, "-o", color=C["fastrho"], markeredgecolor="white", markeredgewidth=1.0,
            zorder=3, label="fastrho")
    j = int(np.argmin(np.abs(nom - 0.95)))
    ax.annotate("%.1f%% at 95%%" % (emp[j] * 100), (nom[j], emp[j]), xytext=(0.36, 0.86),
                fontsize=9, color=C["fastrho"], arrowprops=dict(arrowstyle="->", color=C["fastrho"], lw=1))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.legend(loc="upper left", fontsize=9); panel(ax, "a")
    # (b) cost-accuracy frontier (uniform markers; cost on x, accuracy on y)
    ax = fig.add_subplot(gs[1])
    for m in ["fastrho", "pyrho", "relernn"]:
        acc = np.nanmean([pear(s, c, "25kb", m) for c in head])
        au = np.nanmean([s[c]["scales"]["25kb"].get(m, {}).get("hotspot_auprc", np.nan) for c in head])
        cost = timings[m]
        ax.scatter([cost], [acc], s=240, color=C[m], alpha=0.9, edgecolor="white", linewidth=1.4, zorder=3)
        dy = 0.07 if m != "relernn" else -0.12
        ax.annotate(f"{LAB[m]}\n$r$={acc:.2f}, hotspot AUPRC={au:.2f}", (cost, acc),
                    xytext=(cost, acc + dy), ha="center", fontsize=8.6, color=C[m])
    ax.set_xscale("log"); ax.set_xlim(0.4, 3e4); ax.set_ylim(0, 1.0)
    ax.set_xlabel("relative cost per dataset (log)"); ax.set_ylabel("Pearson $r$ (25\\,kb)")
    ax.text(0.97, 0.04, "cheaper $\\rightarrow$  AND sharper $\\uparrow$", transform=ax.transAxes,
            ha="right", fontsize=8.5, color="#888", style="italic")
    panel(ax, "b")
    # (c) demography misspecification -- the absolute-rate (bias-ratio) collapse (no bars)
    ax = fig.add_subplot(gs[2])
    def biasr(cfg, m):
        try: return s[cfg]["scales"]["100kb"][m].get("bias_ratio", np.nan)
        except KeyError: return np.nan
    labs = ["fastrho\n(no demog.)", "pyrho\n(correct table)", "pyrho\n(wrong table)"]
    cols = [C["fastrho"], C["pyrho"], C["pyrho"]]
    vals = [biasr("bottleneck_n20", "fastrho"), biasr("bottleneck_n20", "pyrho"),
            biasr("bottleneck_n20_wd", "pyrho")]
    xx = np.arange(3)
    ax.axhline(1.0, color="#bbb", lw=1.0, ls="--", zorder=1)
    ax.text(2.45, 1.01, "unbiased", fontsize=8, color="#999", ha="right", va="bottom")
    ax.plot([1, 2], [vals[1], vals[2]], color="#cdcdcd", lw=2.0, zorder=1)   # pyrho correct->wrong drop
    for x_, v, c in zip(xx, vals, cols):
        if v == v:
            ax.scatter(x_, v, s=130, color=c, edgecolor="white", linewidth=1.1, zorder=3)
            ax.annotate("%.2f" % v, (x_, v), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.6, color=c)
    ax.annotate("absolute rate\ncollapses", (2, vals[2]), xytext=(2, vals[2] + 0.28),
                ha="center", fontsize=8.2, color="#777")
    ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=8.8)
    ax.set_xlim(-0.5, 2.6); ax.set_ylim(0, 1.15); ax.set_ylabel("bias ratio (inferred / true rate)")
    panel(ax, "c")
    fig.savefig(os.path.join(OUT, "fig3_trust.pdf")); plt.close(fig); print("wrote fig3_trust.pdf")


def _std(mode):
    return json.load(open(os.path.join(CAMP, "results", f"stdpopsim_{mode}.json")))


def fig5():
    order = ["human", "orangutan", "baboon", "dog", "fly", "worm", "arabidopsis"]
    ph = _std("phased"); keys = [k for k in order if k in ph]
    labels = [ph[k]["common"] for k in keys]
    naive = _std("unphased"); comp = _std("unphased_gt"); fold = _std("unphased_unpol_gt")
    gtm = json.load(open(os.path.join(CAMP, "results", "stdpopsim_unphased_unpol_gt_gtmodel.json")))
    y = np.arange(len(keys))[::-1]
    fig = plt.figure(figsize=(13.0, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0], wspace=0.22)
    # (a) no-retrain rescue across 4 conditions -- dot plot (no bars)
    ax = fig.add_subplot(gs[0])
    S = [(ph, "phased (reference)", "#08519c", "o"),
         (naive, "unphased, naive haplotype features", "#bdbdbd", "X"),
         (comp, "unphased, composite-LD", "#3182bd", "o"),
         (fold, "unphased+unpolarized, composite-LD folded", "#6baed6", "D")]
    for yi, k in zip(y, keys):
        vv = [d.get(k, {}).get("pearson", np.nan) for d, _, _, _ in S]
        good = [v for v in vv if v == v]
        if good:
            ax.plot([min(good), max(good)], [yi, yi], color="#e3e3e3", lw=2.2, zorder=1)
    for d, lab, col, mk in S:
        v = [d.get(k, {}).get("pearson", np.nan) for k in keys]
        ax.scatter(v, y, s=64, color=col, marker=mk, edgecolor="white", linewidth=0.8,
                   zorder=3, label=lab)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.3, 1.02); ax.set_xlabel("genome-wide pooled Pearson $r$")
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.legend(ncol=1, loc="lower left", fontsize=7.8, handletextpad=0.4)
    panel(ax, "a", x=-0.16)
    # (b) dedicated GT model closes the gap -- dot plot
    ax = fig.add_subplot(gs[1])
    G = [(ph, "phased (ceiling)", "#08519c", "o"),
         (fold, "base model + folded (no retrain)", "#9ecae1", "D"),
         (gtm, "dedicated GT model", "#DE8F05", "s")]
    for yi, k in zip(y, keys):
        vv = [d.get(k, {}).get("pearson", np.nan) for d, _, _, _ in G]
        good = [v for v in vv if v == v]
        if good:
            ax.plot([min(good), max(good)], [yi, yi], color="#e3e3e3", lw=2.2, zorder=1)
    for d, lab, col, mk in G:
        v = [d.get(k, {}).get("pearson", np.nan) for k in keys]
        ax.scatter(v, y, s=66, color=col, marker=mk, edgecolor="white", linewidth=0.8,
                   zorder=3, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([ph[k]["common"] for k in keys], fontsize=9)
    ax.set_xlim(0.6, 1.02); ax.set_xlabel("genome-wide pooled Pearson $r$")
    ax.set_ylim(-0.6, len(keys) - 0.4)
    ax.legend(ncol=1, loc="lower left", fontsize=8.2, handletextpad=0.4)
    panel(ax, "b", x=-0.16)
    fig.savefig(os.path.join(OUT, "fig5_unphased.pdf")); plt.close(fig); print("wrote fig5_unphased.pdf")


def fig6():
    bp = json.load(open(os.path.join(CAMP, "results", "between_pop_d50.json")))
    s = bp["scales"]["25kb"]
    ident, floor, pred, true = 1.0, s["within_pop_noise_floor"], s["between_pop_pred"], s["between_pop_true"]
    auc = bp.get("differential_hotspot_auc")
    fig, ax = plt.subplots(figsize=(9.2, 3.7))
    ax.set_xlim(0.74, 1.012); ax.set_ylim(-1.35, 1.75)
    ax.plot([0.74, 1.012], [0, 0], color="#ececec", lw=9, solid_capstyle="round", zorder=0)
    P = [(ident, "identical\nmaps", "#111"),
         (floor, "same population,\ntwo subsamples", "#949494"),
         (pred, "two populations\n(fastrho)", C["fastrho"])]
    for x, lab, c in P:
        ax.plot([x], [0], "o", color=c, ms=15, markeredgecolor="white", markeredgewidth=1.6, zorder=4)
        ax.annotate(lab, (x, 0), xytext=(0, -15), textcoords="offset points", ha="center",
                    va="top", fontsize=8.6, color=c)
        ax.annotate("%.2f" % x, (x, 0), xytext=(0, 15), textcoords="offset points", ha="center",
                    fontsize=9, color=c, fontweight="bold")

    def bracket(x0, x1, yb, txt, col):
        ax.plot([x0, x0, x1, x1], [yb - 0.09, yb, yb, yb - 0.09], color=col, lw=1.4)
        ax.text((x0 + x1) / 2, yb + 0.07, txt, ha="center", va="bottom", fontsize=8.7, color=col)
    bracket(floor, ident, 0.62, "inference noise  $=%.2f$" % (ident - floor), "#777")
    bracket(pred, floor, 1.12, "true divergence (recovered) $=%.2f$\n(true value $%.2f$)"
            % (floor - pred, ident - true), C["fastrho"])
    ax.set_yticks([]); ax.set_xlabel("between-map Spearman correlation (25 kb)")
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    if auc:
        ax.text(0.75, -1.15, "differential hotspots localized: AUROC $%.2f$" % auc,
                ha="left", fontsize=9, color=C["fastrho"], style="italic")
    fig.savefig(os.path.join(OUT, "fig6_betweenpop.pdf")); plt.close(fig); print("wrote fig6_betweenpop.pdf")


if __name__ == "__main__":
    style()
    sys.path.insert(0, "/home/kkor/fastrho")
    todo = sys.argv[1:] or ["fig2"]
    for t in todo:
        globals()[t]()
