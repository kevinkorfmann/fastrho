"""fastrho strength figures (the lunch-money set). Run on sesame (GPU + data).

Produces:
  fig_resolution_showdown.pdf  -- fastrho's per-interval map + calibrated band traces hotspots
                                  that ReLERNN's single-rate-per-window step function flattens.
  fig_cost_accuracy.pdf        -- fastrho is simultaneously ~10^4x cheaper and far sharper.
  fig_capability_matrix.pdf    -- one frozen model does what neither baseline can.

Usage:
  python scripts/fig_fastrho_strengths.py --campaign /home/kkor/fastrho_data/campaign \
      --ckpt <ckpt> --stats <feat_stats.npz> --out /tmp/figs --device cuda:0
"""
from __future__ import annotations
import os, csv, re, json, glob, argparse
import numpy as np

import paper_style as ps

FAST = ps.C["fastrho"]; PYR = ps.C["pyrho"]; REL = ps.C["relernn"]; TRU = "0.35"


# ---------- data loaders ----------
def relernn_step(predict_path, contig_idx):
    """Return (edges, rates) for region i (contig chr{i+1}) from a combined PREDICT.txt."""
    want = contig_idx + 1
    s, e, r = [], [], []
    with open(predict_path) as fh:
        rdr = csv.reader(fh, delimiter="\t"); next(rdr)
        for row in rdr:
            if not row:
                continue
            m = re.search(r"chr(\d+)", row[0])
            if not m or int(m.group(1)) != want:
                continue
            s.append(float(row[1])); e.append(float(row[2])); r.append(float(row[-1]))
    if not s:
        return None
    o = np.argsort(s); s = np.array(s)[o]; e = np.array(e)[o]; r = np.array(r)[o]
    return np.r_[s, e[-1]], r


def pyrho_intervals(rmap_path, four_ne):
    rows = np.loadtxt(rmap_path, ndmin=2)
    if rows.size == 0:
        return None
    start, end, rho = rows[:, -3], rows[:, -2], rows[:, -1]
    return np.r_[start[0], end], rho / four_ne   # rho -> per-bp r


def _smooth_steps(edges, vals):
    x = np.repeat(edges, 2)[1:-1]; y = np.repeat(vals, 2)
    return x, y


# ---------- Fig 1: resolution showdown ----------
def fig_resolution(args, panels):
    import tskit
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from fastrho.translate import load_model, predict_map_from_ts

    model, mcfg, stats = load_model(args.ckpt, args.stats, device=args.device)
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.1 * len(panels)))
    if len(panels) == 1:
        axes = [axes]

    for ax, (cfgname, ridx, title, zoom) in zip(axes, panels):
        C = os.path.join(args.campaign, "configs", cfgname)
        cfg = json.load(open(os.path.join(C, "config.json")))
        Ne = float(cfg["Ne"]); mu = float(cfg["mu"])
        four_ne = 4.0 * Ne
        z = np.load(os.path.join(C, f"region_{ridx:03d}.npz"), allow_pickle=True)
        mpos = np.asarray(z["map_position"], float)
        mrate = np.where(np.isfinite(z["map_rate"]), z["map_rate"], 0.0)
        L = mpos[-1]

        ts = tskit.load(os.path.join(C, f"region_{ridx:03d}.trees"))
        pred = predict_map_from_ts(ts, model, mcfg, stats, mutation_rate=mu,
                                   Ne=Ne, device=args.device)
        fx = np.r_[pred["pos_left"][0], pred["pos_right"]]
        fr = np.asarray(pred["r_per_bp"], float)
        flo = np.asarray(pred.get("r_ci_lo", fr), float)
        fhi = np.asarray(pred.get("r_ci_hi", fr), float)

        rel = relernn_step(os.path.join(C, "relernn_proj", "combined.PREDICT.txt"), ridx)
        rmap = os.path.join(C, f"region_{ridx:03d}.rmap")
        pyr = pyrho_intervals(rmap, four_ne) if os.path.exists(rmap) else None

        scale = 1e8  # plot in cM/Mb-ish (x1e8 c/bp)
        # true map (filled)
        tx, ty = _smooth_steps(mpos, mrate * scale)
        ax.fill_between(tx / 1e6, 0, ty, color=TRU, alpha=0.18, lw=0, zorder=1, label="true map")
        ax.plot(tx / 1e6, ty, color=TRU, lw=1.0, zorder=2)
        # ReLERNN staircase
        if rel is not None:
            rx, ry = _smooth_steps(rel[0], rel[1] * scale)
            ax.plot(rx / 1e6, ry, color=REL, lw=2.4, zorder=3,
                    label="ReLERNN (1 rate / window)")
        # pyrho
        if pyr is not None:
            px, py = _smooth_steps(pyr[0], pyr[1] * scale)
            ax.plot(px / 1e6, py, color=PYR, lw=0.9, alpha=0.8, zorder=3, label="pyrho")
        # fastrho mean + calibrated band
        fc = 0.5 * (fx[:-1] + fx[1:])
        ax.fill_between(fc / 1e6, flo * scale, fhi * scale, color=FAST, alpha=0.25, lw=0,
                        zorder=4, label="fastrho 95% band")
        ax.plot(fc / 1e6, fr * scale, color=FAST, lw=1.6, zorder=5, label="fastrho")

        ax.set_xlim(0, L / 1e6); ax.set_ylim(bottom=0)
        ax.set_ylabel(r"recomb. rate ($\times10^{-8}$ c/bp)")
        ax.set_title(title, fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        if zoom:
            # annotate the tallest true hotspot ReLERNN flattens
            k = int(np.argmax(mrate)); hx = 0.5 * (mpos[k] + mpos[k + 1]) / 1e6
            ax.axvspan(hx - L / 1e6 * 0.012, hx + L / 1e6 * 0.012, color="gold", alpha=0.18, zorder=0)
            ax.annotate("hotspot fastrho resolves,\nReLERNN averages away",
                        xy=(hx, mrate[k] * scale * 0.9),
                        xytext=(hx + L / 1e6 * 0.08, mrate[k] * scale * 0.95),
                        fontsize=8.5, color="0.2",
                        arrowprops=dict(arrowstyle="->", color="0.4", lw=1))
    axes[0].legend(ncol=5, fontsize=8.5, frameon=False, loc="upper center",
                   bbox_to_anchor=(0.5, 1.32))
    axes[-1].set_xlabel("chromosome position (Mb)")
    fig.tight_layout()
    p = os.path.join(args.out, "fig_resolution_showdown.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=160, bbox_inches="tight")
    plt.close(); print("wrote", p)


# ---------- Fig 2: cost vs accuracy ----------
def fig_cost_accuracy(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s = json.load(open(os.path.join(args.campaign, "results", "summary.json")))
    cfgs = ["const_n20", "const_n40", "real_decode", "real_hapmap"]
    cost = {"fastrho": 1, "pyrho": 70, "relernn": 8527}
    col = {"fastrho": FAST, "pyrho": PYR, "relernn": REL}
    def mean25(m):
        v = [s[c]["scales"]["25kb"][m]["pearson"] for c in cfgs
             if m in s[c]["scales"]["25kb"]]
        return np.mean(v)
    def meanau(m):
        v = [s[c]["scales"]["25kb"][m].get("hotspot_auprc") for c in cfgs
             if s[c]["scales"]["25kb"].get(m, {}).get("hotspot_auprc") is not None]
        return np.mean(v) if v else np.nan

    fig, ax = plt.subplots(figsize=(7.2, 5))
    for m in ["fastrho", "pyrho", "relernn"]:
        y = mean25(m); au = meanau(m)
        ax.scatter(cost[m], y, s=900 * (au if np.isfinite(au) else 0.15) + 120,
                   color=col[m], alpha=0.85, edgecolor="white", lw=2, zorder=3)
        lbl = {"fastrho": "fastrho", "pyrho": "pyrho", "relernn": "ReLERNN"}[m]
        ax.annotate(f"{lbl}\n$r$={y:.2f}" + (f", AUPRC={au:.2f}" if np.isfinite(au) else ""),
                    (cost[m], y), textcoords="offset points",
                    xytext=(12 if m != "relernn" else -12, 10),
                    ha="left" if m != "relernn" else "right", fontsize=10, color=col[m],
                    fontweight="bold")
    ax.annotate("", xy=(1.6, 0.86), xytext=(6000, 0.30),
                arrowprops=dict(arrowstyle="-|>", color="0.5", lw=2, ls=":"))
    ax.text(60, 0.62, "cheaper $\\rightarrow$ AND sharper", rotation=-22,
            color="0.45", fontsize=10, ha="center")
    ax.set_xscale("log"); ax.set_xlim(0.5, 2e4); ax.set_ylim(0.1, 0.95)
    ax.set_xlabel("relative wall-clock cost per dataset (log scale)")
    ax.set_ylabel("fine-scale accuracy (25 kb Pearson $r$)")
    ax.set_title("fastrho dominates the cost–accuracy frontier\n"
                 "(marker area $\\propto$ hotspot AUPRC)", fontsize=12, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="both", ls=":", alpha=0.4)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_cost_accuracy.pdf")
    fig.savefig(p, dpi=200); fig.savefig(p[:-4] + ".png", dpi=160); plt.close(); print("wrote", p)


# ---------- Fig 3: capability matrix ----------
def fig_capability(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    s = json.load(open(os.path.join(args.campaign, "results", "summary.json")))
    def m25(c, m, k="pearson"):
        return s[c]["scales"]["25kb"].get(m, {}).get(k, np.nan)
    # columns: (label, fastrho, pyrho, relernn) with score in [0,1] for color + display text
    cols = [
        ("Per-interval\nresolution", 1, 1, 0, ["yes", "yes", "no (windows)"]),
        ("Fine-scale $r$\n(const n=20, 25kb)", m25("const_n20","fastrho"), m25("const_n20","pyrho"), m25("const_n20","relernn"),
         [f"{m25('const_n20','fastrho'):.2f}", f"{m25('const_n20','pyrho'):.2f}", f"{m25('const_n20','relernn'):.2f}"]),
        ("Hotspot AUPRC\n(const n=20)", m25("const_n20","fastrho","hotspot_auprc"), m25("const_n20","pyrho","hotspot_auprc"), m25("const_n20","relernn","hotspot_auprc"),
         [f"{m25('const_n20','fastrho','hotspot_auprc'):.2f}", f"{m25('const_n20','pyrho','hotspot_auprc'):.2f}", f"{m25('const_n20','relernn','hotspot_auprc'):.2f}"]),
        ("1-pass calibrated\nuncertainty", 1, 0.0, 0.3, ["yes", "no", "bootstrap"]),
        ("No per-dataset\ntable / retrain", 1, 0.0, 0.0, ["yes", "no (table)", "no (retrain)"]),
        ("No demography\nrequired", 1, 0.0, 0.5, ["yes", "no (SMC++)", "trained-in"]),
        ("Relative speed", 1, 70/8527, 1/8527, ["1×", "70×", "8527×"]),
    ]
    methods = ["fastrho", "pyrho", "ReLERNN"]
    M = np.array([[c[1], c[2], c[3]] for c in cols]).T  # 3 x ncol
    txt = [[c[4][i] for c in cols] for i in range(3)]
    # colorblind-safe bad->good ramp (vermillion -> light -> blue); avoids red-green
    cmap = LinearSegmentedColormap.from_list("cap", ["#e31a1c", "#f2f2f2", "#1f78b4"])
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols], fontsize=8.5)
    ax.set_yticks(range(3)); ax.set_yticklabels(methods, fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(len(cols)):
            ax.text(j, i, txt[i][j], ha="center", va="center", fontsize=8.5,
                    color="white" if M[i, j] < 0.35 or M[i, j] > 0.8 else "black",
                    fontweight="bold")
    ax.set_title("One frozen model, every capability: fastrho vs the incumbents",
                 fontsize=12, loc="left", pad=10)
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_capability_matrix.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=160, bbox_inches="tight")
    plt.close(); print("wrote", p)


def fig_resolution_zoom(args):
    """Zoom cascade: at ReLERNN's native window scale it matches; zoom in and it degrades into
    a flat step while fastrho's per-interval posterior keeps tracking the true map."""
    import tskit
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, ConnectionPatch
    from fastrho.translate import load_model, predict_map_from_ts
    cfgname, ridx = "real_decode", 17
    C = os.path.join(args.campaign, "configs", cfgname)
    cfg = json.load(open(os.path.join(C, "config.json")))
    Ne = float(cfg["Ne"]); mu = float(cfg["mu"])
    z = np.load(os.path.join(C, f"region_{ridx:03d}.npz"), allow_pickle=True)
    mpos = np.asarray(z["map_position"], float)
    mrate = np.where(np.isfinite(z["map_rate"]), z["map_rate"], 0.0)
    L = mpos[-1]
    model, mcfg, stats = load_model(args.ckpt, args.stats, device=args.device)
    ts = tskit.load(os.path.join(C, f"region_{ridx:03d}.trees"))
    pred = predict_map_from_ts(ts, model, mcfg, stats, mutation_rate=mu, Ne=Ne, device=args.device)
    fx = np.r_[pred["pos_left"][0], pred["pos_right"]]; fc = 0.5 * (fx[:-1] + fx[1:])
    fr = np.asarray(pred["r_per_bp"], float)
    flo = np.asarray(pred.get("r_ci_lo", fr), float); fhi = np.asarray(pred.get("r_ci_hi", fr), float)
    rel = relernn_step(os.path.join(C, "relernn_proj", "combined.PREDICT.txt"), ridx)

    sc = 1e8
    # pick the tallest hotspot for the deepest zoom
    k = int(np.argmax(mrate)); hc = 0.5 * (mpos[k] + mpos[k + 1])
    spans = [(0, L), (hc - 2e5, hc + 2e5), (hc - 4e4, hc + 4e4)]
    titles = ["full region — ReLERNN matches at its native window scale",
              "zoom 5$\\times$ — ReLERNN starts to block up", "zoom 25$\\times$ — ReLERNN flat, fastrho resolves the hotspot"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 8))
    for n, (ax, (lo, hi), title) in enumerate(zip(axes, spans, titles)):
        lo = max(0, lo); hi = min(L, hi)
        tx, ty = _smooth_steps(mpos, mrate * sc)
        ax.fill_between(tx, 0, ty, color=TRU, alpha=0.18, lw=0)
        ax.plot(tx, ty, color=TRU, lw=1.0, label="true map")
        if rel is not None:
            rx, ry = _smooth_steps(rel[0], rel[1] * sc)
            ax.plot(rx, ry, color=REL, lw=2.6, label="ReLERNN")
        ax.fill_between(fc, flo * sc, fhi * sc, color=FAST, alpha=0.25, lw=0, label="fastrho 95% band")
        ax.plot(fc, fr * sc, color=FAST, lw=1.6, label="fastrho")
        ax.set_xlim(lo, hi)
        m = (mpos[:-1] >= lo) & (mpos[1:] <= hi)
        ymax = max((mrate[m].max() if m.any() else mrate.max()), fhi[(fc>=lo)&(fc<=hi)].max() if ((fc>=lo)&(fc<=hi)).any() else 0) * sc
        ax.set_ylim(0, ymax * 1.1 + 1e-9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylabel(r"rate ($\times10^{-8}$)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
        if n < 2:  # draw the next zoom window + connectors
            zlo, zhi = spans[n + 1]; zlo = max(0, zlo); zhi = min(L, zhi)
            ax.axvspan(zlo, zhi, color="gold", alpha=0.18, zorder=0)
            for xz in (zlo, zhi):
                con = ConnectionPatch((xz, 0), (xz, ymax * 1.1), "data", "data",
                                      axesA=ax, axesB=axes[n + 1], color="0.6", lw=0.8, ls="--")
                fig.add_artist(con)
    axes[0].legend(ncol=4, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.4))
    axes[-1].set_xlabel("chromosome position (bp)")
    fig.suptitle("Same prediction, three zoom levels: ReLERNN degrades on zoom, fastrho holds",
                 y=1.0, fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(args.out, "fig_resolution_zoom.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=160, bbox_inches="tight")
    plt.close(); print("wrote", p)


def dump_caldata(args):
    """Forward-pass over held-out shards, dump per-interval (true, pred, lo, hi)."""
    from fastrho.translate import load_model, predict_from_tokens
    model, cfg, stats = load_model(args.ckpt, args.stats, device=args.device)
    files = sorted(glob.glob(os.path.join(args.shards, "ts_*.npz")))[: args.max_shards]
    T, P, LO, HI = [], [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if "tokens" not in z or z["tokens"].shape[0] < 3:
            continue
        meta = json.loads(str(z["meta"]))
        pred = predict_from_tokens(model, cfg, stats, z["tokens"], z["positions"],
                                   2 * int(meta["n_samples"]), float(meta["mutation_rate"]),
                                   Ne=meta.get("Ne"), device=args.device)
        T.append(np.asarray(z["interval_target"], float)); P.append(np.asarray(pred["r_per_bp"], float))
        LO.append(np.asarray(pred["r_ci_lo"], float)); HI.append(np.asarray(pred["r_ci_hi"], float))
    np.savez(os.path.join(args.out, "caldata.npz"),
             true=np.concatenate(T), pred=np.concatenate(P),
             lo=np.concatenate(LO), hi=np.concatenate(HI))
    print("wrote caldata.npz:", sum(len(t) for t in T), "intervals")


def fig_calibration(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    h = json.load(open(os.path.join(args.campaign, "results", "heldout.json")))
    cc = h["coverage_curve"]; nom = np.array(cc["nominal"]); emp = np.array(cc["empirical"])
    cd = os.path.join(args.out, "caldata.npz")
    has_cd = os.path.exists(cd)
    if has_cd:
        d = np.load(cd); true = d["true"]; pred = d["pred"]; lo = d["lo"]; hi = d["hi"]
        ok = np.isfinite(true) & np.isfinite(pred) & (true > 0)
        true, pred, lo, hi = true[ok], pred[ok], lo[ok], hi[ok]

    fig, axes = plt.subplots(1, 3 if has_cd else 2, figsize=(13 if has_cd else 9, 4))
    # A: reliability diagram
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="0.6", lw=1.2, label="perfect")
    ax.plot(nom, emp, "o-", color=FAST, lw=2.2, ms=8, label="fastrho")
    for x, y in zip(nom, emp):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(6, -10), fontsize=8, color=FAST)
    ax.set_xlabel("nominal coverage"); ax.set_ylabel("empirical coverage")
    ax.set_title(f"Calibrated single-pass intervals\n({h['n_intervals']:,} held-out intervals)", fontsize=10, loc="left")
    ax.set_xlim(0.45, 1.0); ax.set_ylim(0.45, 1.0); ax.legend(fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(ls=":", alpha=0.4)
    # B: coverage by true-rate decile at 95%
    ax = axes[1]
    if has_cd:
        cov95 = ((true >= lo) & (true <= hi)).astype(float)
        dec = np.clip((np.searchsorted(np.quantile(true, np.linspace(0, 1, 11)), true) - 1), 0, 9)
        cby = [cov95[dec == d].mean() for d in range(10)]
        ax.bar(range(10), cby, color=FAST, alpha=0.85)
        ax.axhline(0.95, ls="--", color=REL, lw=1.5, label="0.95 target")
        ax.set_ylim(0.8, 1.0); ax.set_xlabel("true-rate decile (low $\\to$ high)")
        ax.set_ylabel("95% interval coverage"); ax.legend(fontsize=9, frameon=False)
        ax.set_title("Calibrated across the rate range", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        # C: sharpness — relative interval width vs rate
        ax = axes[2]
        relw = (hi - lo) / np.maximum(pred, 1e-12)
        order = np.argsort(pred)
        step = max(1, len(order) // 30000)   # subsample + rasterize -> small PDF
        ax.scatter(pred[order][::step] * 1e8, np.clip(relw[order][::step], 0, 20),
                   s=3, alpha=0.2, color=FAST, rasterized=True)
        ax.set_xscale("log"); ax.set_xlabel(r"predicted rate ($\times10^{-8}$ c/bp)")
        ax.set_ylabel("relative 95% width  (hi$-$lo)/mean")
        ax.set_title("Sharper where the rate is higher", fontsize=10, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "pyrho: point estimates\n(no calibrated CI)\n\nReLERNN: separate bootstrap\n(+8.6% cost, retrain)\n\nfastrho: 1 forward pass",
                ha="center", va="center", fontsize=11)
    fig.suptitle("fastrho is the only method with trustworthy, single-pass uncertainty",
                 fontsize=12, fontweight="bold", y=1.04)
    fig.tight_layout()
    p = os.path.join(args.out, "fig_calibration_showcase.pdf")
    fig.savefig(p, dpi=200, bbox_inches="tight"); fig.savefig(p[:-4] + ".png", dpi=160, bbox_inches="tight")
    plt.close(); print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--ckpt", required=True); ap.add_argument("--stats", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--shards", default=None, help="held-out shard dir for calibration data")
    ap.add_argument("--max-shards", type=int, default=300)
    ap.add_argument("--only", default=None,
                    help="resolution|cost|capability|zoom|caldata|calibration")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    panels = [
        ("const_n20", 5, "Synthetic hotspot landscape  ($n{=}20$, constant $N_e$)", True),
        ("real_decode", 17, "Real deCODE hotspot region  ($n{=}20$)", True),
    ]
    if args.only in (None, "resolution"):
        fig_resolution(args, panels)
    if args.only in (None, "cost"):
        fig_cost_accuracy(args)
    if args.only in (None, "capability"):
        fig_capability(args)
    if args.only == "zoom":
        fig_resolution_zoom(args)
    if args.only == "caldata":
        dump_caldata(args)
    if args.only == "calibration":
        fig_calibration(args)


if __name__ == "__main__":
    main()
