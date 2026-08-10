"""Flagship dog figure: village->breed recombination-map transfer under the PRDM9-loss
conserved landscape. Four panels:
  (a) LD decay by regime  -- the recent breed bottleneck inflates LD to ~Mb, saturating the
      per-interval recombination signal a breed's own data could carry.
  (b) one shared locus    -- the village panel recovers the fine-scale map; the breed's own
      (saturated-LD) data cannot. The money shot.
  (c) transfer lift        -- across regions, the breed map rises from ~0.3 (own data) to
      ~0.78 (inferred from the co-located village panel).
  (d) rescue vs severity   -- own-data recovery collapses as the breed Ne shrinks, while the
      transferred map stays high: transfer helps most exactly where it is needed most.

No retraining: one frozen dog model, applied to paired populations sharing a map.
Palette (shared paper_style ps.TRANSFER): truth black, village = fastrho-blue #1f78b4, breed-own
orange #ff7f00 -- deliberately NOT pyrho green (#33a02c), and colorblind-safe.

REPRODUCIBILITY. The figure is a two-stage artifact: ``compute()`` runs msprime + model inference
and writes the plotted arrays to paper/figdata/dog_fig.npz; ``render()`` re-plots from that cache
with NO inference. The paper build renders from the committed cache and is therefore deterministic
and GPU-free. Regenerate the cache deterministically with ``--recompute --device cpu`` (CPU is fp32;
GPU autocast is fp16 and jitters the panel-(b) region selection).

Usage:
  python scripts/fig_dog.py                                  # render from committed cache (default)
  python scripts/fig_dog.py --recompute --device cpu \
      --ckpt <ckpt.txt> --stats <feat_stats.npz> [--np 90]   # regenerate the cache, then render
"""
import os
import sys
import argparse

# Reproducible PDFs: fix the embedded date and the font-subset id hash so a re-render of the SAME
# cache is byte-identical (clean git diffs; the B5 smoke test asserts this).
os.environ.setdefault("SOURCE_DATE_EPOCH", "1420070400")   # 2015-01-01
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "fastrho-dog"
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker

import paper_style as ps

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)   # so compute() can import fastrho.* and scripts.dog_gen
DEFAULT_CACHE = os.path.join(_HERE, "paper", "figdata", "dog_fig.npz")
DEFAULT_OUT = os.path.join(_HERE, "paper", "figures", "fig_dog.pdf")
DEFAULT_STATS_JSON = os.path.join(_HERE, "paper", "figdata", "dog_fig_stats.json")

TRUTH, VIL, BRD = ps.C["truth"], ps.TRANSFER["village"], ps.TRANSFER["breed"]  # black / village-blue / breed-orange


# ============================================================================
# COMPUTE  (msprime + model inference; heavy imports are LOCAL so render() needs none of them)
# ============================================================================
def compute(ckpt_path, stats_path, NP=90, device="cpu"):
    """Run the paired village/breed simulations + frozen-model inference and return the plotted
    arrays as a dict (the dog_fig.npz schema). device='cpu' => fp32 => deterministic."""
    import msprime
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_intervals
    from fastrho.gt_features import GTTokenFeaturizer
    from fastrho.features import FeatureConfig, mean_r2_slice
    from fastrho.preprocess import mean_rate_between
    from fastcxt.sfs import basic_filtering
    from scripts.dog_gen import (_village_traj, _breed_traj, _demography_from_traj, _clean_traj,
                                 make_recombination_map, RecombPriors, L, MAX_RHO)

    ckpt = open(ckpt_path).read().strip()
    model, cfg, stats = load_model(ckpt, stats_path, device=device)

    def build_fc(s):
        kw = {}
        if "ld_radii" in s: kw["ld_radii"] = tuple(int(x) for x in np.asarray(s["ld_radii"]).ravel())
        if "disjoint_bands" in s: kw["disjoint_bands"] = bool(int(s["disjoint_bands"]))
        if "stride_after" in s: kw["stride_after"] = int(s["stride_after"])
        if "max_neighbors" in s: kw["max_neighbors"] = int(s["max_neighbors"])
        return kw, FeatureConfig(**kw)

    FCKW, FC = build_fc(stats)
    RADII = FCKW.get("ld_radii", (300, 2000, 15000, 75000, 300000, 1500000))
    FEAT = GTTokenFeaturizer(config=FC, fold=True)
    EDGES = np.append(np.arange(0, L, 100000), L)          # 100 kb -- the (c)/(d) accuracy metric
    FINE = 50000                                           # finer bins for the (b) example map
    EDGES_F = np.append(np.arange(0, L, FINE), L)
    CENTERS_F = (EDGES_F[:-1] + FINE / 2) / 1e6

    def lp(a, b):
        ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
        return pearsonr(np.log(a[ok]), np.log(b[ok]))[0] if ok.sum() >= 8 else np.nan

    def infer_raw(ts, mu):
        """(bp_edges, r_per_bp) so we can bin at either the 100 kb metric or the display grid."""
        gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position.astype(np.float64)
        gmf, posf = basic_filtering(gm, pos)
        if gmf.shape[1] < 12:
            return None
        p = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=device, featurizer=FEAT)
        bp = np.concatenate([[p["pos_left"][0]], p["pos_right"]])
        return bp, p["r_per_bp"]

    # ---- (a) LD decay by regime: average band mean_r2 for village vs severe breed ----
    def band_r2(mode_traj_fn, severe=False, n=34, seed0=900):
        acc = []
        for i in range(n):
            rng = np.random.default_rng(seed0 + i)
            traj = _clean_traj(mode_traj_fn(rng))
            if severe:  # force a severe breed (small present Ne) for a clean contrast
                traj[0] = (0.0, float(10 ** rng.uniform(np.log10(40), np.log10(70))))
            mu = 10 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))
            mean_r = 10 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))
            rt = 4.0 * max(s for _, s in traj) * mean_r * L
            if rt > MAX_RHO: mean_r *= MAX_RHO / rt
            rm = make_recombination_map(L, rng, kind="gp", mean_rate=mean_r, priors=RecombPriors(sequence_length=L))
            ts = msprime.sim_ancestry(samples=int(rng.choice([30, 50, 67])), demography=_demography_from_traj(traj),
                                      recombination_rate=rm, sequence_length=L, random_seed=int(rng.integers(1, 2**31)))
            ts = msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))
            gm = ts.genotype_matrix().T.astype(np.int8); pos = ts.tables.sites.position.astype(np.float64)
            tok = FEAT(gm, pos, {"sequence_length": float(L)})["tokens"]
            assert tok.shape[1] == FEAT.n_features, (tok.shape[1], FEAT.n_features)
            mr = tok[:, mean_r2_slice(FC)]     # per-radius mean r^2 bands, addressed BY NAME (not tok[:,4:4+R])
            acc.append(np.nanmean(np.where(mr > 0, mr, np.nan), axis=0))
        decay = np.nanmean(np.vstack(acc), axis=0)
        # physical-range guard: mean r^2 in [0,1]. Reading the wrong token columns (e.g. log_npairs)
        # is exactly how panel (a) broke in the 15k regen -- values shot to ~5. Fail loudly instead.
        assert np.all(np.isfinite(decay)) and decay.max() <= 1.0 + 1e-6, f"LD-decay not in [0,1]: {decay}"
        return decay

    vil_decay = band_r2(_village_traj, severe=False)
    brd_decay = band_r2(_breed_traj, severe=True)

    # ---- (b,c,d) paired regions: shared map under village + breed ----
    rows = []   # (Ne_present_breed, lp_own, lp_transfer)
    best = None  # (truth_w, vil_w, brd_w, score, seed)   -- for the (b) example
    for i in range(NP):
        rng = np.random.default_rng(10_000 + i)
        n_dip = int(rng.choice([30, 50, 67]))
        mu = 10 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))
        mean_r = 10 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))
        vtraj = _clean_traj(_village_traj(rng)); btraj = _clean_traj(_breed_traj(rng))
        ne_max = max(max(s for _, s in vtraj), max(s for _, s in btraj))
        rt = 4.0 * ne_max * mean_r * L
        if rt > MAX_RHO: mean_r *= MAX_RHO / rt
        kind = "hotspot" if rng.random() < 0.5 else "gp"
        rm = make_recombination_map(L, rng, kind=kind, mean_rate=mean_r,
                                    priors=RecombPriors(sequence_length=L))
        def sim(traj):
            ts = msprime.sim_ancestry(samples=n_dip, demography=_demography_from_traj(traj),
                                      recombination_rate=rm, sequence_length=L, random_seed=int(rng.integers(1, 2**31)))
            return msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))
        vraw = infer_raw(sim(vtraj), mu); braw = infer_raw(sim(btraj), mu)
        if vraw is None or braw is None: continue
        tw = mean_rate_between(np.asarray(rm.position), np.asarray(rm.rate), EDGES)
        vw = mean_rate_between(vraw[0], vraw[1], EDGES); bw = mean_rate_between(braw[0], braw[1], EDGES)
        lo, lt = lp(bw, tw), lp(vw, tw)
        rows.append((btraj[0][1], lo, lt))
        if np.isfinite(lt) and np.isfinite(lo):
            # (b) example: prefer a clean recovery -- village tracks well (high lt), breed fails
            # (low lo), and a smooth gp landscape reads better than isolated hotspot spikes.
            score = lt - 0.5 * max(lo, 0.0) + (0.20 if kind == "gp" else 0.0)
            if lt > 0.8 and lo < 0.45 and (best is None or score > best[3]):
                tw_f = mean_rate_between(np.asarray(rm.position), np.asarray(rm.rate), EDGES_F)
                vw_f = mean_rate_between(vraw[0], vraw[1], EDGES_F)
                bw_f = mean_rate_between(braw[0], braw[1], EDGES_F)
                best = (tw_f, vw_f, bw_f, score, i)

    Ne = np.array([r[0] for r in rows]); own = np.array([r[1] for r in rows]); trn = np.array([r[2] for r in rows])

    # (d) rescue vs severity: bin BOTH curves by breed Ne
    bins = [(40, 70), (70, 120), (120, 220), (220, 500)]
    bx, b_own, b_trn = [], [], []
    for loo, hii in bins:
        m = (Ne >= loo) & (Ne < hii)
        if m.sum() >= 3:
            bx.append(np.sqrt(loo * hii)); b_own.append(np.nanmedian(own[m])); b_trn.append(np.nanmedian(trn[m]))

    _bt, _bv, _bb = (best[0], best[1], best[2]) if best is not None else (np.array([]),) * 3
    b_seed = int(best[4]) if best is not None else -1
    return dict(
        radii=np.asarray(RADII, float), vil_decay=vil_decay, brd_decay=brd_decay,
        b_centers=CENTERS_F, b_truth=_bt, b_vil=_bv, b_brd=_bb, b_seed=b_seed,
        Ne=Ne, own=own, trn=trn,
        d_bx=np.array(bx), d_own=np.array(b_own), d_trn=np.array(b_trn),
        n_regions=int(len(rows)),
    )


# ============================================================================
# RENDER  (pure matplotlib from the cache dict; NO msprime / torch / GPU)
# ============================================================================
def render(d, out_pdf):
    ps.style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
    fig = plt.figure(figsize=(11.0, 7.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.27)
    axa, axb, axc, axd = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                          fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))

    radii = np.asarray(d["radii"], float)
    vil_decay = np.asarray(d["vil_decay"]); brd_decay = np.asarray(d["brd_decay"])
    b_centers = np.asarray(d["b_centers"]); b_truth = np.asarray(d["b_truth"])
    b_vil = np.asarray(d["b_vil"]); b_brd = np.asarray(d["b_brd"])
    Ne = np.asarray(d["Ne"]); own = np.asarray(d["own"]); trn = np.asarray(d["trn"])
    bx = np.asarray(d["d_bx"]); b_own = np.asarray(d["d_own"]); b_trn = np.asarray(d["d_trn"])

    # (a) LD decay
    axa.plot(radii, vil_decay, "o-", color=VIL, label="village (large $N_e$)")
    axa.plot(radii, brd_decay, "s-", color=BRD, label="breed (bottleneck)")
    axa.set_xscale("log"); axa.set_xlabel("inter-SNP distance (bp)"); axa.set_ylabel(r"mean $r^2$")
    axa.set_title("(a) the breed bottleneck inflates LD to ~Mb", loc="left")
    axa.legend(frameon=False, fontsize=8.5)

    # (b) example locus -- fine grid so the map is a smooth curve
    if b_truth.size:
        axb.plot(b_centers, b_truth, color=TRUTH, lw=2.2, zorder=4, label="true map")
        axb.plot(b_centers, b_vil, color=VIL, lw=1.8, zorder=3, label="village transfer")
        axb.plot(b_centers, b_brd, color=BRD, lw=1.6, alpha=0.9, zorder=2, label="breed, own data")
        axb.set_yscale("log")
        _v = np.concatenate([b_truth, b_vil, b_brd]); _v = _v[_v > 0]
        axb.set_ylim(10.0 ** np.floor(np.log10(_v.min())), 10.0 ** np.ceil(np.log10(_v.max())) * 1.6)
        axb.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        axb.yaxis.set_major_formatter(mticker.LogFormatterMathtext())
        axb.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12))
        axb.yaxis.set_minor_formatter(mticker.NullFormatter())
        axb.annotate("true map", (b_centers[-1], b_truth[-1]), xytext=(4, 6), textcoords="offset points",
                     color=TRUTH, fontsize=8.4, fontweight="medium", va="bottom")
        axb.annotate("village\ntransfer", (b_centers[-1], b_vil[-1]), xytext=(4, -10), textcoords="offset points",
                     color=VIL, fontsize=8.4, va="top")
        axb.annotate("breed, own data\n(no fine-scale signal)", (b_centers[len(b_centers) // 3], b_brd[len(b_centers) // 3]),
                     xytext=(0, -22), textcoords="offset points", color=BRD, fontsize=8.0, ha="center", va="top")
    axb.set_xlabel("position (Mb)"); axb.set_ylabel("recombination rate (per bp)")
    axb.margins(x=0.02)
    axb.set_title("(b) one shared locus: the village panel recovers the map, the breed alone cannot",
                  loc="left", fontsize=9.6)

    # (c) transfer lift -- dumbbell
    mo, mt = np.nanmedian(own), np.nanmedian(trn)
    for o, t in zip(own, trn):
        if np.isfinite(o) and np.isfinite(t):
            axc.plot([0, 1], [o, t], color=(VIL if t >= o else BRD), lw=0.5, alpha=0.18, zorder=1)
    axc.scatter(np.zeros(len(own)), own, color=BRD, s=12, alpha=0.8, zorder=2)
    axc.scatter(np.ones(len(trn)), trn, color=VIL, s=12, alpha=0.8, zorder=2)
    axc.plot([0, 1], [mo, mt], color="black", lw=2.4, zorder=3)
    axc.scatter([0, 1], [mo, mt], color="black", s=40, zorder=4)
    axc.annotate(f"{mo:.2f}", (0, mo), xytext=(-32, -2), textcoords="offset points", fontsize=9)
    axc.annotate(f"{mt:.2f}", (1, mt), xytext=(8, -2), textcoords="offset points", fontsize=9)
    axc.set_xlim(-0.45, 1.45); axc.set_ylim(-0.25, 1.0)
    axc.set_xticks([0, 1]); axc.set_xticklabels(["breed\nown data", "breed via\nvillage transfer"])
    axc.set_ylabel("breed-map accuracy (100 kb log-Pearson)")
    axc.axhline(0, color="0.6", lw=0.6)
    axc.set_title("(c) transfer lifts the breed map", loc="left")

    # (d) rescue vs severity
    axd.scatter(Ne, own, s=9, color=BRD, alpha=0.20, lw=0, zorder=1)
    axd.scatter(Ne, trn, s=9, color=VIL, alpha=0.20, lw=0, zorder=1)
    if bx.size:
        axd.fill_between(bx, b_own, b_trn, color=VIL, alpha=0.13, lw=0, zorder=2, label="rescue (transfer $-$ own)")
        axd.plot(bx, b_trn, "o-", color=VIL, lw=2.4, ms=6, zorder=4, label="village transfer ($N_e$-independent)")
        axd.plot(bx, b_own, "s-", color=BRD, lw=2.2, ms=6, zorder=3, label="breed, own data")
    axd.set_xscale("log"); axd.set_ylim(-0.3, 1.0)
    axd.set_xlabel(r"breed present $N_e$   (bottleneck severity $\rightarrow$)")
    axd.set_ylabel("breed-map accuracy (100 kb log-Pearson)")
    axd.axhline(0, color="0.6", lw=0.6, zorder=0)
    axd.invert_xaxis()
    axd.xaxis.set_major_locator(mticker.FixedLocator([50, 100, 200, 500]))
    axd.xaxis.set_major_formatter(mticker.FixedFormatter(["50", "100", "200", "500"]))
    axd.xaxis.set_minor_locator(mticker.NullLocator())
    if bx.size:
        axd.annotate("rescue widens\nas the bottleneck\ndeepens", (bx[0], (b_own[0] + b_trn[0]) / 2),
                     xytext=(-6, 0), textcoords="offset points", ha="right", va="center",
                     fontsize=8.0, color="#555")
    axd.set_title("(d) transfer rescues most where the breed's own data fails worst", loc="left",
                  fontsize=9.6)
    axd.legend(frameon=False, fontsize=8.0, loc="lower center")

    for ax in (axa, axb, axc, axd):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.savefig(out_pdf, bbox_inches="tight")
    return float(np.nanmedian(own)), float(np.nanmedian(trn)), int(len(own))


def _write_stats_json(d, path):
    import json
    from scipy.stats import spearmanr
    own = np.asarray(d["own"], float)
    transfer = np.asarray(d["trn"], float)
    ne = np.asarray(d["Ne"], float)
    keep = np.isfinite(own) & np.isfinite(transfer) & np.isfinite(ne)
    own, transfer, ne = own[keep], transfer[keep], ne[keep]
    gain = transfer - own
    bootstrap_seed = 20260807
    bootstrap_replicates = 20_000
    rng = np.random.default_rng(bootstrap_seed)
    index = rng.integers(0, gain.size, (bootstrap_replicates, gain.size))
    boot_gain = np.median(gain[index], axis=1)
    own_ne = spearmanr(ne, own)
    gain_ne = spearmanr(ne, gain)
    with open(path, "w") as fh:
        json.dump({"own_median": round(float(np.nanmedian(d["own"])), 4),
                   "trn_median": round(float(np.nanmedian(d["trn"])), 4),
                   "n_regions": int(d.get("n_regions", len(d["own"]))),
                   "n_improved": int(np.sum(gain > 0)),
                   "fraction_improved": float(np.mean(gain > 0)),
                   "paired_median_gain": float(np.median(gain)),
                   "paired_median_gain_ci95": [float(x) for x in np.quantile(boot_gain, [0.025, 0.975])],
                   "bootstrap_replicates": bootstrap_replicates,
                   "bootstrap_seed": bootstrap_seed,
                   "own_ne_spearman": float(own_ne.statistic),
                   "own_ne_p": float(own_ne.pvalue),
                   "gain_ne_spearman": float(gain_ne.statistic),
                   "gain_ne_p": float(gain_ne.pvalue),
                   "b_seed": int(d.get("b_seed", -1))}, fh, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Dog village->breed transfer figure (cache-first).")
    ap.add_argument("--recompute", action="store_true",
                    help="run msprime + inference and rewrite the cache (default: render from cache)")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="dog_fig.npz path")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output PDF path")
    ap.add_argument("--ckpt", default=None, help="dogbn ckpt.txt (required for --recompute)")
    ap.add_argument("--stats", default=None, help="dogbn feat_stats.npz (required for --recompute)")
    ap.add_argument("--np", type=int, default=90, dest="npair", help="paired regions for --recompute")
    ap.add_argument("--device", default=None,
                    help="cpu|cuda[:i]; default cpu (fp32, deterministic). Use cuda only for fast iteration.")
    a = ap.parse_args()

    if a.recompute:
        if not (a.ckpt and a.stats):
            ap.error("--recompute needs --ckpt and --stats")
        device = a.device or "cpu"     # CPU => fp32 => deterministic committed cache
        d = compute(a.ckpt, a.stats, NP=a.npair, device=device)
        np.savez(a.cache, **d)
        _write_stats_json(d, DEFAULT_STATS_JSON)
        print(f"recomputed cache -> {a.cache} (n_regions={d['n_regions']}, device={device})")
    else:
        d = dict(np.load(a.cache, allow_pickle=True))
        _write_stats_json(d, DEFAULT_STATS_JSON)

    mo, mt, n = render(d, a.out)
    print(f"rendered {a.out} | own median={mo:.3f} transfer median={mt:.3f} | n_regions={n}")


if __name__ == "__main__":
    main()
