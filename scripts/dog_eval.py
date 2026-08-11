"""Validate the bottleneck-aware dog model on held-out sims (the clean fine-scale check;
real pedigree maps like Campbell2016 are ~Mb-resolution and the public 'dog' WGS panel is
multi-population/admixed, so sims with EXACT truth are the honest validation).

Reports per-region shape recovery (per-interval + 100kb-windowed logPearson) split by
village vs breed and by bottleneck severity, plus village density-robustness, and writes a
3-panel validation figure.

Usage: python scripts/dog_eval.py <test_shard_dir> <ckpt.txt> <feat_stats.npz> [out.pdf]
"""
import sys, glob, json
import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from fastrho.translate import load_model, predict_from_tokens
from fastrho.preprocess import mean_rate_between

shard_dir = sys.argv[1]
ckpt = open(sys.argv[2]).read().strip()
stats_p = sys.argv[3]
out_pdf = sys.argv[4] if len(sys.argv) > 4 else "/home/kkor/fastrho_data/campaign_dog_bottleneck/dog_validation.pdf"
model, cfg, stats = load_model(ckpt, stats_p, device="cuda:0")


def windowed(pl, pr, rate, W=100000):
    bp = np.concatenate([[pl[0]], pr])
    edges = np.append(np.arange(pl[0], pr[-1], W), pr[-1])
    return (edges[:-1] + W / 2), mean_rate_between(bp, rate, edges)


def lp(a, b):
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return pearsonr(np.log(a[ok]), np.log(b[ok]))[0] if ok.sum() >= 8 else np.nan


rows = []          # mode, Ne_present, num_sites, li, l100
example = None      # (centers, true_w, pred_w) for a representative village region
for f in sorted(glob.glob(shard_dir + "/ts_*.npz")):
    z = np.load(f, allow_pickle=True)
    if "tokens" not in z or z["tokens"].shape[0] < 12 or "interval_target" not in z:
        continue
    meta = json.loads(str(z["meta"]))
    tok = z["tokens"].astype(np.float32); pos = z["positions"].astype(np.float64)
    itgt = z["interval_target"].astype(np.float64)
    try:
        pred = predict_from_tokens(model, cfg, stats, tok, pos, 2 * int(meta["n_samples"]),
                                   float(meta["mutation_rate"]), Ne=None, device="cuda:0")
    except Exception:
        continue
    k = min(len(pred["r_per_bp"]), len(itgt))
    if k < 12:
        continue
    li = lp(pred["r_per_bp"][:k], itgt[:k])
    c, pw = windowed(pred["pos_left"][:k], pred["pos_right"][:k], pred["r_per_bp"][:k])
    _, tw = windowed(pred["pos_left"][:k], pred["pos_right"][:k], itgt[:k])
    l100 = lp(pw, tw)
    rows.append((meta["mode"], meta["Ne_present"], meta["num_sites"], li, l100))
    if example is None and meta["mode"] == "village" and l100 > 0.8 and len(c) > 8:
        example = (c / 1e6, tw, pw, l100)

mode = np.array([r[0] for r in rows]); nep = np.array([r[1] for r in rows])
dens = np.array([r[2] for r in rows]); li = np.array([r[3] for r in rows]); l100 = np.array([r[4] for r in rows])


def med(m):
    return (m.sum(), np.nanmedian(li[m]), np.nanmedian(l100[m]))


groups = [("village", mode == "village"),
          ("breed Ne<70", (mode == "breed") & (nep < 70)),
          ("breed 70-150", (mode == "breed") & (nep >= 70) & (nep < 150)),
          ("breed >=150", (mode == "breed") & (nep >= 150))]
print("held-out SIM shape recovery (median logPearson):")
for name, m in groups:
    n, a, b = med(m)
    print("  %-14s n=%3d | per-interval=%.3f | 100kb=%.3f" % (name, n, a, b))

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.0))
if example is not None:
    c, tw, pw, e = example
    ax[0].plot(c, tw, color="black", lw=1.6, label="truth")
    ax[0].plot(c, pw, color="#1f77b4", lw=1.6, alpha=0.85, label="fastrho-dog")
    ax[0].set_yscale("log"); ax[0].set_xlabel("position (Mb)")
    ax[0].set_ylabel("recombination rate (per bp, 100 kb)")
    ax[0].set_title("village region (logPearson=%.2f)" % e)
    ax[0].legend(frameon=False, fontsize=9)

vil = mode == "village"
bins = [(0, 1200), (1200, 2000), (2000, 3500), (3500, 99999)]
xs = [np.nanmedian(dens[vil & (dens >= lo) & (dens < hi)]) for lo, hi in bins]
ys = [np.nanmedian(l100[vil & (dens >= lo) & (dens < hi)]) for lo, hi in bins]
ax[1].plot(xs, ys, "o-", color="#1f77b4")
ax[1].axhline(0, color="grey", lw=0.6)
ax[1].set_ylim(0, 1); ax[1].set_xlabel("SNP density (per Mb)")
ax[1].set_ylabel("village 100 kb logPearson")
ax[1].set_title("density-robust (real panel ~7800/Mb)")

names = [g[0] for g in groups]
vals = [med(g[1])[2] for g in groups]
colors = ["#1f77b4", "#d62728", "#d62728", "#d62728"]
ax[2].bar(range(len(names)), vals, color=colors)
ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax[2].set_ylim(0, 1); ax[2].set_ylabel("100 kb logPearson")
ax[2].set_title("village resolvable; breed transfer-only")
for s in ["top", "right"]:
    for a in ax:
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out_pdf)
print("wrote", out_pdf)
