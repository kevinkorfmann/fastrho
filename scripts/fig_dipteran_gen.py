"""Generate Fig. (dipteran) figdata for the REFRAMED calibration-strength story.

Old story (campaign2): at extreme Ne fastrho shrank the absolute rate (bias~0.5) and a
genome-mean anchor was needed. With the high-Ne-fixed model (campaign_hidip) the model is
now absolute-rate calibrated in this regime (bias~0.9) with NO anchor. This regenerates
dipteran_bias.npz on an extreme-Ne (Ne=1e6) Drosophila region, recording the hidip recovery
and the old campaign2 prediction for contrast.

Run on sesame (GPU + configs):
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python scripts/fig_dipteran_gen.py
"""
import os, sys, json
import numpy as np
sys.path.insert(0, "/home/kkor/fastrho")
import tskit
from scipy.stats import pearsonr, spearmanr
from fastrho.translate import load_model, predict_map_from_ts
from fastrho.preprocess import mean_rate_between

C = "/home/kkor/fastrho_data/campaign/configs/real_drosophila"
RIDX = 17
OUT = "/home/kkor/fastrho/paper/figdata/dipteran_bias.npz"
HIDIP = ("/home/kkor/fastrho_data/campaign_hidip/train15k/fastrho/version_0/checkpoints/epoch=37-val_loss=-0.178.ckpt",
         "/home/kkor/fastrho_data/campaign_hidip/shards15k/feat_stats.npz")
OLD = ("/home/kkor/fastrho_data/campaign2/train/fastrho/version_0/checkpoints/epoch=49-val_loss=-0.151.ckpt",
       "/home/kkor/fastrho_data/campaign2/shards/feat_stats.npz")


def predict(ck, st, ts, mu, Ne):
    model, cfg, stats = load_model(ck, st, device="cuda:0")
    p = predict_map_from_ts(ts, model, cfg, stats, mutation_rate=mu, Ne=Ne, device="cuda:0")
    return p


def main():
    cfg = json.load(open(os.path.join(C, "config.json")))
    Ne, mu = float(cfg["Ne"]), float(cfg["mu"])
    ts = tskit.load(os.path.join(C, f"region_{RIDX:03d}.trees"))
    z = np.load(os.path.join(C, f"region_{RIDX:03d}.npz"), allow_pickle=True)
    mp = np.asarray(z["map_position"], float); mr = np.where(np.isfinite(z["map_rate"]), z["map_rate"], 0.0)

    hi = predict(*HIDIP, ts, mu, Ne)
    old = predict(*OLD, ts, mu, Ne)
    left, right = hi["pos_left"], hi["pos_right"]
    mid = 0.5 * (left + right)
    rhat = np.asarray(hi["r_per_bp"], float)               # hidip per-SNP rate
    rhat_old = np.asarray(old["r_per_bp"], float)           # old (campaign2) per-SNP rate
    # true rate per SNP interval
    rtrue = np.array([mean_rate_between(mp, mr, np.array([a, b]))[0] for a, b in zip(left, right)])

    ok = np.isfinite(rhat) & np.isfinite(rtrue) & (rtrue > 0) & (rhat > 0)
    hot = (rtrue > 2 * np.median(rtrue[ok])).astype(int)
    baseline = float(np.median(rtrue[ok & (hot == 0)]))
    br = lambda a: float(np.median(a[ok] / rtrue[ok]))
    br_old = float(np.median(rhat_old[ok] / rtrue[ok]))
    anchor_scale = float(np.average(rtrue[ok], weights=(right - left)[ok]) /
                         np.average(rhat[ok], weights=(right - left)[ok]))
    stats = dict(region=f"region_{RIDX:03d}", n_int=int(ok.sum()), Ne=Ne, mu=mu,
                 baseline=baseline,
                 overall_br=br(rhat), overall_br_old=br_old,
                 baseline_br=float(np.median((rhat / rtrue)[ok & (hot == 0)])),
                 hotspot_br=float(np.median((rhat / rtrue)[ok & (hot == 1)])),
                 logpearson=float(pearsonr(np.log(rhat[ok]), np.log(rtrue[ok]))[0]),
                 spearman=float(spearmanr(rhat[ok], rtrue[ok])[0]),
                 anchor_scale=anchor_scale,
                 baseline_br_anchored=float(np.median((rhat * anchor_scale / rtrue)[ok & (hot == 0)])),
                 frac_hot=float(hot[ok].mean()))
    np.savez(OUT, mid=mid, rhat=rhat, rhat_old=rhat_old, rtrue=rtrue, hot=hot,
             mp=mp, mr=mr, scale=anchor_scale, stats=json.dumps(stats),
             meta=json.dumps(dict(model="campaign_hidip", contrast="campaign2")))
    print(f"region_{RIDX:03d} Ne={Ne:.0e}: hidip bias={stats['overall_br']:.3f} "
          f"(old {br_old:.3f}), logPearson={stats['logpearson']:.3f}, anchor_scale={anchor_scale:.3f}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
