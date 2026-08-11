"""Infer a recombination map from REAL genotypes (realdata/hap/<key>.npz) with the matching
fastrho model, and compare to the published stdpopsim genetic map (same assembly).
Saves realdata/maps/<key>.npz with the windowed true vs inferred map + correlations.
"""
import sys, os
import numpy as np
# use the co-located package (keeps an isolated checkout isolated; same result for the
# canonical /home/kkor/fastrho checkout)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import pearsonr, spearmanr
import stdpopsim
from fastrho.translate import load_model, predict_map_from_genotype_matrix, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.preprocess import mean_rate_between

def get_ck(name):
    if name == "base":
        return (open("/home/kkor/fastrho_data/campaign/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz")
    if name == "model2":
        return ("/home/kkor/fastrho_data/campaign2/train/fastrho/version_0/checkpoints/epoch=49-val_loss=-0.151.ckpt",
                "/home/kkor/fastrho_data/campaign2/shards/feat_stats.npz")
    if name == "gt":
        return (open("/home/kkor/fastrho_data/campaign/ckpt_gtf.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign/shards_gtf15k/feat_stats.npz")
    if name == "self":
        return (open("/home/kkor/fastrho_data/campaign_self/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign_self/shards15k/feat_stats.npz")
    if name == "self2":   # prototype selfing model (deployed for the ED selfer figure: 15k self
                          # regressed real A. thaliana recovery 0.27->0.11, like rich=prototype)
        return ("/home/kkor/fastrho_data/campaign_self2/train/fastrho/version_0/checkpoints/epoch=48-val_loss=0.109.ckpt",
                "/home/kkor/fastrho_data/campaign_self2/shards/feat_stats.npz")
    if name == "dogmodel":
        return (open("/home/kkor/fastrho_data/campaign_dog/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign_dog/shards/feat_stats.npz")
    if name == "dogbn":   # bottleneck-aware dog model (isolated campaign)
        return (open("/home/kkor/fastrho_data/campaign_dog_bottleneck/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign_dog_bottleneck/shards15k/feat_stats.npz")
    if name == "hidip":   # broadened-Ne model (dipteran + Anopheles atlas); single source of truth
        return (open("/home/kkor/fastrho_data/campaign_hidip/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign_hidip/shards15k/feat_stats.npz")
    if name == "rich":    # large-n featurizer extension (four-gamete + |D'|), noise-augmented
        return (open("/home/kkor/fastrho_data/campaign_rich/ckpt.txt").read().strip(),
                "/home/kkor/fastrho_data/campaign_rich/feat_stats.npz")
    raise ValueError(name)


def featurizer_from_stats(stats, default_radii):
    """Rebuild the GT featurizer config the model was trained with (parity).
    Reads ld_radii/disjoint_bands/stride_after/max_neighbors from feat_stats if present."""
    from fastrho.features import FeatureConfig
    kw = {}
    if "ld_radii" in stats:
        kw["ld_radii"] = tuple(int(x) for x in np.asarray(stats["ld_radii"]).ravel())
    else:
        kw["ld_radii"] = tuple(default_radii)
    if "disjoint_bands" in stats:
        kw["disjoint_bands"] = bool(int(stats["disjoint_bands"]))
    if "stride_after" in stats:
        kw["stride_after"] = int(stats["stride_after"])
    if "max_neighbors" in stats:
        kw["max_neighbors"] = int(stats["max_neighbors"])
    return FeatureConfig(**kw)


DOG_RADII = (300, 2000, 15000)   # finer LD radii for the short-LD (village-dog) regime
DOGBN_RADII = (5000, 25000, 50000)  # legacy 17-feature bottleneck checkpoint defaults
OUT = "/home/kkor/realdata/maps"
W = 100000
DEV = "cuda:0"


def truth_windows(map_sp, map_id, chrom, edges):
    sp = stdpopsim.get_species(map_sp)
    gm = sp.get_genetic_map(map_id)
    rm = gm.get_chromosome_map(chrom.replace("chr", ""))
    pos = np.asarray(rm.position, float); rate = np.asarray(rm.rate, float)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    return mean_rate_between(pos, rate, edges)


def main():
    key = sys.argv[1]
    z = np.load(f"/home/kkor/realdata/hap/{key}.npz", allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64)
    model_name = sys.argv[2] if len(sys.argv) > 2 else str(z["model"])
    mode = str(z["mode"]); mu = float(z["mu"])
    chrom = str(z["chrom"]); map_sp = str(z["map_sp"]); map_id = str(z["map_id"])
    ckpt, stats_p = get_ck(model_name)
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    if model_name in ("gt", "dogmodel", "dogbn"):
        from fastcxt.sfs import basic_filtering
        from fastrho.features import FeatureConfig
        if model_name == "dogmodel":
            fc = FeatureConfig(ld_radii=DOG_RADII)
        elif model_name == "dogbn":
            # The deployed 17-feature checkpoint predates featurizer metadata in
            # feat_stats and used the base 5/25/50-kb radii. Newer dog campaigns
            # store their radii explicitly and override this fallback.
            fc = featurizer_from_stats(stats, DOGBN_RADII)
        else:
            fc = FeatureConfig()
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)
        pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=DEV,
                                 featurizer=GTTokenFeaturizer(config=fc, fold=True))
    else:
        pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                mutation_rate=mu, Ne=None, device=DEV)
    left = pred["pos_left"]; right = pred["pos_right"]
    bp = np.r_[left[0], right]
    lo = int(left[0]); hi = int(right[-1])
    edges = np.append(np.arange(lo, hi, W), hi)
    pred_r = mean_rate_between(bp, pred["r_per_bp"], edges)
    true_r = truth_windows(map_sp, map_id, chrom, edges)
    centers = (edges[:-1] + W / 2) / 1e6
    k = min(len(pred_r), len(true_r))
    p, t, c = pred_r[:k], true_r[:k], centers[:k]
    ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
    pe = pearsonr(p[ok], t[ok])[0]; sp_ = spearmanr(p[ok], t[ok])[0]
    lpe = pearsonr(np.log(p[ok]), np.log(t[ok]))[0]
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/{key}.npz", centers=c, truth=t, pred=p, ok=ok, chrom=chrom,
             pearson=pe, spearman=sp_, log_pearson=lpe, n_hap=gm.shape[0], n_snp=gm.shape[1],
             map_id=map_id, model=model_name, Ne_est=float(pred["Ne_estimated"]))
    print(f"{key}: {gm.shape[0]}hap x {gm.shape[1]}SNP {chrom} | windows={ok.sum()} | "
          f"Pearson={pe:.3f} Spearman={sp_:.3f} logPearson={lpe:.3f} | vs {map_id}")


if __name__ == "__main__":
    main()
