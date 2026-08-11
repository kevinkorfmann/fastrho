"""Field-guide demonstration on an UNDERSTUDIED taxon with NO published recombination map.

Runs the paper's field-guide checklist end to end on one genotype matrix:
  (1) a single amortized forward pass -> per-interval map with calibrated intervals,
  (2) the blind-QC suite that certifies a map with no ground truth --
        calibration (interval widths / low-information flag),
        subsample reproducibility (disjoint half-maps),
        posterior-predictive LD-decay (diploid coalescent under the inferred map),
  (3) the inversion karyotype PCA + ONE biology-given prediction: recombination is
        suppressed across a segregating inversion.

For unphased (dosage) data the LD-decay curve is computed as genotype-dosage r^2 (composite LD)
on the true diploids, and simulated with DIPLOID SMC' coalescent under the inferred map, matching
the observed MAF>=0.05 ascertainment -- so observed and simulated are directly comparable. (For
the high-diversity redpoll this check is scale-limited; the headline claims are the scale-invariant
reproducibility and inversion ratio.)

Run on sesame:
  PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 ~/venvs/fastrho/bin/python scripts/fieldguide_run.py \
      --hap /home/kkor/realdata/hap/<key>.npz --model gt --mu 4.6e-9 \
      --inv-chrom <scaffold> --inv-start <bp> --inv-end <bp> \
      --out /home/kkor/realdata/fieldguide/<key>.npz
"""
import os, sys, time, json, argparse
import numpy as np
sys.path.insert(0, "/home/kkor/fastrho")
sys.path.insert(0, "/home/kkor/fastrho/scripts")
from scipy.stats import pearsonr, mannwhitneyu
import msprime
from fastrho.translate import load_model, predict_map_from_genotype_matrix, predict_intervals
from fastrho.gt_features import GTTokenFeaturizer
from fastrho.features import FeatureConfig
from fastrho.preprocess import mean_rate_between
from realdata_infer import get_ck, DOG_RADII

DEV = "cuda:0"
BINS = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400])
DOSAGE_MODELS = ("gt", "dogmodel", "dogbn")


def infer(gm, pos, mu, model_name, model, cfg, stats, dev=DEV):
    """One amortized forward pass. Unphased/composite-LD data -> folded GT featurizer path;
    phased/haploid -> the genotype-matrix path."""
    if model_name in DOSAGE_MODELS:
        from fastcxt.sfs import basic_filtering
        fc = FeatureConfig(ld_radii=DOG_RADII) if model_name in ("dogmodel", "dogbn") else FeatureConfig()
        gmf, posf = basic_filtering(gm.astype(np.int8), pos)
        pred = predict_intervals(model, cfg, stats, gmf, posf, mu, Ne=None, device=dev,
                                 featurizer=GTTokenFeaturizer(config=fc, fold=True))
    else:
        pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                                mutation_rate=mu, Ne=None, device=dev)
    return pred


def windowed(pred, W):
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    edges = np.append(np.arange(pred["pos_left"][0], pred["pos_right"][-1], W), pred["pos_right"][-1])
    r = mean_rate_between(bp, pred["r_per_bp"], edges)
    rho = mean_rate_between(bp, pred["rho_per_bp"], edges)
    return edges[:-1], r, rho


def dosage_matrix(gm):
    """Reconstruct the (n_ind, n_snp) 0/1/2 dosage matrix from composite-LD pseudo-haps."""
    return (gm[0::2].astype(np.int16) + gm[1::2].astype(np.int16))


def geno_ld_curve(D, pos, n_ind):
    """Genotype-dosage r^2 (= composite LD) per log-distance bin, finite-sample corrected (1/n_ind)."""
    pos = np.asarray(pos, float)
    Dc = D - D.mean(0, keepdims=True); sd = D.std(0)
    sums = np.zeros(len(BINS) - 1); cnts = np.zeros(len(BINS) - 1)
    for a in np.arange(0, len(pos), max(1, len(pos) // 4000)):
        if sd[a] <= 0:
            continue
        j0 = np.searchsorted(pos, pos[a] + BINS[0]); j1 = np.searchsorted(pos, pos[a] + BINS[-1])
        for b in np.arange(j0, j1, max(1, (j1 - j0) // 30)):
            if sd[b] <= 0:
                continue
            r = float((Dc[:, a] * Dc[:, b]).mean()) / (sd[a] * sd[b])
            k = np.searchsorted(BINS, pos[b] - pos[a]) - 1
            if 0 <= k < len(cnts):
                sums[k] += max(r * r - 1.0 / n_ind, 0.0); cnts[k] += 1
    return sums / np.maximum(cnts, 1)


def posterior_predictive(gm, pos, mu, r_bar, Ne, n_ind, simlen=200_000, obslen=5_000_000, K=8):
    """Compare observed genotype-dosage LD decay to diploid SMC' simulations under the inferred
    map (mean collinear rate r_bar, inferred Ne, MAF>=0.05 ascertainment matched)."""
    lo = float(pos[0]); sel = (pos >= lo) & (pos < lo + obslen)
    D = dosage_matrix(gm)[:, sel].astype(float)
    obs = geno_ld_curve(D, pos[sel] - lo, n_ind)
    sims = []
    for k in range(K):
        ts = msprime.sim_ancestry(samples=n_ind, ploidy=2, population_size=Ne, recombination_rate=r_bar,
                                  sequence_length=simlen, random_seed=100 + k, model="smc_prime")
        ts = msprime.sim_mutations(ts, rate=mu, random_seed=200 + k)
        G = ts.genotype_matrix(); Dg = (G[:, 0::2] + G[:, 1::2]).T.astype(float)
        f = Dg.sum(0) / (2 * n_ind); keep = (f >= 0.05) & (f <= 0.95)
        sims.append(geno_ld_curve(Dg[:, keep], ts.tables.sites.position[keep], n_ind))
    sim = np.nanmean(sims, 0); ok = (obs > 0) & (sim > 0)
    rmse = float(np.sqrt(np.mean((np.log10(obs[ok]) - np.log10(sim[ok])) ** 2)))
    return dict(gof=float(np.exp(-rmse)), rmse=rmse, obs=obs, sim=sim, bins=BINS)


def reproducibility(gm, pos, mu, model_name, model, cfg, stats, W=100000):
    """Blind precision check: infer on two disjoint halves of the sample, correlate half-maps in log space."""
    n = gm.shape[0]; idx = np.arange(n)
    h1, h2 = idx[: n // 2], idx[n // 2:]
    lo = int(pos[0]); hi = int(pos[-1]); edges = np.append(np.arange(lo, hi, W), hi)

    def half(h):
        p = infer(gm[h], pos, mu, model_name, model, cfg, stats)
        bp = np.r_[p["pos_left"][0], p["pos_right"]]
        return mean_rate_between(bp, p["r_per_bp"], edges)

    m1, m2 = half(h1), half(h2)
    k = min(len(m1), len(m2)); a, b = m1[:k], m2[:k]
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    return dict(repro=float(pearsonr(np.log(a[ok]), np.log(b[ok]))[0]), half1=a, half2=b, n_win=int(ok.sum()))


def karyotype_pca(gm, pos, inv_start, inv_end):
    """PCA of the inversion-region genotypes -> PC1/PC2, 3-cluster karyotype labels, cluster sizes."""
    D = dosage_matrix(gm).astype(float)
    reg = (pos >= inv_start) & (pos < inv_end)
    X = D[:, reg] - D[:, reg].mean(0, keepdims=True)
    w, V = np.linalg.eigh(X @ X.T)
    pc1 = V[:, -1] * np.sqrt(max(w[-1], 0)); pc2 = V[:, -2] * np.sqrt(max(w[-2], 0))
    if pc1[np.argmax(np.abs(pc1))] < 0:
        pc1 = -pc1
    s = np.sort(pc1); cut = np.sort(np.argsort(np.diff(s))[-2:])
    thr = [(s[i] + s[i + 1]) / 2 for i in cut]
    lab = np.digitize(pc1, thr)
    sizes = [int((lab == k).sum()) for k in range(3)]
    ev = w[-2:][::-1] / max(w.sum(), 1e-9)
    return dict(pc1=pc1, pc2=pc2, labels=lab, sizes=np.array(sizes), ev=ev)


def inversion_check(pred, inv_start, inv_end):
    """Biology-given prediction: mean rate INSIDE a segregating inversion vs its collinear flanks."""
    left = pred["pos_left"]; rho = pred["rho_per_bp"]
    inside = (left >= inv_start) & (left < inv_end)
    flank = (left < inv_start) | (left >= inv_end)
    ri, rf = rho[inside], rho[flank]
    ri = ri[np.isfinite(ri)]; rf = rf[np.isfinite(rf)]
    out = dict(inv_start=int(inv_start), inv_end=int(inv_end), n_inside=int(inside.sum()),
               n_flank=int(flank.sum()), mean_inside=float(np.nanmean(ri)) if ri.size else float("nan"),
               mean_flank=float(np.nanmean(rf)) if rf.size else float("nan"))
    out["ratio"] = out["mean_inside"] / out["mean_flank"] if out["mean_flank"] else float("nan")
    out["mwu_p"] = float(mannwhitneyu(ri, rf, alternative="less").pvalue) if (ri.size and rf.size) else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hap", required=True)
    ap.add_argument("--model", default="gt")
    ap.add_argument("--mu", type=float, required=True)
    ap.add_argument("--inv-chrom", default="")
    ap.add_argument("--inv-start", type=float, default=0)
    ap.add_argument("--inv-end", type=float, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.hap, allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64)
    n_ind = gm.shape[0] // 2 if args.model in DOSAGE_MODELS else gm.shape[0]
    print(f"[data] {os.path.basename(args.hap)}: {gm.shape[0]} hap x {gm.shape[1]} SNP "
          f"({pos[0]:.0f}-{pos[-1]:.0f} bp) model={args.model} mu={args.mu:g}")

    ckpt, stats_p = get_ck(args.model)
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)

    t0 = time.time()
    pred = infer(gm, pos, args.mu, args.model, model, cfg, stats)
    secs = time.time() - t0
    Ne_est = float(pred["Ne_estimated"])
    print(f"[infer] {secs:.1f}s  Ne_est={Ne_est:.3e}")

    s1, r1, rho1 = windowed(pred, 1_000_000)
    s100, r100, rho100 = windowed(pred, 100_000)
    s50, r50, rho50 = windowed(pred, 50_000)

    sig = pred.get("sigma_log_rho")
    calib = dict(mean_sigma_log_rho=float(np.nanmean(sig)) if sig is not None else float("nan"),
                 median_ci_ratio=float(np.nanmedian(pred["r_ci_hi"] / np.maximum(pred["r_ci_lo"], 1e-30)))
                 if "r_ci_hi" in pred else float("nan"))
    if sig is not None:
        calib["frac_lowinfo"] = float(np.mean(sig > np.nanmedian(sig) + np.nanstd(sig)))
    print(f"[calibration] mean sigma(log rho)={calib['mean_sigma_log_rho']:.3f} "
          f"median CI ratio={calib['median_ci_ratio']:.2f} lowinfo={calib.get('frac_lowinfo', float('nan')):.3f}")

    rep = reproducibility(gm, pos, args.mu, args.model, model, cfg, stats)
    print(f"[reproducibility] half-vs-half log-Pearson={rep['repro']:.3f} (n_win={rep['n_win']})")

    r_bar = float(np.nanmedian(pred["r_per_bp"][pred["pos_left"] < pos[0] + 5_000_000]))
    pp = posterior_predictive(gm, pos, args.mu, r_bar, Ne_est, n_ind)
    print(f"[posterior-predictive] genotype LD-decay GoF={pp['gof']:.3f} (rmse={pp['rmse']:.3f})")

    inv = pca = None
    if args.inv_end > args.inv_start > 0:
        inv = inversion_check(pred, args.inv_start, args.inv_end)
        print(f"[inversion] inside/flank rate ratio={inv['ratio']:.3f} "
              f"(MWU p={inv['mwu_p']:.2e}, n_in={inv['n_inside']}, n_flank={inv['n_flank']})")
        if args.model in DOSAGE_MODELS:
            pca = karyotype_pca(gm, pos, args.inv_start, args.inv_end)
            print(f"[karyotype] PC1={pca['ev'][0]*100:.1f}%  cluster sizes={list(pca['sizes'])}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    bundle = dict(
        n_hap=int(gm.shape[0]), n_snp=int(gm.shape[1]), n_ind=int(n_ind),
        pos0=float(pos[0]), pos1=float(pos[-1]), model=args.model, mu=args.mu, Ne_est=Ne_est,
        infer_secs=secs, inv_chrom=args.inv_chrom,
        pos_left=pred["pos_left"], pos_right=pred["pos_right"], rho_per_bp=pred["rho_per_bp"],
        r_per_bp=pred["r_per_bp"], r_ci_lo=pred.get("r_ci_lo"), r_ci_hi=pred.get("r_ci_hi"),
        sigma_log_rho=sig,
        starts_1m=s1, r_1m=r1, rho_1m=rho1, starts_100k=s100, r_100k=r100, rho_100k=rho100,
        starts_50k=s50, r_50k=r50, rho_50k=rho50,
        repro=rep["repro"], repro_half1=rep["half1"], repro_half2=rep["half2"],
        pp_gof=pp["gof"], pp_rmse=pp["rmse"], pp_obs=pp["obs"], pp_sim=pp["sim"], pp_bins=pp["bins"],
        **{f"calib_{k}": v for k, v in calib.items()},
    )
    if inv is not None:
        bundle.update({f"inv_{k}": v for k, v in inv.items()})
        bundle["inv_start"] = int(args.inv_start); bundle["inv_end"] = int(args.inv_end)
    if pca is not None:
        bundle.update(pca_pc1=pca["pc1"], pca_pc2=pca["pc2"], pca_labels=pca["labels"],
                      pca_ev=pca["ev"], kary_sizes=pca["sizes"])
    np.savez(args.out, **{k: v for k, v in bundle.items() if v is not None})

    summary = dict(n_hap=bundle["n_hap"], n_snp=bundle["n_snp"], n_ind=int(n_ind), Ne_est=Ne_est,
                   infer_secs=secs, repro=rep["repro"], pp_gof=pp["gof"], **calib)
    if inv is not None:
        summary.update({f"inv_{k}": inv[k] for k in ("ratio", "mwu_p", "mean_inside", "mean_flank")})
    if pca is not None:
        summary["kary_sizes"] = [int(x) for x in pca["sizes"]]
    json.dump(summary, open(args.out.replace(".npz", ".json"), "w"), indent=2)
    print(f"[done] -> {args.out}")


if __name__ == "__main__":
    main()
