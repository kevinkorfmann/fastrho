"""Real-data zoom for the linked-selection figure (Fig 4): a canonical human locus.

Extracts 1000G *high-coverage GRCh38* CEU haplotypes over a locus window, runs the frozen
fastrho base model, measures nucleotide diversity, and compares the recovered rate to the
**deCODE pedigree map** (Kong et al. 2010) -- a direct meiotic-crossover map, immune to the
selection distortion that biases every LD-based estimator (fastrho and pyrho alike). This is
the real-data counterpart to the SLiM stress test: at a real sweep, does pyrho's rate distort
away from the pedigree truth more than fastrho's?

Subcommands (mind the venv on sesame):
  extract <key> <chrom> <start> <end> <pop>   [agam venv -- has pysam]
      -> /home/kkor/realdata/hap/<key>.npz  (gm, pos, per-window pi, deCODE truth metadata)
  fastrho <key>                               [fastrho venv]
      -> /home/kkor/realdata/zoom/<key>.npz  (fine-window fastrho + deCODE truth + pi)

pyrho is produced by the existing tested drivers, then binned to the same fine grid:
  fastrho-venv  pyrho_rd.py <key> setup
  pyrho-venv    run_pyrho_config.py /home/kkor/realdata/pyrho/<key>
  fastrho-venv  grch38_zoom.py pyrho <key>    -> adds a 'pyrho' array to zoom/<key>.npz
"""
import os
import sys
import glob

import numpy as np

HAP = "/home/kkor/realdata/hap"
ZOOM = "/home/kkor/realdata/zoom"
PYRHO = "/home/kkor/realdata/pyrho"
WFINE = 25_000                      # fine window for the zoom track (deCODE res is ~10 kb)
MU = 1.29e-8                        # human per-bp mutation rate (matches the rest of the paper)
MAP_ID = "DeCodeSexAveraged_GRCh38"   # deCODE pedigree map (Kong 2010), selection-immune truth
DEV = "cuda:0"

HICOV = ("http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/"
         "working/20220422_3202_phased_SNV_INDEL_SV/"
         "1kGP_high_coverage_Illumina.{c}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz")
# The curated 2504-UNRELATED panel (phase3), used to pick individuals so the sample matches the
# rest of the paper's human analysis and excludes the trio children in the 3202 high-coverage set
# (related haplotypes would inflate LD and bias any recombination estimator).
PANEL = ("https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
         "integrated_call_samples_v3.20130502.ALL.panel")


# ---------------------------------------------------------------- diversity + binning helpers
def pi_windows(gm, pos, edges):
    """Per-bp nucleotide diversity pi in each [edge_i, edge_{i+1}) window.

    Unbiased per-site heterozygosity 2*k*(n-k)/(n*(n-1)) summed over segregating sites and
    divided by the window length in bp.
    """
    n = gm.shape[0]
    k = gm.sum(0).astype(np.float64)
    het = 2.0 * k * (n - k) / (n * (n - 1))          # per-site expected heterozygosity
    idx = np.digitize(pos, edges) - 1
    out = np.full(len(edges) - 1, np.nan)
    for w in range(len(edges) - 1):
        L = edges[w + 1] - edges[w]
        m = idx == w
        out[w] = het[m].sum() / L if L > 0 else np.nan
    return out


def deCODE_windows(chrom, edges):
    import stdpopsim
    from fastrho.preprocess import mean_rate_between
    sp = stdpopsim.get_species("HomSap")
    gm = sp.get_genetic_map(MAP_ID)
    rm = gm.get_chromosome_map(chrom.replace("chr", ""))
    p = np.asarray(rm.position, float)
    r = np.asarray(rm.rate, float)
    r = np.where(np.isfinite(r), r, 0.0)
    return mean_rate_between(p, r, edges)


def fine_edges(pos):
    lo, hi = int(pos[0]), int(pos[-1])
    return np.append(np.arange(lo, hi, WFINE), hi)


# ---------------------------------------------------------------- subcommands
def extract(key, chrom, start, end, pop):
    import ssl
    import urllib.request
    import certifi
    import pysam
    os.environ["CURL_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
    start, end = int(start), int(end)
    ctx = ssl.create_default_context(cafile=certifi.where())
    rows = urllib.request.urlopen(PANEL, context=ctx).read().decode().splitlines()[1:]
    want = {r.split()[0] for r in rows if len(r.split()) > 1 and r.split()[1] == pop}
    vf = pysam.VariantFile(HICOV.format(c=chrom))
    samples = [s for s in vf.header.samples if s in want]
    print(f"{key}: {len(samples)} {pop} samples over {chrom}:{start}-{end}", flush=True)
    cols, gpos = [], []
    n = 0
    for rec in vf.fetch(chrom, start, end):
        n += 1
        if rec.alts is None or len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue
        row = np.zeros(2 * len(samples), np.int8)
        gts = rec.samples
        for k, s in enumerate(samples):
            a = gts[s].get("GT", (None, None))
            row[2 * k] = 1 if a[0] else 0
            row[2 * k + 1] = 1 if (len(a) > 1 and a[1]) else 0
        m = row.sum()
        if 0 < m < len(row):
            cols.append(row)
            gpos.append(rec.pos)
        if n % 200000 == 0:
            print(f"  scanned {n}, kept {len(cols)}", flush=True)
    H = np.array(cols, np.int8).T
    gpos = np.asarray(gpos, np.int64)
    edges = np.append(np.arange(int(gpos[0]), int(gpos[-1]), WFINE), int(gpos[-1]))
    pi = pi_windows(H, gpos, edges)
    os.makedirs(HAP, exist_ok=True)
    np.savez(f"{HAP}/{key}.npz", gm=H, pos=gpos, chrom=chrom, n_ind=len(samples),
             mode="phased2", map_sp="HomSap", map_id=MAP_ID, mu=MU, model="base", pop=pop,
             pi=pi, pi_centers=(edges[:-1] + WFINE / 2) / 1e6)
    print(f"{key}: {H.shape[0]} hap x {H.shape[1]} SNPs -> {HAP}/{key}.npz "
          f"(pi min {np.nanmin(pi)*1e4:.2f} .. max {np.nanmax(pi)*1e4:.2f} x1e-4)", flush=True)


def fastrho(key):
    sys.path.insert(0, "/home/kkor/fastrho")
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_map_from_genotype_matrix
    from fastrho.preprocess import mean_rate_between
    z = np.load(f"{HAP}/{key}.npz", allow_pickle=True)
    gm = z["gm"]
    pos = z["pos"].astype(np.float64)
    chrom = str(z["chrom"])
    ckpt = "/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt"
    stats_p = "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    pred = predict_map_from_genotype_matrix(gm, pos, model, cfg, stats,
                                            mutation_rate=MU, Ne=None, device=DEV)
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    edges = fine_edges(pos)
    fr = mean_rate_between(bp, pred["r_per_bp"], edges)
    truth = deCODE_windows(chrom, edges)
    centers = (edges[:-1] + WFINE / 2) / 1e6
    ok = np.isfinite(fr) & np.isfinite(truth) & (fr > 0) & (truth > 0)
    r_lin = pearsonr(fr[ok], truth[ok])[0]
    r_log = pearsonr(np.log(fr[ok]), np.log(truth[ok]))[0]
    os.makedirs(ZOOM, exist_ok=True)
    np.savez(f"{ZOOM}/{key}.npz", centers=centers, edges=edges, truth=truth, fastrho=fr,
             pi=z["pi"], pi_centers=z["pi_centers"], chrom=chrom, ok=ok,
             fastrho_r=r_lin, fastrho_logr=r_log, Ne_est=float(pred["Ne_estimated"]),
             n_hap=gm.shape[0], n_snp=gm.shape[1])
    print(f"{key}: fastrho vs deCODE  Pearson={r_lin:.3f} logPearson={r_log:.3f}  "
          f"Ne_est={pred['Ne_estimated']:.0f}  windows={int(ok.sum())}", flush=True)


def fastrho2(key, ckpt, stats_p, rich):
    """Run an ARBITRARY checkpoint (e.g. a rich-featurizer prototype) on a real locus and
    report Pearson vs deCODE at 25 kb and 100 kb -- the real-data transfer test for the fix."""
    sys.path.insert(0, "/home/kkor/fastrho")
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_intervals
    from fastrho.features import SNPTokenFeaturizer, FeatureConfig
    from fastrho.preprocess import mean_rate_between
    from fastcxt.sfs import basic_filtering
    rich = bool(int(rich))
    z = np.load(f"{HAP}/{key}.npz", allow_pickle=True)
    gm = z["gm"]
    pos = z["pos"].astype(np.float64)
    chrom = str(z["chrom"])
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    gmf, posf = basic_filtering(gm.astype(np.int8), pos)
    fz = SNPTokenFeaturizer(FeatureConfig(rich_ld=rich))
    pred = predict_intervals(model, cfg, stats, gmf, posf, MU, Ne=None, device=DEV, featurizer=fz)
    bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
    edges = fine_edges(pos)
    fr = mean_rate_between(bp, pred["r_per_bp"], edges)
    truth = deCODE_windows(chrom, edges)

    def rc(f):
        n = (len(fr) // f) * f
        a = np.nanmean(fr[:n].reshape(-1, f), axis=1)
        t = np.nanmean(truth[:n].reshape(-1, f), axis=1)
        ok = np.isfinite(a) & np.isfinite(t) & (a > 0) & (t > 0)
        return pearsonr(a[ok], t[ok])[0]
    if len(sys.argv) > 6 and sys.argv[6] == "save":
        tag = "rich" if rich else "plain"
        np.savez(f"{ZOOM}/{key}_{tag}.npz", centers=(edges[:-1] + WFINE / 2) / 1e6,
                 rate=fr, truth=truth, chrom=chrom)
    print(f"{key}: rich={rich} 25kb={rc(1):.3f} 100kb={rc(4):.3f}", flush=True)


def pyrho_score(key):
    """Score pyrho vs deCODE at 100 kb for a locus directly from its region_*.rmap + hap
    positions (no dependency on a fastrho zoom npz). Prints 100 kb Pearson."""
    sys.path.insert(0, "/home/kkor/fastrho")
    import glob
    from scipy.stats import pearsonr
    from fastrho.preprocess import mean_rate_between
    z = np.load(f"{HAP}/{key}.npz", allow_pickle=True)
    pos = z["pos"].astype(np.float64)
    chrom = str(z["chrom"])
    starts, ends, rates = [], [], []
    for rm in sorted(glob.glob(f"{PYRHO}/{key}/region_*.rmap")):
        for ln in open(rm):
            p = ln.split()
            if len(p) < 3 or not p[0][0].isdigit():
                continue
            starts.append(float(p[0])); ends.append(float(p[1])); rates.append(float(p[2]))
    if not starts:
        print(f"{key}: pyrho NO output"); return
    bp = np.array(starts + [ends[-1]])
    W100 = 100_000
    edges = np.append(np.arange(int(pos[0]), int(pos[-1]), W100), int(pos[-1]))
    py = mean_rate_between(bp, np.array(rates), edges)
    truth = deCODE_windows(chrom, edges)
    ok = np.isfinite(py) & np.isfinite(truth) & (py > 0) & (truth > 0)
    print(f"{key}: pyrho 100kb={pearsonr(py[ok], truth[ok])[0]:.3f}", flush=True)


def select(ckpt_dir, stats_p, rich, *keys):
    """Pick the best-generalizing epoch: eval every checkpoint in ckpt_dir on held-out REAL
    neutral human regions vs deCODE (100 kb), report per-epoch mean, print the best.
    Model loaded once per checkpoint; val data preloaded once."""
    sys.path.insert(0, "/home/kkor/fastrho")
    import glob
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_intervals
    from fastrho.features import SNPTokenFeaturizer, FeatureConfig
    from fastrho.preprocess import mean_rate_between
    from fastcxt.sfs import basic_filtering
    rich = bool(int(rich))
    fz = SNPTokenFeaturizer(FeatureConfig(rich_ld=rich))
    data = {}
    for k in keys:
        z = np.load(f"{HAP}/{k}.npz", allow_pickle=True)
        gmf, posf = basic_filtering(z["gm"].astype(np.int8), z["pos"].astype(np.float64))
        edges = fine_edges(z["pos"].astype(np.float64))
        data[k] = (gmf, posf, edges, deCODE_windows(str(z["chrom"]), edges))
    best = (-2.0, None)
    for ck in sorted(glob.glob(f"{ckpt_dir}/*.ckpt")):
        model, cfg, stats = load_model(ck, stats_p, device=DEV)
        rs = []
        for gmf, posf, edges, truth in data.values():
            pred = predict_intervals(model, cfg, stats, gmf, posf, MU, Ne=None, device=DEV, featurizer=fz)
            bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
            fr = mean_rate_between(bp, pred["r_per_bp"], edges)
            f = 4
            n = (len(fr) // f) * f
            a = np.nanmean(fr[:n].reshape(-1, f), 1)
            t = np.nanmean(truth[:n].reshape(-1, f), 1)
            ok = np.isfinite(a) & np.isfinite(t) & (a > 0) & (t > 0)
            rs.append(pearsonr(a[ok], t[ok])[0])
        m = float(np.mean(rs))
        print(f"{ck.split('/')[-1]}: mean_real_val_r={m:.3f} per-key={[round(x, 2) for x in rs]}", flush=True)
        if m > best[0]:
            best = (m, ck)
    print(f"BEST_CKPT {best[1]} mean_r={best[0]:.3f}", flush=True)


def ensemble(key, n_sub, K):
    """Subsample-ensemble fastrho: run the model on K random n_sub-haplotype subsamples and
    average the per-window maps. fastrho is strongest at small n (its features saturate as n
    grows), so each subsample runs in-distribution; averaging aggregates information across all
    haplotypes -- a no-retraining route to large-n accuracy. Adds fastrho_ens* to zoom/<key>.npz.
    """
    sys.path.insert(0, "/home/kkor/fastrho")
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_map_from_genotype_matrix
    from fastrho.preprocess import mean_rate_between
    n_sub, K = int(n_sub), int(K)
    z = np.load(f"{HAP}/{key}.npz", allow_pickle=True)
    gm = z["gm"]
    pos = z["pos"].astype(np.float64)
    chrom = str(z["chrom"])
    nhap = gm.shape[0]
    ckpt = "/home/kkor/fastrho_data/campaign/train15k/fastrho/version_0/checkpoints/epoch=45-val_loss=-0.019.ckpt"
    stats_p = "/home/kkor/fastrho_data/campaign/shards15k/feat_stats.npz"
    model, cfg, stats = load_model(ckpt, stats_p, device=DEV)
    zoom = dict(np.load(f"{ZOOM}/{key}.npz", allow_pickle=True))
    edges = zoom["edges"]
    acc = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)
    rng = np.random.default_rng(2026)
    nes = []
    for j in range(K):
        idx = np.sort(rng.choice(nhap, n_sub, replace=False))
        sub = gm[idx]
        seg = (sub.sum(0) > 0) & (sub.sum(0) < n_sub)      # segregating within the subsample
        sub = sub[:, seg]
        spos = pos[seg]
        pred = predict_map_from_genotype_matrix(sub, spos, model, cfg, stats,
                                                mutation_rate=MU, Ne=None, device=DEV)
        bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
        w = mean_rate_between(bp, pred["r_per_bp"], edges)
        m = np.isfinite(w)
        acc[m] += w[m]
        cnt[m] += 1
        nes.append(float(pred["Ne_estimated"]))
    ens = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    truth = zoom["truth"]
    ok = np.isfinite(ens) & np.isfinite(truth) & (ens > 0) & (truth > 0)
    r_lin = pearsonr(ens[ok], truth[ok])[0]
    r_log = pearsonr(np.log(ens[ok]), np.log(truth[ok]))[0]
    zoom[f"fastrho_ens_{n_sub}x{K}"] = ens
    zoom[f"fastrho_ens_{n_sub}x{K}_r"] = r_lin
    np.savez(f"{ZOOM}/{key}.npz", **zoom)
    print(f"{key}: ENSEMBLE n_sub={n_sub} K={K}  Pearson={r_lin:.3f} logPearson={r_log:.3f}  "
          f"(single-n{nhap} fastrho={float(zoom['fastrho_r']):.3f}, pyrho={float(zoom.get('pyrho_r', np.nan)):.3f}) "
          f"Ne~{np.mean(nes):.0f}", flush=True)


def pyrho(key):
    """Bin the pyrho region_*.rmap (already optimized) onto the same fine grid and correlate."""
    sys.path.insert(0, "/home/kkor/fastrho")
    from scipy.stats import pearsonr
    from fastrho.preprocess import mean_rate_between
    d = f"{PYRHO}/{key}"
    starts, ends, rates = [], [], []
    for rm in sorted(glob.glob(f"{d}/region_*.rmap")):
        for ln in open(rm):
            p = ln.split()
            if len(p) < 3 or not p[0][0].isdigit():
                continue
            starts.append(float(p[0]))
            ends.append(float(p[1]))
            rates.append(float(p[2]))
    if not starts:
        print(f"{key}: NO pyrho output")
        return
    bp = np.array(starts + [ends[-1]])
    rate = np.array(rates)
    z = dict(np.load(f"{ZOOM}/{key}.npz", allow_pickle=True))
    edges = z["edges"]
    py = mean_rate_between(bp, rate, edges)
    z["pyrho"] = py
    truth = z["truth"]
    ok = np.isfinite(py) & np.isfinite(truth) & (py > 0) & (truth > 0)
    z["pyrho_r"] = pearsonr(py[ok], truth[ok])[0]
    z["pyrho_logr"] = pearsonr(np.log(py[ok]), np.log(truth[ok]))[0]
    np.savez(f"{ZOOM}/{key}.npz", **z)
    print(f"{key}: pyrho vs deCODE  Pearson={z['pyrho_r']:.3f} logPearson={z['pyrho_logr']:.3f} "
          f"| fastrho={float(z['fastrho_r']):.3f}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "extract":
        extract(*sys.argv[2:7])
    elif cmd == "fastrho":
        fastrho(sys.argv[2])
    elif cmd == "pyrho":
        pyrho(sys.argv[2])
    elif cmd == "ensemble":
        ensemble(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "fastrho2":
        fastrho2(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    elif cmd == "select":
        select(*sys.argv[2:])
    elif cmd == "pyrho_score":
        pyrho_score(sys.argv[2])
    else:
        raise SystemExit(f"unknown cmd {cmd}")
