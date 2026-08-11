"""Run pyrho (genotype mode, --ploidy 2) on ONE transect species for the tree-of-life figure's
pyrho overlay + the fastrho-vs-pyrho concordance metric.

For each species the transect reports a fastrho map from unphased genotypes; this script runs the
composite-likelihood gold standard (pyrho) on the same genotypes so the figure can show, as a thin red
overlay, that an independent LD method recovers a concordant landscape -- and quantify the agreement.
pyrho needs a per-dataset lookup table, so we cap the sample size (make_table cost grows steeply with n)
and the SNP count for tractability, build a Watterson-Ne constant-size table, run optimize --ploidy 2 on
the reconstructed diploid genotypes, and window the result onto the committed fastrho track centres.
The reported concordance is the Pearson r between the fastrho track and the pyrho map at 100 kb.

Run on sesame (fastrho venv; shells out to the pyrho binary):
  PYTHONNOUSERSITE=1 /home/kkor/venvs/fastrho/bin/python scripts/transect_pyrho.py <key> [n_cap] [snp_cap]
Writes /home/kkor/realdata/transect_pyrho/<key>.json
"""
import os, sys, json, tempfile, subprocess
import numpy as np
from scipy.stats import pearsonr

PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
W = 100000
HAPDIRS = ["/home/kkor/realdata/hap", "/home/kkor/realdata/hap_specialist"]
TRACKS = "/home/kkor/realdata/transect_tracks.json"
OUTDIR = "/home/kkor/realdata/transect_pyrho"


def find_hap(key):
    for d in HAPDIRS:
        p = os.path.join(d, key + ".npz")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(key)


def dosage_from_pseudohap(gm):
    """(dosage>=1, dosage==2) pseudo-hap pair per diploid -> true dosage in {0,1,2}."""
    return gm[0::2].astype(np.int64) + gm[1::2].astype(np.int64)


def watterson_ne(n_hap, n_seg, span_bp, mu):
    a = sum(1.0 / i for i in range(1, n_hap))
    return max(n_seg / (a * span_bp) / (4.0 * mu), 1e4)


def write_unphased_vcf(dos, pos, path, chrom):
    n_ind = dos.shape[0]
    gt = {0: "0/0", 1: "0/1", 2: "1/1"}
    last = -1
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=%s>\n" % chrom)
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join("s%d" % j for j in range(n_ind)) + "\n")
        for s in range(dos.shape[1]):
            p = int(pos[s]); p = last + 1 if p <= last else p; last = p
            fh.write("%s\t%d\t.\tA\tT\t.\tPASS\t.\tGT\t%s\n"
                     % (chrom, p, "\t".join(gt[int(d)] for d in dos[:, s])))


def window_centers(lo, hi, rate, centers_bp):
    out = np.full(len(centers_bp), np.nan)
    for k, c in enumerate(centers_bp):
        a, b = c - W / 2, c + W / 2
        ov = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
        tot = ov.sum()
        if tot > 0:
            out[k] = float((rate * ov).sum() / tot)
    return out


def run_pyrho(dos, pos, chrom, ne, mu, n_hap, td, threads=12):
    vcf = os.path.join(td, "g.vcf"); write_unphased_vcf(dos, pos, vcf, chrom)
    table = os.path.join(td, "t.hdf")
    subprocess.run([PYRHO, "make_table", "-n", str(n_hap), "-m", str(mu), "-p", str(ne),
                    "--approx", "-N", str(n_hap + 5), "--numthreads", str(threads), "-o", table],
                   check=True, capture_output=True, text=True)
    rmap = os.path.join(td, "g.rmap")
    r = subprocess.run([PYRHO, "optimize", "--vcffile", vcf, "--tablefile", table, "--ploidy", "2",
                        "-w", "50", "-bpen", "25", "--numthreads", str(threads), "-o", rmap],
                       capture_output=True, text=True)
    if not os.path.exists(rmap):
        raise RuntimeError("pyrho optimize failed:\n" + r.stderr[-1500:])
    return np.loadtxt(rmap, ndmin=2)


def main():
    key = sys.argv[1]
    n_cap = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    snp_cap = int(sys.argv[3]) if len(sys.argv) > 3 else 120000
    hap_override = sys.argv[4] if len(sys.argv) > 4 else None   # explicit genotype npz (provenance fix)
    track = json.load(open(TRACKS))[key]
    centers_bp = np.asarray(track["centers"], float) * 1e6
    fastrho_pred = np.asarray(track["pred"], float)

    z = np.load(hap_override or find_hap(key), allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); chrom = str(z["chrom"])
    mu = float(z["mu"]); n_ind = int(z["n_ind"])
    dos = dosage_from_pseudohap(gm)                      # (n_ind, n_sites)
    rng = np.random.default_rng(7)
    if n_ind > n_cap:
        idx = np.sort(rng.choice(n_ind, n_cap, replace=False))
        dos = dos[idx]; n_ind = n_cap
    ac = dos.sum(0); seg = (ac > 0) & (ac < 2 * n_ind)   # keep segregating after subsample
    dos = dos[:, seg]; pos = pos[seg]
    if dos.shape[1] > snp_cap:
        keep = np.sort(rng.choice(dos.shape[1], snp_cap, replace=False))
        dos = dos[:, keep]; pos = pos[keep]
    n_hap = 2 * n_ind
    ne = watterson_ne(n_hap, dos.shape[1], float(pos[-1] - pos[0]), mu)
    with tempfile.TemporaryDirectory() as td:
        rows = run_pyrho(dos, pos, chrom, ne, mu, n_hap, td)
    pyrho_w = window_centers(rows[:, -3], rows[:, -2], rows[:, -1], centers_bp)
    kk = min(len(pyrho_w), len(fastrho_pred))
    pw, fw = pyrho_w[:kk], fastrho_pred[:kk]
    ok = np.isfinite(pw) & np.isfinite(fw) & (pw > 0) & (fw > 0)
    conc = float(pearsonr(pw[ok], fw[ok])[0]) if ok.sum() > 5 else None
    out = dict(key=key, chrom=chrom, n_hap_pyrho=int(n_hap), n_seg=int(dos.shape[1]),
               concordance_r=conc, n_windows=int(ok.sum()),
               pyrho=[None if not np.isfinite(v) else round(float(v), 12) for v in pw])
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump(out, open(os.path.join(OUTDIR, key + ".json"), "w"))
    print(f"{key}: concordance r={conc if conc is None else round(conc,3)} "
          f"n_hap={n_hap} n_seg={dos.shape[1]} windows={int(ok.sum())}", flush=True)


if __name__ == "__main__":
    main()
