"""Real pyrho UNPHASED (--ploidy 2) head-to-head on the SAME canid genotypes fastrho used.

Refutes the (false) claim that "pyrho cannot run on unphased data". pyrho (Spence & Song 2019,
MBE) explicitly supports genotype/unphased data via `optimize --ploidy 2` against a lookup table
built for the haplotype sample size; the authors state genotype-based inference is "indistinguishable
from that on perfectly phased data" and recommend it when phasing may be inaccurate. Here we give
pyrho the EXACT same unphased genotypes fastrho reads (diploid dosages reconstructed from the
phase-invariant pseudo-haplotype encoding), a Watterson-Ne constant-size lookup table (the standard
demography-free pyrho recipe, matched to the data's own diversity), and score the recovered map
against the same published map on the same 100-kb windows fastrho was scored on.

This is a fair head-to-head: identical genotypes, identical windows, identical truth map. It fills
the two "---" cells in paper/tables/pyrho_headtohead.tex (wolf / dog, Plassais vs Campbell).

Run on sesame:
  PYTHONNOUSERSITE=1 /home/kkor/venvs/fastrho/bin/python scripts/pyrho_unphased.py wolf_clean [key ...]
Writes /home/kkor/realdata/pyrho_unphased.json
"""
import os, sys, json, tempfile, subprocess
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stdpopsim
from fastrho.preprocess import mean_rate_between

HAP = "/home/kkor/realdata/hap"
MAPS = "/home/kkor/realdata/maps"
PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
W = 100000
OUT = "/home/kkor/realdata/pyrho_unphased.json"


def watterson_ne(n_hap, n_seg, span_bp, mu):
    a = sum(1.0 / i for i in range(1, n_hap))
    return max(n_seg / (a * span_bp) / (4.0 * mu), 1e4)


def dosage_from_pseudohap(gm):
    """The DR unphased encoding stores each diploid as a pseudo-haplotype pair
    (row 2k = dosage>=1 indicator, row 2k+1 = dosage==2 indicator). The true
    diploid dosage in {0,1,2} is their sum."""
    I1 = gm[0::2].astype(np.int64)
    I2 = gm[1::2].astype(np.int64)
    return I1 + I2                       # (n_ind, n_sites), values in {0,1,2}


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


def window_mean_centers(lo, hi, rate, centers_bp):
    """Mean pyrho rate in each 100-kb window centred at centers_bp (matches
    fastrho's stored windows index-for-index)."""
    out = np.full(len(centers_bp), np.nan)
    for k, c in enumerate(centers_bp):
        a, b = c - W / 2, c + W / 2
        ov = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
        tot = ov.sum()
        if tot > 0:
            out[k] = float((rate * ov).sum() / tot)
    return out


def run_pyrho_unphased(dos, pos, chrom, ne, mu, n_hap, td, ploidy=2):
    vcf = os.path.join(td, "g.vcf")
    write_unphased_vcf(dos, pos, vcf, chrom)
    table = os.path.join(td, "t.hdf")
    subprocess.run([PYRHO, "make_table", "-n", str(n_hap), "-m", str(mu),
                    "-p", str(ne), "--approx", "-N", str(n_hap + 5),
                    "--numthreads", "30", "-o", table], check=True,
                   capture_output=True, text=True)
    rmap = os.path.join(td, "g.rmap")
    # same hyperparameters as the phased benchmark (-w 50 -bpen 25); at 100 kb the choice is immaterial.
    r = subprocess.run([PYRHO, "optimize", "--vcffile", vcf, "--tablefile", table,
                        "--ploidy", str(ploidy), "-w", "50", "-bpen", "25",
                        "--numthreads", "16", "-o", rmap], capture_output=True, text=True)
    if not os.path.exists(rmap):
        raise RuntimeError("pyrho optimize failed:\n" + r.stderr[-2000:])
    return np.loadtxt(rmap, ndmin=2)


def truth_windows(map_sp, map_id, chrom, centers_bp):
    sp = stdpopsim.get_species(map_sp)
    gm = sp.get_genetic_map(map_id)
    rm = gm.get_chromosome_map(chrom.replace("chr", ""))
    pos = np.asarray(rm.position, float); rate = np.asarray(rm.rate, float)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    edges = np.append(centers_bp - W / 2, centers_bp[-1] + W / 2)
    return mean_rate_between(pos, rate, edges)


# label -> (hap genotype npz, fastrho scored-map npz whose windows/truth we match).
# The scored map is the canid specialist run that the head-to-head table reports; its 100-kb
# windows and Campbell truth are identical to the base run (same genotypes -> same edges), so
# pyrho is scored against exactly the same truth fastrho was.
JOBS = {
    "wolf": (f"{HAP}/wolf_clean.npz",            f"{MAPS}/wolf_canid.npz"),
    "dog":  ("/home/kkor/realdata/hap_specialist/dog.npz", f"{MAPS}/dog_canid.npz"),
}


def one(label):
    hap_path, map_path = JOBS[label]
    z = np.load(hap_path, allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(np.float64); chrom = str(z["chrom"])
    map_id = str(z["map_id"]); mu = float(z["mu"]); n_ind = int(z["n_ind"])
    dos = dosage_from_pseudohap(gm)
    n_hap = 2 * n_ind
    m = np.load(map_path, allow_pickle=True)
    centers_bp = np.asarray(m["centers"], float) * 1e6
    fastrho_pred = np.asarray(m["pred"], float)
    truth = np.asarray(m["truth"], float)
    ne = watterson_ne(n_hap, dos.shape[1], float(pos[-1] - pos[0]), mu)
    with tempfile.TemporaryDirectory() as td:
        rows = run_pyrho_unphased(dos, pos, chrom, ne, mu, n_hap, td)
    pyrho_w = window_mean_centers(rows[:, -3], rows[:, -2], rows[:, -1], centers_bp)
    kk = min(len(pyrho_w), len(truth), len(fastrho_pred))
    pw, tw, fw = pyrho_w[:kk], truth[:kk], fastrho_pred[:kk]
    okp = np.isfinite(pw) & np.isfinite(tw) & (pw > 0) & (tw > 0)
    okf = np.isfinite(fw) & np.isfinite(tw) & (fw > 0) & (tw > 0)
    pyrho_r = float(pearsonr(pw[okp], tw[okp])[0])
    fastrho_r = float(pearsonr(fw[okf], tw[okf])[0])
    res = dict(label=label, chrom=chrom, map_id=map_id, n_ind=n_ind, n_hap=n_hap,
               n_seg=int(dos.shape[1]), Ne_watterson=float(ne), ploidy=2,
               n_windows_pyrho=int(okp.sum()), n_windows_fastrho=int(okf.sum()),
               pyrho_pearson=pyrho_r, fastrho_pearson=fastrho_r,
               fastrho_stored=float(m["pearson"]))
    print(json.dumps(res), flush=True)
    return res


def main():
    labels = sys.argv[1:] or ["wolf", "dog"]
    out = [one(k) for k in labels]
    json.dump(out, open(OUT, "w"), indent=2)
    print("wrote", OUT, "->", [r["label"] for r in out])


if __name__ == "__main__":
    main()
