"""Score a dir of pyrho region_*.rmap against the Salome/TAIR10 chr map. fastrho venv.
Usage: score_rmaps.py <dir> <chrom> <hap.npz>  -> prints Pearson/Spearman vs truth at 100 kb."""
import sys, glob
import numpy as np
sys.path.insert(0, "/home/kkor/fastrho")
from scipy.stats import pearsonr, spearmanr
import stdpopsim
from fastrho.preprocess import mean_rate_between

d, chrom, hapnpz = sys.argv[1], sys.argv[2], sys.argv[3]
W = 100000
pos = np.load(hapnpz, allow_pickle=True)["pos"].astype(np.int64)
starts, ends, rates = [], [], []
for rm in sorted(glob.glob(d + "/region_*.rmap")):
    for ln in open(rm):
        p = ln.split()
        if len(p) < 3 or not p[0][0].isdigit():
            continue
        starts.append(float(p[0])); ends.append(float(p[1])); rates.append(float(p[2]))
if not starts:
    print("NO pyrho output"); sys.exit(1)
bp = np.array(starts + [ends[-1]]); rate = np.array(rates)
lo, hi = int(pos[0]), int(pos[-1]); edges = np.append(np.arange(lo, hi, W), hi)
pred = mean_rate_between(bp, rate, edges)
sp = stdpopsim.get_species("AraTha"); gm = sp.get_genetic_map("SalomeAveraged_TAIR10")
rmc = gm.get_chromosome_map(chrom)
tp = np.asarray(rmc.position, float); tr = np.where(np.isfinite(rmc.rate), rmc.rate, 0.0)
true = mean_rate_between(tp, tr, edges)
k = min(len(pred), len(true)); p, t = pred[:k], true[:k]
ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
print("Pearson=%.3f Spearman=%.3f nwin=%d" %
      (pearsonr(p[ok], t[ok])[0], spearmanr(p[ok], t[ok])[0], int(ok.sum())))
