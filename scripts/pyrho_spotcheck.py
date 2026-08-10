"""Main-benchmark spot-check: pyrho 25 kb accuracy at the tuned block penalty vs the original bpen=50.

Compares, on const_n20 and real_decode, the pooled 25 kb Pearson of fastrho (reference), the original
pyrho (pred_pyrho.bpen50.npz backup), and the current tuned pyrho (pred_pyrho.npz) against the truth --
so we can tell whether Table 1's main-benchmark numbers are robust to the block penalty. Run in the
fastrho venv.
"""
import os
import glob

import numpy as np
from scipy.stats import pearsonr

import sys
sys.path.insert(0, "/home/kkor/fastrho")
from fastrho.preprocess import mean_rate_between

L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)
BASE = "/home/kkor/fastrho_data/campaign/configs"
PUB = {"const_n20": (0.88, 0.80), "real_decode": (0.89, 0.81)}  # published (fastrho, pyrho) @25kb


def pool(D, pred, fp, names):
    T, P = [], []
    for n in names:
        if n in pred.files and n in fp.files:
            z = np.load(os.path.join(D, n + ".npz"), allow_pickle=True)
            tr = mean_rate_between(z["map_position"], z["map_rate"], EDGES)
            pr = pred[n]
            k = min(len(tr), len(pr))
            T.append(tr[:k]); P.append(pr[:k])
    tt = np.concatenate(T)
    pp = np.concatenate(P)
    m = np.isfinite(tt) & np.isfinite(pp)
    return pearsonr(tt[m], pp[m])[0], len(T)


def main():
    for c in ["const_n20", "real_decode"]:
        D = os.path.join(BASE, c)
        fp = np.load(os.path.join(D, "pred_fastrho.npz"))
        tuned = np.load(os.path.join(D, "pred_pyrho.npz"))
        orig = np.load(os.path.join(D, "pred_pyrho.bpen50.npz"))
        names = [os.path.basename(v)[:-4] for v in sorted(glob.glob(os.path.join(D, "region_*.vcf")))]
        fr, nreg = pool(D, fp, fp, names)
        po, _ = pool(D, orig, fp, names)
        pt, _ = pool(D, tuned, fp, names)
        pf, pp_pub = PUB[c]
        print("%-12s (n=%d): fastrho=%.3f (pub %.2f)  pyrho_orig(bpen50)=%.3f (pub %.2f)  "
              "pyrho_tuned=%.3f" % (c, nreg, fr, pf, po, pp_pub, pt))


if __name__ == "__main__":
    main()
