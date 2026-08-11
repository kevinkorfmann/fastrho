"""Regenerate paper/figdata/highn.npz for the STANDALONE large-$n$ figure.

fastrho's OWN amortized composite likelihood (a fused-LASSO readout on ONE fixed,
demography-robust two-locus table, data-derived Ne, NO pyrho binary, NO per-dataset
table) MATCHES pyrho at high-$n$ neutral, while fastrho's SSM dominates the sweep.
This replaces the earlier fastrho x pyrho *ensemble* figdata: the high-n win is now
STANDALONE (no pyrho at inference).

The 12 per-locus 100 kb Pearsons vs the deCODE pedigree map (n=198 CEU/GRCh38) are
recomputed here from the saved per-locus fine maps in
``paper/figdata/clmap_lasso_maps.npz`` (rsynced from sesame:/home/kkor/large_n/;
rows per key = [LASSO(=CL-map), rich(=SSM), pyrho, deCODE], 240 fine windows each),
using the same rc(...,f=4) 100 kb-averaging idiom as /home/kkor/large_n/w1_blend.py
and /home/kkor/large_n/pigate_refine.py.  The sim sample-size crossover and the base
SSM per-locus Pearsons are carried as constants (sim / base-model outputs previously
computed on sesame; not present in the maps file).

Run:  PYTHONPATH=scripts python3.13 scripts/highn_standalone_figdata.py
"""
import os
import numpy as np
from scipy.stats import pearsonr

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.join(HERE, "..", "paper", "figdata")
MAPS = os.path.join(FD, "clmap_lasso_maps.npz")
OUT = os.path.join(FD, "highn.npz")

# 12 loci in paper order (label <-> maps-file key). Note bgs10 (10q22) precedes lct (LCT).
KEYS   = ["slc24a5", "ibd5", "fads", "tlr", "bgs10", "lct", "val1", "val2", "val3", "val4", "val5", "val6"]
LABELS = ["SLC24A5", "IBD5", "FADS", "TLR", "10q22", "LCT", "1q23", "6q21", "8p21", "11p14", "14q21", "20p11"]
KIND   = ["sweep", "test", "test", "test", "test", "test", "val", "val", "val", "val", "val", "val"]

# --- carried constants (NOT in the maps file) --------------------------------------
# sim sample-size crossover (neutral benchmark, 25 kb Pearson): base saturates, pyrho overtakes ~n=100.
NCROSS_N     = np.array([20, 40, 100])
NCROSS_BASE  = np.array([0.882, 0.892, 0.941])
NCROSS_PYRHO = np.array([0.748, 0.844, 0.949])
# base SSM (rich_ld=False) per-locus 100 kb Pearson vs deCODE, paper order (carried from prior sesame run).
LOCI_BASE = np.array([0.58761757, 0.75745847, 0.72020162, 0.77821590, 0.78055659, 0.61830305,
                      0.69187295, 0.71077577, 0.77114626, 0.76541630, 0.66359589, 0.51074030])
# SECONDARY, honest operating point of the regime-aware pi-gate (CL-map at neutral windows,
# SSM/rich at low-diversity sweep windows).  Reproducible from sesame /home/kkor/large_n/
# pigate_refine.py at threshold thr=0.295 (pi < 0.295 * locus-median => trust the SSM):
# it recovers most of the SSM's sweep gain over pyrho (0.667 vs 0.553) at a small neutral cost
# (0.825, just below pyrho's 0.839).  A frontier, NOT a "beats pyrho at both" free lunch --
# the headline is the PURE CL-map, which matches pyrho at neutral AND ties it at the sweep.
GATE_NEUTRAL = 0.825529
GATE_SWEEP   = 0.666778
ENS_A = 0.25  # legacy blend weight; unused by the standalone figure, kept for back-compat.


def rc(a, t, f=4):
    """100 kb Pearson: average f adjacent fine windows, keep positive finite pairs."""
    n = (len(a) // f) * f
    aa = np.nanmean(a[:n].reshape(-1, f), 1); tt = np.nanmean(t[:n].reshape(-1, f), 1)
    ok = np.isfinite(aa) & np.isfinite(tt) & (aa > 0) & (tt > 0)
    return float(pearsonr(aa[ok], tt[ok])[0]) if ok.sum() > 3 else np.nan


def main():
    Z = np.load(MAPS, allow_pickle=True)
    assert list(Z["rows"]) == ["LASSO", "rich", "pyrho", "deCODE"], list(Z["rows"])
    clmap, rich, pyr = [], [], []
    for k in KEYS:
        L, R, Py, t = Z[k]            # rows: LASSO(=CL-map), rich(=SSM), pyrho, deCODE(=truth)
        clmap.append(rc(L, t)); rich.append(rc(R, t)); pyr.append(rc(Py, t))
    clmap = np.array(clmap); rich = np.array(rich); pyr = np.array(pyr)
    kind = np.array(KIND); neu = kind != "sweep"

    np.savez(
        OUT,
        ncross_n=NCROSS_N, ncross_base=NCROSS_BASE, ncross_pyrho=NCROSS_PYRHO,
        loci=np.array(LABELS), loci_kind=kind,
        loci_base=LOCI_BASE, loci_rich=rich, loci_clmap=clmap, loci_pyrho=pyr,
        gate_neutral=np.float64(GATE_NEUTRAL), gate_sweep=np.float64(GATE_SWEEP),
        ens_a=np.float64(ENS_A),
    )

    print("per-locus 100 kb Pearson vs deCODE (n=198):")
    for lab, ki, cl, ri, py in zip(LABELS, KIND, clmap, rich, pyr):
        print(f"  {lab:8s} ({ki:5s})  clmap={cl:.3f}  rich={ri:.3f}  pyrho={py:.3f}")
    print()
    print(f"NEUTRAL(11) means:  rich(SSM)={rich[neu].mean():.4f}  "
          f"CL-map={clmap[neu].mean():.4f}  pyrho={pyr[neu].mean():.4f}")
    print(f"SWEEP (slc24a5):    rich(SSM)={rich[0]:.4f}  "
          f"CL-map={clmap[0]:.4f}  pyrho={pyr[0]:.4f}")
    print(f"max|CL-map - pyrho| over 11 neutral loci = {np.max(np.abs((clmap - pyr)[neu])):.4f}")
    print(f"standalone pi-gate:  neutral={GATE_NEUTRAL}  sweep={GATE_SWEEP}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
