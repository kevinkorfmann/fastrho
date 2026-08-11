"""Regenerate paper/figdata/highn.npz with the fastrho x pyrho ENSEMBLE that beats pyrho at large n.

Recomputes base / rich / pyrho / ensemble on all 12 real CEU/GRCh38 loci vs the deCODE pedigree map
(100 kb Pearson), self-consistently from the SSOT models (realdata_infer.get_ck), plus SLC24A5 sweep
tracks. Ensemble = rich^a * pyrho^(1-a), a=0.25 (chosen by leave-one-locus-out CV; see w1_blend.py).
This replaces the stale hand-entered loci_rich (which were the older prototype model)."""
import os, sys, glob, numpy as np
sys.path.insert(0, "/home/kkor/fastrho"); sys.path.insert(0, "/home/kkor/fastrho/scripts")
from scipy.stats import pearsonr
from fastrho.translate import load_model, predict_intervals
from fastrho.features import SNPTokenFeaturizer, FeatureConfig
from fastrho.preprocess import mean_rate_between
from fastcxt.sfs import basic_filtering
import grch38_zoom as gz
from realdata_infer import get_ck

MU = gz.MU; DEV = "cuda:0"; A = 0.25
PYDIR = "/home/kkor/realdata/pyrho"
KEYS   = ["slc24a5", "ibd5", "fads", "tlr", "bgs10", "lct", "val1", "val2", "val3", "val4", "val5", "val6"]
LABELS = ["SLC24A5", "IBD5", "FADS", "TLR", "10q22", "LCT", "1q23", "6q21", "8p21", "11p14", "14q21", "20p11"]
KIND   = ["sweep", "test", "test", "test", "test", "test", "val", "val", "val", "val", "val", "val"]

bck, bst = get_ck("base"); rck, rst = get_ck("rich")
bmodel, bcfg, bstats = load_model(bck, bst, device=DEV)
rmodel, rcfg, rstats = load_model(rck, rst, device=DEV)
bfz = SNPTokenFeaturizer(FeatureConfig(rich_ld=False))
rfz = SNPTokenFeaturizer(FeatureConfig(rich_ld=True))


def fmap(model, cfg, stats, fz, gm, pos, edges):
    gmf, posf = basic_filtering(gm.astype(np.int8), pos.astype(np.float64))
    p = predict_intervals(model, cfg, stats, gmf, posf, MU, Ne=None, device=DEV, featurizer=fz)
    bp = np.r_[p["pos_left"][0], p["pos_right"]]
    return mean_rate_between(bp, p["r_per_bp"], edges)


def pymap(key, edges):
    st, en, rt = [], [], []
    for rm in sorted(glob.glob(f"{PYDIR}/{key}/region_*.rmap")):
        for ln in open(rm):
            p = ln.split()
            if len(p) >= 3 and p[0][0].isdigit():
                st.append(float(p[0])); en.append(float(p[1])); rt.append(float(p[2]))
    return mean_rate_between(np.array(st + [en[-1]]), np.array(rt), edges) if st else None


def rc(a, t, f=4):
    n = (len(a) // f) * f
    aa = np.nanmean(a[:n].reshape(-1, f), 1); tt = np.nanmean(t[:n].reshape(-1, f), 1)
    ok = np.isfinite(aa) & np.isfinite(tt) & (aa > 0) & (tt > 0)
    return float(pearsonr(aa[ok], tt[ok])[0]) if ok.sum() > 3 else np.nan


def ens_map(R, Py, a=A):
    return np.exp(a * np.log(np.clip(R, 1e-12, None)) + (1 - a) * np.log(np.clip(Py, 1e-12, None)))


base, rich, pyr, ens, sw = [], [], [], [], {}
for key, lab, kind in zip(KEYS, LABELS, KIND):
    z = np.load(f"{gz.HAP}/{key}.npz", allow_pickle=True)
    gm = z["gm"]; pos = z["pos"].astype(float); chrom = str(z["chrom"])
    edges = gz.fine_edges(pos); truth = gz.deCODE_windows(chrom, edges)
    B = fmap(bmodel, bcfg, bstats, bfz, gm, pos, edges)
    R = fmap(rmodel, rcfg, rstats, rfz, gm, pos, edges)
    Py = pymap(key, edges); E = ens_map(R, Py)
    base.append(rc(B, truth)); rich.append(rc(R, truth)); pyr.append(rc(Py, truth)); ens.append(rc(E, truth))
    print(f"{lab:8s}({kind}) base={base[-1]:.3f} rich={rich[-1]:.3f} pyrho={pyr[-1]:.3f} ens={ens[-1]:.3f}", flush=True)
    if key == "slc24a5":
        mids = (edges[:-1] + edges[1:]) / 2
        pi = np.interp(mids, np.asarray(z["pi_centers"], float) * 1e6, np.asarray(z["pi"], float))
        sw = dict(sw_centers=mids, sw_truth=truth, sw_base=B, sw_rich=R, sw_pyrho=Py, sw_ens=E, sw_pi=pi)

base = np.array(base); rich = np.array(rich); pyr = np.array(pyr); ens = np.array(ens); kind = np.array(KIND)
neu = kind != "sweep"
ncross_n = np.array([20, 40, 100]); ncross_base = np.array([0.882, 0.892, 0.941]); ncross_pyrho = np.array([0.748, 0.844, 0.949])
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "paper", "figdata", "highn.npz")
np.savez(OUT, ncross_n=ncross_n, ncross_base=ncross_base, ncross_pyrho=ncross_pyrho,
         loci=np.array(LABELS), loci_kind=kind, loci_base=base, loci_rich=rich, loci_pyrho=pyr,
         loci_ens=ens, ens_a=np.float64(A), **sw)
print("\n=== NEUTRAL means (100 kb) ===")
for tag, m in (("base", base), ("rich", rich), ("pyrho", pyr), ("ens", ens)):
    print(f"{tag:6s} neutral={m[neu].mean():.3f}  sweep={m[~neu][0]:.3f}")
print(f"ens>=pyrho neutral: {int((ens[neu] >= pyr[neu] - 1e-6).sum())}/{int(neu.sum())};  "
      f"ens sweep {ens[~neu][0]:.3f} vs pyrho {pyr[~neu][0]:.3f};  rich sweep {rich[~neu][0]:.3f}")
print("HIGHN_FIGDATA_DONE wrote", OUT, flush=True)
