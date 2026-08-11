"""Figure data for the expanded linked-selection analysis (runs on sesame, fastrho venv).

For every condition dir under slim_dr/ it computes, per scoring scale, the pooled Pearson of
fastrho (and pyrho, where present) against the truth, WITH a region-bootstrap 95% CI -- so the
dose-response curves carry error bars. Selection parameters are read from each dir's config.json.
It also extracts the diversity footprint pi(x) and recovery-vs-distance for the footprint conditions.

Writes paper/figdata/selection_dr.json (+ selection_dr_figdata.npz).
"""
import os
import glob
import json

import numpy as np
import tskit
from scipy.stats import pearsonr

import sys
sys.path.insert(0, "/home/kkor/fastrho")
from fastrho.preprocess import mean_rate_between

ROOT = os.environ.get("SLIM_DR_ROOT", "/home/kkor/fastrho_data/slim_dr2")
OUTJSON = "/home/kkor/fastrho/paper/figdata/selection_dr.json"
OUTNPZ = "/home/kkor/fastrho/paper/figdata/selection_dr_figdata.npz"
L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2.0
GRIDS = {"25kb": 1, "100kb": 4, "500kb": 20}
NBOOT = 1000
NDIST = 13          # distance-bin edges (-> 12 bins) for the recovery-vs-distance panel


def block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def load_windows(cdir):
    """Per-region windowed truth, fastrho, pyrho (None if absent) at 25 kb, plus mean SNP count."""
    fpred = np.load(os.path.join(cdir, "pred_fastrho.npz"))
    ppath = os.path.join(cdir, "pred_pyrho.npz")
    ppred = np.load(ppath) if os.path.exists(ppath) else None
    T, Fr, Py, sites = [], [], [], []
    for npz in sorted(glob.glob(os.path.join(cdir, "region_*.npz"))):
        name = os.path.basename(npz)[:-4]
        if name not in fpred.files:
            continue
        z = np.load(npz, allow_pickle=True)
        truth = mean_rate_between(z["map_position"], z["map_rate"], EDGES)
        fr = fpred[name]
        n = min(len(truth), len(fr))
        T.append(truth[:n]); Fr.append(fr[:n])
        Py.append(ppred[name][:n] if (ppred is not None and name in ppred.files) else None)
        sites.append(json.loads(str(z["meta"]))["num_sites"])
    return T, Fr, Py, float(np.mean(sites))


def pooled_with_ci(T, P, f):
    """Pooled Pearson at block factor f, with a region-bootstrap 95% CI."""
    pairs = [(block_mean(t, f), block_mean(p, f)) for t, p in zip(T, P) if p is not None]
    if len(pairs) < 3:
        return None
    full = pearsonr(np.concatenate([t for t, _ in pairs]),
                    np.concatenate([p for _, p in pairs]))[0]
    rng = np.random.default_rng(0)
    boot = []
    idx = np.arange(len(pairs))
    for _ in range(NBOOT):
        s = rng.choice(idx, len(idx), replace=True)
        tt = np.concatenate([pairs[j][0] for j in s])
        pp = np.concatenate([pairs[j][1] for j in s])
        boot.append(pearsonr(tt, pp)[0])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(full), float(lo), float(hi)]


def paired_delta_ci(cm, f):
    """Paired region-bootstrap CI for Delta = pooled_r(fastrho) - pooled_r(pyrho).

    Resamples the SAME regions for both methods on every bootstrap draw, so the
    interval reflects the *paired* fastrho-vs-pyrho difference (accounts for the
    two tools being scored on identical regions) -- the statistically correct way
    to show the gap excludes 0. cm: list of (truth, fastrho, pyrho) per region.
    """
    trip = [(block_mean(t, f), block_mean(fr, f), block_mean(py, f)) for t, fr, py in cm]
    if len(trip) < 3:
        return None

    def pooled_delta(items):
        T = np.concatenate([a for a, _, _ in items])
        F = np.concatenate([b for _, b, _ in items])
        P = np.concatenate([c for _, _, c in items])
        return pearsonr(T, F)[0] - pearsonr(T, P)[0]

    full = pooled_delta(trip)
    rng = np.random.default_rng(0)
    idx = np.arange(len(trip))
    boot = [pooled_delta([trip[j] for j in rng.choice(idx, len(idx), replace=True)])
            for _ in range(NBOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return [float(full), float(lo), float(hi)]


def calib_arrays(cdir):
    """Concatenated (truth, fastrho, pyrho) 25 kb window rates over matched regions.

    Feeds the calibration supplementary figure (estimated-vs-true absolute rate),
    which tests bias that Pearson r -- being scale-invariant -- cannot.
    """
    T, Fr, Py, _ = load_windows(cdir)
    trip = [(t, fr, py) for t, fr, py in zip(T, Fr, Py) if py is not None]
    if not trip:
        return None
    return (np.concatenate([t for t, _, _ in trip]),
            np.concatenate([f for _, f, _ in trip]),
            np.concatenate([p for _, _, p in trip]))


def condition_record(name):
    cdir = os.path.join(ROOT, name)
    sp = json.load(open(os.path.join(cdir, "selection_params.json")))
    T, Fr, Py, mean_sites = load_windows(cdir)
    rec = dict(name=name, demography="slim_" + sp["mode"], sweep_s=sp.get("sweep_s"),
               sweep_target=sp.get("sweep_target"), soft_k=sp.get("soft_k"),
               exon_frac=sp.get("exon_frac"), n_regions=len(T), mean_sites=mean_sites)
    # regions where pyrho also produced an estimate -> fair, matched fastrho-vs-pyrho comparison
    cm = [(t, fr, py) for t, fr, py in zip(T, Fr, Py) if py is not None]
    for sk, f in GRIDS.items():
        rec["fastrho_" + sk] = pooled_with_ci(T, Fr, f)          # all regions (fastrho-only panels)
        if len(cm) >= 3:
            Tc = [c[0] for c in cm]
            rec["fastrho_cmn_" + sk] = pooled_with_ci(Tc, [c[1] for c in cm], f)
            rec["pyrho_" + sk] = pooled_with_ci(Tc, [c[2] for c in cm], f)
            rec["delta_" + sk] = paired_delta_ci(cm, f)   # paired fastrho - pyrho gap
    return rec


def diversity_profile(cdir):
    """Mean diversity footprint over regions, with a 95% CI of the mean (mean +/- 1.96 SEM)."""
    pis = np.array([tskit.load(tp).diversity(windows=EDGES)
                    for tp in sorted(glob.glob(os.path.join(cdir, "region_*.trees")))])
    m = pis.mean(0)
    sem = pis.std(0, ddof=1) / np.sqrt(len(pis))
    return m, m - 1.96 * sem, m + 1.96 * sem


def _binned_logpearson(Tm, Pm, dist, bins, nboot=0):
    """Per-distance-bin log-rate Pearson; with a region-bootstrap 95% CI when nboot>0.

    Returns (bin_centres_Mb, full, lo, hi); lo/hi are NaN when nboot==0.
    """
    bc, full, lo, hi = [], [], [], []
    rng = np.random.default_rng(0)
    nreg = Tm.shape[0]
    idx = np.arange(nreg)
    for k in range(len(bins) - 1):
        cols = np.where((dist >= bins[k]) & (dist < bins[k + 1]))[0]
        if not cols.size:
            continue
        bc.append((bins[k] + bins[k + 1]) / 2.0 / 1e6)
        tf, pf = Tm[:, cols], Pm[:, cols]
        full.append(pearsonr(np.log(tf.ravel() + 1e-12), np.log(pf.ravel() + 1e-12))[0])
        if nboot:
            bs = [pearsonr(np.log(tf[s].ravel() + 1e-12), np.log(pf[s].ravel() + 1e-12))[0]
                  for s in (rng.choice(idx, nreg, replace=True) for _ in range(nboot))]
            l, h = np.percentile(bs, [2.5, 97.5])
        else:
            l = h = np.nan
        lo.append(l); hi.append(h)
    return np.array(bc), np.array(full), np.array(lo), np.array(hi)


def recov_vs_dist(cdir):
    """log-rate Pearson (+bootstrap CI) vs distance from the swept site.

    Returns (bins, (fr, fr_lo, fr_hi), py) where py is None or (pv, py_lo, py_hi).
    """
    T, Fr, Py, _ = load_windows(cdir)
    dist = np.abs(CENTRES - L / 2.0)
    bins = np.linspace(0, L / 2.0, NDIST)
    bc, fr, frlo, frhi = _binned_logpearson(np.array(T), np.array(Fr), dist, bins, nboot=NBOOT)
    cm = [(t, py) for t, py in zip(T, Py) if py is not None]
    py = None
    if len(cm) >= 3:
        _, pv, plo, phi = _binned_logpearson(np.array([c[0] for c in cm]),
                                             np.array([c[1] for c in cm]), dist, bins, nboot=NBOOT)
        py = (pv, plo, phi)
    return bc, (fr, frlo, frhi), py


def main():
    names = sorted(os.path.basename(os.path.dirname(p))
                   for p in glob.glob(os.path.join(ROOT, "*", "config.json")))
    records = [condition_record(n) for n in names]
    json.dump({"conditions": records, "n_boot": NBOOT},
              open(OUTJSON, "w"), indent=2)

    # pick footprint conditions by parameter, not by hardcoded name: neutral, the strongest BGS,
    # and a representative hard sweep (s closest to 0.05)
    params = {n: json.load(open(os.path.join(ROOT, n, "selection_params.json"))) for n in names}
    bgs = [n for n in names if params[n]["mode"] == "bgs"]
    hard = [n for n in names if params[n]["mode"] == "sweep"
            and params[n].get("sweep_target", 1.0) >= 0.999 and int(params[n].get("soft_k", 1)) == 1]

    npz = {"centres": CENTRES}
    if "neutral" in names:
        ndir = os.path.join(ROOT, "neutral")
        npz["pi_neutral"], npz["pi_neutral_lo"], npz["pi_neutral_hi"] = diversity_profile(ndir)
        _, (fn, fnlo, fnhi), _ = recov_vs_dist(ndir)
        npz["recov_neutral"] = fn
        npz["recov_neutral_lo"], npz["recov_neutral_hi"] = fnlo, fnhi
        cn = calib_arrays(ndir)
        if cn is not None:
            npz["calib_true_neutral"], npz["calib_fastrho_neutral"], npz["calib_pyrho_neutral"] = cn
    if bgs:
        bdir = os.path.join(ROOT, max(bgs, key=lambda n: params[n]["exon_frac"]))
        npz["pi_bgs"], npz["pi_bgs_lo"], npz["pi_bgs_hi"] = diversity_profile(bdir)
    if hard:
        sw = min(hard, key=lambda n: abs(params[n]["sweep_s"] - 0.05))
        swdir = os.path.join(ROOT, sw)
        npz["pi_sweep"], npz["pi_sweep_lo"], npz["pi_sweep_hi"] = diversity_profile(swdir)
        bc, (fr, frlo, frhi), py = recov_vs_dist(swdir)
        npz["dist_bins"], npz["recov_sweep"] = bc, fr
        npz["recov_sweep_lo"], npz["recov_sweep_hi"] = frlo, frhi
        if py is not None:
            npz["recov_pyrho_sweep"], npz["recov_pyrho_sweep_lo"], npz["recov_pyrho_sweep_hi"] = py
        cs = calib_arrays(swdir)
        if cs is not None:
            npz["calib_true_sweep"], npz["calib_fastrho_sweep"], npz["calib_pyrho_sweep"] = cs
    np.savez(OUTNPZ, **npz)

    print("conditions:", names)
    for r in records:
        extra = ("s=%s" % r["sweep_s"]) if "sweep" in r["demography"] else ("ef=%s" % r["exon_frac"])
        fr = r.get("fastrho_25kb"); py = r.get("pyrho_25kb")
        print("  %-10s %-8s sites=%5.0f  fastrho25=%.3f  pyrho25=%s"
              % (r["name"], extra, r["mean_sites"], (fr[0] if fr else float("nan")),
                 ("%.3f" % py[0]) if py else "--"))
    print("wrote", OUTJSON, OUTNPZ)


if __name__ == "__main__":
    main()
