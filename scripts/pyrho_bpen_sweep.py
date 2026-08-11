"""Block-penalty sensitivity sweep for pyrho on representative slim_dr2 conditions (rebuttal check).

Motivation: the main selection benchmark runs pyrho with a FIXED block penalty (-bpen 50, -w 50); pyrho's
own guidance is to tune the block penalty per (n, theta, demography). This script re-runs pyrho `optimize`
at several -bpen values (reusing each condition's existing lookup table + VCFs), scores the pooled 25 kb
Pearson r against the truth, and prints it next to fastrho on the same regions -- so we can see whether the
fastrho>pyrho gap is an artifact of an untuned, over-smoothed penalty.

Run in the FASTRHO venv (needs scipy + fastrho.preprocess); it calls the pyrho binary via subprocess:
  /home/kkor/venvs/fastrho/bin/python scripts/pyrho_bpen_sweep.py [--nregions N] [--bpens 10,20,50,100]
"""
import os
import sys
import glob
import json
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, "/home/kkor/fastrho")
from fastrho.preprocess import mean_rate_between

ROOT = "/home/kkor/fastrho_data/slim_dr2"
PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
SCRATCH = "/home/kkor/bpen_sweep"
L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)

# representative conditions: neutral baseline, mid + extreme sweep, strong background selection
DEFAULT_CONDS = "neutral,swstr_5,swstr_9,bgsint_5"
LABEL = {"neutral": "neutral", "swstr_5": "sweep 2Nes=400", "swstr_9": "sweep 2Nes=4000",
         "bgsint_5": "BGS ~32% pi-reduction"}


def optimize(cond, reg_vcf, bpen, w):
    base = os.path.basename(reg_vcf)[:-4]
    outdir = os.path.join(SCRATCH, cond)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "%s.bpen%d.w%d.rmap" % (base, bpen, w))
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    table = os.path.join(ROOT, cond, "pyrho_table.hdf")
    r = subprocess.run([PYRHO, "optimize", "--vcffile", reg_vcf, "--tablefile", table,
                        "--ploidy", "1", "-w", str(w), "-bpen", str(bpen),
                        "--numthreads", "4", "-o", out],
                       capture_output=True, text=True)
    return out if r.returncode == 0 else None


def rmap_to_grid(path, edges):
    """Length-weighted mean of pyrho's per-bp rate (cols: start end rate) onto the 25 kb grid."""
    rows = np.loadtxt(path, ndmin=2)
    if rows.size == 0:
        return None
    start, end, rate = rows[:, -3], rows[:, -2], rows[:, -1]
    out = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        ov = np.clip(np.minimum(end, edges[k + 1]) - np.maximum(start, edges[k]), 0, None)
        tot = ov.sum()
        if tot > 0:
            out[k] = float((rate * ov).sum() / tot)
    return out


def pooled(T, P):
    if not T:
        return float("nan"), 0
    tt = np.concatenate(T)
    pp = np.concatenate(P)
    m = np.isfinite(tt) & np.isfinite(pp)
    return float(pearsonr(tt[m], pp[m])[0]), len(T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default=DEFAULT_CONDS)
    ap.add_argument("--bpens", default="10,20,50,100")
    ap.add_argument("--w", type=int, default=50)
    ap.add_argument("--nregions", type=int, default=0)   # 0 = all regions
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="/home/kkor/fastrho/paper/figdata/pyrho_bpen_sweep.json")
    a = ap.parse_args()
    conds = a.conditions.split(",")
    bpens = [int(x) for x in a.bpens.split(",")]

    jobs = []
    for cond in conds:
        vcfs = sorted(glob.glob(os.path.join(ROOT, cond, "region_*.vcf")))
        if a.nregions:
            vcfs = vcfs[:a.nregions]
        for v in vcfs:
            for b in bpens:
                jobs.append((cond, v, b))
    print("launching %d optimize jobs (%d conditions x %d bpen x regions)..."
          % (len(jobs), len(conds), len(bpens)), flush=True)

    results = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(optimize, c, v, b, a.w): (c, os.path.basename(v)[:-4], b)
                for c, v, b in jobs}
        done = 0
        for f in as_completed(futs):
            key = futs[f]
            results[key] = f.result()
            done += 1
            if done % 40 == 0:
                print("  %d/%d done" % (done, len(jobs)), flush=True)

    fpred = {c: np.load(os.path.join(ROOT, c, "pred_fastrho.npz")) for c in conds}
    report = {"w": a.w, "bpens": bpens, "conditions": {}}
    for cond in conds:
        vcfs = sorted(glob.glob(os.path.join(ROOT, cond, "region_*.vcf")))
        if a.nregions:
            vcfs = vcfs[:a.nregions]
        names = [os.path.basename(v)[:-4] for v in vcfs]
        truth = {n: mean_rate_between(np.load(os.path.join(ROOT, cond, n + ".npz"),
                 allow_pickle=True)["map_position"],
                 np.load(os.path.join(ROOT, cond, n + ".npz"),
                 allow_pickle=True)["map_rate"], EDGES) for n in names}
        rec = {"label": LABEL.get(cond, cond), "bpen": {}}
        for b in bpens:
            Ts, Ps = [], []
            for n in names:
                rp = results.get((cond, n, b))
                grid = rmap_to_grid(rp, EDGES) if rp else None
                if grid is not None:
                    tr = truth[n]
                    k = min(len(tr), len(grid))
                    Ts.append(tr[:k]); Ps.append(grid[:k])
            r, nreg = pooled(Ts, Ps)
            rec["bpen"][b] = {"pyrho_r": r, "n_regions": nreg}
        Tf, Pf = [], []
        for n in names:
            if n in fpred[cond].files:
                tr = truth[n]
                fr = fpred[cond][n]
                k = min(len(tr), len(fr))
                Tf.append(tr[:k]); Pf.append(fr[:k])
        rec["fastrho_r"], _ = pooled(Tf, Pf)
        report["conditions"][cond] = rec
        print("\n%-9s (%s): fastrho=%.3f | %s" % (
            cond, rec["label"], rec["fastrho_r"],
            "  ".join("bpen%d=%.3f" % (b, rec["bpen"][b]["pyrho_r"]) for b in bpens)), flush=True)

    json.dump(report, open(a.out, "w"), indent=2)
    print("\nwrote", a.out, flush=True)


if __name__ == "__main__":
    main()
