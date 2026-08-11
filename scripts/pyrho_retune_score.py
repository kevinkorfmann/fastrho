"""Re-run pyrho at hyperparam-selected penalties across the whole main benchmark, re-score, collate.

Reads paper/figdata/pyrho_bpen_choices.json (from pyrho_hyperparam_all.py). For every benchmark config
with a pyrho baseline it: backs up pred_pyrho.npz -> pred_pyrho.bpen50.npz (once), re-runs `pyrho optimize`
at the config's (bpen, w) and rebuilds pred_pyrho.npz on the 25 kb grid, then re-scores with bench.py
(all methods; fastrho/relernn unchanged) and collates into summary.json. Prints the before/after pyrho
25 kb Pearson per config. Run in the fastrho venv.
"""
import os
import glob
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

VENV = "/home/kkor/venvs/fastrho/bin/python"
PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
BENCH = "/home/kkor/fastrho/scripts/bench.py"
COLLATE = "/home/kkor/fastrho/scripts/collate.py"
CDIR = "/home/kkor/fastrho_data/campaign/configs"
RESULTS = "/home/kkor/fastrho_data/campaign/results"
CHOICES = "/home/kkor/fastrho/paper/figdata/pyrho_bpen_choices.json"
L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)


def optimize(d, vcf, bpen, w):
    base = os.path.basename(vcf)[:-4]
    out = os.path.join(d, base + ".tuned.rmap")
    r = subprocess.run([PYRHO, "optimize", "--vcffile", vcf, "--tablefile",
                        os.path.join(d, "pyrho_table.hdf"), "--ploidy", "1", "-w", str(w),
                        "-bpen", str(bpen), "--numthreads", "4", "-o", out],
                       capture_output=True, text=True)
    return base, (out if r.returncode == 0 else None)


def rmap_to_grid(path):
    rows = np.loadtxt(path, ndmin=2)
    if rows.size == 0:
        return None
    start, end, rate = rows[:, -3], rows[:, -2], rows[:, -1]
    out = np.full(len(EDGES) - 1, np.nan)
    for k in range(len(EDGES) - 1):
        ov = np.clip(np.minimum(end, EDGES[k + 1]) - np.maximum(start, EDGES[k]), 0, None)
        tot = ov.sum()
        if tot > 0:
            out[k] = float((rate * ov).sum() / tot)
    return out


def pyrho25(cfg):
    p = os.path.join(RESULTS, cfg + ".json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    try:
        return d["scales"]["25kb"]["pyrho"]["pearson"]
    except (KeyError, TypeError):
        return None


def main():
    choices = json.load(open(CHOICES))
    configs = [c for c in choices
               if os.path.exists(os.path.join(CDIR, c, "pyrho_table.hdf"))
               and glob.glob(os.path.join(CDIR, c, "region_*.vcf"))]
    before = {c: pyrho25(c) for c in configs}

    # back up originals (once)
    for c in configs:
        d = os.path.join(CDIR, c)
        b = os.path.join(d, "pred_pyrho.bpen50.npz")
        if os.path.exists(os.path.join(d, "pred_pyrho.npz")) and not os.path.exists(b):
            import shutil
            shutil.copy(os.path.join(d, "pred_pyrho.npz"), b)

    # re-run optimize for every (config, region) at that config's chosen (bpen, w)
    jobs = []
    for c in configs:
        d = os.path.join(CDIR, c)
        bp, w = choices[c]["bpen"], choices[c]["w"]
        for v in sorted(glob.glob(os.path.join(d, "region_*.vcf"))):
            jobs.append((c, d, v, bp, w))
    print("re-optimizing %d region-jobs across %d configs..." % (len(jobs), len(configs)), flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(optimize, d, v, bp, w): (c, os.path.basename(v)[:-4])
                for c, d, v, bp, w in jobs}
        done = 0
        for f in as_completed(futs):
            res[futs[f]] = f.result()[1]
            done += 1
            if done % 40 == 0:
                print("  %d/%d" % (done, len(jobs)), flush=True)

    # rebuild pred_pyrho.npz per config, then re-score + collate
    for c in configs:
        d = os.path.join(CDIR, c)
        grids = {}
        for v in sorted(glob.glob(os.path.join(d, "region_*.vcf"))):
            base = os.path.basename(v)[:-4]
            out = res.get((c, base))
            if out and os.path.exists(out):
                g = rmap_to_grid(out)
                if g is not None:
                    grids[base] = g
        if not grids:
            print("  %s: NO rmaps rebuilt, skipping" % c, flush=True)
            continue
        np.savez(os.path.join(d, "pred_pyrho.npz"), **grids)
        subprocess.run([VENV, BENCH, "score", "--config", d, "--methods", "fastrho", "pyrho",
                        "relernn", "--results", RESULTS],
                       env={**os.environ, "PYTHONPATH": "/home/kkor/fastrho"},
                       capture_output=True, text=True)

    # collate -> summary.json
    out = subprocess.run([VENV, COLLATE, RESULTS], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": "/home/kkor/fastrho"})
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        fh.write(out.stdout)

    print("\n=== pyrho 25kb Pearson: before (bpen50) -> after (tuned) ===", flush=True)
    for c in configs:
        aft = pyrho25(c)
        bp = choices[c]["bpen"]
        b = before[c]
        print("  %-18s bpen=%-3d  %s -> %s" % (
            c, bp, ("%.3f" % b) if b else "--", ("%.3f" % aft) if aft else "--"), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
