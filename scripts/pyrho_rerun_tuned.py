"""Re-run pyrho `optimize` at a tuned (bpen, w) across condition dirs and rebuild pred_pyrho.npz.

Used to correct the pyrho baseline after `pyrho hyperparam` selects the block penalty (the fixed
-bpen 50 in run_pyrho_config.py was over-smoothed). For each dir it re-optimizes every region VCF at
the tuned (bpen, w) -> region_*.tuned.rmap, then length-weight-bins pyrho's per-bp rate onto the 25 kb
grid and overwrites pred_pyrho.npz (callers back up the original to pred_pyrho.bpen50.npz first). The
gridding reproduces the original pipeline exactly (bpen=50 -> published pyrho_25kb to 3 dp).

Run in the FASTRHO venv; calls the pyrho binary via subprocess:
  python scripts/pyrho_rerun_tuned.py --bpen 25 --w 50 --dirs <dir> [<dir> ...] [--workers 16]
"""
import os
import sys
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
L = 2_000_000
W = 25_000
EDGES = np.append(np.arange(0, L, W), L)


def optimize(d, vcf, bpen, w):
    base = os.path.basename(vcf)[:-4]
    out = os.path.join(d, base + ".tuned.rmap")
    table = os.path.join(d, "pyrho_table.hdf")
    r = subprocess.run([PYRHO, "optimize", "--vcffile", vcf, "--tablefile", table, "--ploidy", "1",
                        "-w", str(w), "-bpen", str(bpen), "--numthreads", "4", "-o", out],
                       capture_output=True, text=True)
    return base, (out if r.returncode == 0 else None)


def rmap_to_grid(path, edges):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpen", type=int, required=True)
    ap.add_argument("--w", type=int, required=True)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    # keep only real condition dirs: must have a lookup table AND region VCFs
    dirs = [d for d in a.dirs if os.path.exists(os.path.join(d, "pyrho_table.hdf"))
            and glob.glob(os.path.join(d, "region_*.vcf"))]
    skipped = [d for d in a.dirs if d not in dirs]
    if skipped:
        print("skipping %d dirs without table/VCFs: %s" % (len(skipped),
              ", ".join(os.path.basename(d.rstrip("/")) for d in skipped)), flush=True)
    jobs = [(d, v) for d in dirs for v in sorted(glob.glob(os.path.join(d, "region_*.vcf")))]
    print("%d optimize jobs at bpen=%d w=%d across %d dirs" % (len(jobs), a.bpen, a.w, len(dirs)),
          flush=True)
    res = {}
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(optimize, d, v, a.bpen, a.w): (d, os.path.basename(v)[:-4])
                for d, v in jobs}
        done = 0
        for f in as_completed(futs):
            d, base = futs[f]
            res[(d, base)] = f.result()[1]
            done += 1
            if done % 50 == 0:
                print("  %d/%d optimized" % (done, len(jobs)), flush=True)

    for d in dirs:
        grids = {}
        for v in sorted(glob.glob(os.path.join(d, "region_*.vcf"))):
            base = os.path.basename(v)[:-4]
            out = res.get((d, base))
            if out and os.path.exists(out):
                g = rmap_to_grid(out, EDGES)
                if g is not None:
                    grids[base] = g
        if not grids:
            print("  WARNING: no valid rmaps for %s -- leaving pred_pyrho.npz untouched" % d, flush=True)
            continue
        np.savez(os.path.join(d, "pred_pyrho.npz"), **grids)
        print("  wrote %s (%d regions)" % (os.path.join(d, "pred_pyrho.npz"), len(grids)), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
