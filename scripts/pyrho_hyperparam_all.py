"""Run `pyrho hyperparam` for each distinct (n, demography) table in the main benchmark, in parallel,
and select a block penalty by a uniform, non-edge rule.

Selection rule (applied identically to every setting): fix window=50 (pyrho's standard, validated on
the n=20 sweep) and choose the block penalty maximizing fine-scale log-correlation (Log_Pearson_Corr_10kb)
over bpen in [20,40]. The bpen>=20 floor avoids the numerically fragile low-penalty edge (bpen=15 caused
`optimize` failures and degraded the real deCODE hotspot map). This reproduces the validated bpen=25 for
n=20 constant-Ne. Writes the chosen (bpen, w) per config to paper/figdata/pyrho_bpen_choices.json.

Run in the fastrho venv; calls the pyrho binary. Each config's own pyrho_table.hdf + config.json are used.
"""
import os
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np

PYRHO = "/home/kkor/venvs/pyrho/bin/pyrho"
CDIR = "/home/kkor/fastrho_data/campaign/configs"
OUT = "/home/kkor/fastrho/paper/figdata"
# configs needing a NEW hyperparam (n=20 constant is already done -> bpen 25)
NEW = ["const_n40", "const_n100", "bottleneck_n20", "expansion_n20", "real_dog"]
WFIX = 50
BPEN_MIN = 20


def run_one(cfg):
    d = os.path.join(CDIR, cfg)
    c = json.load(open(os.path.join(d, "config.json")))
    n = c["n_dip"] * 2
    logf = os.path.join(OUT, "hyperparam_%s.txt" % cfg)
    cmd = [PYRHO, "hyperparam", "-n", str(n), "-m", str(c["mu"]), "--ploidy", "1",
           "-p", ",".join(str(x) for x in c["popsizes"]),
           "--tablefile", os.path.join(d, "pyrho_table.hdf"),
           "--num_sims", "15", "-bpen", "20,25,30,35", "-w", "50",
           "--numthreads", "14", "-o", logf, "--logfile", "/tmp/hp_%s.log" % cfg]
    if c.get("epochtimes"):
        cmd += ["-t", ",".join(str(x) for x in c["epochtimes"])]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return cfg, logf, r.returncode


def pick_bpen(logf):
    rows = [l.split("\t") for l in open(logf).read().splitlines() if l.strip()]
    hdr = rows[0]
    ci = {h: i for i, h in enumerate(hdr)}
    b, w, m = ci["Block_Penalty"], ci["Window_Size"], ci["Log_Pearson_Corr_10kb"]
    best, best_val = None, -1
    for r in rows[1:]:
        bp, ws, val = float(r[b]), float(r[w]), float(r[m])
        if ws == WFIX and bp >= BPEN_MIN and val > best_val:
            best_val, best = val, int(bp)
    return best


def main():
    print("launching %d hyperparam runs in parallel..." % len(NEW), flush=True)
    with ThreadPoolExecutor(max_workers=len(NEW)) as ex:
        results = list(ex.map(run_one, NEW))
    choices = {"const_n20": {"bpen": 25, "w": 50},
               "real_decode": {"bpen": 25, "w": 50},
               "real_hapmap": {"bpen": 25, "w": 50},
               "bottleneck_n20_wd": {"bpen": 25, "w": 50}}
    for cfg, logf, rc in results:
        if rc != 0 or not os.path.exists(logf):
            print("  %-16s FAILED (rc=%s)" % (cfg, rc), flush=True)
            continue
        bp = pick_bpen(logf)
        choices[cfg] = {"bpen": bp, "w": WFIX}
        print("  %-16s -> bpen=%d w=%d" % (cfg, bp, WFIX), flush=True)
    json.dump(choices, open(os.path.join(OUT, "pyrho_bpen_choices.json"), "w"), indent=2)
    print("\nwrote", os.path.join(OUT, "pyrho_bpen_choices.json"), flush=True)
    print(json.dumps(choices), flush=True)


if __name__ == "__main__":
    main()
