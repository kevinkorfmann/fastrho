"""Run pyrho on a bench config dir (executed by the pyrho venv).

Reads config.json, builds a lookup table matched to (n, size-history), runs `optimize`
per region VCF -> region_*.rmap.  Usage: pyrho_python run_pyrho_config.py <config_dir>
"""
import json, os, sys, subprocess, glob

cfgdir = sys.argv[1]
cfg = json.load(open(os.path.join(cfgdir, "config.json")))
PYRHO = os.path.join(os.path.dirname(sys.executable), "pyrho")
n = 2 * cfg["n_dip"]
table = os.path.join(cfgdir, "pyrho_table.hdf")

cmd = [PYRHO, "make_table", "-n", str(n), "-m", str(cfg["mu"]),
       "-p", ",".join(str(x) for x in cfg["popsizes"]),
       "--approx", "-N", str(n + 5), "--numthreads", "30", "-o", table]
if cfg.get("epochtimes"):
    cmd += ["-t", ",".join(str(x) for x in cfg["epochtimes"])]
print("make_table:", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

ok = 0
for v in sorted(glob.glob(os.path.join(cfgdir, "region_*.vcf"))):
    base = v[:-4]
    # block penalty / window selected by `pyrho hyperparam` for this n=20, constant-Ne setting
    # (was a fixed -bpen 50, which over-smooths at fine scale; see scripts/pyrho_hyperparam_n20.txt).
    r = subprocess.run([PYRHO, "optimize", "--vcffile", v, "--tablefile", table,
                        "--ploidy", "1", "-w", "50", "-bpen", "25",
                        "--numthreads", "8", "-o", base + ".rmap"],
                       capture_output=True, text=True)
    if r.returncode == 0:
        ok += 1
    else:
        print("FAIL", os.path.basename(base), r.stderr.strip().splitlines()[-1:], flush=True)
print("pyrho done: %d/%d regions optimized" % (ok, len(glob.glob(os.path.join(cfgdir, "region_*.vcf")))))
