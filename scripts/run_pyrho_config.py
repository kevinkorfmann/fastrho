"""Run pyrho on a bench config dir (executed by the pyrho venv).

Reads config.json, builds a lookup table matched to (n, size-history), runs `optimize`
per region VCF -> region_*.rmap.  Usage: pyrho_python run_pyrho_config.py <config_dir>
"""
import json, os, sys, subprocess, glob, time

cfgdir = sys.argv[1]
cfg = json.load(open(os.path.join(cfgdir, "config.json")))
PYRHO = os.path.join(os.path.dirname(sys.executable), "pyrho")
n = 2 * cfg["n_dip"]
table = os.path.join(cfgdir, "pyrho_table.hdf")
table_threads = int(os.environ.get("PYRHO_TABLE_THREADS", "30"))
optimize_threads = int(os.environ.get("PYRHO_OPTIMIZE_THREADS", "8"))

cmd = [PYRHO, "make_table", "-n", str(n), "-m", str(cfg["mu"]),
       "-p", ",".join(str(x) for x in cfg["popsizes"]),
       "--approx", "-N", str(n + 5), "--numthreads", str(table_threads), "-o", table]
if cfg.get("epochtimes"):
    cmd += ["-t", ",".join(str(x) for x in cfg["epochtimes"])]
print("make_table:", " ".join(cmd), flush=True)
table_start = time.perf_counter()
subprocess.run(cmd, check=True)
table_wall = time.perf_counter() - table_start

ok = 0
optimize_wall = 0.0
region_wall = {}
for v in sorted(glob.glob(os.path.join(cfgdir, "region_*.vcf"))):
    base = v[:-4]
    # block penalty / window selected by `pyrho hyperparam` for this n=20, constant-Ne setting
    # (was a fixed -bpen 50, which over-smooths at fine scale; see scripts/pyrho_hyperparam_n20.txt).
    region_start = time.perf_counter()
    r = subprocess.run([PYRHO, "optimize", "--vcffile", v, "--tablefile", table,
                        "--ploidy", "1", "-w", "50", "-bpen", "25",
                        "--numthreads", str(optimize_threads), "-o", base + ".rmap"],
                       capture_output=True, text=True)
    elapsed = time.perf_counter() - region_start
    optimize_wall += elapsed
    region_wall[os.path.basename(base)] = elapsed
    if r.returncode == 0:
        ok += 1
    else:
        print("FAIL", os.path.basename(base), r.stderr.strip().splitlines()[-1:], flush=True)
print("pyrho done: %d/%d regions optimized" % (ok, len(glob.glob(os.path.join(cfgdir, "region_*.vcf")))))
with open(os.path.join(cfgdir, "pyrho_runtime.json"), "w") as handle:
    json.dump(
        {
            "lookup_table_wall_seconds": table_wall,
            "optimize_wall_seconds": optimize_wall,
            "total_wall_seconds": table_wall + optimize_wall,
            "successful_regions": ok,
            "region_wall_seconds": region_wall,
        },
        handle,
        indent=2,
        sort_keys=True,
    )
    handle.write("\n")
