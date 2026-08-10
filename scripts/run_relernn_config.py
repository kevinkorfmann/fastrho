"""Run the full ReLERNN pipeline on a bench config dir (executed by the relernn2 venv).

Builds one combined multi-contig VCF from the per-region VCFs, then
SIMULATE -> TRAIN -> PREDICT.  nTrain = 100000 if config.relernn_full else 30000.
Usage: relernn_python run_relernn_config.py <config_dir>
"""
import json, os, sys, subprocess, glob, shutil

cfgdir = sys.argv[1]
cfg = json.load(open(os.path.join(cfgdir, "config.json")))
bindir = os.path.dirname(sys.executable)
proj = os.path.join(cfgdir, "relernn_proj")
shutil.rmtree(proj, ignore_errors=True)

# combined VCF from per-region VCFs (consistent tsk_* samples; one header)
regions = sorted(glob.glob(os.path.join(cfgdir, "region_*.vcf")))
combined = os.path.join(cfgdir, "combined.vcf")
contigs, sample_line, body = [], None, []
for v in regions:
    with open(v) as fh:
        for line in fh:
            if line.startswith("##contig"):
                contigs.append(line)
            elif line.startswith("#CHROM"):
                sample_line = line
            elif not line.startswith("#"):
                body.append(line)
with open(combined, "w") as fh:
    fh.write("##fileformat=VCFv4.2\n")
    fh.writelines(contigs)
    fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    fh.write(sample_line)
    fh.writelines(body)
print("combined.vcf: %d contigs, %d sites" % (len(contigs), len(body)), flush=True)

nTrain = 100000 if cfg.get("relernn_full") else 30000
# FAIRNESS: ReLERNN's per-base rate prior is r ~ U(0, mu*upperRhoThetaRatio)
# (ReLERNN_SIMULATE: priorHighsRho = assumedMu*upRTR). The default upRTR=1.0 caps the
# prior at mu, far below hotspot rates, so it saturates and underpredicts. But setting it
# too high (e.g. a flat 50) inflates the prior MEAN and biases predictions ~2.5-6x high.
# Best: tune upRTR to the data's actual peak rate -> upRTR = ceil(SAFETY * max_r / mu).
SAFETY = 1.15
QUANTILE = 0.999   # cover 99.9% of the map BY LENGTH; using max() overshoots on real maps
                   # whose rare extreme hotspot peaks (>>p99) blow up the uniform prior mean.
def _auto_uprtr():
    import numpy as np, math
    rate_chunks, len_chunks = [], []
    for npz in sorted(glob.glob(os.path.join(cfgdir, "region_*.npz"))):
        try:
            z = np.load(npz, allow_pickle=True)
            pos = np.asarray(z["map_position"], float)
            rate = np.asarray(z["map_rate"], float)
            seglen = np.diff(pos)
            m = np.isfinite(rate) & (seglen > 0)
            rate_chunks.append(rate[m]); len_chunks.append(seglen[m])
        except Exception:
            pass
    if not rate_chunks:
        return None
    rate = np.concatenate(rate_chunks); seglen = np.concatenate(len_chunks)
    order = np.argsort(rate)
    rate, seglen = rate[order], seglen[order]
    cdf = np.cumsum(seglen) / seglen.sum()
    peak = float(rate[np.searchsorted(cdf, QUANTILE)])     # length-weighted p99.9 rate
    return int(math.ceil(SAFETY * peak / float(cfg["mu"])))

_arg = sys.argv[2] if len(sys.argv) > 2 else "auto"
if str(_arg).lower() == "auto":
    upRTR = _auto_uprtr()
    if upRTR is None:
        upRTR = 50.0
        print("WARN: no region_*.npz truth found; falling back to upRTR=50", flush=True)
    else:
        print("AUTO upRTR from truth peak rate = %d" % upRTR, flush=True)
else:
    upRTR = float(_arg)
maxSites = sys.argv[3] if len(sys.argv) > 3 else None     # finer windows if set
ncpu = sys.argv[4] if len(sys.argv) > 4 else "40"
gpuID = sys.argv[5] if len(sys.argv) > 5 else "0"         # ReLERNN overrides CUDA_VISIBLE_DEVICES
# make BOTH GPUs visible so ReLERNN's internal --gpuID selects the right physical device;
# without this, ReLERNN forces gpuID=0 and concurrent runs collide on GPU0.
env = dict(os.environ, CUDA_VISIBLE_DEVICES=os.environ.get("CUDA_VISIBLE_DEVICES", "0,1"),
           PYTHONWARNINGS="ignore", TF_CPP_MIN_LOG_LEVEL="3")
genome = os.path.join(cfgdir, "genome.bed")


def run(args, log):
    with open(os.path.join(cfgdir, log), "w") as fh:
        subprocess.run(args, env=env, stdout=fh, stderr=subprocess.STDOUT, check=True)


sim_args = [bindir + "/ReLERNN_SIMULATE", "-v", combined, "-g", genome, "-d", proj,
            "-u", str(cfg["mu"]), "-l", "1", "-t", ncpu, "-s", "1", "--phased",
            "-r", str(upRTR),
            "--nTrain", str(nTrain), "--nVali", str(nTrain // 10), "--nTest", str(nTrain // 10)]
if maxSites:
    sim_args += ["--maxSites", str(maxSites)]
print("upperRhoThetaRatio=%.1f maxSites=%s gpuID=%s" % (upRTR, maxSites, gpuID), flush=True)
run(sim_args, "relernn_sim.log")
print("SIMULATE done (nTrain=%d)" % nTrain, flush=True)
run([bindir + "/ReLERNN_TRAIN", "-d", proj, "--nEpochs", "100", "-t", "8", "-s", "1",
     "--gpuID", str(gpuID)], "relernn_train.log")
print("TRAIN done", flush=True)
run([bindir + "/ReLERNN_PREDICT", "-v", combined, "-d", proj, "--phased", "-s", "1",
     "--gpuID", str(gpuID)], "relernn_pred.log")
pred = glob.glob(os.path.join(proj, "*PREDICT*txt"))
print("PREDICT done ->", pred[0] if pred else "NONE", flush=True)
