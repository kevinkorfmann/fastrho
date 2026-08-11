"""SLiM forward selfing + background-selection generator for the selfer training simulator.

Closes the sim-to-real gap that caps A. thaliana recovery
(research/archive/model-zoo-15k-regression-postmortem.md):
the incumbent selfer sim (scripts/selfing_gen.py) is a NEUTRAL, constant-Ne, panmictic rescaled
coalescent, so it omits background selection in low-recombination pericentromeres -- exactly what
shapes real selfer diversity. This generator forward-simulates selfing + BGS in SLiM on a REAL
A. thaliana recombination map + real exon architecture + species DFE, then emits the SAME ts_*
training contract as selfing_gen.py so fastrho/preprocess.py consumes it unchanged.

Seam (forward selfing vs panmictic recap): SLiM gets the FULL meiotic map (x Q) + p1.setSelfingRate(s)
[unscaled s]; the panmictic recapitation cannot self, so it receives EFFECTIVE params
(rate x Q x (1-F), sizes /(1+F)/Q, times /Q), F=s/(2-s) -- the mechanism_sim.py recipe. Because s
(hence F) is Q-invariant, the standard "x Q selection coefficient" rescaling still holds. Validated
by scripts/selfing_seam_check.py (neutral selfing SLiM must match the neutral coalescent).

Requires SLiM 5.x. Point SLIM_BIN at the binary (sesame: /home/kkor/.local/bin/slim).
Usage: SLIM_BIN=/home/kkor/.local/bin/slim python scripts/selfing_slim_gen.py <outdir> <n> [off] [mode] [nproc]
  mode: bgs (default) | neutral
"""
import os
import sys
import json
import subprocess
import tempfile

import numpy as np
import msprime
import tskit
import pyslim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import selfer_arch as A

SLIM = os.environ.get("SLIM_BIN", "slim")

# --- operating point (env-overridable; calibrate N' against real athal SNP density) ---
def _envf(k, d): return float(os.environ.get(k, d))
def _envi(k, d): return int(os.environ.get(k, d))

Q = _envf("SELF_Q", 10.0)              # SLiM rescaling: census N = N' * Q
NPRIME = _envi("SELF_NPRIME", 5000)    # forward WF census (per-individual cost ~ N' * burnin)
L = _envi("SELF_L", 500_000)
MU = _envf("SELF_MU", A.ATHAL["mu"])   # 7e-9
U_DEL_FRAC = _envf("SELF_UDEL_FRAC", 1.0)   # deleterious rate = MU * dfe_del_prop * this, inside exons
CFG = dict(A.ATHAL)


def _write_slim(path, mode, rmap_tsv, exon_tsv, out_trees, seed, s, dfe, burnin):
    """Emit a SLiM 5.x hermaphrodite WF recipe with selfing (+ optional BGS DFE over real exons).
    Recombination is the FULL meiotic rate x Q (read from rmap_tsv). Selfing rate s is UNSCALED."""
    mean_s, shape, del_prop = dfe
    if mode == "bgs":
        u_del = MU * del_prop * U_DEL_FRAC * Q
        sel_block = ('initializeMutationType("m1", 0.5, "g", %r, %r);\n'
                     '    initializeGenomicElementType("g1", m1, 1.0);\n' % (mean_s * Q, shape))
        elem_block = ('lines = readFile("%s");\n'
                      '    for (line in lines) {\n'
                      '        f = strsplit(line, "\\t");\n'
                      '        initializeGenomicElement(g1, asInteger(f[0]), asInteger(f[1]));\n'
                      '    }\n' % exon_tsv)
        mut_rate = repr(u_del)
    else:  # neutral: a spanning neutral element, no deleterious mutations
        sel_block = ('initializeMutationType("m1", 0.5, "f", 0.0);\n'
                     '    initializeGenomicElementType("g1", m1, 1.0);\n')
        elem_block = 'initializeGenomicElement(g1, 0, %d);\n' % (L - 1)
        mut_rate = "0.0"

    script = (
        'initialize() {\n'
        '    setSeed(%d);\n'
        '    initializeTreeSeq();\n'
        '    initializeMutationRate(%s);\n'
        '    %s'
        '    rl = readFile("%s");\n'
        '    rends = NULL; rrates = NULL;\n'
        '    for (line in rl) {\n'
        '        g = strsplit(line, "\\t");\n'
        '        rends = c(rends, asInteger(g[0]));\n'
        '        rrates = c(rrates, asFloat(g[1]));\n'
        '    }\n'
        '    initializeRecombinationRate(rrates, rends);\n'
        '    %s'
        '}\n'
        '1 early() { sim.addSubpop("p1", %d); p1.setSelfingRate(%g); }\n'
        '%d late() {\n'
        '    sim.treeSeqOutput("%s");\n'
        '    sim.simulationFinished();\n'
        '}\n'
        % (seed, mut_rate, sel_block, rmap_tsv, elem_block, NPRIME, s, burnin, out_trees)
    )
    with open(path, "w") as fh:
        fh.write(script)


def slim_selfing_ts(pos, rate, s, n, mode, exons, dfe, demog_or_ne, seed, sp=None):
    """Core: forward SLiM (selfing + optional BGS) on the FULL meiotic map (pos,rate in per-bp),
    recapitate under EFFECTIVE params, sample ONE haplotype per individual, overlay neutral muts.
    Returns a tskit.TreeSequence with n haploid samples. `demog_or_ne` is an msprime.Demography
    (already F/Q-rescaled) or a float ancestral effective Ne."""
    F = A.selfing_F(s)
    Lloc = float(pos[-1])
    N_eff_slim = NPRIME / (1.0 + F)
    burnin = max(2000, int(round(4.0 * N_eff_slim)))
    rng = np.random.default_rng(seed)

    with tempfile.TemporaryDirectory() as td:
        rmap_tsv = os.path.join(td, "rmap.tsv")
        ends = (pos[1:].astype(int) - 1); ends[-1] = int(Lloc) - 1
        with open(rmap_tsv, "w") as fh:
            for e, r in zip(ends, rate):
                fh.write("%d\t%g\n" % (int(e), float(r) * Q))       # full meiotic rate x Q
        exon_tsv = os.path.join(td, "exons.tsv")
        if mode == "bgs" and len(exons):
            with open(exon_tsv, "w") as fh:
                for a0, b0 in exons:
                    fh.write("%d\t%d\n" % (int(a0), int(b0)))
        slim_path = os.path.join(td, "s.slim")
        out_trees = os.path.join(td, "out.trees")
        _write_slim(slim_path, mode if (mode == "neutral" or len(exons)) else "neutral",
                    rmap_tsv, exon_tsv, out_trees, seed, s, dfe, burnin)
        res = subprocess.run([SLIM, slim_path], capture_output=True, text=True)
        if not os.path.exists(out_trees):
            raise RuntimeError("SLiM failed (seed %d):\n%s\n%s" % (seed, res.stdout[-1500:], res.stderr[-1500:]))
        sts = tskit.load(out_trees)

    # recapitate the deep past under the EFFECTIVE recombination map (selfing-suppressed)
    eff_map = msprime.RateMap(position=pos, rate=rate * Q * (1.0 - F))
    reck = dict(recombination_rate=eff_map, random_seed=seed + 1)
    if isinstance(demog_or_ne, (int, float)):
        reck["ancestral_Ne"] = float(demog_or_ne)
    else:
        reck["demography"] = demog_or_ne
    rts = pyslim.recapitate(sts, **reck)

    alive = pyslim.individuals_alive_at(rts, 0)
    keep = rng.choice(alive, size=min(n, len(alive)), replace=False)
    nodes = [int(rts.individual(j).nodes[0]) for j in keep]           # ONE haplotype / individual
    sts2 = rts.simplify(nodes, filter_sites=True)
    # drop SLiM's selected mutations; overlay fresh neutral variation on the selection-shaped tree
    tb = sts2.dump_tables(); tb.sites.clear(); tb.mutations.clear()
    ts = msprime.sim_mutations(tb.tree_sequence(), rate=MU * Q, random_seed=seed + 2)
    return ts


def gen_one(i, outdir, mode="bgs", exclude_chrom=None):
    try:
        _gen_one(i, outdir, mode, exclude_chrom)
    except Exception as e:      # never let one region kill the pool
        print("ts_%08d FAILED (%s): %s" % (i, mode, e), flush=True)


def _gen_one(i, outdir, mode="bgs", exclude_chrom=None):
    base = os.path.join(outdir, "ts_%08d" % i)
    if os.path.exists(base + ".trees"):
        return
    import stdpopsim
    sp = stdpopsim.get_species(CFG["species"])
    seed = 700_000 + i
    rng = np.random.default_rng(seed)
    s = A.draw_selfing(rng); F = A.selfing_F(s)
    n = int(rng.choice([50, 80, 120, 156, 200]))

    pos, rate, chrom, lo = A.load_real_map_slice(sp, CFG["map_id"], L, rng,
                                                 peri_bias=0.5, exclude_chrom=exclude_chrom)
    exons = A.exon_intervals_for_slice(sp, CFG["annotation_id"], chrom, lo, L) if mode == "bgs" else np.zeros((0, 2), int)
    dfe = A.dfe_gamma(sp, CFG["dfe_id"])
    # Recap the deep past under a CONSTANT effective Ne = N'/(1+F) (the validated seam convention;
    # a 1-pop msprime.Demography cannot be passed to pyslim.recapitate on a 2-pop SLiM ts). The
    # SLiM fraction is thus constant-Ne + BGS on the real map; time-varying Ne(t) is the coalescent
    # fraction's job (scripts/selfing_blend_gen.py), keeping BGS and demography as separate axes.
    anc_ne = NPRIME / (1.0 + F)

    ts = slim_selfing_ts(pos, rate, s, n, mode, exons, dfe, anc_ne, seed, sp=sp)

    # diversity-implied effective size (consistent scalar across the blend + correct under BGS /
    # time-varying demography): the Ne that the observed polymorphism supports, so 4*Ne*r is the
    # population-scaled rho the LD reflects and the Ne head learns to estimate.
    Ne_eff = max(1.0, float(ts.diversity(span_normalise=True)) / (4.0 * MU))
    eff_rate = rate * (1.0 - F)                                      # EFFECTIVE meiotic (the target)
    meta = dict(seed=int(seed), n_samples=int(ts.num_samples), mutation_rate=float(MU),
                Ne=float(Ne_eff), selfing=float(s), sequence_length=float(pos[-1]),
                window_size=2000, num_sites=int(ts.num_sites),
                mode="slim_selfing_" + mode, source_map=CFG["map_id"], source_chrom=str(chrom),
                source_lo=float(lo), dfe_mean_s=float(dfe[0]), rescale_Q=float(Q), Nprime=int(NPRIME))
    ts.dump(base + ".trees")
    np.savez(base + ".npz", map_position=pos, map_rate=eff_rate, meta=json.dumps(meta))
    print("ts_%08d [%s] chr%s lo=%.2fMb s=%.3f %dhap %dsites"
          % (i, mode, chrom, lo / 1e6, s, ts.num_samples, ts.num_sites), flush=True)


def main():
    outdir = sys.argv[1]; n = int(sys.argv[2])
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    mode = sys.argv[4] if len(sys.argv) > 4 else "bgs"
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    exclude = os.environ.get("SELF_EXCLUDE_CHROM")   # LOCO fold: hold out this chromosome's map
    os.makedirs(outdir, exist_ok=True)
    if nproc <= 1:
        for i in range(off, off + n):
            gen_one(i, outdir, mode, exclude)
    else:
        from multiprocessing import get_context
        with get_context("fork").Pool(nproc) as pool:
            pool.starmap(gen_one, [(i, outdir, mode, exclude) for i in range(off, off + n)])
    print("done", outdir, mode, n)


if __name__ == "__main__":
    main()
