"""SLiM linked-selection stress-test generator for fastrho.

Closes the paper's stated future-work gap: "controlled robustness to linked selection
(a forward-simulation SLiM stress test) remains future work" (discussion.tex). We test whether
the FROZEN base model -- never told about selection -- still recovers the true recombination map
when the LD it reads from is distorted by background selection (BGS) or a hard sweep.

Design: identical to the const_n20 benchmark cell (Ne=1e4, n_dip=10, mu=1.5e-8, 2 Mb, variable
truth map from fastrho.simulate.make_recombination_map) EXCEPT the ancestry is forward-simulated
in SLiM with linked selection instead of neutral msprime. The TRUTH map scored against is the same
spatially-varying RateMap; only the genealogy/LD changes. Three regimes:

  * neutral : SLiM forward, no selected mutations  (within-simulator control; must match msprime const_n20)
  * bgs     : deleterious gamma DFE on ~25% exonic sequence  (classic background selection)
  * sweep   : a single strong beneficial mutation at region centre, conditioned on fixation

To keep forward simulation tractable at Ne=1e4 we use SLiM's standard rescaling by Q: simulate
N'=Ne/Q diploids with mutation/recombination/selection rates multiplied by Q, which preserves the
population-scaled dynamics (theta=4N'mu', rho=4N'r', 2N's all invariant). The deep neutral past is
filled by msprime recapitation under the SAME (Q-scaled) RateMap; neutral variation is overlaid
afterwards at mu*Q on the N'-clock tree, so the realised theta and rho match Ne=1e4, mu=1.5e-8.
The truth map written to disk is the UNSCALED rate (correlation metrics are scale-invariant).

Output per region matches the bench.py contract exactly:
  region_%03d.trees  + region_%03d.npz (map_position, map_rate, meta) + config.json + genome.bed
so the existing `bench.py fastrho` / `bench.py score` are reused with no changes.

Usage:  python scripts/slim_gen.py <outdir> <neutral|bgs|sweep> <n_regions> [offset] [n_proc]
Requires: SLiM 5.x on PATH, pyslim, msprime, tskit, the fastrho package.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.simulate import make_recombination_map, RecombPriors

# --- benchmark-matched constants (const_n20 cell) ---------------------------
L = 2_000_000
NE = 10_000.0
N_DIP = int(os.environ.get("SLIM_N_DIP", 10))   # 2*N_DIP haplotypes; default 20 (matches const_n20). Set SLIM_N_DIP for the sample-size series.
MU = 1.5e-8
MEAN_R = 1.0e-8

# --- rescaling + forward-sim controls ---------------------------------------
Q = 10.0             # SLiM rescaling factor; N' = NE/Q = 1000
N_SLIM = int(round(NE / Q))
BURNIN = 8 * N_SLIM  # forward generations (a few coalescent units so selection shapes LD)

# --- selection architecture (env-overridable for dose-response sweeps) ------
def _envf(key, default):
    return float(os.environ.get(key, default))


EXON_LEN = 2_000
EXON_FRAC = _envf("SLIM_EXON_FRAC", 0.25)    # fraction of the 2 Mb under purifying selection (BGS)
DFE_MEAN_S = _envf("SLIM_DFE_MEAN_S", -0.025)  # gamma mean selection coeff (deleterious), unscaled
DFE_SHAPE = 0.20
U_DEL = 1.5e-8                   # deleterious mutation rate per bp inside exons (unscaled)
SWEEP_S = _envf("SLIM_SWEEP_S", 0.05)        # beneficial selection coeff for the sweep (unscaled)
SWEEP_TARGET = _envf("SLIM_SWEEP_TARGET", 1.0)  # <1.0 -> partial sweep, sample at this frequency
SOFT_K = int(_envf("SLIM_SOFT_K", 1))        # >1 -> soft sweep: introduce on K haplotypes at once


def _write_vcf(path, ts, chrom):
    """Diploid phased VCF (drop non-segregating sites) so pyrho can be run on the same regions."""
    gm = ts.genotype_matrix().T.astype(np.int8)       # (n_hap, n_sites)
    positions = ts.tables.sites.position
    n_hap = gm.shape[0]
    seg = (gm.sum(0) > 0) & (gm.sum(0) < n_hap)
    gm = gm[:, seg]
    positions = positions[seg]
    last = -1
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=%s,length=%d>\n" % (chrom, int(L)))
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                 + "\t".join("tsk_%d" % j for j in range(n_hap // 2)) + "\n")
        for s in range(gm.shape[1]):
            p = int(positions[s])
            p = last + 1 if p <= last else p
            last = p
            h = gm[:, s]
            fh.write("%s\t%d\t.\tA\tT\t.\tPASS\t.\tGT\t%s\n"
                     % (chrom, p, "\t".join("%d|%d" % (h[2 * j], h[2 * j + 1])
                                            for j in range(n_hap // 2))))


def _exons():
    """Evenly spaced exons covering EXON_FRAC of the region (BGS architecture)."""
    n_ex = int((L * EXON_FRAC) / EXON_LEN)
    gap = (L - n_ex * EXON_LEN) // (n_ex + 1)
    starts, ends = [], []
    x = gap
    for _ in range(n_ex):
        starts.append(int(x))
        ends.append(int(x + EXON_LEN - 1))
        x += EXON_LEN + gap
    return starts, ends


def _write_slim(slim_path, mode, rmap_path, exon_path, out_trees, seed):
    """Emit a SLiM 5.x WF recipe; recombination map + exons are read from files."""
    sel_block = ""
    elem_block = ""
    mut_rate = "0.0"
    if mode == "bgs":
        # background selection: deleterious gamma DFE over the exon architecture
        mut_rate = repr(U_DEL * Q)
        sel_block += (
            'initializeMutationType("m1", 0.5, "g", %r, %r);\n'
            '    initializeGenomicElementType("g1", m1, 1.0);\n'
            % (DFE_MEAN_S * Q, DFE_SHAPE)
        )
        elem_block = (
            'lines = readFile("%s");\n'
            '    for (line in lines) {\n'
            '        f = strsplit(line, "\\t");\n'
            '        initializeGenomicElement(g1, asInteger(f[0]), asInteger(f[1]));\n'
            '    }\n' % exon_path
        )
    else:
        # neutral and (pure) sweep: neutral background; SLiM still needs a mutation type,
        # element type and a spanning element. The sweep is isolated from BGS this way.
        sel_block += (
            'initializeMutationType("m1", 0.5, "f", 0.0);\n'
            '    initializeGenomicElementType("g1", m1, 1.0);\n'
        )
        elem_block = 'initializeGenomicElement(g1, 0, %d);\n' % (L - 1)
    if mode == "sweep":
        sel_block += 'initializeMutationType("m2", 0.5, "f", %r);\n    ' % (SWEEP_S * Q)

    # population init (SLiM 5 renamed 'genomes' -> 'haplosomes'); for the sweep, keep the
    # beneficial mutation queryable after fixation so we can detect it.
    pop_block = 'sim.addSubpop("p1", %d);' % N_SLIM
    if mode == "sweep":
        pop_block += ' m2.convertToSubstitution = F;'

    # sweep machinery: introduce the beneficial mutation at the centre after burn-in (on SOFT_K
    # haplotypes for a soft sweep), reintroduce on stochastic loss, stop at the target frequency
    # (SWEEP_TARGET<1 -> partial sweep; ==1 -> hard sweep run to fixation).
    sweep_events = ""
    end_gen = BURNIN
    if mode == "sweep":
        intro = BURNIN
        end_gen = intro + 40 * N_SLIM
        sweep_events = (
            '%d late() {\n'
            '    target = sample(p1.haplosomes, %d);\n'
            '    target.addNewDrawnMutation(m2, asInteger(%d));\n'
            '}\n'
            '%d:%d late() {\n'
            '    mut = sim.mutationsOfType(m2);\n'
            '    if (mut.size() == 0) {\n'
            '        target = sample(p1.haplosomes, %d);\n'
            '        target.addNewDrawnMutation(m2, asInteger(%d));\n'
            '    } else if (sum(sim.mutationFrequencies(NULL, mut)) >= %r) {\n'
            '        sim.treeSeqOutput("%s");\n'
            '        sim.simulationFinished();\n'
            '    }\n'
            '}\n'
            '%d late() {\n'
            '    sim.treeSeqOutput("%s");\n'
            '    sim.simulationFinished();\n'
            '}\n'
            % (intro, SOFT_K, L // 2, intro + 1, end_gen, SOFT_K, L // 2,
               SWEEP_TARGET, out_trees, end_gen, out_trees)
        )

    final_block = ""
    if mode != "sweep":
        final_block = (
            '%d late() {\n'
            '    sim.treeSeqOutput("%s");\n'
            '    sim.simulationFinished();\n'
            '}\n' % (BURNIN, out_trees)
        )

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
        '1 early() { %s }\n'
        '%s'
        '%s'
        % (seed, mut_rate, sel_block, rmap_path, elem_block,
           pop_block, sweep_events, final_block)
    )
    with open(slim_path, "w") as fh:
        fh.write(script)
    return end_gen


def gen_one(i, outdir, mode):
    base = os.path.join(outdir, "region_%03d" % i)
    if os.path.exists(base + ".trees"):
        return
    seed = 20260 + i
    rng = np.random.default_rng(seed)

    # truth recombination map (same construction as bench.py gen)
    priors = RecombPriors(sequence_length=L)
    kind = "hotspot" if i % 2 else "gp"
    rm = make_recombination_map(L, np.random.default_rng(seed), kind=kind,
                                mean_rate=MEAN_R, priors=priors)
    pos = np.asarray(rm.position)
    rate = np.asarray(rm.rate)

    with tempfile.TemporaryDirectory() as td:
        # SLiM recombination map: end (0-based last base of segment) \t Q-scaled rate
        rmap_path = os.path.join(td, "rmap.tsv")
        ends_bp = (pos[1:].astype(int) - 1)
        ends_bp[-1] = L - 1
        with open(rmap_path, "w") as fh:
            for e, r in zip(ends_bp, rate):
                fh.write("%d\t%g\n" % (int(e), float(r) * Q))

        exon_path = os.path.join(td, "exons.tsv")
        if mode == "bgs":
            es, ee = _exons()
            with open(exon_path, "w") as fh:
                for a, b in zip(es, ee):
                    fh.write("%d\t%d\n" % (a, b))

        slim_path = os.path.join(td, "rec.slim")
        out_trees = os.path.join(td, "slim_out.trees")
        _write_slim(slim_path, mode, rmap_path, exon_path, out_trees, seed)

        res = subprocess.run(["slim", slim_path], capture_output=True, text=True)
        if not os.path.exists(out_trees):
            raise RuntimeError("SLiM failed region %d (%s):\n%s\n%s"
                               % (i, mode, res.stdout[-2000:], res.stderr[-2000:]))

        sts = tskit.load(out_trees)

    # recapitate the deep past under the SAME Q-scaled map, then sample n_dip, overlay neutral muts
    rec_ratemap = msprime.RateMap(position=pos, rate=rate * Q)
    rts = pyslim.recapitate(sts, recombination_rate=rec_ratemap, ancestral_Ne=N_SLIM,
                            random_seed=seed + 1)

    alive = pyslim.individuals_alive_at(rts, 0)
    keep = rng.choice(alive, size=N_DIP, replace=False)
    nodes = np.concatenate([rts.individual(j).nodes for j in keep])
    sts2 = rts.simplify(nodes.tolist(), filter_sites=True)
    # Drop SLiM's selected mutations (their allele alphabet is incompatible with the neutral
    # overlay model). The linked-selection signal lives in the genealogy -- shortened coalescent
    # times near selected sites -- so overlaying fresh neutral mutations on this selection-shaped
    # tree reproduces the BGS/sweep LD pattern that fastrho reads.
    tb = sts2.dump_tables()
    tb.sites.clear()
    tb.mutations.clear()
    sts2 = tb.tree_sequence()
    ts = msprime.sim_mutations(sts2, rate=MU * Q, random_seed=seed + 2)

    meta = dict(Ne=NE, mutation_rate=MU, n_samples=N_DIP,
                sequence_length=int(L), window_size=2000, contig="chr%d" % (i + 1),
                demography="slim_" + mode, rescale_Q=Q, num_sites=int(ts.num_sites),
                sweep_s=SWEEP_S, sweep_target=SWEEP_TARGET, soft_k=SOFT_K,
                exon_frac=EXON_FRAC, dfe_mean_s=DFE_MEAN_S)
    ts.dump(base + ".trees")
    np.savez(base + ".npz", map_position=pos, map_rate=rate, meta=json.dumps(meta))
    _write_vcf(base + ".vcf", ts, "chr%d" % (i + 1))
    print("region %03d [%s]: %d sites, %d sample haps" % (i, mode, ts.num_sites, ts.num_samples),
          flush=True)


def main():
    outdir = sys.argv[1]
    mode = sys.argv[2]
    n = int(sys.argv[3])
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    assert mode in ("neutral", "bgs", "sweep")
    os.makedirs(outdir, exist_ok=True)

    name = os.environ.get("SLIM_NAME", "slim_" + mode)
    # config.json must stay bench.py-Config compatible -- selection params go in a sidecar + meta
    cfg = dict(name=name, demography="slim_" + mode, n_dip=N_DIP, mu=MU,
               Ne=NE, seq_len=int(L), n_regions=n, popsizes=[NE], epochtimes=[],
               genetic_map=None, species="HomSap", relernn_full=False)
    with open(os.path.join(outdir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(outdir, "selection_params.json"), "w") as fh:
        json.dump(dict(name=name, mode=mode, sweep_s=SWEEP_S, sweep_target=SWEEP_TARGET,
                       soft_k=SOFT_K, exon_frac=EXON_FRAC, dfe_mean_s=DFE_MEAN_S), fh, indent=2)
    with open(os.path.join(outdir, "genome.bed"), "w") as fh:
        for i in range(n):
            fh.write("chr%d\t0\t%d\n" % (i + 1, L))

    if nproc <= 1:
        for i in range(off, off + n):
            gen_one(i, outdir, mode)
    else:
        from multiprocessing import get_context
        with get_context("fork").Pool(nproc) as pool:
            pool.starmap(gen_one, [(i, outdir, mode) for i in range(off, off + n)])
    print("done", outdir, mode, n)


if __name__ == "__main__":
    main()
