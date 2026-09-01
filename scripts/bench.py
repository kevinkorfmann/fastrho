"""Config-driven benchmark engine for the fastrho vs pyrho vs ReLERNN campaign.

One Config = one cell of the experiment matrix (demography x n, or a real genetic map).
Each config dir holds: region_*.trees, region_*.npz (true map + meta), region_*.vcf
(diploid phased, non-segregating sites dropped), genome.bed, config.json. Methods write
pred_*.npz keyed by region name (rates per fixed bp GRID); score reads truth + preds and
emits metrics JSON at multiple scales.

Subcommands: gen | fastrho | ingest | score   (drivers for pyrho/ReLERNN are bash).
"""

from __future__ import annotations

import os
import json
import glob
import argparse
from dataclasses import dataclass, field, fields, asdict

import numpy as np

GRID = 25_000          # base scoring window (bp); coarser scales via block-averaging
SYNTHETIC_MEDIAN_RATE = 1e-8


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    name: str
    demography: str = "constant"          # constant | bottleneck | expansion | realmap
    n_dip: int = 10                       # diploid individuals (2x = haploids)
    mu: float = 1.5e-8
    Ne: float = 10_000.0
    seq_len: int = 2_000_000
    n_regions: int = 20
    popsizes: list = field(default_factory=lambda: [10_000.0])   # for pyrho size history
    epochtimes: list = field(default_factory=list)
    genetic_map: str | None = None        # realmap mode (stdpopsim HomSap)
    species: str = "HomSap"
    relernn_full: bool = False            # True -> nTrain 100000


def load_config(cfg_dir: str) -> Config:
    with open(os.path.join(cfg_dir, "config.json")) as fh:
        raw = json.load(fh)
    allowed = {item.name for item in fields(Config)}
    return Config(**{key: value for key, value in raw.items() if key in allowed})


# ---------------------------------------------------------------------------
# Demography
# ---------------------------------------------------------------------------

def _demography(cfg: Config):
    import msprime
    Ne = cfg.Ne
    if cfg.demography == "constant":
        return None, Ne
    d = msprime.Demography()
    d.add_population(initial_size=Ne)
    if cfg.demography == "bottleneck":
        d.add_population_parameters_change(time=1000, initial_size=Ne / 10)
        d.add_population_parameters_change(time=3000, initial_size=Ne)
    elif cfg.demography == "expansion":
        d.add_population_parameters_change(time=2000, initial_size=Ne / 10)
    else:
        raise ValueError(cfg.demography)
    return d, None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _write_vcf(path, chrom, gm, positions, seq_len):
    """Diploid phased VCF; retain only segregating, strictly biallelic sites."""
    n_hap = gm.shape[0]
    # ``msprime.sim_mutations`` can occasionally create recurrent/multiallelic
    # sites, whose genotype states exceed 1.  Writing those states with a single
    # ALT allele produces an invalid VCF and can be interpreted as a fixed allele
    # by downstream methods.  Keep only sites represented exactly by states 0/1.
    seg = (gm.min(0) == 0) & (gm.max(0) == 1)
    gm = gm[:, seg]; positions = positions[seg]
    last = -1
    with open(path, "w") as fh:
        fh.write("##fileformat=VCFv4.2\n##contig=<ID=%s,length=%d>\n" % (chrom, int(seq_len)))
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" +
                 "\t".join("tsk_%d" % j for j in range(n_hap // 2)) + "\n")
        for s in range(gm.shape[1]):
            p = int(positions[s]); p = last + 1 if p <= last else p; last = p
            h = gm[:, s]
            fh.write("%s\t%d\t.\tA\tT\t.\tPASS\t.\tGT\t%s\n" %
                     (chrom, p, "\t".join("%d|%d" % (h[2 * j], h[2 * j + 1])
                                          for j in range(n_hap // 2))))
    return int(seg.sum())


def cmd_gen(args):
    import msprime
    from fastrho.simulate import make_recombination_map, RecombPriors
    cfg = load_config(args.config) if os.path.exists(
        os.path.join(args.config, "config.json")) else None
    if cfg is None:  # build from CLI + save
        cfg = Config(name=os.path.basename(args.config.rstrip("/")),
                     demography=args.demography, n_dip=args.n_dip, mu=args.mu,
                     Ne=args.Ne, seq_len=args.seq_len, n_regions=args.n_regions,
                     genetic_map=args.genetic_map, species=args.species,
                     relernn_full=args.relernn_full,
                     popsizes=_popsizes(args.demography, args.Ne),
                     epochtimes=_epochtimes(args.demography))
        os.makedirs(args.config, exist_ok=True)
        with open(os.path.join(args.config, "config.json"), "w") as fh:
            json.dump(asdict(cfg), fh, indent=2)

    priors = RecombPriors(sequence_length=cfg.seq_len)
    for i in range(cfg.n_regions):
        seed = args.seed + i
        if cfg.demography == "realmap":
            ts, rm = _sim_realmap(cfg, seed, i)
        else:
            rng = np.random.default_rng(seed)
            kind = "hotspot" if i % 2 else "gp"
            rm = make_recombination_map(
                cfg.seq_len,
                rng,
                kind=kind,
                mean_rate=SYNTHETIC_MEDIAN_RATE,
                priors=priors,
            )
            demo, popsize = _demography(cfg)
            anc = dict(samples=cfg.n_dip, recombination_rate=rm,
                       sequence_length=cfg.seq_len, random_seed=seed + 1)
            if demo is not None:
                anc["demography"] = demo
            else:
                anc["population_size"] = popsize
            ts = msprime.sim_ancestry(**anc)
            ts = msprime.sim_mutations(ts, rate=cfg.mu, random_seed=seed + 2)
        base = os.path.join(args.config, "region_%03d" % i)
        ts.dump(base + ".trees")
        meta = dict(Ne=cfg.Ne, mutation_rate=cfg.mu, n_samples=cfg.n_dip,
                    sequence_length=int(cfg.seq_len), window_size=2000,
                    contig="chr%d" % (i + 1))
        np.savez(base + ".npz", map_position=rm.position, map_rate=rm.rate,
                 meta=json.dumps(meta))
        gm = ts.genotype_matrix().T.astype(np.int8)
        pos = ts.tables.sites.position
        nseg = _write_vcf(base + ".vcf", "chr%d" % (i + 1), gm, pos, cfg.seq_len)
        print("region %d: %d sites (%d seg) demo=%s" % (i, ts.num_sites, nseg, cfg.demography))
    with open(os.path.join(args.config, "genome.bed"), "w") as fh:
        for i in range(cfg.n_regions):
            fh.write("chr%d\t0\t%d\n" % (i + 1, cfg.seq_len))
    print("done config %s (%d regions)" % (cfg.name, cfg.n_regions))


def _popsizes(demo, Ne):
    return {"constant": [Ne], "bottleneck": [Ne, Ne / 10, Ne],
            "expansion": [Ne, Ne / 10], "realmap": [Ne]}.get(demo, [Ne])


def _epochtimes(demo):
    return {"bottleneck": [1000, 3000], "expansion": [2000]}.get(demo, [])


def _sim_realmap(cfg: Config, seed: int, i: int):
    """HomSap under a REAL genetic map (deCODE/HapMapII): slice a real chromosome
    region's RateMap and simulate with plain msprime. True map = the sliced map."""
    import stdpopsim, msprime
    sp = stdpopsim.get_species(cfg.species)
    gmap = sp.get_genetic_map(cfg.genetic_map)
    rng = np.random.default_rng(seed)
    autos = [c for c in sp.genome.chromosomes
             if c.id not in ("X", "Y", "Z", "W", "MT", "Mt", "Pt", "M")
             and "scaffold" not in c.id.lower()]
    for attempt in range(200):
        try:
            c = autos[rng.integers(len(autos))]
            rm = gmap.get_chromosome_map(c.id)
            L = float(np.asarray(rm.position)[-1])
            if L <= cfg.seq_len:
                continue
            lo = float(rng.uniform(0, L - cfg.seq_len))
            sub = rm.slice(left=lo, right=lo + cfg.seq_len, trim=True)   # -> [0, seq_len]
            pos = np.asarray(sub.position, float)
            rate = np.where(np.isfinite(sub.rate), sub.rate, 0.0)
            # require coverage with real recombination (skip all-missing/all-zero segments)
            if np.diff(pos).sum() <= 0 or np.average(rate, weights=np.diff(pos)) <= 1e-10:
                continue
            clean = msprime.RateMap(position=pos, rate=rate)
            ts = msprime.sim_ancestry(samples=cfg.n_dip, population_size=cfg.Ne,
                                      recombination_rate=clean, sequence_length=pos[-1],
                                      random_seed=seed + attempt + 1)
            ts = msprime.sim_mutations(ts, rate=cfg.mu, random_seed=seed + 2)
            if ts.num_sites >= 50:
                return ts, clean
        except Exception:
            continue
    raise RuntimeError("no valid real-map segment after 200 tries")


# ---------------------------------------------------------------------------
# Scoring grid helpers (per-config seq_len)
# ---------------------------------------------------------------------------

def _edges(seq_len):
    return np.append(np.arange(0, seq_len, GRID), seq_len)


def _resample(positions, rates, seq_len):
    from fastrho.preprocess import mean_rate_between
    pos = np.asarray(positions, float); rr = np.asarray(rates, float)
    if pos[0] > 0:
        pos = np.r_[0.0, pos]; rr = np.r_[rr[0], rr]
    if pos[-1] < seq_len:
        pos = np.r_[pos, seq_len]; rr = np.r_[rr, rr[-1]]
    return mean_rate_between(pos, rr, _edges(seq_len))


# ---------------------------------------------------------------------------
# fastrho prediction
# ---------------------------------------------------------------------------

def cmd_fastrho(args):
    import tskit, time
    from fastrho.translate import load_model, predict_map_from_ts
    cfg = load_config(args.config)
    total_start = time.perf_counter()
    model, mcfg, stats = load_model(args.checkpoint, args.stats, device=args.device)
    model_load_wall = time.perf_counter() - total_start
    out = {}; prediction_start = time.perf_counter()
    for tp in sorted(glob.glob(os.path.join(args.config, "region_*.trees"))):
        name = os.path.basename(tp)[:-6]
        ts = tskit.load(tp)
        pred = predict_map_from_ts(ts, model, mcfg, stats, mutation_rate=cfg.mu,
                                   Ne=cfg.Ne, device=args.device)
        bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
        out[name] = _resample(bp, pred["r_per_bp"], cfg.seq_len)
    name = getattr(args, "save_as", "fastrho") or "fastrho"
    prediction_wall = time.perf_counter() - prediction_start
    total_wall = time.perf_counter() - total_start
    np.savez(os.path.join(args.config, "pred_%s.npz" % name),
             _wall=total_wall, _model_load_wall=model_load_wall,
             _prediction_wall=prediction_wall, **out)
    print("%s: %d regions in %.1fs (load %.1fs, predict %.1fs) -> pred_%s.npz" %
          (name, len(out), total_wall, model_load_wall, prediction_wall, name))


# ---------------------------------------------------------------------------
# ingest competitor outputs
# ---------------------------------------------------------------------------

def cmd_ingest(args):
    cfg = load_config(args.config)
    out = {}
    if args.kind == "pyrho":
        for rmap in sorted(glob.glob(os.path.join(args.config, "region_*.rmap"))):
            name = os.path.basename(rmap)[:-5]
            rows = np.loadtxt(rmap, ndmin=2)
            if rows.size == 0:
                continue
            out[name] = _resample(np.r_[rows[0, -3], rows[:, -2]], rows[:, -1], cfg.seq_len)
    elif args.kind == "relernn":
        import csv, re
        by = {}
        with open(args.predict) as fh:
            rdr = csv.reader(fh, delimiter="\t"); next(rdr)
            for row in rdr:
                if row:
                    by.setdefault(row[0], []).append((float(row[1]), float(row[2]), float(row[-1])))
        for chrom, recs in by.items():
            i = int(re.search(r"chr(\d+)", chrom).group(1)) - 1
            recs.sort()
            s = np.array([r[0] for r in recs]); e = np.array([r[1] for r in recs])
            rt = np.array([r[2] for r in recs])
            out["region_%03d" % i] = _resample(np.r_[s[0], e], rt, cfg.seq_len)
    np.savez(os.path.join(args.config, "pred_%s.npz" % args.kind), **out)
    print("ingested %s: %d regions" % (args.kind, len(out)))


# ---------------------------------------------------------------------------
# scoring -> JSON
# ---------------------------------------------------------------------------

def _block_mean(x, f):
    if f <= 1:
        return x
    n = (len(x) // f) * f
    return x[:n].reshape(-1, f).mean(1)


def cmd_score(args):
    from fastrho.evaluate import score_rates
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        average_precision_score = None
    cfg = load_config(args.config)
    regions = sorted(os.path.basename(p)[:-4]
                     for p in glob.glob(os.path.join(args.config, "region_*.npz")))
    # true windowed rate per region
    from fastrho.preprocess import mean_rate_between
    truth = {}
    for n in regions:
        z = np.load(os.path.join(args.config, n + ".npz"), allow_pickle=True)
        truth[n] = mean_rate_between(z["map_position"], z["map_rate"], _edges(cfg.seq_len))

    methods = {k: np.load(os.path.join(args.config, "pred_%s.npz" % k), allow_pickle=True)
               for k in args.methods
               if os.path.exists(os.path.join(args.config, "pred_%s.npz" % k))}
    common = set(truth)
    for k, mp in methods.items():
        common &= set(n for n in mp.files if not n.startswith("_"))
    common = sorted(common)

    result = {"config": cfg.name, "demography": cfg.demography, "n_hap": 2 * cfg.n_dip,
              "Ne": cfg.Ne, "n_regions_scored": len(common), "scales": {}}
    for k, mp in methods.items():
        if "_wall" in mp.files:
            result.setdefault("wall_clock_s", {})[k] = float(mp["_wall"])
    for grid in args.grids:
        f = max(1, grid // GRID)
        sk = "%dkb" % (grid // 1000)
        result["scales"][sk] = {}
        for k, mp in methods.items():
            P, T, ht, hs = [], [], [], []
            for n in common:
                m = min(len(mp[n]), len(truth[n]))
                pr = _block_mean(mp[n][:m], f); tr = _block_mean(truth[n][:m], f)
                P.append(pr); T.append(tr)
                ht.append((tr > 2 * np.median(tr)).astype(int)); hs.append(pr)
            s = score_rates(np.concatenate(P), np.concatenate(T))
            if average_precision_score is not None and grid <= 100_000:
                yt = np.concatenate(ht)
                if yt.sum() > 0:
                    s["hotspot_auprc"] = float(average_precision_score(yt, np.concatenate(hs)))
            result["scales"][sk][k] = s
    os.makedirs(args.results, exist_ok=True)
    outp = os.path.join(args.results, cfg.name + ".json")
    with open(outp, "w") as fh:
        json.dump(result, fh, indent=2)
    print("scored %s (%d regions) -> %s" % (cfg.name, len(common), outp))
    for sk, md in result["scales"].items():
        row = "  %6s " % sk + " ".join(
            "%s=%.3f" % (k, md[k].get("pearson", float("nan"))) for k in md)
        print(row)


def main():
    ap = argparse.ArgumentParser(description="fastrho benchmark campaign engine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("--config", required=True)
    g.add_argument("--demography", default="constant")
    g.add_argument("--n-dip", type=int, default=10); g.add_argument("--mu", type=float, default=1.5e-8)
    g.add_argument("--Ne", type=float, default=10000.0); g.add_argument("--seq-len", type=int, default=2000000)
    g.add_argument("--n-regions", type=int, default=20); g.add_argument("--seed", type=int, default=7)
    g.add_argument("--genetic-map", default=None); g.add_argument("--species", default="HomSap")
    g.add_argument("--relernn-full", action="store_true")
    f = sub.add_parser("fastrho"); f.add_argument("--config", required=True)
    f.add_argument("--checkpoint", required=True); f.add_argument("--stats", required=True)
    f.add_argument("--device", default="cuda:0")
    f.add_argument("--save-as", default="fastrho", help="method name for the output npz")
    ig = sub.add_parser("ingest"); ig.add_argument("--config", required=True)
    ig.add_argument("--kind", required=True, choices=["pyrho", "relernn"])
    ig.add_argument("--predict", default=None)
    s = sub.add_parser("score"); s.add_argument("--config", required=True)
    s.add_argument("--methods", nargs="+", default=["fastrho", "pyrho", "relernn"])
    s.add_argument("--grids", nargs="+", type=int, default=[25000, 100000, 500000])
    s.add_argument("--results", required=True)
    args = ap.parse_args()
    {"gen": cmd_gen, "fastrho": cmd_fastrho, "ingest": cmd_ingest, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
