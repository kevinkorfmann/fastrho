"""Variable-recombination-map simulation for fastrho.

Recombination rate varies along each simulated region through an msprime
``RateMap``. Because the map is retained directly, the training target is exact
(no tree traversal needed) -- see ``fastrho.preprocess.windowed_recombination_rate``.

Each simulated region produces:
  * ``ts_{i}.trees``    -- the tree sequence (genotypes + positions live here)
  * ``ts_{i}.npz``      -- the generative RateMap (position, rate) + scalar metadata
                           (Ne, mutation_rate, mean_rate, n_samples, demography, map_kind)

Demography priors implement constant, sawtooth, island, and bottleneck histories locally;
stdpopsim species + empirical genetic maps are reserved mostly for
held-out realistic evaluation (``--stdpopsim``).

Usage:
    python -m fastrho.simulate --data-dir ./sims/train --num-ts 2000 --num-processes 40
    python -m fastrho.simulate --data-dir ./sims/test_real --stdpopsim HomSap \
        --genetic-map HapMapII_GRCh38 --num-ts 200
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass, replace
from functools import partial
from multiprocessing import get_context

import msprime
import numpy as np

# ---------------------------------------------------------------------------
# Priors for the amortized training distribution
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RecombPriors:
    """Broad joint prior over (demography, mutation rate, recombination map, n).

    Wide ranges are deliberate: an amortized estimator must read Ne/diversity from
    the data to make the population-scaled rho -> per-base r conversion (see plan).
    """
    sequence_length: float = 1_000_000.0
    window_size: int = 2000
    map_resolution: int = 500            # bp grid the synthetic map is drawn on

    n_samples_choices: tuple[int, ...] = (10, 20, 50, 100)   # diploid individuals
    # Ranges chosen to stay realistic AND keep rho*L tractable: the upper corner
    # (high Ne x high recombination) makes the coalescent-with-recombination
    # pathologically slow and produces huge SNP counts.
    log10_mu_range: tuple[float, float] = (-8.5, -7.7)       # ~3e-9 .. 2e-8 per bp/gen
    log10_mean_rec_range: tuple[float, float] = (-8.7, -7.7)  # ~2e-9 .. 2e-8 per bp/gen
    log10_Ne_range: tuple[float, float] = (3.5, 4.8)         # ~3k .. 63k

    demographies: tuple[str, ...] = ("constant", "sawtooth", "island", "bottleneck")
    map_kinds: tuple[str, ...] = ("gp", "hotspot", "constant")
    map_kind_weights: tuple[float, ...] = (0.5, 0.4, 0.1)

    # synthetic-map shape priors
    gp_log10_sigma_range: tuple[float, float] = (0.2, 0.9)   # sd of log10 rate field
    gp_phi_range: tuple[float, float] = (0.85, 0.985)        # AR(1) smoothness
    hotspot_count_range: tuple[int, int] = (1, 6)
    hotspot_width_range: tuple[float, float] = (1000.0, 4000.0)
    hotspot_log10_intensity_range: tuple[float, float] = (0.7, 1.5)  # ~5x .. 30x

    rate_clip: tuple[float, float] = (1e-10, 2e-7)
    # hard cap on rho_total = 4*Ne*mean_rate*L: the coalescent-with-recombination is
    # pathologically slow in the high-Ne x high-recombination corner. If a draw exceeds it:
    #   cap_mode="scale_rate" (legacy) -- shrink mean_rate to the cap. SIDE EFFECT: at high Ne
    #       this drives the per-bp rate to ~0, so the model never sees realistic high rho and
    #       systematically under-predicts the absolute rate on real dipteran data (bias<<1).
    #   cap_mode="shorten" -- keep the realistic per-bp rate, shorten the region (down to
    #       min_sequence_length) instead. The model then trains on realistic high per-bp rho at
    #       bounded simulation cost; inference is SNP-chunked so short regions are fine.
    max_rho_total: float = 2000.0
    cap_mode: str = "scale_rate"
    min_sequence_length: float = 8000.0
    # per-cell rho cap (shorten mode only): the mean-rho cap does NOT bound LOCAL peaks, and at
    # high Ne a single hotspot cell at the rate-clip can be a local rho of ~1e6 -> the coalescent
    # explodes there. Clipping local rate so 4*Ne*rate*map_resolution <= max_local_rho keeps the
    # simulation tractable while preserving high BACKGROUND rho (0 disables).
    max_local_rho: float = 0.0
    # above this Ne, force CONSTANT demography (shorten mode): structured/time-varying demographies
    # (island migration, sawtooth) with recombination at high Ne are intractable. The high-Ne
    # Large, approximately panmictic populations can require a broader range. 0=off.
    highne_constant_above: float = 0.0


# ---------------------------------------------------------------------------
# Recombination map generation
# ---------------------------------------------------------------------------

def _ratemap_from_grid(rate_grid: np.ndarray, res: int, L: int) -> msprime.RateMap:
    """Build an msprime RateMap from a per-cell rate array on a uniform `res` grid."""
    n = len(rate_grid)
    pos = np.minimum(np.arange(n + 1, dtype=np.float64) * res, float(L))
    pos[0] = 0.0
    pos[-1] = float(L)
    # guard strict monotonicity at the tail (when L is a multiple of res)
    if pos[-1] <= pos[-2]:
        pos = pos[:-1]
        rate_grid = rate_grid[: len(pos) - 1]
    return msprime.RateMap(position=pos, rate=np.asarray(rate_grid, dtype=np.float64))


def make_recombination_map(
    sequence_length: float,
    rng: np.random.Generator,
    kind: str = "gp",
    mean_rate: float = 1e-8,
    priors: RecombPriors = RecombPriors(),
) -> msprime.RateMap:
    """Sample a spatially-varying recombination map as an msprime ``RateMap``.

    kind:
      * ``constant`` -- flat map at ``mean_rate``.
      * ``gp``       -- log10-rate as a unit-variance AR(1) field around log10(mean_rate),
                        giving smooth hills/valleys (the common case for real maps).
      * ``hotspot``  -- flat background with a few narrow high-rate hotspots.
    """
    L = int(sequence_length)
    res = priors.map_resolution
    n = max(1, int(np.ceil(L / res)))
    lo, hi = priors.rate_clip

    if kind == "constant":
        grid = np.full(n, mean_rate, dtype=np.float64)

    elif kind == "gp":
        sigma = rng.uniform(*priors.gp_log10_sigma_range)
        phi = rng.uniform(*priors.gp_phi_range)
        eps = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = eps[0]
        s = np.sqrt(1.0 - phi * phi)
        for i in range(1, n):  # AR(1); n is small (~2k for 1Mb @ 500bp)
            x[i] = phi * x[i - 1] + s * eps[i]
        x -= x.mean()
        log10_rate = np.log10(mean_rate) + sigma * x
        grid = np.power(10.0, log10_rate)

    elif kind == "hotspot":
        grid = np.full(n, mean_rate, dtype=np.float64)
        k = int(rng.integers(priors.hotspot_count_range[0],
                             priors.hotspot_count_range[1] + 1))
        for _ in range(k):
            center = rng.uniform(0, L)
            width = rng.uniform(*priors.hotspot_width_range)
            intensity = 10.0 ** rng.uniform(*priors.hotspot_log10_intensity_range)
            a = int(max(0, (center - width / 2) // res))
            b = int(min(n, (center + width / 2) // res + 1))
            grid[a:b] *= intensity

    elif kind == "pericentromere":
        # a gp field with a broad multiplicative recombination VALLEY -- the pericentromeric
        # suppression where a selfer's linked-selection signal concentrates and LD methods fail.
        # (Synthetic analogue of a real AraTha pericentromere; used for the map-shape ablation
        # arm so the model isn't fit only to the one real SalomeAveraged map.)
        sigma = rng.uniform(*priors.gp_log10_sigma_range)
        phi = rng.uniform(*priors.gp_phi_range)
        eps = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = eps[0]
        s = np.sqrt(1.0 - phi * phi)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + s * eps[i]
        x -= x.mean()
        grid = np.power(10.0, np.log10(mean_rate) + sigma * x)
        gx = (np.arange(n) + 0.5) / n
        c = rng.uniform(0.35, 0.65)      # trough centre (fraction of L)
        w = rng.uniform(0.12, 0.35)      # trough half-width (fraction of L)
        depth = rng.uniform(0.7, 0.97)   # fractional suppression at the centre
        grid *= (1.0 - depth * np.exp(-0.5 * ((gx - c) / w) ** 2))
    else:
        raise ValueError(f"unknown map kind {kind!r}")

    np.clip(grid, lo, hi, out=grid)
    return _ratemap_from_grid(grid, res, L)


# ---------------------------------------------------------------------------
# Demography
# ---------------------------------------------------------------------------

def _bottleneck_demography(Ne: float, rng: np.random.Generator) -> msprime.Demography:
    d = msprime.Demography()
    d.add_population(initial_size=Ne)
    t_bot = rng.uniform(500, 5000)
    depth = rng.uniform(3, 20)            # fold reduction during bottleneck
    t_rec = t_bot + rng.uniform(500, 5000)
    d.add_population_parameters_change(time=t_bot, initial_size=Ne / depth)
    d.add_population_parameters_change(time=t_rec, initial_size=Ne)
    return d


def _sawtooth_demography(Ne: float = 2e4, magnitude: int = 3) -> msprime.Demography:
    """Return a deterministic sawtooth population-size history."""
    d = msprime.Demography()
    d.add_population(initial_size=Ne)
    scale = magnitude * 1e4
    schedule = [
        (20, 6437.7516497364 / (4 * 1e4), None),
        (30, None, -378.691273513906 / scale),
        (200, None, -643.77516497364 / scale),
        (300, None, 37.8691273513906 / scale),
        (2_000, None, 64.377516497364 / scale),
        (3_000, None, -3.78691273513906 / scale),
        (20_000, None, -6.4377516497364 / scale),
        (30_000, None, 0.378691273513906 / scale),
        (200_000, None, 0.64377516497364 / scale),
        (300_000, None, -0.0378691273513906 / scale),
        (2_000_000, None, -0.064377516497364 / scale),
        (3_000_000, None, 0.00378691273513906 / scale),
        (20_000_000, Ne, 0.0),
    ]
    for time, initial_size, growth_rate in schedule:
        kwargs = {"time": time, "population": None}
        if initial_size is not None:
            kwargs["initial_size"] = initial_size
        if growth_rate is not None:
            kwargs["growth_rate"] = growth_rate
        d.add_population_parameters_change(**kwargs)
    return d


def _build_demography(kind: str, Ne: float, n_samples: int,
                      rng: np.random.Generator):
    """Return (demography_or_None, samples_arg, population_size_or_None)."""
    if kind == "constant":
        return None, n_samples, Ne
    if kind == "sawtooth":
        return _sawtooth_demography(Ne, magnitude=int(rng.integers(2, 5))), n_samples, None
    if kind == "bottleneck":
        return _bottleneck_demography(Ne, rng), n_samples, None
    if kind == "island":
        mig = 10.0 ** rng.uniform(-1.5, -0.3)
        dem = msprime.Demography.island_model([Ne, Ne / 2, Ne / 2], migration_rate=mig)
        n = n_samples
        samples = {0: int(n * 0.6), 1: int(n * 0.2),
                   2: n - int(n * 0.6) - int(n * 0.2)}
        return dem, samples, None
    raise ValueError(f"unknown demography {kind!r}")


# ---------------------------------------------------------------------------
# One region from the amortized prior
# ---------------------------------------------------------------------------

def simulate_region(seed: int, priors: RecombPriors = RecombPriors()):
    """Draw one labeled region from the prior.

    Returns (ts, rate_map, meta) where meta carries the scalars a downstream model
    conditions on / needs for the rho<->r conversion.
    """
    rng = np.random.default_rng(seed)

    n_samples = int(rng.choice(priors.n_samples_choices))
    mu = 10.0 ** rng.uniform(*priors.log10_mu_range)
    mean_rate = 10.0 ** rng.uniform(*priors.log10_mean_rec_range)
    Ne = 10.0 ** rng.uniform(*priors.log10_Ne_range)
    # cap rho_total for tractability. "shorten" keeps the realistic per-bp rate (so the model
    # learns to predict high rho) and trims the region instead; "scale_rate" is the legacy mode.
    L = float(int(priors.sequence_length))
    rho_total = 4.0 * Ne * mean_rate * L
    if rho_total > priors.max_rho_total:
        if priors.cap_mode == "shorten":
            # integer length (msprime discrete_genome requires it); keep the realistic per-bp rate
            L = float(int(max(priors.min_sequence_length,
                              priors.max_rho_total / (4.0 * Ne * mean_rate))))
        else:
            mean_rate *= priors.max_rho_total / rho_total
    if priors.highne_constant_above > 0 and Ne > priors.highne_constant_above:
        demo_kind = "constant"          # structured demographies are intractable here
    else:
        demo_kind = str(rng.choice(priors.demographies))
    map_kind = str(rng.choice(priors.map_kinds,
                              p=np.asarray(priors.map_kind_weights) /
                              np.sum(priors.map_kind_weights)))

    rate_map = make_recombination_map(L, rng,
                                      kind=map_kind, mean_rate=mean_rate, priors=priors)
    # bound LOCAL recombination per cell for tractability (shorten mode): the mean-rho cap above
    # does not stop a single high-Ne hotspot cell from exploding the coalescent.
    if priors.cap_mode == "shorten" and priors.max_local_rho > 0:
        clip = priors.max_local_rho / (4.0 * Ne * priors.map_resolution)
        rate_map = msprime.RateMap(position=rate_map.position,
                                   rate=np.minimum(np.asarray(rate_map.rate, float), clip))
    demo, samples, pop_size = _build_demography(demo_kind, Ne, n_samples, rng)

    anc = dict(samples=samples, recombination_rate=rate_map,
               sequence_length=L,
               random_seed=int(rng.integers(1, 2**31)))
    if demo is not None:
        anc["demography"] = demo
    else:
        anc["population_size"] = pop_size
    ts = msprime.sim_ancestry(**anc)
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))

    meta = dict(seed=seed, n_samples=n_samples, mutation_rate=mu,
                mean_rate=mean_rate, Ne=Ne, demography=demo_kind, map_kind=map_kind,
                sequence_length=float(L),
                window_size=int(priors.window_size),
                num_sites=int(ts.num_sites))
    return ts, rate_map, meta


# ---------------------------------------------------------------------------
# stdpopsim path (realistic demographies + empirical genetic maps; held-out eval)
# ---------------------------------------------------------------------------

def simulate_stdpopsim_region(seed: int, species: str, n_samples: int,
                              sequence_length: float, genetic_map: str | None,
                              window_size: int = 2000):
    """Realistic region from stdpopsim while retaining the exact generative map.

    The previous implementation tried to reconstruct a rate from realized tree
    breakpoints when provenance parsing failed. A realized breakpoint density is
    not a per-generation recombination rate. We now keep the ``Contig`` RateMap
    used by the simulator and fail if it cannot be represented exactly.
    """
    import stdpopsim

    rng = np.random.default_rng(seed)
    seed_value = int(rng.integers(1, 2**31))
    sp = stdpopsim.get_species(species)
    excluded = {
        "Multi-population model of ancient Eurasia",
        "Out-of-Africa with archaic admixture into Papuans",
        "Multi-population model of ancient Europe",
    }
    models = [m for m in sp.demographic_models if m.description not in excluded]
    if not models:
        models = [stdpopsim.PiecewiseConstantSize(10_000)]
    model = models[int(rng.integers(len(models)))]
    populations = [
        p for p in model.populations
        if not (isinstance(getattr(p, "default_sampling_time", None), (int, float))
                and p.default_sampling_time > 0)
        and getattr(p, "allow_samples", True)
    ]
    if not populations:
        raise RuntimeError(f"stdpopsim model for {species} has no present-day sampling population")
    counts = {p.name: 0 for p in populations}
    for p in random.Random(seed_value).choices(populations, k=n_samples):
        counts[p.name] += 1

    autosomes = [
        c for c in sp.genome.chromosomes
        if c.id not in ("Mt", "Pt", "X", "Y", "Z", "W")
        and any(ch.isdigit() for ch in c.id)
    ]
    if not autosomes:
        autosomes = [c for c in sp.genome.chromosomes if c.id not in ("Mt", "Pt")]
    if not autosomes:
        raise RuntimeError(f"stdpopsim species {species} has no usable chromosome")

    contig = None
    for _ in range(100):
        chrom = autosomes[int(rng.integers(len(autosomes)))]
        segment = min(float(sequence_length), float(chrom.length))
        left = float(rng.uniform(0, max(float(chrom.length) - segment, 1.0)))
        right = left + segment
        mutation_rate = float(
            model.mutation_rate or chrom.mutation_rate or sp.genome.mean_mutation_rate
        )
        try:
            if genetic_map:
                source_map = sp.get_genetic_map(genetic_map)
                trimmed = source_map.get_chromosome_map(chrom.id).slice(
                    left, right, trim=True
                )
                candidate = stdpopsim.Contig(
                    recombination_map=trimmed,
                    mutation_rate=mutation_rate,
                    ploidy=chrom.ploidy,
                    genetic_map=source_map,
                    coordinates=(chrom.id, int(left), int(right)),
                )
            else:
                candidate = stdpopsim.Contig.basic_contig(
                    length=segment,
                    mutation_rate=mutation_rate,
                    recombination_rate=float(chrom.recombination_rate),
                    ploidy=chrom.ploidy,
                )
            rates = np.asarray(candidate.recombination_map.rate, dtype=float)
            if rates.size and np.isfinite(rates).all() and np.all(rates >= 0):
                contig = candidate
                break
        except (KeyError, ValueError):
            continue
    if contig is None:
        raise RuntimeError(f"no valid stdpopsim segment found for {species} map={genetic_map!r}")

    rate_map = msprime.RateMap(
        position=np.asarray(contig.recombination_map.position, dtype=float),
        rate=np.asarray(contig.recombination_map.rate, dtype=float),
    )
    ts = stdpopsim.get_engine("msprime").simulate(model, contig, counts, seed=seed_value)
    if not np.isclose(rate_map.sequence_length, ts.sequence_length):
        raise RuntimeError(
            "stdpopsim contig and simulated tree sequence have different lengths; "
            "refusing to create an approximate target"
        )
    mutation_rate = float(getattr(contig, "mutation_rate", None)
                          or model.mutation_rate or sp.genome.mean_mutation_rate)
    meta = dict(seed=seed, n_samples=n_samples, mutation_rate=mutation_rate,
                mean_rate=float(np.average(rate_map.rate,
                                           weights=np.diff(rate_map.position))),
                Ne=None, demography=f"stdpopsim:{species}",
                map_kind="empirical" if genetic_map else "stdpopsim_flat",
                sequence_length=float(ts.sequence_length),
                window_size=int(window_size), num_sites=int(ts.num_sites))
    return ts, rate_map, meta


def _ratemap_from_ts_metadata(ts) -> msprime.RateMap:
    """Best-effort recovery of the generative rate map from ts provenance.

    This compatibility helper only accepts an explicitly serialized RateMap.
    It never estimates a generative rate from realized tree breakpoints.
    """
    try:
        for prov in ts.provenances():
            rec = json.loads(prov.record)
            params = rec.get("parameters", {})
            rm = params.get("recombination_map") or params.get("recombination_rate")
            if isinstance(rm, dict) and "position" in rm and "rate" in rm:
                return msprime.RateMap(position=np.asarray(rm["position"], float),
                                       rate=np.asarray(rm["rate"], float))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    raise ValueError("tree-sequence provenance does not contain an explicit recombination RateMap")


# ---------------------------------------------------------------------------
# Parallel dataset generation
# ---------------------------------------------------------------------------

def _dump_one(i: int, outdir: str, priors: RecombPriors,
              std_kwargs: dict | None) -> str:
    if std_kwargs is None:
        ts, rate_map, meta = simulate_region(i, priors)
    else:
        ts, rate_map, meta = simulate_stdpopsim_region(i, **std_kwargs)
    base = os.path.join(outdir, f"ts_{i:08d}")
    ts.dump(base + ".trees")
    np.savez(base + ".npz",
             map_position=rate_map.position, map_rate=rate_map.rate,
             meta=json.dumps(meta))
    return base


def generate_dataset(num_ts: int, output_dir: str, priors: RecombPriors,
                     num_processes: int = 8, std_kwargs: dict | None = None,
                     seed_offset: int = 0) -> None:
    """Generate ``num_ts`` regions using the explicit half-open seed range.

    Output names contain the simulation seed rather than an array-local counter. This makes
    independent scheduler chunks safe to write into one directory and leaves the complete seed
    schedule visible in the resulting filenames and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)
    from tqdm import tqdm
    worker = partial(_dump_one, outdir=output_dir, priors=priors, std_kwargs=std_kwargs)
    ctx = get_context("fork")
    with ctx.Pool(num_processes) as pool:
        seeds = range(seed_offset, seed_offset + num_ts)
        for _ in tqdm(pool.imap_unordered(worker, seeds),
                      total=num_ts, desc="Simulating"):
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Simulate variable-recombination-map regions for fastrho")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-ts", type=int, default=1000)
    ap.add_argument("--num-processes", type=int, default=8)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first deterministic region seed (filenames use the same seed)")
    ap.add_argument("--sequence-length", type=float, default=1e6)
    ap.add_argument("--window-size", type=int, default=2000)
    # stdpopsim (held-out realistic) path
    ap.add_argument("--stdpopsim", type=str, default=None, help="species code, e.g. HomSap")
    ap.add_argument("--genetic-map", type=str, default=None)
    ap.add_argument("--n-samples", type=int, default=50)
    # Broadened-prior overrides for high-Ne regimes; use short regions.
    ap.add_argument("--log10-ne-max", type=float, default=None,
                    help="raise the upper Ne bound, e.g. 6.3 for approximately 2e6")
    ap.add_argument("--log10-mean-rec-max", type=float, default=None,
                    help="raise the upper mean-rec bound, e.g. -7.3 for ~5e-8 (dipteran rates)")
    ap.add_argument("--max-rho-total", type=float, default=None,
                    help="cap on 4*Ne*meanrate*L for tractability (default 2000)")
    ap.add_argument("--cap-mode", choices=["scale_rate", "shorten"], default=None,
                    help="how to enforce the rho_total cap: 'shorten' keeps realistic per-bp "
                         "rates (correct for high-Ne models), 'scale_rate' is legacy")
    ap.add_argument("--min-sequence-length", type=float, default=None,
                    help="floor on the shortened region length in 'shorten' cap-mode")
    ap.add_argument("--max-local-rho", type=float, default=None,
                    help="per-cell rho cap in 'shorten' mode (bounds local hotspot peaks for "
                         "tractability at high Ne; e.g. 300)")
    ap.add_argument("--highne-constant-above", type=float, default=None,
                    help="force constant demography above this Ne (shorten mode), e.g. 2e5; "
                         "structured demographies at high Ne are intractable with recombination")
    args = ap.parse_args()

    priors = RecombPriors(sequence_length=args.sequence_length, window_size=args.window_size)
    if args.log10_ne_max is not None:
        priors = replace(priors, log10_Ne_range=(priors.log10_Ne_range[0], args.log10_ne_max))
    if args.log10_mean_rec_max is not None:
        priors = replace(priors, log10_mean_rec_range=(priors.log10_mean_rec_range[0],
                                                       args.log10_mean_rec_max))
    if args.max_rho_total is not None:
        priors = replace(priors, max_rho_total=args.max_rho_total)
    if args.cap_mode is not None:
        priors = replace(priors, cap_mode=args.cap_mode)
    if args.min_sequence_length is not None:
        priors = replace(priors, min_sequence_length=args.min_sequence_length)
    if args.max_local_rho is not None:
        priors = replace(priors, max_local_rho=args.max_local_rho)
    if args.highne_constant_above is not None:
        priors = replace(priors, highne_constant_above=args.highne_constant_above)
    std_kwargs = None
    if args.stdpopsim:
        std_kwargs = dict(species=args.stdpopsim, n_samples=args.n_samples,
                          sequence_length=args.sequence_length,
                          genetic_map=args.genetic_map, window_size=args.window_size)
        print(f"stdpopsim={args.stdpopsim} genetic_map={args.genetic_map} "
              f"n={args.n_samples} L={args.sequence_length:.0f}")
    else:
        print(f"synthetic prior: L={args.sequence_length:.0f} window={args.window_size} "
              f"demographies={priors.demographies} map_kinds={priors.map_kinds}")

    generate_dataset(args.num_ts, args.data_dir, priors,
                     num_processes=args.num_processes, std_kwargs=std_kwargs,
                     seed_offset=args.seed_offset)


if __name__ == "__main__":
    main()
