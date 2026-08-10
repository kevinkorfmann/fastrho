"""Generate dog training regions with REALISTIC canine demography (village + breed).

The previous dog_gen.py simulated only constant-Ne "village dogs" (no bottleneck). Dogs are
the low-Ne / severe-bottleneck extreme: a single scalar Ne cannot convert population-scaled
rho -> per-base r when Ne(t) varies that strongly (see paper identifiability arc). This
generator draws BOTH regimes over a shared deep trunk:

  * "village": large present Ne, no recent crash -> LD decays over ~kb..tens-of-kb
        present Ne 5k-13k -> domestication floor ~2k (~12-15k gen) -> ancestral wolf 15k-30k
  * "breed":   same deep trunk + a RECENT severe crash (last 30-70 gen, Ne 30-300)
        -> LD extends to ~Mb (the hard, currently-broken case)

Times are in generations (3 yr/gen). Calibrated to dog mu~4e-9 and mean rate ~0.9 cM/Mb
(Campbell2016_CanFam3_1). Hotspots are narrow/promoter-like (dogs carry a PRDM9 pseudogene).

Each region writes ts_{i}.trees + ts_{i}.npz; meta carries:
  * Ne        = LD-EFFECTIVE (recent-weighted harmonic-mean) Ne -- the best single scalar for
                the rho<->r conversion under the bottleneck (P0 heuristic; relocates, does not
                remove, the absolute-scale bias -- the principled fix is the joint Ne(t) head).
  * ne_traj   = the exact piecewise Ne(t) [[t_gen, size], ...] (for the future Ne(t) head).
  * demography ("dog_village"/"dog_breed"), Ne_present, Ne_anc, mode.

The downstream GT (folded composite-LD) featurizer pairs haplotypes into dosages, so phase is
irrelevant. Use long-range disjoint-band radii in preprocessing so the breed LD is visible.

Usage:  python scripts/dog_gen.py <outdir> <n> [offset] [village_frac]
"""
import os, sys, json
import numpy as np
import msprime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastrho.simulate import make_recombination_map, RecombPriors

L = 1_000_000
MAX_RHO = 5000.0          # trunk-aware cap on 4*Ne_max*mean_r*L (ancestral size, not present)


# ---------------------------------------------------------------------------
# Canine Ne(t)
# ---------------------------------------------------------------------------
# IMPORTANT: real dog WGS is DENSE (~2500-8000 SNP/Mb). A deep/long domestication floor
# collapses diversity-Ne and starves the recombination signal (sparse SNPs). So the trunk
# stays high-Ne with only a MILD, brief domestication dip -- this matches real density and
# the diversity regime the model is actually applied to. The recombination identifiability
# signal of the bottleneck lives in the RECENT crash (breed), not in deep diversity.

def _exp_pieces(ne0, ne1, t0, t1, nsteps=14):
    """Fine piecewise-constant approximation of an exponential Ne change t0->t1 (ne0->ne1)."""
    ts = np.linspace(t0, t1, nsteps)
    sizes = ne0 * (ne1 / ne0) ** ((ts - t0) / (t1 - t0))
    return [(float(t), float(s)) for t, s in zip(ts, sizes)]


def _village_traj(rng):
    """High-diversity village trunk: large present Ne, mild brief domestication dip, wolf.
    (Narrow specialist prior; a broadened-shape variant was tried and reverted -- it cost
    in-family accuracy 0.78->0.71 for no out-of-family gain. See fastrho-dog-model memory.)"""
    ne_pres = 10.0 ** rng.uniform(np.log10(1.5e4), np.log10(6.0e4))
    ne_dip = ne_pres / rng.uniform(2.0, 4.0)                       # mild dip (not a deep floor)
    ne_anc = 10.0 ** rng.uniform(np.log10(3.0e4), np.log10(6.0e4))
    return [(0.0, ne_pres), (10000.0, ne_dip), (16000.0, ne_anc)]


def _breed_traj(rng):
    """Same high-diversity trunk + a recent severe crash (Ne 50-500 over last 30-70 gen)."""
    ne_breed = 10.0 ** rng.uniform(np.log10(50.0), np.log10(500.0))
    t_bot = rng.uniform(30.0, 70.0)
    ne_pre = 10.0 ** rng.uniform(np.log10(1.5e4), np.log10(6.0e4))  # pre-breed (village-like)
    ne_dip = ne_pre / rng.uniform(2.0, 4.0)
    ne_anc = 10.0 ** rng.uniform(np.log10(3.0e4), np.log10(6.0e4))
    return [(0.0, ne_breed), (t_bot, ne_pre), (10000.0, ne_dip), (16000.0, ne_anc)]


def _clean_traj(traj):
    """Sort by time and enforce strictly increasing breakpoints (msprime requirement)."""
    traj = sorted(traj, key=lambda x: x[0])
    out = [traj[0]]
    for t, s in traj[1:]:
        if t > out[-1][0] + 1e-6:
            out.append((t, s))
    if out[0][0] != 0.0:
        out = [(0.0, out[0][1])] + out
    return out


def _demography_from_traj(traj):
    traj = _clean_traj(traj)
    d = msprime.Demography()
    d.add_population(initial_size=traj[0][1])
    for t, size in traj[1:]:
        d.add_population_parameters_change(time=t, initial_size=size)
    return d


def _ne_at(traj, t):
    """Piecewise-constant Ne(t): size set at the largest breakpoint <= t."""
    size = traj[0][1]
    for bt, bs in traj:
        if bt <= t:
            size = bs
        else:
            break
    return size


def ld_effective_ne(traj):
    """Recent-weighted harmonic-mean Ne -- the single scalar that best rescales rho->r under
    a bottleneck. Weight w(t)=exp(-t/(2*Ne_present)) concentrates on the recent coalescent
    timescale, so a breed crash dominates (small Ne_eff) while a village stays ~present Ne."""
    ne_pres = traj[0][1]
    tcap = max(8.0 * ne_pres, 800.0)
    ts = np.linspace(0.0, tcap, 4000)
    ne = np.array([_ne_at(traj, t) for t in ts])
    w = np.exp(-ts / (2.0 * ne_pres))
    return float(w.sum() / (w / ne).sum())


# ---------------------------------------------------------------------------
# One region
# ---------------------------------------------------------------------------

def gen_one(seed, village_frac=0.6):
    rng = np.random.default_rng(seed)
    mode = "village" if rng.random() < village_frac else "breed"
    traj = _clean_traj(_village_traj(rng) if mode == "village" else _breed_traj(rng))
    ne_max = max(s for _, s in traj)

    n_dip = int(rng.choice([30, 50, 67]))
    mu = 10.0 ** rng.uniform(np.log10(2e-9), np.log10(6e-9))      # dog ~4e-9
    mean_r = 10.0 ** rng.uniform(np.log10(5e-9), np.log10(5e-8))  # dog mean ~3.3e-8
    rho_total = 4.0 * ne_max * mean_r * L                         # cap on ANCESTRAL size
    if rho_total > MAX_RHO:
        mean_r *= MAX_RHO / rho_total

    priors = RecombPriors(sequence_length=L)
    kind = "hotspot" if rng.random() < 0.5 else "gp"
    rm = make_recombination_map(L, rng, kind=kind, mean_rate=mean_r, priors=priors)

    demo = _demography_from_traj(traj)
    ts = msprime.sim_ancestry(samples=n_dip, demography=demo, recombination_rate=rm,
                              sequence_length=L, random_seed=int(rng.integers(1, 2**31)))
    ts = msprime.sim_mutations(ts, rate=mu, random_seed=int(rng.integers(1, 2**31)))

    ne_eff = ld_effective_ne(traj)
    meta = dict(n_samples=n_dip, mutation_rate=float(mu),
                mean_rate=float(np.average(rm.rate, weights=np.diff(rm.position))),
                Ne=float(ne_eff),                       # LD-effective scalar for rho<->r
                Ne_present=float(traj[0][1]), Ne_anc=float(traj[-1][1]),
                demography="dog_" + mode, mode=mode,
                ne_traj=[[float(t), float(s)] for t, s in traj],
                sequence_length=float(L), window_size=2000, num_sites=int(ts.num_sites))
    return ts, rm, meta


def dump(i, outdir, village_frac):
    base = os.path.join(outdir, "ts_%08d" % i)
    if os.path.exists(base + ".trees"):
        return
    ts, rm, meta = gen_one(2_000 + i, village_frac)
    ts.dump(base + ".trees")
    np.savez(base + ".npz", map_position=np.asarray(rm.position), map_rate=np.asarray(rm.rate),
             meta=json.dumps(meta))


def main():
    outdir = sys.argv[1]
    n = int(sys.argv[2])
    off = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    village_frac = float(sys.argv[4]) if len(sys.argv) > 4 else 0.6
    os.makedirs(outdir, exist_ok=True)
    from multiprocessing import get_context
    with get_context("fork").Pool(40) as pool:
        pool.starmap(dump, [(off + i, outdir, village_frac) for i in range(n)])
    print("done", outdir, n, "village_frac", village_frac)


if __name__ == "__main__":
    main()
