"""Realistic genome-architecture helpers for the selfing-aware training simulator.

Shared by the coalescent path (scripts/selfing_blend_gen.py) and the SLiM+BGS path
(scripts/selfing_slim_gen.py). Everything is parameterised by (species, map_id, dfe_id,
demography_id) so the SAME machinery transfers from A. thaliana to another selfer (e.g.
C. elegans CaeEle / RockmanRIAIL_ce11) by swapping the config.

Uses REAL stdpopsim data (verified present for AraTha 0.3.0):
  * recombination map : SalomeAveraged_TAIR10  -- sliced, optionally pericentromere-biased
  * exon architecture : araport_11_exons       -- real CDS/exon intervals for SLiM BGS elements
  * DFE               : GammaAdditive_H18       -- Huber et al. 2018 deleterious gamma
  * demography Ne(t)  : SouthMiddleAtlas_1D17 / African{2,3}Epoch_1H18

Selfing / Q scaling matches scripts/mechanism_sim.py:  F = s/(2-s);  Ne_eff = Ne/(1+F);
r_eff = r*(1-F).  For the SLiM forward phase pass the FULL meiotic rate (x Q) + setSelfingRate(s);
for the panmictic recapitation pass EFFECTIVE params (rate x Q x (1-F), sizes /(1+F)/Q, times /Q).
"""
from __future__ import annotations

import numpy as np


# --- A. thaliana default config (swap these for another selfer) --------------
ATHAL = dict(species="AraTha", map_id="SalomeAveraged_TAIR10",
             dfe_id="GammaAdditive_H18", annotation_id="araport_11_exons",
             demography_id="SouthMiddleAtlas_1D17", mu=7.0e-9)


def selfing_F(s: float) -> float:
    """Wright's inbreeding coefficient at equilibrium for selfing rate s."""
    return s / (2.0 - s)


def draw_selfing(rng: np.random.Generator, broad_frac: float = 0.10) -> float:
    """Selfing rate prior: mostly the real A. thaliana regime s~U(0.95,0.99); a `broad_frac`
    tail of U(0.80,0.99) for transfer robustness to other selfers."""
    if rng.random() < broad_frac:
        return float(rng.uniform(0.80, 0.99))
    return float(rng.uniform(0.95, 0.99))


# ---------------------------------------------------------------------------
# Real recombination map: slice a chromosome, optionally over the pericentromere
# ---------------------------------------------------------------------------
def _autosomes(sp):
    skip = ("X", "Y", "Z", "W", "MT", "Mt", "Pt", "M")
    return [c for c in sp.genome.chromosomes
            if c.id not in skip and "scaffold" not in c.id.lower()]


def _trough_center_bp(rm, coarse=200):
    """bp of the deepest recombination trough (pericentromere) on a chromosome RateMap."""
    pos = np.asarray(rm.position, float)
    rate = np.where(np.isfinite(rm.rate), rm.rate, 0.0)
    L = pos[-1]
    edges = np.linspace(0, L, coarse + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    # span-weighted mean rate per coarse window
    binned = np.array([_win_mean(pos, rate, a, b) for a, b in zip(edges[:-1], edges[1:])])
    # smooth a little, then argmin over the interior (avoid telomere edge zeros)
    k = max(1, coarse // 20)
    sm = np.convolve(binned, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    interior = slice(coarse // 8, coarse - coarse // 8)
    j = np.argmin(sm[interior]) + (coarse // 8)
    return float(mids[j])


def _win_mean(pos, rate, a, b):
    lo = np.searchsorted(pos, a); hi = np.searchsorted(pos, b)
    lo = max(1, lo); hi = min(len(pos) - 1, hi)
    if hi <= lo:
        i = min(len(rate) - 1, np.searchsorted(pos, a))
        return float(rate[max(0, i - 1)])
    seg_pos = np.r_[a, pos[lo:hi], b]
    seg_rate = rate[lo - 1:hi]
    w = np.diff(seg_pos)
    return float(np.average(seg_rate, weights=w)) if w.sum() > 0 else 0.0


def load_real_map_slice(sp, map_id, L, rng, peri_bias=0.5, exclude_chrom=None,
                        min_mean_rate=1e-10, tries=200):
    """Slice a length-L region from a real chromosome genetic map.

    With prob `peri_bias`, place the slice to STRADDLE the pericentromeric trough (where the
    selfer signal lives and pyrho fails) while still keeping a mean rate > min_mean_rate (a slice
    sitting entirely in the zero-rate centromere is degenerate). Returns
    (pos, rate, chrom_id, lo) with pos in [0, L] and rate the FULL meiotic rate (per bp)."""
    gmap = sp.get_genetic_map(map_id)
    autos = [c for c in _autosomes(sp) if c.id != str(exclude_chrom)]
    for attempt in range(tries):
        c = autos[int(rng.integers(len(autos)))]
        rm = gmap.get_chromosome_map(c.id)
        clen = float(np.asarray(rm.position)[-1])
        if clen <= L:
            continue
        if rng.random() < peri_bias:
            tc = _trough_center_bp(rm)
            lo = tc - L * float(rng.uniform(0.3, 0.7))
            lo = float(np.clip(lo, 0.0, clen - L))
        else:
            lo = float(rng.uniform(0, clen - L))
        sub = rm.slice(left=lo, right=lo + L, trim=True)
        pos = np.asarray(sub.position, float).copy()
        rate = np.where(np.isfinite(sub.rate), sub.rate, 0.0)
        # msprime discrete_genome requires an INTEGER sequence length; a slice that hits the map's
        # last mapped position (< requested right edge) ends on a non-integer bp. Coerce the endpoint.
        pos[-1] = float(int(round(pos[-1])))
        if len(pos) < 2 or pos[-1] <= pos[-2] or np.diff(pos).sum() <= 0:
            continue
        if np.average(rate, weights=np.diff(pos)) <= min_mean_rate:
            continue
        return pos, rate, c.id, lo
    raise RuntimeError(f"no usable {L}bp slice after {tries} tries (map {map_id})")


# ---------------------------------------------------------------------------
# Real exon architecture (for SLiM BGS genomic elements)
# ---------------------------------------------------------------------------
def exon_intervals_for_slice(sp, annotation_id, chrom_id, lo, L):
    """Real exon [start,end] intervals (0-based, relative to the slice [0,L)) intersecting the
    window [lo, lo+L) on `chrom_id`. Empty if the annotation is unavailable."""
    try:
        ann = sp.get_annotations(annotation_id)
        iv = np.asarray(ann.get_chromosome_annotations(chrom_id), float)
    except Exception:
        return np.zeros((0, 2), int)
    hi = lo + L
    keep = (iv[:, 1] > lo) & (iv[:, 0] < hi)
    iv = iv[keep].copy()
    iv[:, 0] = np.clip(iv[:, 0] - lo, 0, L - 1)
    iv[:, 1] = np.clip(iv[:, 1] - lo, 0, L - 1)
    iv = iv[iv[:, 1] > iv[:, 0]]
    return iv.astype(int)


# ---------------------------------------------------------------------------
# DFE (deleterious gamma) -> SLiM mutation-type args
# ---------------------------------------------------------------------------
def dfe_gamma(sp, dfe_id):
    """(mean_s, shape, del_proportion) for the deleterious gamma component of a stdpopsim DFE.
    mean_s is returned NEGATIVE (deleterious). Falls back to the slim_gen default if unavailable."""
    try:
        dfe = sp.get_dfe(dfe_id)
        props = list(getattr(dfe, "proportions", []))
        for i, mt in enumerate(dfe.mutation_types):
            if getattr(mt, "distribution_type", "f") == "g":
                mean_s = -abs(float(mt.distribution_args[0]))
                shape = float(mt.distribution_args[1])
                dp = float(props[i]) if i < len(props) else 1.0
                return mean_s, shape, dp
    except Exception:
        pass
    return -0.025, 0.20, 1.0   # slim_gen fallback (A.-thaliana-plausible)


# ---------------------------------------------------------------------------
# Demography Ne(t): rescale for selfing (/(1+F)) and SLiM's N'-clock (/Q, times /Q)
# ---------------------------------------------------------------------------
def demography_epochs(sp, demography_id):
    """[(start_time_gen, size), ...] present->past for a 1-population stdpopsim model."""
    dm = sp.get_demographic_model(demography_id)
    epochs = dm.model.debug().epochs
    out = []
    for ep in epochs:
        t = float(ep.start_time)
        if not np.isfinite(t):
            t = out[-1][0] if out else 0.0
        out.append((max(0.0, t), float(ep.populations[0].start_size)))
    return out


def build_demography(sp, demography_id, F=0.0, Q=1.0):
    """msprime.Demography with sizes scaled by 1/((1+F)*Q) and times by 1/Q.
      * coalescent selfer path: F=F, Q=1   (real generations, effective sizes)
      * SLiM recapitation:      F=F, Q=Q   (N'-clock, effective sizes)"""
    import msprime
    scale = 1.0 / ((1.0 + F) * Q)
    eps = demography_epochs(sp, demography_id)
    d = msprime.Demography()
    d.add_population(initial_size=eps[0][1] * scale)
    seen = set()
    for t, size in eps:
        if t <= 0:
            continue
        tt = round(t / Q, 6)
        if tt in seen:
            continue
        seen.add(tt)
        d.add_population_parameters_change(time=tt, initial_size=size * scale)
    return d
