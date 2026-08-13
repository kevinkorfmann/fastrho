"""Verify every stdpopsim genetic map the paper validates against.

Every accuracy claim in the paper is scored against a stdpopsim reference map through one loader
(realdata_infer.truth_windows) with no orientation or resolution guard. This script fetches each
map and reports, per representative chromosome:

  * n_segments / n_distinct_rates  -- how many rate intervals the map actually carries;
  * median/mean segment length     -- the map's true resolution (a coarse pedigree map cannot
                                      validate fine-scale inference no matter the correlation);
  * frac_largest_constant          -- fraction of the chromosome covered by a single rate value;
  * arm_centre_ratio               -- mean rate in the outer 25%+25% vs the middle 50%
                                      (orientation: >1 = arm-high, <1 = centre-high).

It flags a map as COARSE (resolution coarser than ~250 kb, or <50 segments over a whole
chromosome) or MIS-ORIENTED (a species with a known arm-high landscape whose reference is
centre-high). This is how the C. elegans RockmanRIAIL_ce11 artefact was caught (6 segments,
centre-high) and how we confirm the dog Campbell and human deCODE maps that drive validated
claims are not similarly degenerate.

Run (any env with stdpopsim):
  python3.13 scripts/audit_reference_maps.py            # representative chromosome per map
  python3.13 scripts/audit_reference_maps.py --all-chroms   # C. elegans: every chromosome
Writes paper/figdata/reference_map_audit.json
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import stdpopsim

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "paper", "figdata", "reference_map_audit.json")

# (species_id, map_id, representative_chrom, expected_orientation)
# expected_orientation: "arm_high" where the canonical fine-scale landscape concentrates
# recombination on chromosome arms with a suppressed centre (holocentric/large-genome selfers);
# None where no simple arm/centre expectation applies.
MAPS = [
    ("HomSap", "HapMapII_GRCh38",         "1",  None),
    ("HomSap", "DeCodeSexAveraged_GRCh38", "1",  None),
    ("PonAbe", "NaterPA_PonAbe3",         "1",  None),
    ("PapAnu", "Pyrho_PAnubis1_0",        "1",  None),
    ("CanFam", "Campbell2016_CanFam3_1",  "1",  None),
    ("DroMel", "ComeronCrossover_dm6",    "2L", None),
    ("CaeEle", "RockmanRIAIL_ce11",       "I",  "arm_high"),
    ("AraTha", "SalomeAveraged_TAIR10",   "1",  None),
]

COARSE_RES_KB = 250.0   # median segment coarser than this cannot resolve fine-scale structure
MIN_SEGMENTS = 50       # fewer real rate intervals than this over a chromosome = coarse


def audit_chrom(species_id, map_id, chrom):
    sp = stdpopsim.get_species(species_id)
    gm = sp.get_genetic_map(map_id)
    rm = gm.get_chromosome_map(chrom)
    pos = np.asarray(rm.position, float)
    rate = np.asarray(rm.rate, float)
    rate = np.where(np.isfinite(rate), rate, 0.0)
    seg_len = np.diff(pos)                       # bp length of each interval
    span = float(pos[-1] - pos[0])
    # distinct rate values (round to avoid float noise) and their covered length
    key = np.round(rate, 12)
    distinct, inv = np.unique(key, return_inverse=True)
    cover = np.zeros(len(distinct))
    np.add.at(cover, inv, seg_len)
    frac_largest = float(cover.max() / span) if span > 0 else float("nan")
    # arm vs centre (length-weighted mean rate)
    mid = pos[:-1] + seg_len / 2
    x = (mid - pos[0]) / span
    arm = (x < 0.25) | (x > 0.75)
    ctr = (x >= 0.25) & (x <= 0.75)
    def wmean(msk):
        w = seg_len[msk]
        return float((rate[msk] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    arm_r, ctr_r = wmean(arm), wmean(ctr)
    ratio = arm_r / ctr_r if ctr_r and ctr_r > 0 else float("nan")
    return dict(
        chrom=chrom, n_intervals=int(len(seg_len)), n_distinct_rates=int(len(distinct)),
        span_mb=round(span / 1e6, 2),
        median_seg_kb=round(float(np.median(seg_len)) / 1e3, 3),
        mean_seg_kb=round(float(np.mean(seg_len)) / 1e3, 3),
        frac_largest_constant=round(frac_largest, 3),
        arm_rate=arm_r, centre_rate=ctr_r, arm_centre_ratio=round(ratio, 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-chroms", action="store_true",
                    help="also verify every C. elegans chromosome")
    a = ap.parse_args()
    report = []
    for species_id, map_id, chrom, expect in MAPS:
        try:
            m = audit_chrom(species_id, map_id, chrom)
        except Exception as e:
            print(f"[{species_id}/{map_id}] FAILED: {e}", flush=True)
            report.append(dict(species=species_id, map=map_id, error=str(e)))
            continue
        coarse = (m["median_seg_kb"] > COARSE_RES_KB) or (m["n_intervals"] < MIN_SEGMENTS)
        misoriented = (expect == "arm_high" and np.isfinite(m["arm_centre_ratio"])
                       and m["arm_centre_ratio"] < 1.0)
        flags = []
        if coarse: flags.append("COARSE")
        if misoriented: flags.append("MIS-ORIENTED")
        row = dict(species=species_id, map=map_id, expected_orientation=expect,
                   flags=flags, **m)
        report.append(row)
        print(f"{species_id:7s} {map_id:26s} chr{chrom:<3s} "
              f"intervals={m['n_intervals']:>7d} distinct={m['n_distinct_rates']:>6d} "
              f"res(med)={m['median_seg_kb']:>8.2f}kb arm/centre={m['arm_centre_ratio']:>6.2f} "
              f"{'  <<< ' + '+'.join(flags) if flags else ''}", flush=True)
        if a.all_chroms and species_id == "CaeEle":
            for c in ["I", "II", "III", "IV", "V", "X"]:
                mc = audit_chrom(species_id, map_id, c)
                print(f"        {map_id} chr{c:<3s} intervals={mc['n_intervals']:>6d} "
                      f"res(med)={mc['median_seg_kb']:>8.2f}kb arm/centre={mc['arm_centre_ratio']:>6.2f}",
                      flush=True)
                report.append(dict(species=species_id, map=map_id, **mc))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", os.path.relpath(OUT, os.path.join(HERE, "..")))
    flagged = [r for r in report if r.get("flags")]
    if flagged:
        print("FLAGGED:", ", ".join(f"{r['species']}/{r['map']} ({'+'.join(r['flags'])})"
                                    for r in flagged))
    else:
        print("no maps flagged")


if __name__ == "__main__":
    main()
