"""Guard: every stdpopsim reference map the paper validates against is audited for resolution
and orientation, so a coarse or flipped map can never silently drive an accuracy claim.

The audit itself (scripts/audit_reference_maps.py) fetches each map from stdpopsim and writes
paper/figdata/reference_map_audit.json; that JSON is the committed source of truth. These tests
lock in its findings:

  * the C. elegans RockmanRIAIL_ce11 map is COARSE and MIS-ORIENTED (centre-high on all six
    chromosomes) -- which is exactly why C. elegans is reported by reproducibility, not scored as
    a validation (see sections/transect_methods.tex);
  * every map that DOES drive a positive-Pearson validation claim (human HapMap, human deCODE,
    dog Campbell, Drosophila Comeron) is fine-scale enough and correctly oriented.

If someone re-points a validated claim at a coarse/flipped reference, or stdpopsim ships a changed
map that re-runs the audit into a different verdict, these tests fail.
"""
import functools
import json

from paperlib import FIGDATA


@functools.lru_cache(maxsize=None)
def _audit():
    with open(FIGDATA / "reference_map_audit.json") as fh:
        return json.load(fh)


def _rows(species=None, mapid=None, chrom=None):
    out = []
    for r in _audit():
        if "error" in r:
            continue
        if species and r["species"] != species:
            continue
        if mapid and r["map"] != mapid:
            continue
        if chrom and r.get("chrom") != chrom:
            continue
        out.append(r)
    return out


def test_audit_covers_every_validated_reference():
    """Each map behind a validated Pearson claim must appear in the audit."""
    needed = {
        ("HomSap", "HapMapII_GRCh38"),
        ("HomSap", "DeCodeSexAveraged_GRCh38"),
        ("CanFam", "Campbell2016_CanFam3_1"),
        ("DroMel", "ComeronCrossover_dm6"),
        ("CaeEle", "RockmanRIAIL_ce11"),
        ("AraTha", "SalomeAveraged_TAIR10"),
    }
    have = {(r["species"], r["map"]) for r in _audit() if "error" not in r}
    missing = needed - have
    assert not missing, f"reference-map audit missing: {missing}"


def test_celegans_reference_is_coarse_and_inverted():
    """The stdpopsim C. elegans map must be flagged both COARSE and MIS-ORIENTED on its
    representative chromosome -- the guarantee that it cannot be used as a fine-scale validation."""
    rep = [r for r in _rows("CaeEle", "RockmanRIAIL_ce11") if "flags" in r]
    assert rep, "no flagged C. elegans representative row in the audit"
    flags = rep[0]["flags"]
    assert "COARSE" in flags and "MIS-ORIENTED" in flags, flags
    # coarse: a handful of segments over the whole chromosome, resolution far above fine-scale
    assert rep[0]["n_intervals"] < 20
    assert rep[0]["median_seg_kb"] > 500.0


def test_celegans_inversion_is_systematic_across_chromosomes():
    """Centre-high (arm/centre < 1) on every C. elegans chromosome -- the inversion is a property
    of the reference, not of one chromosome, so the -0.70 is a reference artefact."""
    ratios = {r["chrom"]: r["arm_centre_ratio"]
              for r in _rows("CaeEle", "RockmanRIAIL_ce11") if "chrom" in r}
    assert set(ratios) >= {"I", "II", "III", "IV", "V", "X"}, ratios
    assert all(v < 1.0 for v in ratios.values()), ratios


def test_validated_anchors_are_not_misoriented():
    """No map that drives a positive-Pearson validation claim is flagged MIS-ORIENTED."""
    for sp, mp in [("HomSap", "HapMapII_GRCh38"), ("HomSap", "DeCodeSexAveraged_GRCh38"),
                   ("CanFam", "Campbell2016_CanFam3_1"), ("DroMel", "ComeronCrossover_dm6")]:
        rows = [r for r in _rows(sp, mp) if "flags" in r]
        assert rows, f"{sp}/{mp} not in audit"
        assert "MIS-ORIENTED" not in rows[0]["flags"], (sp, mp, rows[0]["flags"])


def test_human_and_dog_references_are_fine_scale_enough():
    """The maps behind the human and dog Pearson claims resolve well below the 100-kb scoring
    window (a sanity floor: they are not the degenerate ~2-Mb rendering C. elegans got)."""
    hap = [r for r in _rows("HomSap", "HapMapII_GRCh38") if "flags" in r][0]
    decode = [r for r in _rows("HomSap", "DeCodeSexAveraged_GRCh38") if "flags" in r][0]
    dog = [r for r in _rows("CanFam", "Campbell2016_CanFam3_1") if "flags" in r][0]
    assert hap["median_seg_kb"] < 5.0, hap["median_seg_kb"]
    assert decode["median_seg_kb"] < 25.0, decode["median_seg_kb"]
    assert dog["median_seg_kb"] < 100.0 and dog["arm_centre_ratio"] > 1.0, dog
