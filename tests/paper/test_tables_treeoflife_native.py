"""Structural checks for the tree-of-life pyrho head-to-head table."""

import paperlib as P
import pytest

pytestmark = pytest.mark.consistency


# --------------------------------------------------------------------------
# Tree-of-life (pyrho_headtohead.tex)
# --------------------------------------------------------------------------
def test_treeoflife_seven_species_well_formed():
    rows = P.tree_of_life_rows()
    labels = [r[0] for r in rows]
    assert labels == ["Human", "Orangutan", "Olive baboon", "Dog",
                      "Fruit fly", "Nematode", "Thale cress"]
    for lbl, f, p in rows:
        assert -1.0 <= f <= 1.0 and -1.0 <= p <= 1.0, lbl


def test_real_genotype_rows_well_formed():
    rows = {r[0]: r for r in P.real_genotype_rows()}
    assert {"Human", "Drosophila", "Arabidopsis", "Wolf", "Dog"} <= set(rows)
    # the selfer is the headline: fastrho positive, pyrho negative
    _, f_athal, p_athal = rows["Arabidopsis"]
    assert f_athal > 0 > p_athal, "Arabidopsis selfer: fastrho>0>pyrho expected"
    # unphased wolf/dog: pyrho DOES run in its genotype mode (--ploidy 2), but is less accurate
    # than fastrho on the same genotypes (refutes the old "pyrho cannot run unphased" claim)
    for canid in ("Wolf", "Dog"):
        _, f, p = rows[canid]
        assert p is not None, f"{canid}: pyrho genotype-mode number should be present"
        assert f > p, f"{canid}: fastrho ({f}) should beat pyrho unphased ({p})"


def test_athal_headtohead_matches_independent_selfer_data():
    """De-tautologized guard: the selfer head-to-head cells must equal the INDEPENDENT 5-chromosome
    Salome recovery means committed in selfer_ceiling.json, not merely themselves. This is what
    catches a stale table value (e.g. the old 0.41/-0.40) that no data source backs."""
    import json
    rows = {r[0]: r for r in P.real_genotype_rows()}
    _, f, p = rows["Arabidopsis"]
    d = json.loads((P.FIGDATA / "selfer_ceiling.json").read_text())
    assert round(d["fastrho_real_recovery"]["vs_salome_mean"], 2) == round(f, 2), (f, d["fastrho_real_recovery"]["vs_salome_mean"])
    assert round(d["pyrho_real_recovery"]["vs_salome_mean"], 2) == round(p, 2), (p, d["pyrho_real_recovery"]["vs_salome_mean"])
