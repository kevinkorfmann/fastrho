"""Bundle the real-genotype recovery tracks that Figure 5 (fig_identifiability.py) needs.

Reads the per-species recovered maps produced on sesame (truth / fastrho / pyrho, per
position) and writes a single compact ``paper/figdata/realmaps.npz`` so the figure builds
locally without touching the sesame filesystem. Run this on sesame whenever the underlying
maps change:

    PYTHONNOUSERSITE=1 python scripts/make_realmaps_bundle.py

Bundled arrays (per species SP in {athal, human, dmel, dog}):
    <SP>_centers   window centres (Mb)
    <SP>_truth     published/meiotic map rate (/bp)
    <SP>_pred      fastrho recovered rate (/bp)   -- athal is the selfing-aware model
    <SP>_r         genome-wide Pearson r (fastrho vs truth)
    <SP>_pyrho     pyrho recovered rate (/bp)     -- where pyrho can run (not dog)
    <SP>_pyrho_r   genome-wide Pearson r (pyrho vs truth)

Also bundles the four scalars for the dog/wolf identifiability points (Fig 5a; the bottleneck
transfer itself is Extended Data fig_dogtransfer, scripts/fig_dog.py), read from
``gof_ld.json``:
    dog_rawld / wolf_rawld   raw-LD <-> map Pearson r (identifiability in the data itself)
    dog_r     / wolf_r       fastrho recovered Pearson r  (dog_r already bundled above)
"""
import os
import json
import numpy as np

MAPS = os.environ.get("FASTRHO_MAPS", "/home/kkor/realdata/maps")
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_HERE, "paper", "figdata", "realmaps.npz")

# (bundle key, fastrho map file, pyrho map file or None)
SPECIES = [("athal", "athal", "athal_pyrho"),
           ("human", "human", "human_pyrho"),
           ("dmel", "dmel", "dmel_pyrho"),
           ("dog", "dog", None)]


def _load(name):
    return np.load(os.path.join(MAPS, name + ".npz"), allow_pickle=True)


def main():
    out = {}
    for sp, fr, py in SPECIES:
        a = _load(fr)
        out[f"{sp}_centers"] = np.asarray(a["centers"], float)
        out[f"{sp}_truth"] = np.asarray(a["truth"], float)
        out[f"{sp}_pred"] = np.asarray(a["pred"], float)
        out[f"{sp}_r"] = float(a["pearson"])
        if py:
            b = _load(py)
            out[f"{sp}_pyrho"] = np.asarray(b["pred"], float)
            out[f"{sp}_pyrho_r"] = float(b["pearson"])

    # dog/wolf bottleneck panel: raw-LD<->map r (what is identifiable in the data) and the
    # fastrho recovery. wolf_r is taken from gof_ld (fastrho-vs-truth pooled r), matching the text.
    gof = json.load(open(os.path.join(os.path.dirname(MAPS), "gof_ld.json")))
    out["dog_rawld"] = float(gof["dog"]["obs_vs_truth"])
    out["wolf_rawld"] = float(gof["wolf"]["obs_vs_truth"])
    out["wolf_r"] = float(gof["wolf"]["truth_pearson"])

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez(OUT, **out)
    print("wrote", OUT, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
