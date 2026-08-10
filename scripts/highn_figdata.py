"""Assemble paper/figdata/highn.npz for the high-n scaling / rich-featurizer figure.

All numbers are the live results measured this session (see MODEL_MANIFEST on sesame):
  * sample-size crossover on the const_n sim benchmark (base fastrho vs pyrho, 25 kb)
  * real 1000G high-coverage GRCh38 CEU recovery vs the deCODE pedigree map (100 kb),
    12 loci: base vs the rich_ld+noise fix vs pyrho
  * SLC24A5 sweep-zoom tracks (deCODE / base / rich / pyrho + diversity)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH = ("/private/tmp/claude-501/-Users-kevinkorfmann-Desktop-lunch-money/"
           "93b512a7-4195-439f-ae37-17fba416df7c/scratchpad/zoom")
OUT = os.path.join(HERE, "paper", "figdata", "highn.npz")

# --- sample-size crossover on the const_n benchmark (25 kb Pearson, pooled) ---------------
ncross_n = np.array([20, 40, 100])
ncross_base = np.array([0.882, 0.892, 0.941])
ncross_pyrho = np.array([0.748, 0.844, 0.949])

# --- real-data recovery vs deCODE (100 kb Pearson), base / rich / pyrho -------------------
loci = ["SLC24A5", "IBD5", "FADS", "TLR", "10q22", "LCT",
        "1q23", "6q21", "8p21", "11p14", "14q21", "20p11"]     # 6 test + 6 neutral val
loci_kind = ["sweep", "test", "test", "test", "test", "test",
             "val", "val", "val", "val", "val", "val"]
# order matches `loci`
loci_base = np.array([0.674, 0.751, 0.729, 0.671, 0.760, 0.656,
                      0.625, 0.676, 0.723, 0.820, 0.676, 0.486])
loci_rich = np.array([0.740, 0.923, 0.825, 0.830, 0.807, 0.734,
                      0.785, 0.696, 0.879, 0.845, 0.882, 0.549])
loci_pyrho = np.array([0.553, 0.887, 0.896, 0.906, 0.851, 0.805,
                       0.933, 0.762, 0.882, 0.882, 0.914, 0.511])

# --- SLC24A5 sweep zoom -------------------------------------------------------------------
b = np.load(f"{SCRATCH}/slc24a5_base.npz", allow_pickle=True)
r = np.load(f"{SCRATCH}/slc24a5_noisyrich.npz", allow_pickle=True)
sw = dict(
    sw_centers=np.asarray(b["centers"], float),
    sw_truth=np.asarray(b["truth"], float),        # deCODE pedigree
    sw_base=np.asarray(b["fastrho"], float),        # base fastrho
    sw_pyrho=np.asarray(b["pyrho"], float),
    sw_rich=np.asarray(r["rate"], float),           # rich_ld + noise fix
    sw_pi=np.asarray(b["pi"], float),
    sw_pi_centers=np.asarray(b["pi_centers"], float),
)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, ncross_n=ncross_n, ncross_base=ncross_base, ncross_pyrho=ncross_pyrho,
         loci=np.array(loci), loci_kind=np.array(loci_kind),
         loci_base=loci_base, loci_rich=loci_rich, loci_pyrho=loci_pyrho, **sw)

# quick summary
for tag, m in (("base", loci_base), ("rich", loci_rich), ("pyrho", loci_pyrho)):
    print(f"{tag}: all12={m.mean():.3f}  test={m[:6].mean():.3f}  val={m[6:].mean():.3f}")
print("rich beats base:", int((loci_rich > loci_base).sum()), "/12; "
      "rich beats pyrho:", int((loci_rich > loci_pyrho).sum()), "/12")
print("wrote", OUT)
