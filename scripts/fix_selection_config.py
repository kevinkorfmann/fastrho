"""Repair a slim_dr condition dir whose config.json carried extra (selection) keys that
bench.py's Config rejects: strip config.json back to Config-compatible keys and write a
selection_params.json sidecar reconstructed from the region meta. Usage: fix_selection_config.py DIR
"""
import sys
import os
import json
import glob

import numpy as np

KNOWN = {"name", "demography", "n_dip", "mu", "Ne", "seq_len", "n_regions",
         "popsizes", "epochtimes", "genetic_map", "species", "relernn_full"}

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
json.dump({k: v for k, v in cfg.items() if k in KNOWN},
          open(os.path.join(d, "config.json"), "w"), indent=2)

reg = sorted(glob.glob(os.path.join(d, "region_*.npz")))[0]
m = json.loads(str(np.load(reg, allow_pickle=True)["meta"]))
sp = {k: m.get(k) for k in ("sweep_s", "sweep_target", "soft_k", "exon_frac", "dfe_mean_s")}
sp["name"] = cfg["name"]
sp["mode"] = cfg["demography"].replace("slim_", "")
json.dump(sp, open(os.path.join(d, "selection_params.json"), "w"), indent=2)
print("fixed", os.path.basename(d), "->", sp)
