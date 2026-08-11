"""Force a pyrho config dir to the genome-wide Ne so the shared lookup table is reused.
Usage: python athal_ed_fixcfg.py <config.json> <Ne>
"""
import json, sys
p, ne = sys.argv[1], float(sys.argv[2])
c = json.load(open(p))
c["popsizes"] = [ne]
json.dump(c, open(p, "w"))
print("popsizes ->", c["popsizes"])
