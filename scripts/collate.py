"""Merge per-config result JSONs into one summary.json. Usage: collate.py <results_dir>"""
import json, sys, glob, os

res = {}
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if os.path.basename(f) == "summary.json":
        continue
    res[os.path.basename(f)[:-5]] = json.load(open(f))
print(json.dumps(res, indent=2))
