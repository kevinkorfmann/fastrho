"""Merge transect species-metadata JSONs into a base file (last wins). Avoids inline-python quoting.
Usage: python scripts/merge_meta.py <base.json> <extra1.json> [extra2.json ...]"""
import os
import sys
import json

base = sys.argv[1]
m = json.load(open(base)) if os.path.exists(base) else {}
for f in sys.argv[2:]:
    if os.path.exists(f):
        try:
            m.update(json.load(open(f)))
        except Exception as e:
            print(f"skip {f}: {e}")
json.dump(m, open(base, "w"), indent=1)
print(f"merged -> {len(m)} species")
