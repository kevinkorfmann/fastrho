"""Fetch black organism silhouettes from PhyloPic (CC) for the transect species and save them to
paper/figdata/silhouettes/<key>.png. Resolution: lowercase name -> /nodes -> primaryImage -> raster PNG,
falling back species -> genus -> a clade-representative taxon. Idempotent (skips existing).

Run on sesame: python scripts/phylopic_fetch.py <transect_meta.json> <out_dir>
"""
import os
import sys
import json
import time
import urllib.parse
import urllib.request

API = "https://api.phylopic.org"
CLADE_REP = {"Mammals": "mammalia", "Primates": "primates", "Birds": "aves", "Reptiles": "reptilia",
             "Amphibians": "amphibia", "Fish": "actinopterygii", "Molluscs": "mollusca",
             "Arthropods": "arthropoda", "Insects": "insecta", "Nematodes": "nematoda",
             "Cnidaria": "cnidaria", "Plants": "magnoliophyta", "Fungi": "fungi"}


def get(url, raw=False):
    req = urllib.request.Request(url, headers={"User-Agent": "fastrho-treeoflife/1.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read() if raw else json.load(r)


def build():
    return get(f"{API}/images?filter_name=x").get("build")


def _abs(h):
    return h if h.startswith("http") else API + h


def raster_url(name, B):
    q = urllib.parse.quote(name.lower())
    try:
        base = get(f"{API}/nodes?build={B}&filter_name={q}")
        fp = base.get("_links", {}).get("firstPage")
        if not fp:
            return None
        items = get(API + fp["href"]).get("_links", {}).get("items", [])
    except Exception:
        return None
    for it in items[:3]:
        try:
            node = get(_abs(it["href"]))
            pi = node.get("_links", {}).get("primaryImage")
            if not pi:
                continue
            img = get(_abs(pi["href"]))
            rasters = img.get("_links", {}).get("rasterFiles", [])
            if rasters:
                return _abs(rasters[min(len(rasters) - 1, len(rasters) // 2 + 1)]["href"])
        except Exception:
            continue
    return None


def main():
    meta = json.load(open(sys.argv[1])); outd = sys.argv[2]
    os.makedirs(outd, exist_ok=True)
    B = build(); print("build", B)
    ok = miss = 0
    for key, m in meta.items():
        dst = os.path.join(outd, f"{key}.png")
        if os.path.exists(dst):
            ok += 1; continue
        latin = m.get("latin", ""); clade = m.get("clade", "")
        cands = [latin] if latin else []
        if latin and " " in latin:
            cands.append(latin.split()[0])           # genus
        cands.append(CLADE_REP.get(clade, clade))    # clade fallback
        url = None
        for c in cands:
            if not c:
                continue
            url = raster_url(c, B)
            if url:
                break
            time.sleep(0.15)
        if not url:
            print(f"MISS {key} ({latin})"); miss += 1; continue
        try:
            data = get(url, raw=True)
            open(dst, "wb").write(data)
            print(f"OK {key} <- {url.split('/')[-2][:8]}"); ok += 1
        except Exception as e:
            print(f"ERR {key}: {e}"); miss += 1
        time.sleep(0.1)
    print(f"done: {ok} ok, {miss} miss")


if __name__ == "__main__":
    main()
