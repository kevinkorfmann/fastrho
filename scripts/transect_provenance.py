"""Generate the tree-of-life transect DATA-PROVENANCE log: for every species, exactly where the
genotypes came from and how they were accessed. Joins the ingest configs (URL, contig, mode, mu, thin,
population subset), the species metadata, the extracted npz (actual sample/SNP counts), and the
inference/QC results. Emits a human-readable Markdown table + a machine-readable JSON, both of which
regenerate from scratch each run so they stay in sync with the campaign.

Run on sesame: python scripts/transect_provenance.py <realdata_dir> <out.md> <out.json>
"""
import os
import sys
import glob
import json

import numpy as np

CFG_COLS = ["key", "url", "chrom", "start", "end", "mode", "mu", "thin", "max_samples",
            "sample_regex", "map_sp", "map_id", "max_missing", "indexed"]


def read_cfgs(d):
    """key -> config dict, from every *.cfg in the dir."""
    out = {}
    for cf in glob.glob(os.path.join(d, "*.cfg")):
        for ln in open(cf):
            ln = ln.rstrip("\n")
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split("|")
            if len(parts) < 3:
                continue
            row = dict(zip(CFG_COLS, parts + [""] * len(CFG_COLS)))
            row["_cfg"] = os.path.basename(cf)
            out[row["key"]] = row
    return out


def read_meta(d):
    m = {}
    # base + all eva meta
    for f in ([os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "paper", "figdata", "transect_meta.json")]
              + glob.glob(os.path.join(d, "transect_meta*.json"))):
        if os.path.exists(f):
            try:
                m.update(json.load(open(f)))
            except Exception:
                pass
    return m


def source_kind(url):
    if "ebi.ac.uk/pub/databases/eva" in url:
        return "EBI EVA"
    if "ebi.ac.uk/pub/databases/nextgen" in url:
        return "EBI NextGen"
    if "ftp.sra.ebi.ac.uk" in url or "ena" in url.lower():
        return "ENA analysis"
    if "1000genomes" in url:
        return "1000 Genomes"
    if "caendr" in url:
        return "CaeNDR (AWS S3)"
    if "zenodo" in url:
        return "Zenodo"
    if "datadryad" in url:
        return "Dryad (cookie-jar)"
    if "eichlerlab" in url:
        return "GAGP"
    if "solgenomics" in url:
        return "SolGenomics"
    if "gwdguser" in url or "wildmouse" in url:
        return "MPI wild-mouse"
    if url in ("-", "", None):
        return "in-repo anchor"
    return url.split("/")[2] if "//" in url else "?"


def main():
    d = sys.argv[1]; out_md = sys.argv[2]; out_json = sys.argv[3]
    cfgs = read_cfgs(d); meta = read_meta(d)
    keys = sorted(set(list(cfgs) + list(meta)))
    rows = []
    for k in keys:
        c = cfgs.get(k, {}); m = meta.get(k, {})
        hap = os.path.join(d, "hap", f"{k}.npz")
        n_hap = n_snp = None; chrom = c.get("chrom", "")
        if os.path.exists(hap):
            try:
                z = np.load(hap, allow_pickle=True)
                n_hap = int(z["gm"].shape[0]); n_snp = int(z["gm"].shape[1])
                chrom = str(z["chrom"]) if "chrom" in z.files else chrom
            except Exception:
                pass
        tj = os.path.join(d, f"transect_{k}.json"); r = win = None
        if os.path.exists(tj):
            try:
                t = json.load(open(tj)); r = t.get("pearson_vs_map"); win = t.get("windows")
            except Exception:
                pass
        qj = os.path.join(d, f"qc_{k}.json"); repro = None
        if os.path.exists(qj):
            try:
                repro = json.load(open(qj)).get("log_reproducibility")
            except Exception:
                pass
        url = c.get("url", m.get("source", "-"))
        rows.append(dict(
            key=k, common=m.get("common", k), latin=m.get("latin", ""), clade=m.get("clade", ""),
            source=source_kind(url), accession=m.get("source", ""), url=url,
            contig=chrom, mode=c.get("mode", ""), mu=c.get("mu", ""),
            population=(c.get("sample_regex") if c.get("sample_regex") not in ("-", "", None) else "whole panel"),
            n_hap=n_hap, n_dip=(n_hap // 2 if n_hap else m.get("n_dip")), n_snp=n_snp,
            windows=win, validation_map=(c.get("map_id") if c.get("map_id") not in ("-", "", None) else ""),
            pearson_vs_map=r, log_reproducibility=repro))
    json.dump(rows, open(out_json, "w"), indent=1)

    # markdown
    with open(out_md, "w") as f:
        f.write("# Tree-of-life recombination transect — data provenance & access log\n\n")
        f.write("Every species below: a fine-scale recombination map inferred by ONE frozen "
                "domain-randomized model from **real, unphased** population genotypes, one forward pass, "
                "no retraining. pyrho cannot run on any of these unphased inputs. Access method and full "
                "URL given for reproducibility; validation `r` is Pearson vs a published map at 100 kb, "
                "`log-ρ` is blind subsample-reproducibility (novel species).\n\n")
        f.write(f"**{len([r for r in rows if r['n_hap']])} species with extracted genotypes** "
                f"({len([r for r in rows if r['pearson_vs_map'] is not None])} validated vs a published "
                f"map).\n\n")
        f.write("| Species | Latin | Clade | Source | Accession/URL | Contig | n(dip) | SNPs | mode | μ | "
                "Population | Val. r | Blind ρ |\n")
        f.write("|---|---|---|---|---|---|--:|--:|---|---|---|--:|--:|\n")
        for r in sorted(rows, key=lambda x: (x["clade"], x["common"])):
            u = r["url"] if len(r["url"]) < 70 else r["url"][:67] + "…"
            f.write(f"| {r['common']} | *{r['latin']}* | {r['clade']} | {r['source']} | {u} | "
                    f"{r['contig']} | {r['n_dip'] or ''} | {r['n_snp'] or ''} | {r['mode']} | {r['mu']} | "
                    f"{r['population']} | {r['pearson_vs_map'] if r['pearson_vs_map'] is not None else ''} | "
                    f"{r['log_reproducibility'] if r['log_reproducibility'] is not None else ''} |\n")
    print(f"wrote {out_md} + {out_json}: {len(rows)} species ({len([r for r in rows if r['n_hap']])} extracted)")


if __name__ == "__main__":
    main()
