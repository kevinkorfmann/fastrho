"""Bundle the per-chromosome A. thaliana selfer recovery tracks for the Extended Data figure.

Reads maps/athal_c{1..5}.npz (fastrho self2: truth/pred/pearson) and maps/athal_c{1..5}_pyrho.npz
(pyrho: truth/pred/pearson) produced on sesame, and writes one compact
paper/figdata/selfer_chroms.npz so the figure builds locally.

Per chromosome c:
    c{c}_centers   window centres (Mb)         (fastrho windows -- the common x)
    c{c}_truth     Salome/TAIR10 map rate (/bp)
    c{c}_pred      fastrho self2 recovered rate (/bp)
    c{c}_pyrho     pyrho recovered rate (/bp), interpolated onto the fastrho window centres
    c{c}_r         fastrho self2 genome-wide Pearson r vs truth
    c{c}_pyrho_r   pyrho genome-wide Pearson r vs truth
    c{c}_nhap      n haplotypes ; c{c}_nsnp  n SNPs

Run in the fastrho venv on sesame:
    /home/kkor/venvs/fastrho/bin/python bundle_selfer_chroms.py
"""
import os
import numpy as np

MAPS = "/home/kkor/realdata/maps"
OUT = "/home/kkor/realdata/selfer_chroms.npz"


def main():
    out = {}
    for c in ["1", "2", "3", "4", "5"]:
        fr = np.load(f"{MAPS}/athal_c{c}.npz", allow_pickle=True)
        py = np.load(f"{MAPS}/athal_c{c}_pyrho.npz", allow_pickle=True)
        cen = np.asarray(fr["centers"], float)
        truth = np.asarray(fr["truth"], float)
        pred = np.asarray(fr["pred"], float)

        # pyrho onto the fastrho window centres: interpolate over its valid (>0) windows
        pcen = np.asarray(py["centers"], float)
        ppred = np.asarray(py["pred"], float)
        good = np.isfinite(ppred) & (ppred > 0)
        pyi = np.interp(cen, pcen[good], ppred[good], left=np.nan, right=np.nan)

        out[f"c{c}_centers"] = cen
        out[f"c{c}_truth"] = truth
        out[f"c{c}_pred"] = pred
        out[f"c{c}_pyrho"] = pyi
        out[f"c{c}_r"] = float(fr["pearson"])
        out[f"c{c}_pyrho_r"] = float(py["pearson"])
        out[f"c{c}_nhap"] = int(fr["n_hap"])
        out[f"c{c}_nsnp"] = int(fr["n_snp"])
        print(f"chr{c}: fastrho r={float(fr['pearson']):+.3f}  pyrho r={float(py['pearson']):+.3f}"
              f"  ({int(fr['n_hap'])} hap x {int(fr['n_snp'])} SNP, {len(cen)} windows)")

    np.savez(OUT, **out)
    print("wrote", OUT, "with", len(out), "arrays")


if __name__ == "__main__":
    main()
