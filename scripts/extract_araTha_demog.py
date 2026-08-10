"""Emit pyrho make_table -p/-t strings for the three stdpopsim A. thaliana demographies.
Writes /home/kkor/realdata/demog_params.sh (sourced by pyrho_demog_check.sh). fastrho venv."""
import warnings; warnings.filterwarnings("ignore")
import stdpopsim

sp = stdpopsim.get_species("AraTha")
out = []
for tag, mid in [("SMA", "SouthMiddleAtlas_1D17"),
                 ("AF3", "African3Epoch_1H18"),
                 ("AF2", "African2Epoch_1H18")]:
    dbg = sp.get_demographic_model(mid).model.debug()
    sizes = [ep.populations[0].start_size for ep in dbg.epochs]
    times = [ep.start_time for ep in dbg.epochs]          # first is 0
    p = ",".join(f"{s:.4f}" for s in sizes)
    t = ",".join(f"{x:.4f}" for x in times[1:])           # -t excludes t=0; len = len(sizes)-1
    out.append(f'{tag}_P="{p}"')
    out.append(f'{tag}_T="{t}"')
    print(f"{tag} {mid}: {len(sizes)} epochs")
    # pyrho make_table is intractable with 33 epochs -> emit a ~6-epoch coarsening of the
    # SouthMiddleAtlas piecewise-constant model that preserves its shape (modern ~79k, the
    # ~247k mid-history peak, and the ~74k ancient floor).
    if tag == "SMA":
        idx = [0, 10, 18, 22, 28, 32]
        cp = ",".join(f"{sizes[i]:.4f}" for i in idx)
        ct = ",".join(f"{times[i]:.4f}" for i in idx[1:])
        out.append(f'SMAC_P="{cp}"')
        out.append(f'SMAC_T="{ct}"')
        print(f"SMAC (coarsened SMA): {len(idx)} epochs")

open("/home/kkor/realdata/demog_params.sh", "w").write("\n".join(out) + "\n")
print("wrote /home/kkor/realdata/demog_params.sh")
