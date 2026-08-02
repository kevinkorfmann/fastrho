# Evaluate on a known msprime map

A known-answer simulation checks the full path from population history and recombination landscape
to genotypes, fastrho inference, aligned windows, and quantitative evaluation.

:::{important}
This example requires a compatible model bundle. `feat_stats.npz` is the checkpoint's unchanged
companion archive, not statistics computed from this simulation. Download and verify both files
with `scripts/fetch_model_release.py`; see {ref}`feat-stats-file`.
:::

## Simulate, infer, and score

The script below simulates 50 diploid samples across a heterogeneous 2 Mb map. It evaluates the
prediction on the same physical windows as the rate map supplied to `msprime` and saves both the
inferred BED file and an evaluation table.

The final plotting block uses Matplotlib (`python -m pip install matplotlib`) and writes a PNG next
to the table.

```python
from pathlib import Path
import numpy as np
import msprime
import fastrho
import matplotlib.pyplot as plt

L = 2_000_000
Ne = 10_000
mu = 1.5e-8
window_size = 50_000
bundle = Path("downloaded-models/domain-randomized-v1")

# A simple known map: background, hotspot, background, warm region.
true_map = msprime.RateMap(
    position=[0, 700_000, 950_000, 1_300_000, L],
    rate=[1e-8, 8e-8, 1e-8, 3e-8],
)

ts = msprime.sim_ancestry(
    samples=50,
    ploidy=2,
    population_size=Ne,
    sequence_length=L,
    recombination_rate=true_map,
    random_seed=17,
)
ts = msprime.sim_mutations(
    ts,
    rate=mu,
    model=msprime.BinaryMutationModel(),
    random_seed=18,
)

model, cfg, stats = fastrho.load_model(
    bundle / "model.ckpt",
    bundle / "feat_stats.npz",
    device="cuda:0",
)
pred = fastrho.predict_map_from_ts(
    ts,
    model,
    cfg,
    stats,
    mutation_rate=mu,
    Ne=Ne,
    device="cuda:0",
    input_mode="phased",
)

# Put prediction and truth on the exact same physical windows.
starts, estimated = fastrho.rebin_to_windows(
    pred,
    window_size=window_size,
    key="r_per_bp",
)
ends = np.append(starts[1:], pred["pos_right"][-1])

def mean_true_rate(left, right):
    mass = true_map.get_cumulative_mass(float(right))
    mass -= true_map.get_cumulative_mass(float(left))
    return mass / (right - left)

truth = np.array(
    [mean_true_rate(left, right) for left, right in zip(starts, ends)]
)

valid = np.isfinite(truth) & np.isfinite(estimated) & (truth > 0) & (estimated > 0)
pearson = np.corrcoef(truth[valid], estimated[valid])[0, 1]
scale_ratio = np.median(estimated[valid] / truth[valid])
log10_rmse = np.sqrt(
    np.mean((np.log10(estimated[valid]) - np.log10(truth[valid])) ** 2)
)

print(f"retained SNPs: {ts.num_sites:,}")
print(f"shape Pearson r: {pearson:.3f}")
print(f"median estimated / true rate: {scale_ratio:.3f}")
print(f"log10 RMSE: {log10_rmse:.3f}")

ts.dump("known-map.trees")
fastrho.write_bed(
    pred,
    "known-map.50kb.bed",
    chrom="sim",
    window_size=window_size,
)
np.savetxt(
    "known-map-evaluation.tsv",
    np.column_stack([starts, ends, truth, estimated]),
    delimiter="\t",
    header="start\tend\ttrue_r_per_bp\testimated_r_per_bp",
    comments="",
)

# Draw the same aligned truth-vs-prediction check for this run.
_, lower = fastrho.rebin_to_windows(pred, window_size=window_size, key="r_ci_lo")
_, upper = fastrho.rebin_to_windows(pred, window_size=window_size, key="r_ci_hi")
x = starts / 1e6

fig, ax = plt.subplots(figsize=(7.2, 3.6))
ax.fill_between(
    x,
    lower * 1e8,
    upper * 1e8,
    step="post",
    color="#8fd0ee",
    alpha=0.30,
    label="Mean conditional limits",
)
ax.step(x, truth * 1e8, where="post", color="#657786", label="Simulated truth")
ax.step(x, estimated * 1e8, where="post", color="#0072b2", label="fastrho")
ax.set(xlabel="Position (Mb)", ylabel="Rate (cM/Mb)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
ax.text(
    0.01,
    0.97,
    f"Pearson r={pearson:.2f}  ·  median ratio={scale_ratio:.2f}  ·  log10 RMSE={log10_rmse:.2f}",
    transform=ax.transAxes,
    va="top",
)
fig.tight_layout()
fig.savefig("known-map-evaluation.png", dpi=180)
```

## Reference output

:::{figure} _static_public/msprime_evaluation.png
:alt: Simulated truth and inferred recombination maps from a held-out msprime benchmark with evaluation metrics.
:class: hero-figure
:width: 760px

Reference output from the repository's committed held-out `msprime` benchmark. Its simulated truth
is a deCODE-derived rate landscape; the code writes the same diagnostic for your chosen map and
checkpoint. Metrics shown here describe this single displayed region.
:::

`BinaryMutationModel` keeps the simulated genotype matrix biallelic. Supplying the known `Ne`
places `r_per_bp` and its interval on the intended absolute scale; omit `Ne` only when the auxiliary
point estimate is itself part of the evaluation.

The plotted limits are span-weighted means of the adjacent-interval limits for visualization; they
are not a simultaneous confidence band for the rebinned map.

## Read the evaluation

| Quantity | What it checks | Better result |
|---|---|---|
| Shape Pearson $r$ | Whether high- and low-recombination windows are recovered in the right places | Closer to 1 |
| Median estimated / true rate | Absolute-rate scaling after conditioning on the supplied $N_e$ | Closer to 1 |
| Log10 RMSE | Combined multiplicative shape and scale error | Closer to 0 |

A single region is a useful installation and plumbing check. For a scientific evaluation, repeat
the workflow across held-out seeds, demographic histories, sample sizes, mutation rates, and map
families. Keep the checkpoint frozen, aggregate metrics across independent regions, and retain the
input `RateMap` used to simulate every replicate.

Change `true_map` or pass an `msprime.Demography` to `sim_ancestry`; the fastrho inference and
window-alignment code stays the same.
