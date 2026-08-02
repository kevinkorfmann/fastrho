# Interpret the output

fastrho estimates the historical recombination signal encoded by population linkage disequilibrium.
The result is a population recombination map that can be compared across genomic regions,
populations, and species; pedigree or gamete data provide complementary evidence about contemporary
crossovers.

## Two rates appear in every result

| Quantity | Meaning | Best use |
|---|---|---|
| `rho_per_bp` | Population-scaled rate $\rho=4N_e r$ | Shape comparisons when $N_e$ is uncertain |
| `r_per_bp` | Absolute rate $r=\rho/(4N_e)$ per bp per generation | Biological scale when $N_e$ is justified |

Convert absolute rate to centimorgans per megabase with:

```python
df["cM_per_Mb"] = df["r_per_bp"] * 1e8
```

If `Ne` is omitted, fastrho uses an auxiliary point estimate. Report `Ne_used` with every
absolute-rate result so its scaling is explicit and reproducible.

## What the interval means

The model predicts a mean and dispersion for $\log\rho$ in the same forward pass. The reported limits:

- express model uncertainty relative to the checkpoint's training distribution;
- are converted to absolute $r$ using the supplied or point-estimated $N_e$;
- quantify uncertainty for each adjacent-SNP interval;
- are most informative when reported with the demographic, mutation-rate, gene-conversion, and
  model assumptions used for inference;
- provide conditional model intervals that can be calibrated with simulations or independent maps.

Intervals highlight local variation in model certainty. For a new species or demographic regime,
dataset-specific simulations or an independent map provide a useful calibration benchmark.

## Biological effects that can change an LD map

- **Demographic history:** bottlenecks can flatten or erase recoverable fine-scale structure.
- **Selfing or inbreeding:** changes effective recombination and the relationship between LD and
  crossover rate.
- **Population structure and admixture:** create LD unrelated to local crossover rate.
- **Inversions:** mixed arrangements create long-range LD and apparent cold regions.
- **Linked selection:** sweeps and background selection can preserve map ordering while shifting
  absolute scale.
- **Gene conversion:** a crossover-only checkpoint cannot identify crossover and non-crossover
  exchange separately; long or frequent conversion can distort both shape and scale.

For these reasons, describe the output as a **population recombination map** unless direct crossover
data independently establish a meiotic interpretation.

## Validate before making a biological claim

Use at least two of the following:

1. infer maps from disjoint sample subsets and compare them on the same windows;
2. recover a feature specified before looking at the result, such as a known inversion or broad
   centromeric pattern;
3. compare with a pedigree, gamete-sequencing, cytological, or published crossover map;
4. compare with a second LD estimator on the same cohort and coordinates;
5. simulate under a plausible dataset-specific regime and score against the exact supplied map.

Split-sample agreement measures **repeatability**, not accuracy: shared bias can reproduce itself.
Windows within a chromosome are also not independent biological replicates.

## A safe result sentence

> Using the declared checkpoint and genotype view, fastrho inferred a population recombination map
> for the specified cohort. Rates were aggregated to fixed physical windows; absolute estimates and
> intervals are conditional on the reported $N_e$. The map was evaluated with the stated
> repeatability and external-validation checks.

For a new organism or regime, treat this guide as a starting protocol rather than a substitute for
dataset-specific qualification.
