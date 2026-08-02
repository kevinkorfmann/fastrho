# fastrho domain-randomized-v1

## Summary

`domain-randomized-v1` is the primary general-purpose release. It was trained jointly on aligned
phased-haplotype, unphased-genotype, and folded unphased/unpolarized views. The checkpoint is
intended for compatible cohorts without dataset-specific retraining.

## Model and inputs

- Architecture: bidirectional Mamba encoder-decoder with six encoder and four decoder layers.
- Hidden width: 256; state size: 64; context length: 1,024 SNP intervals.
- Input width: 18 features plus four conditioning values.
- Supported views: `phased`, `unphased`, and `unpolarized`.
- Outputs: mean and dispersion of log population-scaled recombination rate per adjacent-SNP
  interval, plus an auxiliary effective-population-size point estimate.

The matching `feat_stats.npz` is part of the model, not an interchangeable convenience file. The
loader must use the view-specific statistics stored in that archive. Inference users should use the
released file unchanged; it is not recomputed from, or adapted to, the cohort being mapped.

## Training record

The model used seed 0 and a three-view domain-randomized workflow with 15,000 simulated training
regions. The archived checkpoint is epoch 53 (exact recorded validation Pearson correlation
0.8618586659431458; global step 8,424). Its embedded Lightning metadata records batch size 48,
learning rate 0.0004, weight decay 0.1, and Lightning 2.6.5.

The historical simulator started both directories at seed zero, so the 400 validation seeds overlap
the first 400 training seeds. That score documents checkpoint selection and is not evidence for
generalization. Evaluate performance with independent, regime-matched simulations before use.

## Intended use

Use this model to infer population recombination maps from compatible SNP data when phase or
ancestral polarization may be unavailable. Record the input view, cohort, filtering, mutation
rate, effective population size, reporting scale, checkpoint checksum, and statistics checksum.

## Limitations

The output reflects historical linkage disequilibrium and is not automatically a direct meiotic
crossover map. Demographic misspecification, selfing, population structure, inversions, selection,
gene conversion, relatedness, missingness, and ascertainment can change map shape or absolute
scale. Predominantly selfing, severe-bottleneck, and very high-diversity regimes may require a
qualified specialist model. Absolute rates remain conditional on the supplied or estimated
effective population size.

## Integrity and license

The expected filenames, byte counts, SHA-256 digests, public release links, and frozen training
metadata are in `manifest.json`. Code and model files are released under the repository's MIT
License. Cite the versioned software and model releases when using these weights.
