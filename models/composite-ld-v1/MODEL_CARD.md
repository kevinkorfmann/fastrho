# fastrho composite-ld-v1

## Summary

`composite-ld-v1` is the single-view specialist for unphased, unpolarized diploid genotypes. It
uses folded composite-LD tokens, making the representation invariant to haplotype phase and allele
orientation.

## Intended use

Use this checkpoint when a cohort is diploid but phase and ancestral polarization are unavailable.
Set `input_mode="unpolarized"`. It is useful for experiments that need peak accuracy in this one
input condition; `domain-randomized-v1` is the safer default when one network must support several
input views.

The checkpoint expects 17 folded composite-LD features and its released `feat_stats.npz`. The
companion archive identifies the `gtf` featurizer explicitly so current fastrho versions reject an
incompatible input mode.

## Training record

The model reused 15,000 base-prior simulations re-featurized as folded genotype dosages. The retained
checkpoint is epoch 44 (global step 7,020). Exact artifact hashes are in `manifest.json`.

## Limitations

Folding discards unfolded site-frequency-spectrum information. The model does not recover signal
that is absent after severe bottlenecks, extreme inbreeding, sparse ascertainment, or mixture of
structured populations. It is not a substitute for the selfing, dog-bottleneck, or high-effective-
population-size specialists.

## Integrity and license

The release bundle contains the checkpoint, matching statistics, this model card, a machine
manifest, the MIT license, and per-file SHA-256 checksums. Cite the fastrho paper and this versioned
model release when using the weights.
