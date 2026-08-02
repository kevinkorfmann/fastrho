# fastrho selfing-v1

## Summary

`selfing-v1` is the selfing-aware checkpoint used for the paper's Arabidopsis analysis. Its
training simulations rescaled effective recombination and population size under predominant
self-fertilization, with one haplotype sampled per near-homozygous accession.

## Intended use

Use this checkpoint for predominantly selfing populations represented as phased, ancestrally
polarized haplotypes. Set `input_mode="phased"`. Despite the biology, this artifact is not an
unphased/unpolarized genotype model: the public registry previously stated that incorrectly and the
release corrects the contract.

The model expects 17 haplotype-LD features and the released `feat_stats.npz`. In the historical
prior, selfing reduced effective recombination by `1 - F` and effective population size by
`1 + F`, where `F = s / (2 - s)`.

## Training and selection record

This is the earlier 4,000-region `self2` checkpoint, retained at epoch 48 (global step 2,058), not
the nominal 15,000-region retrain. The larger retrain improved simulated validation but reduced
mean real Arabidopsis map recovery from 0.27 to 0.11; the paper therefore kept this frozen checkpoint.
That empirical comparison is part of the selection history and should be considered when evaluating
new species. Exact artifact hashes are in `manifest.json`.

## Limitations

The checkpoint is most directly qualified for highly inbred, Arabidopsis-like panels. Population
structure, background selection, hyperdivergent haplotypes, ascertainment, and disagreement among
meiotic reference maps remain important. Clean-simulation recovery does not guarantee transfer to a
new selfer. Do not use it on ordinary diploid unphased VCFs or treat its output as contemporary
crossover rate without validation.

## Integrity and license

The release bundle contains the checkpoint, matching statistics, this model card, a machine
manifest, the MIT license, and per-file SHA-256 checksums. Cite the fastrho paper and this versioned
model release when using the weights.
