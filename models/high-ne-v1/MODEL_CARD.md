# fastrho high-ne-v1

## Summary

`high-ne-v1` is the phased specialist for very large effective population size and high diversity,
including mosquito and fly regimes. It is the retained broadened-Nₑ checkpoint used for the
Anopheles population atlas and dipteran analyses.

## Intended use

Use this model for phased, ancestrally polarized haplotypes from populations whose effective size
and population-scaled recombination exceed the ordinary base prior. Set `input_mode="phased"`.
The training regime extended to approximately `Ne = 2e6` and mean recombination rates near `5e-8`
per base per generation, using shortened regions to keep coalescent simulation tractable without
artificially lowering the per-base rate.

The checkpoint expects 17 haplotype-LD features and the released `feat_stats.npz`.

## Training record

The retained 15,000-region checkpoint is epoch 37 (global step 5,928). It is identified by the same
checkpoint hash recorded for the paper's Anopheles atlas. Exact artifact hashes are in
`manifest.json`.

## Limitations

This is a qualified research specialist, not a universal mosquito model. Population structure,
inversions, selection, and cohort composition can dominate LD. On the most extreme dipteran tests,
map shape was more reliable than absolute rate calibration; users should validate the genome-wide
level against simulations or an external map and report any calibration. The model requires phased,
polarized input and is not qualified for selfers or severely bottlenecked populations.

## Integrity and license

The release bundle contains the checkpoint, matching statistics, this model card, a machine
manifest, the MIT license, and per-file SHA-256 checksums. Cite the fastrho paper and this versioned
model release when using the weights.
