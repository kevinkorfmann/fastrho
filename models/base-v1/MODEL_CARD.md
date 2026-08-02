# fastrho base-v1

## Summary

`base-v1` is the ordinary-demography specialist for phased, ancestrally polarized haplotypes. It
uses the same bidirectional Mamba encoder-decoder family as the general release, but is trained on a
single haplotype-LD feature view rather than sharing capacity across input representations.

## Intended use

Use this checkpoint for outbred populations whose effective population size, recombination rate,
sample size, and demographic history are reasonably represented by the base training prior. Set
`input_mode="phased"`. Use `domain-randomized-v1` if phase or ancestral polarization is uncertain,
and use a regime specialist for predominant selfing, severe recent bottlenecks, or very high
effective population size.

The checkpoint expects 17 SNP-token features, two conditioning values, and the released
`feat_stats.npz`. The companion archive contains the original numerical standardization arrays plus
release metadata identifying the phased haplotype featurizer. Do not recompute it from a cohort.

## Training record

The retained 15,000-region retrain was selected at epoch 45 (global step 7,176). Its prior covered
constant, sawtooth, island, and bottleneck histories over ordinary population-genetic scales and
multiple sample sizes. The exact checkpoint and statistics hashes are recorded in `manifest.json`.

## Limitations

This is an LD-based historical population map, not a direct meiotic crossover map. Population
structure, recent selection, inversions, gene conversion, relatedness, missingness, ascertainment,
and demographic misspecification can distort map shape or absolute scale. The model is not qualified
for unphased or unpolarized input.

## Integrity and license

The release bundle contains the checkpoint, matching statistics, this model card, a machine
manifest, the MIT license, and per-file SHA-256 checksums. Cite the fastrho paper and this versioned
model release when using the weights.
