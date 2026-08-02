"""fastrho — fast amortized recombination-map estimation with Mamba SSMs.

A self-contained research package for dense per-interval recombination maps with
single-pass heteroscedastic uncertainty.

Python API
----------
The inference entry points are re-exported here for convenience. The simplest path from a
VCF to a recombination map is a single call::

    import fastrho

    # one call: load a pretrained checkpoint + predict, return a tidy DataFrame
    df = fastrho.quick_map_from_vcf(
        "sample.vcf.gz", "best.ckpt", "feat_stats.npz",
        contig="chr1", as_dataframe=True,
    )

or, keeping the loaded model around to reuse across regions::

    model, cfg, stats = fastrho.load_model("best.ckpt", "feat_stats.npz")
    pred = fastrho.predict_map_from_vcf("sample.vcf.gz", model, cfg, stats, contig="chr1")
    df   = fastrho.to_dataframe(pred)                      # one row per SNP interval
    fastrho.write_bed(pred, "map.bed", chrom="chr1", window_size=50_000)

``predict_map_from_ts`` / ``predict_map_from_genotype_matrix`` are the tree-sequence and
in-memory equivalents. See ``docs/python-api.md`` for complete call patterns and
``docs/your-data.md`` for the input and cohort contract.
"""

from __future__ import annotations

__version__ = "0.1.1"

# The inference code lives in ``fastrho.translate``, which imports torch at module top.
# Re-export everything lazily (PEP 562) so a bare ``import fastrho`` -- used by the docs
# build, the test suite, and any CPU-only tooling -- never imports torch/mamba-ssm.
# Torch-free helpers (VCF reading, tidy output) resolve from ``fastrho.io`` so they keep
# working with no GPU stack installed; the rest resolve from ``fastrho.translate``.
_IO_EXPORTS = (
    "read_vcf",
    "vcf_contigs",
    "to_dataframe",
)
_TRANSLATE_EXPORTS = (
    "load_model",
    "predict_map_from_vcf",
    "predict_map_from_ts",
    "predict_map_from_genotype_matrix",
    "predict_intervals",
    "predict_from_tokens",
    "rebin_to_windows",
    "quick_map_from_vcf",
    "write_bed",
)

__all__ = ["__version__", *_IO_EXPORTS, *_TRANSLATE_EXPORTS]


def __getattr__(name: str):
    if name in _IO_EXPORTS:
        from fastrho import io
        return getattr(io, name)
    if name in _TRANSLATE_EXPORTS:
        from fastrho import translate
        return getattr(translate, name)
    raise AttributeError(f"module 'fastrho' has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
