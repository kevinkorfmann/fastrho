"""Inference for fastrho: genotypes -> per-SNP-interval recombination map.

A trained model + its saved standardization stats turn a genotype matrix (or tree
sequence, or VCF) into a per-interval estimate of population-scaled rho and (given Ne or
the auxiliary Ne head) the absolute per-bp rate r. Rate intervals are conditional on the
supplied or point-estimated Ne.

Long SNP sequences are processed in overlapping context-length chunks and stitched.
"""

from __future__ import annotations

import numpy as np
import torch

from fastrho.features import FeatureConfig, SNPTokenFeaturizer
from fastrho.stitching import combine_gaussian_moments, positive_hann_weights

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(checkpoint: str, stats_path: str, device: str = "cuda:0"):
    """Load a model and its unchanged, release-matched companion archive.

    ``stats_path`` is the ``feat_stats.npz`` shipped with ``checkpoint``. It
    contains training-time feature/target scaling and featurizer metadata; it
    must not be recomputed from the cohort being mapped.
    """
    from fastrho.train import LitFastRho
    lit = LitFastRho.load_from_checkpoint(checkpoint, map_location=device)
    model = lit.model.eval().to(device)
    s = np.load(stats_path)
    stats = {k: s[k] for k in s.files}
    expected = int(np.asarray(stats.get("n_features", model.config.n_features)).item())
    if expected != model.config.n_features:
        raise ValueError(
            f"checkpoint expects {model.config.n_features} features but stats declare {expected}"
        )
    return model, lit.model_config, stats


def _stat_scalar(stats: dict, key: str, default=None):
    if key not in stats:
        return default
    value = np.asarray(stats[key])
    return value.item() if value.ndim == 0 else value.tolist()


def _feature_config_from_stats(stats: dict) -> FeatureConfig:
    """Reconstruct the exact featurizer configuration stored with training stats."""
    kwargs = {}
    if "ld_radii" in stats:
        kwargs["ld_radii"] = tuple(int(x) for x in np.asarray(stats["ld_radii"]).tolist())
    for key in ("neigh_snps", "max_neighbors", "stride_after"):
        if key in stats:
            kwargs[key] = int(_stat_scalar(stats, key))
    for key in ("disjoint_bands", "rich_ld", "r2_debias", "sfs_shape"):
        if key in stats:
            kwargs[key] = bool(_stat_scalar(stats, key))
    return FeatureConfig(**kwargs)


def featurizer_from_metadata(stats: dict, cfg, input_mode: str = "auto"):
    """Return ``(featurizer, selected_stats, cond_extra, resolved_mode)``.

    Domain-randomized archives contain per-view standardization arrays. For a
    single-view archive, the saved ``featurizer_kind`` is authoritative. Legacy
    archives without metadata are treated conservatively as phased haplotype
    models rather than silently applying a different token definition.
    """
    valid = {"auto", "phased", "unphased", "unpolarized", "raw"}
    if input_mode not in valid:
        raise ValueError(f"input_mode must be one of {sorted(valid)}")
    fcfg = _feature_config_from_stats(stats)
    variants = [str(x) for x in np.asarray(stats.get("variants", [])).tolist()]
    if variants:
        if input_mode == "auto":
            raise ValueError("domain-randomized models require a resolved input mode")
        variant = {"phased": "hap", "unphased": "gt", "unpolarized": "gtf"}.get(input_mode)
        if variant not in variants:
            raise ValueError(f"model archive has no {variant!r} view for {input_mode!r} input")
        selected = dr_variant_stats(stats, variant)
        if variant == "hap":
            feat, cond = SNPTokenFeaturizer(fcfg), [0.0, 0.0]
        else:
            from fastrho.gt_features import GTTokenFeaturizer
            feat = GTTokenFeaturizer(fcfg, fold=variant == "gtf")
            cond = [1.0, 1.0 if variant == "gtf" else 0.0]
        return feat, selected, cond, input_mode

    kind = str(_stat_scalar(stats, "featurizer_kind", "raw" if getattr(cfg, "featurizer", "ld") == "raw" else "hap"))
    supported = {
        "hap": "phased", "ld": "phased", "gt": "unphased",
        "gtf": "unpolarized", "raw": "raw",
    }
    resolved = supported.get(kind)
    if resolved is None:
        raise ValueError(f"unknown saved featurizer_kind {kind!r}")
    if input_mode not in ("auto", resolved):
        raise ValueError(
            f"stats were trained with {kind!r} tokens ({resolved} input), not {input_mode!r}"
        )
    if kind == "raw":
        from fastrho.raw_features import RawGenotypeFeaturizer
        feat = RawGenotypeFeaturizer()
    elif kind in ("gt", "gtf"):
        from fastrho.gt_features import GTTokenFeaturizer
        feat = GTTokenFeaturizer(fcfg, fold=kind == "gtf")
    else:
        feat = SNPTokenFeaturizer(fcfg)
    if getattr(feat, "n_features", cfg.n_features) != cfg.n_features:
        raise ValueError(
            "saved featurizer metadata does not match checkpoint input dimension: "
            f"{getattr(feat, 'n_features', '?')} != {cfg.n_features}"
        )
    return feat, stats, None, resolved


# ---------------------------------------------------------------------------
# Core chunked inference over the SNP token sequence
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_from_tokens(model, cfg, stats, tokens_raw, positions, n_hap,
                        mutation_rate, Ne=None, device="cuda:0", overlap=None,
                        cond_extra=None):
    """Chunked inference from raw (unstandardized) SNP tokens -> per-interval map dict.

    cond_extra: optional extra conditioning values appended after [log10 mu, log10 n_hap]
    (e.g. the domain-randomized model's (is_gt, is_folded) view bits); must match the
    model's cond_dim.
    """
    tokens = np.asarray(tokens_raw, np.float32)
    positions = np.asarray(positions, dtype=np.float64)
    S = tokens.shape[0]
    if S < 2:
        raise ValueError("need >=2 SNPs to estimate a recombination map")

    if positions.shape != (S,) or np.any(np.diff(positions) <= 0):
        raise ValueError("positions must match tokens and be strictly increasing")
    feat_mean = np.asarray(stats["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(stats["feat_std"], dtype=np.float32)
    if tokens.ndim != 2 or tokens.shape[1] != feat_mean.size or feat_std.shape != feat_mean.shape:
        raise ValueError("token feature dimension does not match standardization statistics")
    if not np.isfinite(tokens).all() or not np.isfinite(feat_std).all() or np.any(feat_std <= 0):
        raise ValueError("tokens and feature standard deviations must be finite and positive")
    tokens = (tokens - feat_mean) / feat_std
    cvals = [np.log10(mutation_rate), np.log10(n_hap)]
    if cond_extra is not None:
        cvals += [float(x) for x in np.atleast_1d(cond_extra)]
    cond = torch.tensor([cvals], dtype=torch.float32, device=device)

    K = cfg.context_len
    overlap = overlap if overlap is not None else K // 4
    stride = max(1, K - overlap)

    mu_sum = np.zeros(S)
    second_sum = np.zeros(S)
    wsum = np.zeros(S)
    ne_vals = []
    starts = list(range(0, max(1, S - K + 1), stride))
    if starts[-1] != max(0, S - K):
        starts.append(max(0, S - K))
    for st in starts:
        chunk = tokens[st:st + K]
        m = chunk.shape[0]
        xt = torch.from_numpy(chunk[None, ...]).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            out = model(xt, cond, None)
        mu = out["rho"][0, :m, 0].float().cpu().numpy()
        logv = out["rho"][0, :m, 1].float().clamp(-10, 10).cpu().numpy()
        # Positive Hann interior weights downweight edges without ever discarding
        # the first or last token of a single/terminal chunk.
        w = positive_hann_weights(m)
        mu_sum[st:st + m] += mu * w
        second_sum[st:st + m] += (np.exp(logv) + mu * mu) * w
        wsum[st:st + m] += w
        ne_vals.append(float(out["log_Ne"][0].float().cpu()))

    mu_std, var_std = combine_gaussian_moments(mu_sum, second_sum, wsum)

    # de-standardize log rho
    log_rho = mu_std * stats["tgt_std"] + stats["tgt_mean"]
    sigma_log = np.sqrt(var_std) * stats["tgt_std"]

    # Ne: user-supplied or aux head
    ne_std_mean = float(np.average(ne_vals))
    ne_log_sd_chunks = float(np.std(ne_vals) * float(stats["ne_std"]))
    Ne_est = float(np.exp(ne_std_mean * stats["ne_std"] + stats["ne_mean"]))
    Ne_use = Ne if Ne is not None else Ne_est

    rho = np.exp(log_rho)                       # population-scaled, per bp
    r = rho / (4.0 * Ne_use)                    # absolute per-bp rate

    # keep only valid intervals (token i -> interval [pos i, pos i+1]); drop last token
    sel = slice(0, S - 1)
    return {
        "pos_left": positions[:-1].astype(np.float64),
        "pos_right": positions[1:].astype(np.float64),
        "rho_per_bp": rho[sel],
        "r_per_bp": r[sel],
        "log_rho": log_rho[sel],
        "sigma_log_rho": sigma_log[sel],
        "rho_ci_lo": np.exp(log_rho[sel] - 1.96 * sigma_log[sel]),
        "rho_ci_hi": np.exp(log_rho[sel] + 1.96 * sigma_log[sel]),
        "r_ci_lo": np.exp(log_rho[sel] - 1.96 * sigma_log[sel]) / (4.0 * Ne_use),
        "r_ci_hi": np.exp(log_rho[sel] + 1.96 * sigma_log[sel]) / (4.0 * Ne_use),
        "Ne_estimated": Ne_est,
        "Ne_used": Ne_use,
        "Ne_log_sd_between_chunks": ne_log_sd_chunks,
        "r_interval_is_conditional_on_Ne": True,
    }


def predict_intervals(model, cfg, stats, gm, positions, mutation_rate,
                      Ne=None, device="cuda:0", featurizer=None, cond_extra=None,
                      input_mode="auto", n_hap_condition=None):
    """Featurize a genotype matrix then run per-interval inference.

    The featurizer matches the model: raw genotypes for the GRU-seq2seq steelman
    (cfg.featurizer == 'raw'), SNP-token LD features for fastrho otherwise.
    """
    if featurizer is None:
        featurizer, stats, saved_cond, _ = featurizer_from_metadata(stats, cfg, input_mode)
        if cond_extra is None:
            cond_extra = saved_cond
    feats = featurizer(gm, positions.astype(np.float64),
                       {"sequence_length": float(positions[-1] + 1)})
    n_hap = gm.shape[0] if n_hap_condition is None else int(n_hap_condition)
    if n_hap < 2:
        raise ValueError("n_hap_condition must be at least 2")
    return predict_from_tokens(model, cfg, stats, feats["tokens"], positions,
                               n_hap, mutation_rate, Ne=Ne, device=device,
                               cond_extra=cond_extra)


def dr_variant_stats(raw: dict, name: str) -> dict:
    """Pull one variant's standardization stats out of a feat_stats_dr.npz dict.

    `raw` is the loaded npz (per-variant <name>_feat_mean/std + shared tgt/ne stats).
    Returns the standard {feat_mean, feat_std, tgt_*, ne_*} dict predict_* expects.
    """
    return {"feat_mean": np.asarray(raw[f"{name}_feat_mean"], np.float32),
            "feat_std": np.asarray(raw[f"{name}_feat_std"], np.float32),
            "tgt_mean": float(raw["tgt_mean"]), "tgt_std": float(raw["tgt_std"]),
            "ne_mean": float(raw["ne_mean"]), "ne_std": float(raw["ne_std"])}


# ---------------------------------------------------------------------------
# Convenience entry points
# ---------------------------------------------------------------------------

def predict_map_from_genotype_matrix(gm, positions, model, cfg, stats,
                                     mutation_rate=1.5e-8, Ne=None, device="cuda:0",
                                     input_mode="phased", n_hap_condition=None):
    from fastrho.filtering import basic_filtering
    gm, positions = basic_filtering(gm.astype(np.int8), positions.astype(np.float64))
    return predict_intervals(model, cfg, stats, gm, positions, mutation_rate,
                             Ne=Ne, device=device, input_mode=input_mode,
                             n_hap_condition=n_hap_condition)


def predict_map_from_ts(ts, model, cfg, stats, mutation_rate=1.5e-8, Ne=None,
                        device="cuda:0", input_mode="phased"):
    gm = ts.genotype_matrix().T.astype(np.int8)
    positions = ts.tables.sites.position.astype(np.float64)
    return predict_map_from_genotype_matrix(gm, positions, model, cfg, stats,
                                            mutation_rate=mutation_rate, Ne=Ne, device=device,
                                            input_mode=input_mode)


def predict_map_from_vcf(vcf_path, model, cfg, stats, contig=None,
                         mutation_rate=1.5e-8, Ne=None, device="cuda:0",
                         input_mode="auto", missing="drop-site"):
    """Predict a recombination map directly from a phased VCF.

    All sample columns are used. No automatic sample cap or training-range subsampling is applied;
    subset a large VCF to a deliberate biological cohort before calling this function.

    Reads genotypes with :func:`fastrho.io.read_vcf`, which works on both real ACGT VCFs and
    **tskit/msprime numeric-allele (0/1) VCFs** (the kind ``tskit.TreeSequence.write_vcf``
    emits) and needs no extra packages when cyvcf2 is absent. Pass ``contig`` to restrict to
    one chromosome.
    """
    from fastrho.io import read_vcf
    gm, positions, meta = read_vcf(
        vcf_path, contig=contig, missing=missing, return_metadata=True
    )
    if input_mode == "auto":
        input_mode = "phased" if meta["phased"] else "unphased"
    pred = predict_map_from_genotype_matrix(
        gm, positions, model, cfg, stats, mutation_rate=mutation_rate,
        Ne=Ne, device=device, input_mode=input_mode
    )
    pred["contig"] = meta["contig"]
    pred["input_mode"] = input_mode
    pred["coordinate_system"] = "0-based-half-open"
    return pred


# ---------------------------------------------------------------------------
# Rebin per-interval map to a fixed bp grid (BED-friendly map)
# ---------------------------------------------------------------------------

def rebin_to_windows(pred: dict, window_size: int, key: str = "r_per_bp"):
    """Span-weighted mean of a per-interval quantity over fixed bp windows."""
    from fastrho.preprocess import mean_rate_between
    left = pred["pos_left"]
    right = pred["pos_right"]
    rate = pred[key]
    # step function over breakpoints = [left[0], right...]
    bp = np.concatenate([[left[0]], right])
    L = right[-1]
    starts = np.arange(left[0], L, window_size)
    edges = np.append(starts, L)
    binned = mean_rate_between(bp, rate, edges)
    return starts, binned


# ---------------------------------------------------------------------------
# Tidy / one-call convenience for interactive use (notebooks, scripts)
# ---------------------------------------------------------------------------

from fastrho.io import to_dataframe  # noqa: E402  (torch-free tidy-table helper, re-exported)


def quick_map_from_vcf(vcf_path, checkpoint, stats, *, contig=None,
                       mutation_rate=1.5e-8, Ne=None, device="cuda:0",
                       as_dataframe=False, input_mode="auto", missing="drop-site"):
    """One call: load a checkpoint and predict a recombination map from a VCF.

    Equivalent to :func:`load_model` followed by :func:`predict_map_from_vcf`. Returns the
    per-interval prediction dict, or a tidy :func:`to_dataframe` table when
    ``as_dataframe=True``. ``stats`` is the unchanged ``feat_stats.npz`` companion
    shipped with the checkpoint, not statistics computed from ``vcf_path``.
    """
    model, cfg, st = load_model(checkpoint, stats, device=device)
    pred = predict_map_from_vcf(vcf_path, model, cfg, st, contig=contig,
                                mutation_rate=mutation_rate, Ne=Ne, device=device,
                                input_mode=input_mode, missing=missing)
    return to_dataframe(pred, chrom=contig) if as_dataframe else pred


def write_bed(pred: dict, path: str, chrom: str = "chr", window_size: int | None = None):
    if window_size:
        starts, r = rebin_to_windows(pred, window_size, "r_per_bp")
        final_end = int(pred["pos_right"][-1])
        with open(path, "w") as fh:
            fh.write("chrom\tstart\tend\trecomb_rate_per_bp\n")
            for s, val in zip(starts, r):
                fh.write(f"{chrom}\t{int(s)}\t{min(int(s + window_size), final_end)}\t{val:.6e}\n")
    else:
        with open(path, "w") as fh:
            fh.write("chrom\tstart\tend\trho_per_bp\tr_per_bp\tr_ci_lo\tr_ci_hi\n")
            for i in range(len(pred["pos_left"])):
                fh.write(f"{chrom}\t{int(pred['pos_left'][i])}\t{int(pred['pos_right'][i])}\t"
                         f"{pred['rho_per_bp'][i]:.6e}\t{pred['r_per_bp'][i]:.6e}\t"
                         f"{pred['r_ci_lo'][i]:.6e}\t{pred['r_ci_hi'][i]:.6e}\n")
