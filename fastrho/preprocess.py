"""Preprocessing for fastrho: turn (tree sequence + generative RateMap) into
per-SNP-token shards (token features, per-interval rate target, positions, metadata).

The recombination target is exact because we know the generative map. The per-interval
target is the span-weighted mean per-bp rate between consecutive SNP positions
(``mean_rate_between``); ``windowed_recombination_rate`` is the same machinery on a fixed
bp grid, kept for output rebinning and ReLERNN-style evaluation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Exact span-weighted rate between arbitrary edges
# ---------------------------------------------------------------------------

def _integral_fn(map_position: np.ndarray, map_rate: np.ndarray):
    """Return f(x) = integral of rate over [0, x] as a vectorized callable."""
    pos = np.asarray(map_position, dtype=np.float64)
    rate = np.asarray(map_rate, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(rate * np.diff(pos))])

    def integral(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), pos[0], pos[-1])
        k = np.clip(np.searchsorted(pos, x, side="right") - 1, 0, len(rate) - 1)
        return cum[k] + rate[k] * (x - pos[k])

    return integral


def mean_rate_between(map_position, map_rate, edges) -> np.ndarray:
    """Span-weighted mean per-bp rate in each interval [edges[k], edges[k+1]].

    Returns array of length len(edges) - 1.
    """
    edges = np.asarray(edges, dtype=np.float64)
    integral = _integral_fn(map_position, map_rate)
    width = np.diff(edges)
    width = np.where(width == 0, 1.0, width)
    return (integral(edges[1:]) - integral(edges[:-1])) / width


def windowed_recombination_rate(map_position, map_rate, window_size, sequence_length):
    """Span-weighted mean rate per fixed-width window (for output/eval)."""
    L = float(sequence_length)
    starts = np.arange(0, L, window_size, dtype=np.float64)
    edges = np.append(starts, L)
    return mean_rate_between(map_position, map_rate, edges)


# ---------------------------------------------------------------------------
# Genotype matrix
# ---------------------------------------------------------------------------

def genotype_matrix(ts):
    """Return (gm, positions): gm (n_samples, n_sites) int8 0/1, biallelic, unfixed."""
    from fastrho.filtering import basic_filtering
    gm = ts.genotype_matrix().T.astype(np.int8)
    positions = ts.tables.sites.position.astype(np.float64)
    return basic_filtering(gm, positions)


# ---------------------------------------------------------------------------
# Process one region
# ---------------------------------------------------------------------------

def process_one(base_path: str, featurizer=None):
    """Load region; return dict with tokens/positions/interval_target/y_window/meta.

    If featurizer is None, tokens are omitted (Phase-1 raw mode: gm/positions kept).
    """
    import tskit
    ts = tskit.load(base_path + ".trees")
    npz = np.load(base_path + ".npz", allow_pickle=True)
    meta = json.loads(str(npz["meta"]))
    map_pos, map_rate = npz["map_position"], npz["map_rate"]

    gm, positions = genotype_matrix(ts)
    # fixed-grid target for eval / output rebinning
    y_window = windowed_recombination_rate(
        map_pos, map_rate, int(meta["window_size"]), float(meta["sequence_length"]))

    out = {"positions": positions, "y_window": y_window.astype(np.float32),
           "meta": meta, "gm": gm}
    if featurizer is not None:
        feats = featurizer(gm, positions, meta)
        out["tokens"] = feats["tokens"]                       # (S, F), maybe (0, F)
        # per-interval target aligned to tokens 0..S-2 (interval to the right)
        if len(positions) >= 2:
            out["interval_target"] = mean_rate_between(
                map_pos, map_rate, positions).astype(np.float32)
        else:
            out["interval_target"] = np.zeros(0, np.float32)
    return out


def _process_and_save(base_path: str, out_dir: str, featurizer) -> str:
    r = process_one(base_path, featurizer)
    out = os.path.join(out_dir, os.path.basename(base_path))
    meta = dict(r["meta"])
    if featurizer is not None:
        from dataclasses import asdict, is_dataclass
        cfg = getattr(featurizer, "cfg", None)
        meta["featurizer"] = {
            "kind": ("gtf" if featurizer.__class__.__name__ == "GTTokenFeaturizer"
                     and getattr(featurizer, "fold", False)
                     else "gt" if featurizer.__class__.__name__ == "GTTokenFeaturizer"
                     else "raw" if featurizer.__class__.__name__ == "RawGenotypeFeaturizer"
                     else "hap"),
            "config": asdict(cfg) if cfg is not None and is_dataclass(cfg) else {},
        }
    save = dict(positions=r["positions"].astype(np.float64),
                y_window=r["y_window"], meta=json.dumps(meta))
    if "tokens" in r:
        save["tokens"] = r["tokens"].astype(np.float32)
        save["interval_target"] = r["interval_target"]
    else:
        save["gm"] = r["gm"]
    np.savez(out + ".npz", **save)
    return out


def main():
    ap = argparse.ArgumentParser(description="Preprocess fastrho simulations into shards")
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--with-features", action="store_true",
                    help="compute SNP-token LD features (fastrho)")
    ap.add_argument("--raw", action="store_true",
                    help="compute raw-genotype per-SNP tokens (ReLERNN-seq2seq steelman)")
    ap.add_argument("--gt", action="store_true",
                    help="phase-invariant composite-LD (dosage) tokens")
    ap.add_argument("--gt-fold", action="store_true",
                    help="phase- AND polarization-invariant composite-LD tokens (minor-allele)")
    ap.add_argument("--radii", default=None,
                    help="comma-separated LD radii in bp (override defaults; for short-LD regimes)")
    ap.add_argument("--disjoint-bands", action="store_true",
                    help="assign each LD pair to a single distance band (long-range/bottleneck)")
    ap.add_argument("--stride-after", type=int, default=0,
                    help="geometric neighbour subsampling: double the index step every N "
                         "comparisons so a bounded budget reaches the widest radius (e.g. 64)")
    ap.add_argument("--max-neighbors", type=int, default=None,
                    help="override max LD comparisons per direction (default 200)")
    ap.add_argument("--sfs-shape", action="store_true",
                    help="append the polarization-invariant local_rare_frac token (18-dim)")
    ap.add_argument("--r2-debias", action="store_true",
                    help="subtract the 1/n_hap finite-sample floor from r^2 (de-noise LD)")
    ap.add_argument("--num-processes", type=int, default=8)
    args = ap.parse_args()

    modes = sum(bool(x) for x in (args.with_features, args.raw, args.gt, args.gt_fold))
    if modes > 1:
        ap.error("choose only one of --with-features, --raw, --gt, or --gt-fold")

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = ".trees"
    bases = sorted(p[: -len(suffix)]
                   for p in glob.glob(os.path.join(args.sim_dir, "ts_*.trees")))
    print(f"{len(bases)} regions to preprocess")

    from fastrho.features import FeatureConfig
    _fkw = dict(disjoint_bands=args.disjoint_bands, stride_after=args.stride_after,
                sfs_shape=args.sfs_shape, r2_debias=args.r2_debias)
    if args.radii:
        _fkw["ld_radii"] = tuple(int(x) for x in args.radii.split(","))
    if args.max_neighbors is not None:
        _fkw["max_neighbors"] = args.max_neighbors
    fcfg = FeatureConfig(**_fkw)
    featurizer = None
    if args.raw:
        from fastrho.raw_features import RawGenotypeFeaturizer
        featurizer = RawGenotypeFeaturizer()
    elif args.gt or args.gt_fold:
        from fastrho.gt_features import GTTokenFeaturizer
        featurizer = GTTokenFeaturizer(config=fcfg, fold=args.gt_fold)
    elif args.with_features:
        from fastrho.features import SNPTokenFeaturizer
        featurizer = SNPTokenFeaturizer(fcfg)

    from functools import partial
    from multiprocessing import get_context

    from tqdm import tqdm
    worker = partial(_process_and_save, out_dir=args.out_dir, featurizer=featurizer)
    ctx = get_context("fork")
    with ctx.Pool(args.num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(worker, bases), total=len(bases),
                      desc="Preprocessing"):
            pass


if __name__ == "__main__":
    main()
