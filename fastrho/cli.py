"""Unified ``fastrho`` command-line interface.

    fastrho predict  --trees region.trees --checkpoint m.ckpt --stats feat_stats.npz \
                     --mutation-rate 1.5e-8 [--ne 10000] --out map.bed
    fastrho predict  --vcf sample.vcf.gz --chrom chr1 --checkpoint m.ckpt --stats s.npz --out map.bed
    fastrho evaluate --checkpoint m.ckpt --stats s.npz --shards test/
    fastrho simulate | preprocess | train    (thin pass-throughs to the module CLIs)

The headline ease-of-use win over ReLERNN: ``predict`` runs a *pretrained* amortized
model in one pass — no per-dataset simulate/train/bootstrap.
"""

from __future__ import annotations

import argparse
import sys


def _cmd_predict(args):
    from fastrho.translate import load_model, predict_map_from_ts, predict_map_from_vcf, write_bed
    device = args.device
    if device == "auto":
        import torch
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model, cfg, stats = load_model(args.checkpoint, args.stats, device=device)
    if args.trees:
        import tskit
        ts = tskit.load(args.trees)
        pred = predict_map_from_ts(ts, model, cfg, stats,
                                   mutation_rate=args.mutation_rate, Ne=args.ne,
                                   device=device, input_mode=args.input_mode)
        chrom = args.chrom or "chr"
    elif args.vcf:
        pred = predict_map_from_vcf(args.vcf, model, cfg, stats, contig=args.chrom,
                                    mutation_rate=args.mutation_rate, Ne=args.ne,
                                    device=device, input_mode=args.input_mode,
                                    missing=args.missing)
        chrom = args.chrom or pred["contig"]
    else:
        sys.exit("provide --trees or --vcf")
    write_bed(pred, args.out, chrom=chrom, window_size=args.window_size)
    n = len(pred["pos_left"])
    print(f"wrote {args.out}  ({n} intervals; Ne_used={pred['Ne_used']:.0f}, "
          f"Ne_estimated={pred['Ne_estimated']:.0f})")


def _delegate(module_main, argv):
    old = sys.argv
    sys.argv = [old[0]] + argv
    try:
        module_main()
    finally:
        sys.argv = old


def main():
    ap = argparse.ArgumentParser(prog="fastrho",
                                 description="Fast amortized recombination-map estimation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="estimate a recombination map from data")
    p.add_argument("--trees")
    p.add_argument("--vcf")
    p.add_argument("--chrom")
    p.add_argument("--checkpoint", required=True)
    p.add_argument(
        "--stats",
        required=True,
        help="unchanged feat_stats.npz companion from the same model bundle as --checkpoint",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--mutation-rate", type=float, default=1.5e-8)
    p.add_argument("--ne", type=float, default=None,
                   help="diploid Ne for absolute rate; omit to use the model's aux Ne head")
    p.add_argument("--window-size", type=int, default=None,
                   help="rebin the per-interval map to fixed bp windows")
    p.add_argument("--input-mode", default="auto",
                   choices=("auto", "phased", "unphased", "unpolarized", "raw"),
                   help="token view; VCF phase is detected when set to auto")
    p.add_argument("--missing", default="drop-site", choices=("drop-site", "error"),
                   help="drop sites with missing genotypes or fail")
    p.add_argument("--device", default="auto")

    sub.add_parser("evaluate", help="evaluate a checkpoint on held-out shards (see --help)",
                   add_help=False)
    sub.add_parser("simulate", help="simulate training regions", add_help=False)
    sub.add_parser("preprocess", help="preprocess simulations into shards", add_help=False)
    sub.add_parser("train", help="train a model", add_help=False)

    # split so sub-CLIs parse their own args
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "predict":
        _cmd_predict(p.parse_args(sys.argv[2:]))
    elif cmd == "evaluate":
        from fastrho.evaluate import main as m
        _delegate(m, sys.argv[2:])
    elif cmd == "simulate":
        from fastrho.simulate import main as m
        _delegate(m, sys.argv[2:])
    elif cmd == "preprocess":
        from fastrho.preprocess import main as m
        _delegate(m, sys.argv[2:])
    elif cmd == "train":
        from fastrho.train import main as m
        _delegate(m, sys.argv[2:])
    else:
        ap.parse_args()


if __name__ == "__main__":
    main()
