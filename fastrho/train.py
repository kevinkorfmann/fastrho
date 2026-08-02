"""Training for fastrho (SNP-token recombination model).

Loss = heteroscedastic NLL on log population-scaled rho per SNP-interval
       + lambda_ne * MSE(log Ne)            (auxiliary, deconfounds rho -> r)
       + lambda_coarse * MSE(region means)  (coarse-scale anchor).

V100-friendly: fp16 ("16-mixed"), not bf16.

Usage:
    python -m fastrho.train --model base --dataset-path /path/to/shards --gpus 0 1 --epochs 20
"""

from __future__ import annotations

import argparse
import math
import os

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from fastrho.config import PRESETS, FastRhoConfig, TrainingConfig
from fastrho.dataset import (
    DR_VARIANTS,
    DomainRandomizedDataset,
    RegionTokenDataset,
    fit_stats,
    fit_stats_per_variant,
)
from fastrho.model import build_model

torch.serialization.add_safe_globals([FastRhoConfig, TrainingConfig])


def hetero_nll(mu, log_sigma2, target, mask, beta=0.5):
    log_sigma2 = torch.clamp(log_sigma2, -10, 10)
    sigma2 = torch.exp(log_sigma2)
    mse = (target - mu) ** 2
    w = sigma2.detach() ** beta
    per = 0.5 * w * (log_sigma2 + mse / sigma2)
    return (per * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_mean(x, m):
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class LitFastRho(L.LightningModule):
    def __init__(self, model_config, training_config: dict | None = None):
        super().__init__()
        self.model_config = model_config
        self.model = build_model(model_config)
        tc = training_config or TrainingConfig().__dict__
        self.training_config = tc.__dict__ if isinstance(tc, TrainingConfig) else tc
        self.save_hyperparameters(ignore=["model"])

    def _step(self, batch):
        out = self.model(batch["tokens"], batch["cond"], batch["token_mask"])
        mu = out["rho"][..., 0]
        logv = out["rho"][..., 1]
        tgt = batch["target"]
        m = batch["target_mask"]
        tc = self.training_config

        nll = hetero_nll(mu, logv, tgt, m, beta=tc["beta_nll"])
        pred_mean = _masked_mean(mu, m)
        tgt_mean = _masked_mean(tgt, m)
        coarse = ((pred_mean - tgt_mean) ** 2).mean()
        nem = batch["ne_mask"]
        ne_loss = ((out["log_Ne"] - batch["log_Ne"]) ** 2 * nem).sum() / nem.sum().clamp(min=1.0)
        loss = nll + tc["lambda_ne"] * ne_loss + tc["lambda_coarse"] * coarse

        with torch.no_grad():
            muf, tgtf, mf = mu.float(), tgt.float(), m.float()
            rmse = (((muf - tgtf) ** 2 * mf).sum() / mf.sum().clamp(min=1)).sqrt()
            sigma = torch.exp(0.5 * torch.clamp(logv.float(), -10, 10))
            cov = (((tgtf >= muf - 1.96 * sigma) & (tgtf <= muf + 1.96 * sigma)).float() * mf
                   ).sum() / mf.sum().clamp(min=1)
            # Pearson on valid intervals (relative-accuracy / map-shape proxy)
            sel = mf > 0
            if sel.any():
                a = muf[sel] - muf[sel].mean()
                b = tgtf[sel] - tgtf[sel].mean()
                pear = (a * b).sum() / (a.norm() * b.norm() + 1e-8)
            else:
                pear = torch.zeros((), device=mu.device)
        return loss, {"nll": nll, "ne": ne_loss, "coarse": coarse,
                      "rmse": rmse, "cov95": cov, "pearson": pear}

    def training_step(self, batch, _):
        loss, mt = self._step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rmse", mt["rmse"], prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, mt = self._step(batch)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", mt["rmse"], prog_bar=True, sync_dist=True)
        self.log("val_cov95", mt["cov95"], prog_bar=True, sync_dist=True)
        self.log("val_pearson", mt["pearson"], prog_bar=True, sync_dist=True)
        self.log("val_ne", mt["ne"], sync_dist=True)
        return loss

    def configure_optimizers(self):
        tc = self.training_config
        opt = self.model.configure_optimizers(
            weight_decay=tc["weight_decay"], learning_rate=tc["max_lr"],
            betas=tc["betas"], device_type=self.device.type)
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step", "frequency": 1}]

    def _lr_lambda(self, step):
        tc = self.training_config
        if tc["lr_decay_iters"] <= tc["warmup_iters"]:
            return 1.0
        if step < tc["warmup_iters"]:
            return step / max(1, tc["warmup_iters"])
        if step > tc["lr_decay_iters"]:
            return tc["min_lr"] / tc["max_lr"]
        ratio = (step - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return tc["min_lr"] / tc["max_lr"] + coeff * (1.0 - tc["min_lr"] / tc["max_lr"])


def _trainer_runtime(args):
    devices = args.gpus if args.gpus is not None else (
        int(args.devices) if str(args.devices).isdigit() else args.devices
    )
    precision = args.precision or ("16-mixed" if torch.cuda.is_available() else "32-true")
    return devices, precision


def _build_trainer(args, tc, n_gpus):
    devices, precision = _trainer_runtime(args)
    return L.Trainer(
        max_epochs=args.epochs, accelerator=args.accelerator, devices=devices,
        precision=precision,
        strategy="ddp" if n_gpus > 1 else "auto",
        accumulate_grad_batches=tc.grad_accum_steps,
        limit_train_batches=args.limit_train_batches,
        default_root_dir=args.log_dir or "lightning_logs",
        logger=CSVLogger(save_dir=args.log_dir or "lightning_logs", name="fastrho"),
        callbacks=[ModelCheckpoint(monitor="val_pearson", mode="max", save_top_k=args.save_top_k,
                                   filename="{epoch}-{val_pearson:.3f}")],
    )


def _main_dr(args):
    """Domain-randomized training: one model over hap/gt/gtf views (cond_dim=4, 18 feats)."""
    from dataclasses import replace as _replace
    base = args.dr_base
    cfg = PRESETS[args.model].for_training(batch_size=args.batch_size)
    # shards are built sfs_shape=True (18-dim); two extra conditioning bits (is_gt,is_folded)
    cfg = _replace(cfg, n_features=18, cond_dim=4)
    print(f"[DR] n_features={cfg.n_features} cond_dim={cfg.cond_dim} "
          f"variants={[v[0] for v in DR_VARIANTS]}")

    has_split = os.path.isdir(os.path.join(base, DR_VARIANTS[0][0], "train"))
    if not has_split:
        raise ValueError("domain-randomized training requires aligned train/ and val/ or test/ splits")
    tsplit = "train" if has_split else None
    stats_pv = fit_stats_per_variant(base, split=tsplit)
    # persist: per-variant feature stats + shared target/Ne stats + featurizer meta
    save = {"stats_schema_version": np.int64(1),
            "featurizer_kind": np.array("domain_randomized"),
            "variants": np.array([v[0] for v in DR_VARIANTS]),
            "n_features": cfg.n_features, "cond_dim": cfg.cond_dim,
            "ld_radii": np.asarray((5000, 25000, 50000), np.int64),
            "neigh_snps": np.int64(8), "max_neighbors": np.int64(200),
            "disjoint_bands": np.int64(0), "stride_after": np.int64(0),
            "rich_ld": np.int64(0), "sfs_shape": np.int64(1),
            "r2_debias": np.int64(1)}
    ref = stats_pv[DR_VARIANTS[0][0]]
    for k in ("tgt_mean", "tgt_std", "ne_mean", "ne_std"):
        save[k] = ref[k]
    for name, _, _ in DR_VARIANTS:
        save[f"{name}_feat_mean"] = stats_pv[name]["feat_mean"]
        save[f"{name}_feat_std"] = stats_pv[name]["feat_std"]
    np.savez(os.path.join(base, "feat_stats_dr.npz"), **save)
    print(f"[DR] stats fit per variant | tgt_mean={ref['tgt_mean']:.2f} "
          f"tgt_std={ref['tgt_std']:.2f} ne_mean={ref['ne_mean']:.2f}")

    vsplit = ("test" if os.path.isdir(os.path.join(base, DR_VARIANTS[0][0], "test"))
              else ("val" if os.path.isdir(os.path.join(base, DR_VARIANTS[0][0], "val"))
                    else None))
    if vsplit is None:
        raise FileNotFoundError("domain-randomized training requires aligned val/ or test/ shards")
    train_ds = DomainRandomizedDataset(base, split=tsplit, context_len=cfg.context_len,
                                       stats=stats_pv, train=True)
    val_ds = DomainRandomizedDataset(base, split=vsplit, context_len=cfg.context_len,
                                     stats=stats_pv, train=False)
    print(f"[DR] train={len(train_ds)} val={len(val_ds)} regions x {len(DR_VARIANTS)} views")

    n_gpus = len(args.gpus) if args.gpus is not None else 1
    steps = max(1, len(train_ds) // (args.batch_size * n_gpus * args.grad_accum))
    total = steps * args.epochs
    tc = TrainingConfig(max_lr=args.lr, batch_size=args.batch_size,
                        grad_accum_steps=args.grad_accum, num_workers=args.workers,
                        lr_decay_iters=total, warmup_iters=min(100, max(1, total // 10)))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=tc.batch_size, num_workers=tc.num_workers,
                          shuffle=shuffle, pin_memory=True, drop_last=shuffle,
                          persistent_workers=tc.num_workers > 0,
                          prefetch_factor=tc.prefetch_factor if tc.num_workers > 0 else None)

    lit = (LitFastRho.load_from_checkpoint(args.checkpoint, model_config=cfg,
                                           training_config=tc.__dict__)
           if args.checkpoint else LitFastRho(cfg, training_config=tc.__dict__))
    if args.compile:
        lit.model = torch.compile(lit.model)
    torch.set_float32_matmul_precision("medium")
    _build_trainer(args, tc, n_gpus).fit(lit, loader(train_ds, True), loader(val_ds, False))


def main():
    ap = argparse.ArgumentParser(description="Train fastrho")
    ap.add_argument("--model", default="base", choices=list(PRESETS))
    ap.add_argument("--dataset-path", default=None,
                    help="shard root for single-featurizer training; omit when using --dr-base")
    ap.add_argument("--gpus", type=int, nargs="+", default=None,
                    help="deprecated CUDA device list; prefer --devices")
    ap.add_argument("--devices", default="auto", help="Lightning device count or 'auto'")
    ap.add_argument("--accelerator", default="auto", choices=("auto", "cpu", "gpu"))
    ap.add_argument("--precision", default=None,
                    help="Lightning precision; defaults to 16-mixed on CUDA and 32-true on CPU")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-fraction", type=float, default=0.1,
                    help="deterministic validation fraction when shards have no split directories")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--limit-train-batches", type=float, default=1.0)
    ap.add_argument("--save-top-k", type=int, default=1,
                    help="checkpoints to keep (Lightning ModelCheckpoint). -1 = keep EVERY epoch, "
                         "which is required to select a checkpoint on REAL data afterwards "
                         "(scripts/self_epoch_select.py) instead of on simulated val_pearson.")
    # featurizer config recorded with the model so inference rebuilds an identical
    # featurizer (train/inference parity); also sets the model input dim n_features.
    ap.add_argument("--radii", default=None,
                    help="comma-separated LD radii used in preprocessing (sets n_features)")
    ap.add_argument("--disjoint-bands", action="store_true")
    ap.add_argument("--stride-after", type=int, default=0)
    ap.add_argument("--max-neighbors", type=int, default=None)
    ap.add_argument("--rich-ld", action="store_true")
    ap.add_argument("--sfs-shape", action="store_true")
    ap.add_argument("--r2-debias", action="store_true")
    ap.add_argument("--featurizer-kind", choices=("hap", "gt", "gtf", "raw"), default=None,
                    help="token view used to create the shards; saved for safe inference")
    # Domain-randomized training: base dir holding hap/gt/gtf variant subdirs (each a
    # train/[test] tree of aligned 18-dim shards). Trains ONE model robust to phased,
    # unphased and unpolarized data via two conditioning bits (is_gt, is_folded).
    ap.add_argument("--dr-base", default=None)
    args = ap.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        ap.error("--val-fraction must be between 0 and 1")
    L.seed_everything(args.seed, workers=True)

    if args.dr_base:
        return _main_dr(args)
    if not args.dataset_path:
        ap.error("--dataset-path is required unless --dr-base is given")

    cfg = PRESETS[args.model].for_training(batch_size=args.batch_size)

    # If the featurizer differs from the default 3-radius config, recompute n_features
    # and capture the featurizer params to store alongside the stats.
    from dataclasses import replace as _replace

    from fastrho.features import FeatureConfig
    from fastrho.features import n_features as _nfeat
    _fkw = dict(disjoint_bands=args.disjoint_bands, stride_after=args.stride_after,
                rich_ld=args.rich_ld, sfs_shape=args.sfs_shape,
                r2_debias=args.r2_debias)
    if args.radii:
        _fkw["ld_radii"] = tuple(int(x) for x in args.radii.split(","))
    if args.max_neighbors is not None:
        _fkw["max_neighbors"] = args.max_neighbors
    fcfg = FeatureConfig(**_fkw)
    featurizer_kind = args.featurizer_kind or (
        "raw" if getattr(cfg, "featurizer", "ld") == "raw" else "hap"
    )
    if featurizer_kind != "raw":
        cfg = _replace(cfg, n_features=_nfeat(fcfg))
    feat_meta = dict(stats_schema_version=np.int64(1),
                     featurizer_kind=np.array(featurizer_kind),
                     ld_radii=np.asarray(fcfg.ld_radii, np.int64),
                     neigh_snps=np.int64(fcfg.neigh_snps),
                     disjoint_bands=np.int64(fcfg.disjoint_bands),
                     stride_after=np.int64(fcfg.stride_after),
                     max_neighbors=np.int64(fcfg.max_neighbors),
                     rich_ld=np.int64(fcfg.rich_ld),
                     sfs_shape=np.int64(fcfg.sfs_shape),
                     r2_debias=np.int64(fcfg.r2_debias))
    print(f"featurizer={featurizer_kind} radii={fcfg.ld_radii} "
          f"disjoint={fcfg.disjoint_bands} stride_after={fcfg.stride_after} "
          f"max_neigh={fcfg.max_neighbors} -> n_features={cfg.n_features}")

    # standardization stats (features + log-rho target + log Ne), fit on train
    has_split = os.path.isdir(os.path.join(args.dataset_path, "train"))
    if has_split:
        train_files = RegionTokenDataset(args.dataset_path, split="train").files
        val_split = "test" if os.path.isdir(os.path.join(args.dataset_path, "test")) else \
            ("val" if os.path.isdir(os.path.join(args.dataset_path, "val")) else None)
        if val_split is None:
            raise FileNotFoundError("training split exists but neither val/ nor test/ shards exist")
        val_files = RegionTokenDataset(args.dataset_path, split=val_split).files
    else:
        all_files = RegionTokenDataset(args.dataset_path).files
        if len(all_files) < 2:
            raise ValueError("at least two shards are required to create train/validation splits")
        order = np.random.default_rng(args.seed).permutation(len(all_files))
        n_val = max(1, int(round(len(all_files) * args.val_fraction)))
        val_index = set(order[:n_val].tolist())
        train_files = [f for i, f in enumerate(all_files) if i not in val_index]
        val_files = [f for i, f in enumerate(all_files) if i in val_index]
    stats = fit_stats(train_files)
    np.savez(os.path.join(args.dataset_path, "feat_stats.npz"),
             n_features=cfg.n_features, **stats, **feat_meta)
    print(f"stats fit on {len(train_files)} shards | "
          f"tgt_mean={stats['tgt_mean']:.2f} tgt_std={stats['tgt_std']:.2f} "
          f"ne_mean={stats['ne_mean']:.2f}")

    ds_kw = dict(context_len=cfg.context_len, stats=stats)
    train_ds = RegionTokenDataset(args.dataset_path, files=train_files, train=True,
                                  seed=args.seed, **ds_kw)
    val_ds = RegionTokenDataset(args.dataset_path, files=val_files, train=False,
                                seed=args.seed, **ds_kw)
    print(f"train={len(train_files)} regions; val={len(val_files)} regions / {len(val_ds)} crops")

    n_gpus = len(args.gpus) if args.gpus is not None else 1
    steps = max(1, len(train_ds) // (args.batch_size * n_gpus * args.grad_accum))
    total = steps * args.epochs
    tc = TrainingConfig(max_lr=args.lr, batch_size=args.batch_size,
                        grad_accum_steps=args.grad_accum, num_workers=args.workers,
                        lr_decay_iters=total, warmup_iters=min(100, max(1, total // 10)))

    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=tc.batch_size, num_workers=tc.num_workers,
                          shuffle=shuffle, pin_memory=True, drop_last=shuffle,
                          persistent_workers=tc.num_workers > 0,
                          prefetch_factor=tc.prefetch_factor if tc.num_workers > 0 else None)

    if args.checkpoint:
        lit = LitFastRho.load_from_checkpoint(args.checkpoint, model_config=cfg,
                                              training_config=tc.__dict__)
    else:
        lit = LitFastRho(cfg, training_config=tc.__dict__)
    if args.compile:
        lit.model = torch.compile(lit.model)

    torch.set_float32_matmul_precision("medium")
    trainer = _build_trainer(args, tc, n_gpus)
    trainer.fit(lit, loader(train_ds, True), loader(val_ds, False))


if __name__ == "__main__":
    main()
