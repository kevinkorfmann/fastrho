"""Prototype: does enriching fastrho's featurizer with n-sensitive two-locus statistics
fix its high-n underperformance vs pyrho?

Diagnosis (verified): fastrho's features (mean r^2 in bands, config fractions, adjacent-pair
two-locus config) are deliberately sample-size-robust and SATURATE as n grows, so pyrho's exact
two-locus likelihood overtakes fastrho around n~100 (real human is n~198). The rich_ld featurizer
adds, per LD band, the four-gamete fraction and mean |D'| -- two-locus statistics that KEEP
sharpening with n (the information pyrho exploits).

This script trains a plain (17-feat) and a rich (23-feat) model at IDENTICAL budget on the same
sims and compares them at low vs high n, to decide whether to commit to the full retrain.

Subcommands (fastrho venv on sesame):
  preprocess <sim_dir> <out_dir> <rich:0|1> [nproc]     re-featurize .trees -> shards
  train <train_dir> <val_dir> <out_dir> <epochs> [gpu]  train; auto n_features from shards
  eval  <ckpt> <stats> <test_dir> [tag]                 Pearson @25/100kb, split by sample size
"""
import os
import sys
import glob
import json
from functools import partial
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, "/home/kkor/fastrho")


def inject_noise(gm, rng, switch, geno):
    """Add realistic real-data noise to simulated haplotypes so the (noise-sensitive)
    four-gamete/|D'| features learn to be robust: per-diploid phase SWITCH errors
    (haplotypes paired 2i,2i+1) at rate `switch`/site, and genotyping errors at rate `geno`.
    Clean sims otherwise let the model memorise artifacts absent in real data."""
    n_hap, S = gm.shape
    g = gm.astype(np.int8).copy()
    nd = n_hap // 2
    if switch > 0 and nd > 0:
        flips = rng.random((nd, S)) < switch
        state = (np.cumsum(flips, axis=1) % 2).astype(bool)      # phase state along the seq
        gg = g[:2 * nd].reshape(nd, 2, S)
        a = gg[:, 0, :].copy(); b = gg[:, 1, :].copy()
        gg[:, 0, :] = np.where(state, b, a)
        gg[:, 1, :] = np.where(state, a, b)
        g[:2 * nd] = gg.reshape(2 * nd, S)
    if geno > 0:
        err = rng.random(g.shape) < geno
        g[err] = 1 - g[err]
    return g


class _NoisyFeaturizer:
    """Wrap a featurizer to inject noise into the genotype matrix before featurizing.
    Per-region deterministic RNG (seeded by the sim seed) so shards are reproducible."""
    def __init__(self, fz, switch, geno):
        self.fz, self.switch, self.geno = fz, switch, geno

    @property
    def n_features(self):
        return self.fz.n_features

    def __call__(self, gm, positions, meta):
        # draw per-region noise levels U(0, max) so training spans clean->noisy conditions
        rng = np.random.default_rng(int(meta.get("seed", 0)) * 2 + 101)
        sw = rng.uniform(0.0, self.switch)
        ge = rng.uniform(0.0, self.geno)
        return self.fz(inject_noise(gm, rng, sw, ge), positions, meta)


# ------------------------------------------------------------------ preprocess
def cmd_preprocess(sim_dir, out_dir, rich, nproc=24, switch=0.0, geno=0.0):
    from fastrho.features import SNPTokenFeaturizer, FeatureConfig
    from fastrho.preprocess import _process_and_save
    rich = bool(int(rich))
    nproc = int(nproc)
    switch, geno = float(switch), float(geno)
    os.makedirs(out_dir, exist_ok=True)
    bases = sorted(p[:-6] for p in glob.glob(os.path.join(sim_dir, "ts_*.trees")))
    fz = SNPTokenFeaturizer(FeatureConfig(rich_ld=rich))
    if switch > 0 or geno > 0:
        fz = _NoisyFeaturizer(fz, switch, geno)
    print(f"preprocess {len(bases)} regions rich={rich} noise(switch={switch},geno={geno}) "
          f"-> {out_dir} (n_features={fz.n_features})", flush=True)
    worker = partial(_process_and_save, out_dir=out_dir, featurizer=fz)
    ctx = get_context("fork")
    with ctx.Pool(nproc) as pool:
        for i, _ in enumerate(pool.imap_unordered(worker, bases)):
            if i % 1000 == 0:
                print(f"  {i}/{len(bases)}", flush=True)
    print("PREPROCESS_DONE", out_dir, flush=True)


# ------------------------------------------------------------------ train
def cmd_train(train_dir, val_dir, out_dir, epochs, gpu=0):
    from dataclasses import replace as _replace
    import torch
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader
    from fastrho.config import PRESETS, TrainingConfig
    from fastrho.dataset import RegionTokenDataset, fit_stats
    from fastrho.train import LitFastRho
    epochs, gpu = int(epochs), int(gpu)
    os.makedirs(out_dir, exist_ok=True)

    # auto-detect feature dim from a shard
    probe = np.load(sorted(glob.glob(os.path.join(train_dir, "ts_*.npz")))[0], allow_pickle=True)
    F = int(probe["tokens"].shape[1])
    rich = (F == 23)
    cfg = PRESETS["base"].for_training(batch_size=64)
    cfg = _replace(cfg, n_features=F)
    print(f"[train] n_features={F} (rich={rich}) epochs={epochs} gpu={gpu}", flush=True)

    train_files = RegionTokenDataset(train_dir).files
    stats = fit_stats(train_files)
    np.savez(os.path.join(out_dir, "feat_stats.npz"), n_features=F,
             rich_ld=np.int64(rich), ld_radii=np.asarray((5000, 25000, 50000), np.int64), **stats)

    tc = TrainingConfig(max_lr=3e-4, batch_size=64, grad_accum_steps=2, num_workers=8)
    train_ds = RegionTokenDataset(train_dir, context_len=cfg.context_len, stats=stats, train=True)
    val_ds = RegionTokenDataset(val_dir, context_len=cfg.context_len, stats=stats, train=False)
    steps = max(1, len(train_ds) // (tc.batch_size * tc.grad_accum_steps))
    tc = _replace(tc, lr_decay_iters=steps * epochs, warmup_iters=min(100, max(1, steps * epochs // 10)))
    print(f"[train] train={len(train_ds)} val={len(val_ds)}", flush=True)

    def loader(ds, sh):
        return DataLoader(ds, batch_size=tc.batch_size, num_workers=tc.num_workers, shuffle=sh,
                          pin_memory=True, drop_last=sh, persistent_workers=True, prefetch_factor=2)

    lit = LitFastRho(cfg, training_config=tc.__dict__)
    torch.set_float32_matmul_precision("medium")
    # save EVERY epoch (save_top_k=-1) so the best-generalizing epoch can be chosen on a
    # real-data proxy afterwards -- sim-val Pearson keeps rising while real transfer peaks early.
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, save_top_k=-1,
                              filename="ep{epoch:02d}-{val_pearson:.3f}")
    trainer = L.Trainer(max_epochs=epochs, accelerator="gpu", devices=[gpu], precision="16-mixed",
                        logger=CSVLogger(save_dir=out_dir, name="log"),
                        enable_progress_bar=False, callbacks=[ckpt_cb])
    trainer.fit(lit, loader(train_ds, True), loader(val_ds, False))
    with open(os.path.join(out_dir, "best_ckpt.txt"), "w") as fh:
        fh.write(ckpt_cb.best_model_path)
    print(f"TRAIN_DONE best={ckpt_cb.best_model_path} val_pearson={ckpt_cb.best_model_score}",
          flush=True)


# ------------------------------------------------------------------ eval (by n)
def cmd_eval(ckpt, stats_path, test_dir, tag="model"):
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_from_tokens
    from fastrho.preprocess import mean_rate_between
    model, cfg, stats = load_model(ckpt, stats_path, device="cuda:0")

    def rebin(pos, rate, w):
        L = pos[-1]
        edges = np.append(np.arange(pos[0], L, w), L)
        return mean_rate_between(pos, rate, edges)

    groups = {}  # n_hap -> {scale: {p:[],t:[]}}
    for f in sorted(glob.glob(os.path.join(test_dir, "ts_*.npz"))):
        z = np.load(f, allow_pickle=True)
        if "tokens" not in z or z["tokens"].shape[0] < 3:
            continue
        meta = json.loads(str(z["meta"]))
        nh = 2 * int(meta["n_samples"])
        pos = z["positions"].astype(float)
        true_r = z["interval_target"].astype(float)
        pred = predict_from_tokens(model, cfg, stats, z["tokens"], pos, nh,
                                   float(meta["mutation_rate"]), device="cuda:0")
        pr = pred["r_per_bp"]
        g = groups.setdefault(nh, {"25kb": {"p": [], "t": []}, "100kb": {"p": [], "t": []}})
        for w, key in ((25000, "25kb"), (100000, "100kb")):
            g[key]["p"].append(rebin(pos, pr, w))
            g[key]["t"].append(rebin(pos, true_r, w))
    print(f"=== EVAL {tag} ({ckpt.split('/')[-1]}) ===", flush=True)
    for nh in sorted(groups):
        row = []
        for key in ("25kb", "100kb"):
            p = np.concatenate(groups[nh][key]["p"]); t = np.concatenate(groups[nh][key]["t"])
            ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
            row.append(f"{key}={pearsonr(p[ok], t[ok])[0]:.3f}")
        print(f"  n_hap={nh:4d}  " + "  ".join(row), flush=True)


def cmd_eval_ts(ckpt, stats_p, rich, sim_dir, pattern="region_*.trees"):
    """Evaluate a checkpoint directly on a dir of tree-sequences (on-the-fly rich featurization),
    pooled Pearson vs the generative map at 25/100 kb. For the const_n sample-size benchmark."""
    import tskit
    from scipy.stats import pearsonr
    from fastrho.translate import load_model, predict_intervals
    from fastrho.features import SNPTokenFeaturizer, FeatureConfig
    from fastrho.preprocess import mean_rate_between
    from fastcxt.sfs import basic_filtering
    rich = bool(int(rich))
    model, cfg, stats = load_model(ckpt, stats_p, device="cuda:0")
    fz = SNPTokenFeaturizer(FeatureConfig(rich_ld=rich))
    pools = {"25kb": {"p": [], "t": []}, "100kb": {"p": [], "t": []}}
    for tf in sorted(glob.glob(os.path.join(sim_dir, pattern))):
        z = np.load(tf[:-6] + ".npz", allow_pickle=True)
        ts = tskit.load(tf)
        gm = ts.genotype_matrix().T.astype(np.int8)
        pos = ts.tables.sites.position.astype(np.float64)
        gm, pos = basic_filtering(gm, pos)
        meta = json.loads(str(z["meta"]))
        pred = predict_intervals(model, cfg, stats, gm, pos, float(meta["mutation_rate"]),
                                 Ne=None, device="cuda:0", featurizer=fz)
        bp = np.r_[pred["pos_left"][0], pred["pos_right"]]
        for w, key in ((25000, "25kb"), (100000, "100kb")):
            edges = np.append(np.arange(pos[0], pos[-1], w), pos[-1])
            pools[key]["p"].append(mean_rate_between(bp, pred["r_per_bp"], edges))
            pools[key]["t"].append(mean_rate_between(z["map_position"], z["map_rate"], edges))
    tag = sim_dir.rstrip("/").split("/")[-1]
    for key in ("25kb", "100kb"):
        p = np.concatenate(pools[key]["p"]); t = np.concatenate(pools[key]["t"])
        ok = np.isfinite(p) & np.isfinite(t) & (p > 0) & (t > 0)
        print(f"{tag} {key} rich_pearson={pearsonr(p[ok], t[ok])[0]:.4f}", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "preprocess":
        cmd_preprocess(*sys.argv[2:])
    elif c == "eval_ts":
        cmd_eval_ts(*sys.argv[2:])
    elif c == "train":
        cmd_train(*sys.argv[2:])
    elif c == "eval":
        cmd_eval(*sys.argv[2:])
    else:
        raise SystemExit(f"unknown cmd {c}")
