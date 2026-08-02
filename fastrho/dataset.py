"""Dataset for fastrho: SNP-token shards -> fixed-length training chunks.

Each item is a context of K SNP tokens (random crop in training, head crop in eval),
with the per-interval target = log of population-scaled rho (= 4 * Ne * r_interval), two
masks (real tokens; valid intervals), conditioning [log10 mu, log10 n_haplotypes], and the
region log Ne for the auxiliary head.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset


def _list_shards(root: str, split: str | None) -> list[str]:
    if split:
        d = os.path.join(root, split)
        if os.path.isdir(d):
            return sorted(glob.glob(os.path.join(d, "ts_*.npz")))
    return sorted(glob.glob(os.path.join(root, "ts_*.npz")))


def fit_stats(files: list[str], max_shards: int = 300, seed: int = 0) -> dict:
    """Standardization stats over a sample of shards: token features, the log
    population-scaled-rho target, and log Ne. Predicting z-scored targets is what
    lets the model learn the (large, negative) rho offset quickly.
    """
    import json
    rng = np.random.default_rng(seed)
    sel = files if len(files) <= max_shards else [
        files[i] for i in rng.choice(len(files), max_shards, replace=False)]
    feats, tgts, nes = [], [], []
    for f in sel:
        z = np.load(f, allow_pickle=True)
        if "tokens" not in z or z["tokens"].shape[0] == 0:
            continue
        feats.append(z["tokens"].astype(np.float64))
        meta = json.loads(str(z["meta"]))
        Ne = meta.get("Ne", None)
        if "interval_target" in z and z["interval_target"].shape[0] > 0 and Ne is not None:
            r = np.clip(z["interval_target"].astype(np.float64), 1e-12, None)
            tgts.append(np.log(4.0 * Ne * r))
            nes.append(np.log(Ne))
    X = np.concatenate(feats, axis=0)
    T = np.concatenate(tgts) if tgts else np.array([0.0, 1.0])
    N = np.asarray(nes) if nes else np.array([0.0, 1.0])
    return {
        "feat_mean": X.mean(0).astype(np.float32),
        "feat_std": (X.std(0) + 1e-6).astype(np.float32),
        "tgt_mean": np.float32(T.mean()),
        "tgt_std": np.float32(T.std() + 1e-6),
        "ne_mean": np.float32(N.mean()),
        "ne_std": np.float32(N.std() + 1e-6),
    }


class RegionTokenDataset(Dataset):
    def __init__(self, root: str, split: str | None = None, context_len: int = 1024,
                 stats: dict | None = None, train: bool = True, seed: int = 0,
                 files: list[str] | None = None):
        self.files = sorted(files) if files is not None else _list_shards(root, split)
        if not self.files:
            raise FileNotFoundError(f"no ts_*.npz shards under {root} (split={split})")
        self.K = context_len
        self.train = train
        s = stats or {}
        self.feat_mean = None if s.get("feat_mean") is None else np.asarray(s["feat_mean"], np.float32)
        self.feat_std = None if s.get("feat_std") is None else np.asarray(s["feat_std"], np.float32)
        self.tgt_mean = float(s.get("tgt_mean", 0.0))
        self.tgt_std = float(s.get("tgt_std", 1.0))
        self.ne_mean = float(s.get("ne_mean", 0.0))
        self.ne_std = float(s.get("ne_std", 1.0))
        self.seed = int(seed)
        self._eval_items = None
        if not train:
            self._eval_items = []
            for file_index, path in enumerate(self.files):
                with np.load(path, allow_pickle=True) as z:
                    length = int(z["tokens"].shape[0]) if "tokens" in z else 0
                for start in self._evaluation_starts(length, self.K):
                    self._eval_items.append((file_index, start))

    def __len__(self):
        return len(self.files) if self.train else len(self._eval_items)

    @staticmethod
    def _evaluation_starts(length: int, context_len: int) -> list[int]:
        """Deterministic crops that cover every token, including the tail."""
        if length <= context_len:
            return [0]
        starts = list(range(0, length - context_len + 1, context_len))
        tail = length - context_len
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    def _fit_len(self, tokens, target, tmask, start=None):
        S, F = tokens.shape
        K = self.K
        if S >= K:
            if start is None:
                start = int(torch.randint(0, S - K + 1, (1,)).item()) if self.train else 0
            sl = slice(start, start + K)
            return tokens[sl], target[sl], tmask[sl], np.ones(K, np.float32)
        # pad at end
        tok = np.zeros((K, F), np.float32)
        tgt = np.zeros(K, np.float32)
        tm = np.zeros(K, np.float32)
        tokmask = np.zeros(K, np.float32)
        tok[:S] = tokens
        tgt[:S] = target
        tm[:S] = tmask
        tokmask[:S] = 1.0
        return tok, tgt, tm, tokmask

    def __getitem__(self, idx):
        if self.train:
            file_index, start = idx, None
        else:
            file_index, start = self._eval_items[idx]
        z = np.load(self.files[file_index], allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tokens = z["tokens"].astype(np.float32) if "tokens" in z else \
            np.zeros((0, 1), np.float32)
        S = tokens.shape[0]

        target = np.zeros(S, np.float32)
        tmask = np.zeros(S, np.float32)
        Ne = meta.get("Ne", None)
        if S >= 2:
            itarget = z["interval_target"].astype(np.float64)   # (S-1,)
            r = np.clip(itarget, 1e-12, None)
            raw = np.log(4.0 * Ne * r) if Ne is not None else np.log(r)
            target[: S - 1] = (raw - self.tgt_mean) / self.tgt_std   # z-scored log rho
            tmask[: S - 1] = 1.0

        if self.feat_mean is not None and S > 0:
            tokens = (tokens - self.feat_mean) / self.feat_std

        tok, tgt, tm, tokmask = self._fit_len(tokens, target, tmask, start=start)

        mu = float(meta["mutation_rate"])
        n_hap = int(meta.get("n_haplotypes", 2 * int(meta["n_samples"])))
        cond = np.array([np.log10(mu), np.log10(n_hap)], np.float32)
        log_Ne_raw = np.log(Ne) if Ne is not None else 0.0
        log_Ne = np.float32((log_Ne_raw - self.ne_mean) / self.ne_std)   # z-scored log Ne
        ne_mask = np.float32(1.0 if Ne is not None else 0.0)

        return {
            "tokens": torch.from_numpy(np.ascontiguousarray(tok)),
            "target": torch.from_numpy(tgt),
            "target_mask": torch.from_numpy(tm),
            "token_mask": torch.from_numpy(tokmask),
            "cond": torch.from_numpy(cond),
            "log_Ne": torch.tensor(log_Ne),
            "ne_mask": torch.tensor(ne_mask),
        }


# ---------------------------------------------------------------------------
# Domain-randomized dataset (one model robust to phased / unphased / unpolarized)
# ---------------------------------------------------------------------------

# (name, is_gt, is_folded): the data "view" the model is told it is looking at, via two
# extra conditioning bits appended to [log10 mu, log10 n_hap]. hap == phased haplotype
# features; gt == phase-invariant composite-LD; gtf == phase- AND polarization-invariant.
DR_VARIANTS = [("hap", 0.0, 0.0), ("gt", 1.0, 0.0), ("gtf", 1.0, 1.0)]


def fit_stats_per_variant(base: str, variants=DR_VARIANTS, split: str | None = "train",
                          **kw) -> dict:
    """Fit standardization stats separately for each variant subdir of `base`.

    Returns {name: stats_dict}. Feature stats differ per variant (different featurizer);
    target/Ne stats are featurization-independent, so any variant's are usable.
    """
    out = {}
    for name, _, _ in variants:
        root = os.path.join(base, name)
        out[name] = fit_stats(_list_shards(root, split), **kw)
    return out


class DomainRandomizedDataset(Dataset):
    """Aligned per-variant shards (same regions featurized 3 ways) -> one training stream.

    Each region exists under base/<variant>/<split>/ts_*.npz with identical positions and
    interval_target across variants. In training, every __getitem__ draws a random variant
    so the model sees all of phased / unphased / unpolarized for every region across epochs;
    in eval the variant is fixed by index for a stable, balanced validation metric. Tokens
    are z-scored with that variant's own stats, and the two view-bits (is_gt, is_folded) are
    appended to the conditioning vector so the model can specialize its readout per view.
    """

    def __init__(self, base: str, split: str | None = None, context_len: int = 1024,
                 stats: dict | None = None, train: bool = True, seed: int = 0,
                 variants=DR_VARIANTS):
        self.base = base
        self.variants = list(variants)
        self.split = split
        self.train = train
        self.K = context_len
        # region basenames from the first variant; assume aligned across variants
        ref_root = os.path.join(base, self.variants[0][0])
        self.names = [os.path.basename(f) for f in _list_shards(ref_root, split)]
        if not self.names:
            raise FileNotFoundError(f"no ts_*.npz under {ref_root} (split={split})")
        self.stats = stats or {}
        self._std = {}
        for name, _, _ in self.variants:
            s = self.stats.get(name, {})
            self._std[name] = dict(
                feat_mean=None if s.get("feat_mean") is None else np.asarray(s["feat_mean"], np.float32),
                feat_std=None if s.get("feat_std") is None else np.asarray(s["feat_std"], np.float32),
                tgt_mean=float(s.get("tgt_mean", 0.0)), tgt_std=float(s.get("tgt_std", 1.0)),
                ne_mean=float(s.get("ne_mean", 0.0)), ne_std=float(s.get("ne_std", 1.0)))
        self.seed = int(seed)
        self._reuse = RegionTokenDataset.__dict__["_fit_len"]
        self._eval_items = None
        if not train:
            self._eval_items = []
            ref_dir = os.path.join(ref_root, split) if split else ref_root
            for region_index, filename in enumerate(self.names):
                with np.load(os.path.join(ref_dir, filename), allow_pickle=True) as z:
                    length = int(z["tokens"].shape[0])
                starts = RegionTokenDataset._evaluation_starts(length, self.K)
                for variant_index in range(len(self.variants)):
                    for start in starts:
                        self._eval_items.append((region_index, variant_index, start))

    def __len__(self):
        return len(self.names) if self.train else len(self._eval_items)

    def __getitem__(self, idx):
        if self.train:
            region_index = idx
            variant_index = int(torch.randint(0, len(self.variants), (1,)).item())
            start = None
        else:
            region_index, variant_index, start = self._eval_items[idx]
        name, is_gt, is_fold = self.variants[variant_index]
        st = self._std[name]
        filename = self.names[region_index]
        path = os.path.join(self.base, name, self.split, filename) if self.split \
            else os.path.join(self.base, name, filename)
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        tokens = z["tokens"].astype(np.float32) if "tokens" in z else np.zeros((0, 1), np.float32)
        S = tokens.shape[0]
        Ne = meta.get("Ne", None)
        target = np.zeros(S, np.float32)
        tmask = np.zeros(S, np.float32)
        if S >= 2:
            r = np.clip(z["interval_target"].astype(np.float64), 1e-12, None)
            raw = np.log(4.0 * Ne * r) if Ne is not None else np.log(r)
            target[: S - 1] = (raw - st["tgt_mean"]) / st["tgt_std"]
            tmask[: S - 1] = 1.0
        if st["feat_mean"] is not None and S > 0:
            tokens = (tokens - st["feat_mean"]) / st["feat_std"]
        tok, tgt, tm, tokmask = self._reuse(self, tokens, target, tmask, start=start)
        mu = float(meta["mutation_rate"])
        n_hap = int(meta.get("n_haplotypes", 2 * int(meta["n_samples"])))
        cond = np.array([np.log10(mu), np.log10(n_hap), is_gt, is_fold], np.float32)
        log_Ne_raw = np.log(Ne) if Ne is not None else 0.0
        log_Ne = np.float32((log_Ne_raw - st["ne_mean"]) / st["ne_std"])
        ne_mask = np.float32(1.0 if Ne is not None else 0.0)
        return {
            "tokens": torch.from_numpy(np.ascontiguousarray(tok)),
            "target": torch.from_numpy(tgt),
            "target_mask": torch.from_numpy(tm),
            "token_mask": torch.from_numpy(tokmask),
            "cond": torch.from_numpy(cond),
            "log_Ne": torch.tensor(log_Ne),
            "ne_mask": torch.tensor(ne_mask),
        }
