"""Configuration for fastrho models and training."""

from __future__ import annotations

from dataclasses import dataclass, replace

from fastrho.features import n_features as _n_features
from fastrho.raw_features import n_features as _raw_n_features


@dataclass
class FastRhoConfig:
    """Config for the SNP-token bidirectional-Mamba recombination model."""

    # input
    n_features: int = _n_features()      # per-token feature dim (17)
    context_len: int = 1024              # K SNP tokens per chunk
    cond_dim: int = 2                    # [log10 mutation_rate, log10 n_haplotypes]

    # architecture selection
    backbone: str = "mamba"              # "mamba" (fastrho) | "gru" (ReLERNN-seq2seq steelman)
    featurizer: str = "ld"               # "ld" (SNP-token LD feats) | "raw" (raw genotype matrix)
    n_gru_layers: int = 3                # for the GRU backbone

    # Mamba dims
    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    n_enc_layers: int = 6
    n_dec_layers: int = 4

    # token stem (multi-scale depthwise conv over the SNP sequence)
    stem_kernels: tuple[int, ...] = (3, 7, 15)

    dropout: float = 0.1

    # runtime
    device: str = "cpu"
    batch_size: int = 64

    def for_training(self, batch_size: int = 64, device: str = "cuda") -> "FastRhoConfig":
        return replace(self, batch_size=batch_size, device=device)

    def for_inference(self, batch_size: int = 1, device: str = "cpu") -> "FastRhoConfig":
        return replace(self, batch_size=batch_size, device=device)


# NOTE: with mamba-ssm's fused Mamba2 path (default headdim=64, ngroups=1) the in-proj
# dim must be a multiple of 8, i.e. nheads = d_model*expand/64 must be divisible by 8 ->
# d_model must be a multiple of 256. Otherwise causal_conv1d raises a stride error.
PRESETS: dict[str, FastRhoConfig] = {
    "small": FastRhoConfig(d_model=256, n_enc_layers=4, n_dec_layers=2, context_len=512),
    "base":  FastRhoConfig(d_model=256, n_enc_layers=6, n_dec_layers=4, context_len=1024),
    "large": FastRhoConfig(d_model=512, n_enc_layers=8, n_dec_layers=6, context_len=2048),
    # ReLERNN-seq2seq steelman: raw genotype matrix + bidirectional GRU (per-SNP output)
    "gru_seq2seq": FastRhoConfig(backbone="gru", featurizer="raw",
                                 n_features=_raw_n_features(), n_gru_layers=4,
                                 d_model=512, context_len=1024),
}


@dataclass
class TrainingConfig:
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 0
    batch_size: int = 64
    grad_accum_steps: int = 2
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    num_workers: int = 8
    prefetch_factor: int = 2
    # loss weights
    lambda_ne: float = 0.1           # auxiliary Ne regression
    lambda_coarse: float = 0.1       # region-mean (coarse-scale) anchor
    beta_nll: float = 0.5
