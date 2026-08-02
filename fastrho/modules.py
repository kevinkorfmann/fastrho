"""Self-contained neural-network blocks used by :mod:`fastrho.model`.

Mamba remains an optional inference and training dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba sequence mixing followed by an FFN residual block."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, ffn_expand: int = 4):
        super().__init__()
        try:
            from mamba_ssm import Mamba2
        except ImportError as exc:  # pragma: no cover - exercised on GPU installs
            raise ImportError(
                "Mamba models require the optional 'inference' dependencies; "
                "install fastrho[inference] on a supported CUDA system"
            ) from exc
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba2(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.mamba_bwd = Mamba2(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
        )
        self.merge = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)
        h = self.merge(torch.cat([h_fwd, h_bwd], dim=-1))
        x = x + self.drop(h)
        return x + self.ffn(self.norm2(x))


class FiLMLayer(nn.Module):
    """Feature-wise affine modulation from global conditioning variables."""

    def __init__(self, d_model: int, cond_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return h + gamma.unsqueeze(1) * h + beta.unsqueeze(1)


class UncertaintyHead(nn.Module):
    """Independent mean and log-variance heads for a Gaussian log-rate output."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mid = d_model // 2
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Linear(mid, 1)
        )
        self.var_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Linear(mid, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm(h)
        return torch.cat([self.mu_head(h), self.var_head(h)], dim=-1)
