"""SNP-token bidirectional-Mamba recombination-map estimator.

The model is self-contained and has one recombination distribution head plus an
auxiliary effective-population-size regression head:

  tokens (B,K,F) --stem--> (B,K,D) (+pos)
       --> [BiMambaBlock + FiLM(cond)] x n_enc   (cond = log mu, log n)
       --> [BiMambaBlock + skip]      x n_dec
       --> rho head:  (mu_log_rho, log_sigma2) per SNP interval
       --> Ne head:   point log Ne (region mean pool)

Absolute rate r = rho / (4 Ne) is conditional on supplied or point-estimated Ne.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastrho.modules import BiMambaBlock, FiLMLayer, UncertaintyHead


class SNPTokenStem(nn.Module):
    """Embed per-token features and mix local context with multi-scale depthwise convs."""

    def __init__(self, n_features: int, d_model: int,
                 kernels: tuple[int, ...] = (3, 7, 15), dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU())
        self.branches = nn.ModuleList([
            nn.Conv1d(d_model, d_model, k, padding=k // 2, groups=d_model)
            for k in kernels
        ])
        self.merge = nn.Sequential(
            nn.Linear(d_model * (len(kernels) + 1), d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)                       # (B, K, D)
        hc = h.transpose(1, 2)                  # (B, D, K)
        outs = [h] + [b(hc).transpose(1, 2) for b in self.branches]
        return self.merge(torch.cat(outs, dim=-1))


class FastRhoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.stem = SNPTokenStem(config.n_features, d, config.stem_kernels, config.dropout)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.context_len, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.enc_blocks = nn.ModuleList([
            BiMambaBlock(d, config.d_state, config.d_conv, config.expand, config.dropout)
            for _ in range(config.n_enc_layers)
        ])
        self._film_indices = {0, config.n_enc_layers - 1}
        self.enc_films = nn.ModuleDict({
            str(i): FiLMLayer(d, cond_dim=config.cond_dim) for i in self._film_indices
        })

        self.dec_blocks = nn.ModuleList([
            BiMambaBlock(d, config.d_state, config.d_conv, config.expand, config.dropout)
            for _ in range(config.n_dec_layers)
        ])
        n_skips = min(config.n_dec_layers, config.n_enc_layers)
        self.skip_projs = nn.ModuleList([nn.Linear(d, d) for _ in range(n_skips)])

        self.rho_head = UncertaintyHead(d)       # -> (mu_log_rho, log_sigma2)
        self.ne_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _forward_unpadded(self, tokens: torch.Tensor, cond: torch.Tensor) -> dict:
        B, K, _ = tokens.shape
        h = self.stem(tokens) + self.pos_embed[:, :K, :]

        enc_hiddens = []
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            if str(i) in self.enc_films:
                h = self.enc_films[str(i)](h, cond)
            enc_hiddens.append(h)

        for i, block in enumerate(self.dec_blocks):
            if i < len(self.skip_projs):
                h = h + self.skip_projs[i](enc_hiddens[len(enc_hiddens) - 1 - i])
            h = block(h)

        rho = self.rho_head(h)                   # (B, K, 2)

        pooled = h.mean(dim=1)
        log_Ne = self.ne_head(pooled).squeeze(-1)  # (B,)
        return {"rho": rho, "log_Ne": log_Ne}

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor,
                mask: torch.Tensor | None = None) -> dict:
        """Run only valid prefixes so backward Mamba never scans padded tokens."""
        B, K, _ = tokens.shape
        if mask is None or bool(torch.all(mask > 0.5)):
            return self._forward_unpadded(tokens, cond)
        rows, ne_rows = [], []
        for b in range(B):
            length = int((mask[b] > 0.5).sum().item())
            if length < 1:
                raise ValueError("each sequence must contain at least one valid token")
            expected = torch.arange(K, device=mask.device) < length
            if not bool(torch.equal(mask[b] > 0.5, expected)):
                raise ValueError("token masks must be contiguous valid prefixes")
            out = self._forward_unpadded(tokens[b:b + 1, :length], cond[b:b + 1])
            rows.append(F.pad(out["rho"], (0, 0, 0, K - length)))
            ne_rows.append(out["log_Ne"])
        return {"rho": torch.cat(rows, dim=0), "log_Ne": torch.cat(ne_rows, dim=0)}

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        params = {n: p for n, p in self.named_parameters() if p.requires_grad}
        decay = [p for p in params.values() if p.dim() >= 2]
        nodecay = [p for p in params.values() if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = "fused" in inspect.signature(torch.optim.AdamW).parameters
        extra = dict(fused=True) if (fused and device_type == "cuda") else {}
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **extra)


class GRUSeq2SeqModel(nn.Module):
    """ReLERNN-seq2seq steelman: raw genotype matrix -> bidirectional GRU
    (return_sequences) -> per-SNP-interval head. Same forward signature, heads, loss, and
    training data as FastRhoModel; the only differences from fastrho are the raw-genotype
    input (vs LD-aware SNP-token features) and the GRU backbone (vs bidirectional Mamba)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model
        self.embed = nn.Sequential(nn.Linear(config.n_features, d), nn.GELU(),
                                   nn.Dropout(config.dropout))
        self.pos_embed = nn.Parameter(torch.zeros(1, config.context_len, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nl = getattr(config, "n_gru_layers", 3)
        self.gru = nn.GRU(input_size=d, hidden_size=d // 2, num_layers=nl,
                          batch_first=True, bidirectional=True,
                          dropout=config.dropout if nl > 1 else 0.0)
        self.norm = nn.LayerNorm(d)
        self.film = FiLMLayer(d, cond_dim=config.cond_dim)
        self.rho_head = UncertaintyHead(d)
        self.ne_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_unpadded(self, tokens, cond):
        B, K, _ = tokens.shape
        h = self.embed(tokens) + self.pos_embed[:, :K, :]
        h, _ = self.gru(h)                       # (B, K, d) -- per-token (return_sequences)
        h = self.film(self.norm(h), cond)
        rho = self.rho_head(h)
        pooled = h.mean(dim=1)
        return {"rho": rho, "log_Ne": self.ne_head(pooled).squeeze(-1)}

    def forward(self, tokens, cond, mask=None):
        B, K, _ = tokens.shape
        if mask is None or bool(torch.all(mask > 0.5)):
            return self._forward_unpadded(tokens, cond)
        rows, ne_rows = [], []
        for b in range(B):
            length = int((mask[b] > 0.5).sum().item())
            if length < 1:
                raise ValueError("each sequence must contain at least one valid token")
            expected = torch.arange(K, device=mask.device) < length
            if not bool(torch.equal(mask[b] > 0.5, expected)):
                raise ValueError("token masks must be contiguous valid prefixes")
            out = self._forward_unpadded(tokens[b:b + 1, :length], cond[b:b + 1])
            rows.append(F.pad(out["rho"], (0, 0, 0, K - length)))
            ne_rows.append(out["log_Ne"])
        return {"rho": torch.cat(rows, dim=0), "log_Ne": torch.cat(ne_rows, dim=0)}

    configure_optimizers = FastRhoModel.configure_optimizers


def build_model(config):
    """Return the model selected by config.backbone ('mamba' -> fastrho, 'gru' -> steelman)."""
    if getattr(config, "backbone", "mamba") == "gru":
        return GRUSeq2SeqModel(config)
    return FastRhoModel(config)
