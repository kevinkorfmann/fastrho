import torch, numpy as np
from fastrho.config import PRESETS
from fastrho.model import build_model
from fastrho.train import hetero_nll
from fastrho.raw_features import RawGenotypeFeaturizer, n_features

cfg = PRESETS["gru_seq2seq"]
print("gru cfg: backbone", cfg.backbone, "featurizer", cfg.featurizer, "n_features", cfg.n_features)
m = build_model(cfg).to("cuda:0")
print("model:", type(m).__name__, "params %.2fM" % (sum(p.numel() for p in m.parameters()) / 1e6))
B, K, F = 4, cfg.context_len, cfg.n_features
x = torch.randn(B, K, F, device="cuda:0")
cond = torch.randn(B, 2, device="cuda:0")
mask = torch.ones(B, K, device="cuda:0")
out = m(x, cond, mask)
print("rho", tuple(out["rho"].shape), "log_Ne", tuple(out["log_Ne"].shape))
loss = hetero_nll(out["rho"][..., 0], out["rho"][..., 1], torch.randn(B, K, device="cuda:0"), mask)
loss.backward()
print("loss", float(loss), "grad_ok",
      all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
gm = (np.random.rand(20, 60) < 0.3).astype(np.int8)
pos = np.sort(np.random.uniform(0, 1e5, 60))
t = RawGenotypeFeaturizer()(gm, pos, {})
print("raw tokens", t["tokens"].shape, "expect", (60, n_features()),
      "finite", bool(np.isfinite(t["tokens"]).all()))
print("SMOKE_OK")
