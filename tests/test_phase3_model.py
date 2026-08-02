"""Phase-3: FastRhoModel forward/backward + loss (GPU-only; Mamba2 needs CUDA)."""

import pytest

torch = pytest.importorskip("torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba2 requires CUDA")


@cuda
def test_forward_and_loss_backward():
    from fastrho.config import PRESETS
    from fastrho.model import FastRhoModel
    from fastrho.train import hetero_nll

    cfg = PRESETS["small"]
    dev = "cuda:0"
    model = FastRhoModel(cfg).to(dev)

    B, K, F = 3, cfg.context_len, cfg.n_features
    tokens = torch.randn(B, K, F, device=dev)
    cond = torch.randn(B, cfg.cond_dim, device=dev)
    token_mask = torch.ones(B, K, device=dev)
    token_mask[0, K // 2:] = 0.0                   # a padded sample
    target = torch.randn(B, K, device=dev) * 3 - 8
    target_mask = token_mask.clone()
    target_mask[:, -1] = 0.0

    with torch.autocast("cuda", dtype=torch.float16):
        out = model(tokens, cond, token_mask)
    assert out["rho"].shape == (B, K, 2)
    assert out["log_Ne"].shape == (B,)

    mu, logv = out["rho"][..., 0].float(), out["rho"][..., 1].float()
    loss = hetero_nll(mu, logv, target, target_mask)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
    print(f"forward/loss/backward OK; loss={float(loss):.3f} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
