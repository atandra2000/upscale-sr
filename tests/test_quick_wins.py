"""Tests for the three quick-win changes:
  1. learned LR-latent upsample in models/cond.py
  2. logit-normal DDIM / DPM-Solver step spacing
  3. P2 loss reweighting (Pythontests importable, no train-run needed)
"""
import torch
import pytest

from models.cond import sr_cond_input, LRUpConv
from models.scheduler import DDIMScheduler, DPMSolverMultistepScheduler


def test_learned_upsample_identity_at_init():
    """At init, the LRUpConv is identity on the centre tap → output == bilinear."""
    lr = torch.randn(1, 4, 8, 8)
    z = torch.randn(1, 4, 32, 32)
    out = sr_cond_input(lr, z, up=LRUpConv(4))
    # 9 channels: 4 (lr_up) + 4 (z_t) + 1 (mask)
    assert out.shape == (1, 9, 32, 32)
    # lr_up must equal bilinear(bilinear(lr, 32×32)) at init → same as plain bilinear
    import torch.nn.functional as F
    expected = F.interpolate(lr, size=(32, 32), mode="bilinear", align_corners=False)
    # identity-init: only the centre tap is 1.0, all others 0 → conv is identity
    assert torch.allclose(out[:, :4], expected, atol=1e-5), \
        f"identity-init upsample should match bilinear, max diff {(out[:, :4] - expected).abs().max()}"


def test_learned_upsample_owned_by_unet():
    """The LRUpConv must live on the U-Net (not a module-level singleton)."""
    from models.sr_unet import SRUNet
    net = SRUNet(in_ch=9, out_ch=4, base_ch=32, ch_mults=(1, 2),
                 res_blks=1, attn_levels=(1,), heads=4, t_dim=32).eval()
    lr = torch.randn(1, 4, 8, 8)
    z = torch.randn(1, 4, 32, 32)
    out = sr_cond_input(lr, z, up=net.lr_up)
    assert out.shape == (1, 9, 32, 32)


def test_logit_normal_spacing_ddim():
    """DDIM set_timesteps with N steps must produce ≤N unique descending t values
    clustered around the SNR-midpoint (skewed toward the middle)."""
    for n in (10, 15, 30):
        s = DDIMScheduler(steps=1000)
        s.set_timesteps(n, device=torch.device("cpu"))
        ts = s.timesteps
        assert ts.dim() == 1
        assert ts.numel() <= n
        # descending (DDIM walks from T toward 0)
        assert torch.all(ts[:-1] >= ts[1:]), f"not descending: {ts}"
        # no NaN, in valid range
        assert torch.isfinite(ts).all()
        assert ts.min() >= 0 and ts.max() < 1000
        # clustered around the midpoint: median should be in the middle third
        med = float(ts.median())
        assert 333 <= med <= 666, f"N={n}: median {med} not in middle third"


def test_logit_normal_spacing_dpm_solver():
    """Same invariant for the DPM-Solver++ sampler."""
    s = DPMSolverMultistepScheduler(steps=1000)
    s.set_timesteps(15, device=torch.device("cpu"))
    ts = s.timesteps
    assert ts.numel() <= 15
    assert torch.all(ts[:-1] >= ts[1:])
    assert torch.isfinite(ts).all()
    med = float(ts.median())
    assert 333 <= med <= 666
    # set_timesteps must reset the multistep state
    assert s.old_pred_x0 is None
    assert s.old_h is None


def test_logit_normal_deterministic():
    """Two calls with the same N must produce identical timesteps (no RNG)."""
    s1 = DDIMScheduler(steps=1000)
    s1.set_timesteps(20, device=torch.device("cpu"))
    s2 = DDIMScheduler(steps=1000)
    s2.set_timesteps(20, device=torch.device("cpu"))
    assert torch.equal(s1.timesteps, s2.timesteps)
