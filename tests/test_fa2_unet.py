"""test_fa2_unet.py — SRUNet forward under autocast (BF16 on CUDA, no-op on
CPU) with FP32 LayerNorm/GroupNorm produces (B,4,H,W) and is NaN-free.

EXECUTION-PLAN Phase 2 acceptance: "FA2 U-Net forward path under autocast +
FP32 LN".  On CPU / non-FA2 boxes the FA2SelfAttention falls back to
``F.scaled_dot_product_attention`` (flash backend where available), so the
test still exercises the full attention path.
"""
import pytest
import torch

from models.sr_unet import SRUNet, build_sr_unet, fa2_available


def _autocast(device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.no_grad()


def _small_unet():
    # base_ch must be divisible by 32 (GroupNorm(32, ...)); keep it tiny for CPU.
    return SRUNet(in_ch=9, out_ch=4, base_ch=64, ch_mults=(1, 2, 4),
                 res_blks=1, attn_levels=(1, 2), heads=8, t_dim=64,
                 grad_ckpt=False).eval()


def test_unet_output_shape():
    net = _small_unet().to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    Hl, Wl = 8, 8  # latent spatial size (== HR/8)
    x = torch.randn(2, 9, Hl, Wl, device=device)
    t = torch.randint(0, 1000, (2,), device=device)
    with torch.no_grad(), _autocast(device):
        eps = net(x, t)
    assert eps.shape == (2, 4, Hl, Wl), f"bad shape {eps.shape}"


def test_unet_no_nan_under_autocast():
    net = _small_unet().to(device := torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    x = torch.randn(1, 9, 8, 8, device=device)
    t = torch.tensor([500], device=device)
    with torch.no_grad(), _autocast(device):
        eps = net(x, t)
    assert not torch.isnan(eps).any()
    assert not torch.isinf(eps).any()


def test_unet_zero_init_predicts_zero_noise():
    """conv_out is zero-init → U-Net predicts ~0 noise at init (residual start)."""
    net = _small_unet().eval()
    x = torch.randn(1, 9, 8, 8)
    t = torch.tensor([250])
    with torch.no_grad():
        eps = net(x, t)
    assert torch.allclose(eps, torch.zeros_like(eps), atol=1e-5), \
        f"zero-init U-Net should predict ~0, got max {eps.abs().max()}"


def test_unet_channels_last_compatible():
    """channels_last memory format (Blackwell sm_120 mandate) must not break
    the forward."""
    if not torch.cuda.is_available():
        pytest.skip("channels_last is a CUDA memory-format test")
    net = _small_unet().to("cuda").to(memory_format=torch.channels_last).eval()
    x = torch.randn(1, 9, 16, 16, device="cuda").to(memory_format=torch.channels_last)
    t = torch.tensor([100], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eps = net(x, t)
    assert eps.shape == (1, 4, 16, 16)
    assert eps.is_contiguous(memory_format=torch.channels_last)


def test_fa2_self_attention_runs():
    """Directly exercise FA2SelfAttention — picks flash_attn if available, else
    sdpa flash backend, else sdpa math."""
    from models.sr_unet import FA2SelfAttention
    attn = FA2SelfAttention(channels=64, num_heads=8).eval()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        y = attn(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
    # sanity: FA2 availability flag is a boolean
    assert isinstance(fa2_available(), bool)


def test_build_from_config():
    from utils.config import load_config
    net = build_sr_unet(load_config())
    assert net.conv_in.in_channels == 9
    assert net.conv_out.out_channels == 4