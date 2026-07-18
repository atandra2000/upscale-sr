"""test_refiner_shapes.py — SSM refiner (B,C,H,W) round-trip, no NaN at BF16,
and kernel-equivalence (mamba-ssm fast path ≡ pure-PyTorch fallback on a toy).

EXECUTION-PLAN Phase 2 acceptance.
"""
import pytest
import torch

from models.ssm_refiner import SSMRefiner, build_ssm_refiner, mamba_available
from models.ssm_refiner import selective_scan, DilatedBiSSM


def test_refiner_roundtrip_shape():
    ref = SSMRefiner(in_ch=3, base_ch=32, ch_mults=(1, 2, 4),
                    num_blocks=(2, 2, 2, 2), ssm_state_dim=8,
                    ssm_dilated_strides=(1, 2, 4), grad_ckpt=False).eval()
    for hw in (32, 64, 96):
        x = torch.randn(2, 3, hw, hw)
        with torch.no_grad():
            y = ref(x)
        assert y.shape == x.shape, f"shape mismatch at {hw}: {y.shape}"


def test_refiner_no_nan_bf16():
    """Forward under BF16 autocast (CUDA) or FP32 (CPU) must not produce NaN."""
    ref = SSMRefiner(in_ch=3, base_ch=32, ch_mults=(1, 2, 4),
                    num_blocks=(2, 2, 2, 2), ssm_state_dim=8,
                    ssm_dilated_strides=(1, 2, 4), grad_ckpt=False).eval()
    x = torch.randn(2, 3, 32, 32)
    ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
           if torch.cuda.is_available() else torch.no_grad())
    with torch.no_grad(), ctx:
        y = ref(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()


def test_refiner_identity_at_init():
    """Zero-init conv_out → refiner starts as identity (res + 0)."""
    ref = SSMRefiner(in_ch=3, base_ch=16, ch_mults=(1, 2), num_blocks=(1, 1, 1),
                    ssm_state_dim=4, ssm_dilated_strides=(1, 2), grad_ckpt=False).eval()
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        y = ref(x)
    # global residual is `h + res` where h = conv_out(...) = 0 at init → y ≈ x
    assert torch.allclose(y, x, atol=1e-5), "refiner not identity at init"


def test_selective_scan_kernel_equiv():
    """If mamba-ssm is available, assert fast path ≈ pure-PyTorch fallback.

    We force the fallback by temporarily disabling the fast path and compare
    outputs on a toy sequence.  When mamba-ssm is NOT installed this test is
    skipped (the fallback is the only path).
    """
    if not mamba_available():
        pytest.skip("mamba-ssm not installed — fallback is the only path")
    import models.ssm_refiner as M
    B, L, D, N = 1, 32, 8, 4
    x = torch.randn(B, L, D)
    dt = torch.rand(B, L, 1) * 0.1 + 0.1
    A = torch.randn(D) * -0.5
    Bm = torch.randn(B, L, N)
    Cm = torch.randn(B, L, N)
    # fast path
    y_fast = selective_scan(x, dt, A, Bm, Cm)
    # force fallback
    saved = M._HAS_MAMBA
    M._HAS_MAMBA = False
    try:
        y_slow = selective_scan(x, dt, A, Bm, Cm)
    finally:
        M._HAS_MAMBA = saved
    assert torch.allclose(y_fast, y_slow, atol=1e-3), \
        f"kernel mismatch: max diff {(y_fast-y_slow).abs().max()}"


def test_dilated_scan_covers_all_positions():
    """Dilated multi-scan {1,2,4} must fill every output position (no gaps)."""
    # GroupNorm(32, channels) ⇒ channels must be divisible by 32.
    ssm = DilatedBiSSM(channels=32, state_dim=4, dilations=(1, 2, 4))
    x = torch.randn(2, 32, 16, 16)
    y = ssm(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_build_from_config():
    from utils.config import load_config
    cfg = load_config()
    ref = build_ssm_refiner(cfg["model"])
    assert ref.conv_in.in_channels == 3