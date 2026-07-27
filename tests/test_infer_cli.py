"""test_infer_cli.py — end-to-end LR → SR round-trip on CPU with the stub VAE.

Builds a tiny LR image, runs ``infer.upscale`` with the stub VAE + freshly
init U-Net/refiner + DDIM, and asserts the output is exactly ``scale``× the
input spatial size.  This is the EXECUTION-PLAN Phase 2 "infer round-trip"
gate (also run on the GPU pod before training).
"""
import os
import pytest
import torch
import torch.nn.functional as F

from infer import upscale
from models.sr_unet import SRUNet
from models.ssm_refiner import SSMRefiner
from models.scheduler import DDIMScheduler
from training.train import _StubVAE
from utils.config import load_config


def _pipeline(device, scale=4):
    vae = _StubVAE().to(device)
    unet = SRUNet(in_ch=9, out_ch=4, base_ch=64, ch_mults=(1, 2, 4),
                 res_blks=1, attn_levels=(1, 2), heads=8, t_dim=64,
                 grad_ckpt=False).to(device).eval()
    refiner = SSMRefiner(in_ch=3, base_ch=32, ch_mults=(1, 2, 4),
                        num_blocks=(2, 2, 2, 2), ssm_state_dim=8,
                        ssm_dilated_strides=(1, 2, 4),
                        grad_ckpt=False).to(device).eval()
    ddim = DDIMScheduler().to(device)
    return unet, refiner, vae, ddim


@pytest.mark.parametrize("hw", [16, 32, 48])
def test_upscale_shape(hw):
    """Output H/W == scale × input H/W for several input sizes (incl. ones
    that need reflect-padding to a multiple of scale*8)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet, refiner, vae, ddim = _pipeline(device, scale=4)
    lr = torch.randn(1, 3, hw, hw, device=device).clamp(-1, 1)
    with torch.no_grad():
        sr = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=4,
                     crop=0)
    assert sr.shape == (1, 3, hw * 4, hw * 4), \
        f"expected (1,3,{hw*4},{hw*4}), got {tuple(sr.shape)}"


def test_upscale_no_nan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet, refiner, vae, ddim = _pipeline(device)
    lr = torch.randn(1, 3, 32, 32, device=device).clamp(-1, 1)
    with torch.no_grad():
        sr = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=4,
                     crop=0)
    assert not torch.isnan(sr).any()
    assert not torch.isinf(sr).any()


def test_upscale_finite_range():
    """At init the output is finite (no NaN/Inf) and bounded to a sane
    magnitude — the refiner is identity-init so it does not blow up the
    VAE decode, and the DDIM trajectory with ~0 predicted noise is stable."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet, refiner, vae, ddim = _pipeline(device)
    lr = torch.zeros(1, 3, 32, 32, device=device)  # mid-gray input
    with torch.no_grad():
        sr = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=4,
                     crop=0)
    assert torch.isfinite(sr).all()
    # identity-init refiner ⇒ output magnitude stays within a few× of the
    # stub-VAE decode range (not unbounded growth).
    assert sr.abs().max() < 50.0


def test_cli_writes_output(tmp_path, monkeypatch):
    """Run the actual CLI entrypoint end-to-end: ``infer.main()`` reads a
    stub safetensors ckpt + stub VAE and writes an sr.png at 4× the input.

    This is the EXECUTION-PLAN Phase 2 product contract: "inference CLI
    ships and produces a 4× image".

    Uses a small config (configs/sr_smoke_test.yaml) so the test runs on
    any CUDA card, including the sm_75 dev box that cannot dispatch the
    production 1024-group depthwise conv under BF16.  The full CLI plumbing
    (argparse, model build, ckpt load, DDIM step loop, refiner, image
    save) is still exercised.  The production-shape refiner is validated
    on the 2× RTX 5090 (sm_120) pod.
    """
    from PIL import Image
    import numpy as np
    import infer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build a tiny RGB LR image (16×16)
    lr_np = (np.random.default_rng(0).integers(0, 255, (16, 16, 3))
             ).astype("uint8")
    lr_path = tmp_path / "lr.jpg"
    Image.fromarray(lr_np).save(str(lr_path))

    # small config so the test runs on dev-box GPUs (avoids the
    # production 1024-group depthwise conv that sm_75 cannot dispatch).
    from pathlib import Path
    cfg_path = Path(infer.__file__).parent / "configs" / "sr_smoke_test.yaml"
    mcfg_small = load_config(str(cfg_path))["model"]

    # build a small safetensors ckpt with the small config
    from safetensors.torch import save_file
    unet = SRUNet(in_ch=mcfg_small["sr_unet"]["in_ch"],
                 out_ch=mcfg_small["sr_unet"]["out_ch"],
                 base_ch=mcfg_small["sr_unet"]["base_ch"],
                 ch_mults=tuple(mcfg_small["sr_unet"]["ch_mults"]),
                 res_blks=mcfg_small["sr_unet"]["res_blks"],
                 attn_levels=tuple(mcfg_small["sr_unet"]["attn_levels"]),
                 heads=mcfg_small["sr_unet"]["heads"],
                 t_dim=mcfg_small["sr_unet"]["t_dim"],
                 grad_ckpt=False).to(device).eval()
    refiner = SSMRefiner(in_ch=mcfg_small["refiner"]["in_ch"],
                        base_ch=mcfg_small["refiner"]["base_ch"],
                        ch_mults=tuple(mcfg_small["refiner"]["ch_mults"]),
                        num_blocks=tuple(mcfg_small["refiner"]["num_blocks"]),
                        ssm_state_dim=mcfg_small["refiner"]["ssm_state_dim"],
                        ssm_dilated_strides=tuple(mcfg_small["refiner"]["ssm_dilated_strides"]),
                        grad_ckpt=False).to(device).eval()
    sd = {}
    for k, v in unet.state_dict().items():
        sd[f"unet.{k}"] = v.contiguous()
    for k, v in refiner.state_dict().items():
        sd[f"refiner.{k}"] = v.contiguous()
    ckpt_path = tmp_path / "stub.safetensors"
    save_file(sd, str(ckpt_path))

    out_path = tmp_path / "sr.png"
    monkeypatch.setattr("sys.argv", [
        "infer.py", "--in", str(lr_path), "--out", str(out_path),
        "--stub-vae", "--ckpt", str(ckpt_path), "--scale", "4",
        "--steps", "2", "--crop", "0",
        "--config", str(cfg_path),
    ])
    infer.main()

    assert out_path.exists(), "CLI did not write the output image"
    out = Image.open(out_path)
    # SR output is 4× the LR input spatial size
    assert out.size == (16 * 4, 16 * 4), \
        f"expected (64,64), got {out.size}"


def test_tile_upscale_shape():
    """Tile path (crop>0) must produce the same overall output shape as the
    whole-image path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet, refiner, vae, ddim = _pipeline(device)
    hw = 48
    lr = torch.randn(1, 3, hw, hw, device=device).clamp(-1, 1)
    with torch.no_grad():
        whole = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=2,
                        crop=0)
        tiled = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=2,
                        crop=32)
    assert whole.shape == tiled.shape == (1, 3, hw * 4, hw * 4)