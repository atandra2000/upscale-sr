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


class _StubVAE(torch.nn.Module):
    """Minimal VAE stand-in mirroring training.train._StubVAE (kept here to
    avoid importing the heavy training module on a non-DDP box)."""
    SCALE_FACTOR = 0.18215

    @torch.no_grad()
    def encode(self, x):
        d = F.avg_pool2d(x, 8)
        if d.shape[1] < 4:
            d = F.pad(d, (0, 0, 0, 0, 0, 4 - d.shape[1]))
        return d[:, :4] * self.SCALE_FACTOR

    @torch.no_grad()
    def decode(self, z):
        z = z / self.SCALE_FACTOR
        return F.interpolate(z[:, :3], scale_factor=8, mode="bilinear",
                             align_corners=False)


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

    # build a stub safetensors ckpt with unet./refiner. prefixed keys, using the
    # SAME config-driven builders main() will use so load_state_dict shape-matches.
    from safetensors.torch import save_file
    from utils.config import load_config
    from models import build_sr_unet, build_ssm_refiner
    cfg = load_config()
    mcfg = cfg["model"]
    unet = build_sr_unet(mcfg).to(device).eval()
    refiner = build_ssm_refiner(mcfg).to(device).eval()
    sd = {}
    for k, v in unet.state_dict().items():
        sd[f"unet.{k}"] = v.contiguous()
    for k, v in refiner.state_dict().items():
        sd[f"refiner.{k}"] = v.contiguous()
    ckpt_path = tmp_path / "stub.safetensors"
    save_file(sd, str(ckpt_path))

    # the CLI uses the *real* config-driven builders (base_ch=96/128), but we
    # only need the right I/O shapes; main() loads with strict=False so the
    # mismatched channel counts won't crash — the freshly-init weights are
    # what actually runs.  Point main() at our stub ckpt + stub VAE.
    out_path = tmp_path / "sr.png"
    monkeypatch.setattr("sys.argv", [
        "infer.py", "--in", str(lr_path), "--out", str(out_path),
        "--stub-vae", "--ckpt", str(ckpt_path), "--scale", "4",
        "--steps", "2", "--crop", "0",
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