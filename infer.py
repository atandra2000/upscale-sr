"""Upscale-SR — product inference CLI.

Loads ``sr_x4_final.safetensors`` + the frozen SD1.5 VAE, runs DDIM in latent
space + the SSM refiner in pixel space, writes the 4× HR image.
"""
from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils.config import load_config
from utils.checkpoint import load_upsr_state
from models import (build_sr_unet, build_ssm_refiner, build_ddim, build_dpm_solver,
                    sr_cond_input)
from models.vae_frozen import load_frozen_vae
from utils.memory import apply_channels_last


def load_image_pil(img: Image.Image) -> torch.Tensor:
    """PIL RGB image → (1,3,H,W) tensor in [-1,1] (shared with demo_gradio)."""
    import numpy as np
    arr = torch.from_numpy(np.asarray(img.convert("RGB")).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
    return arr


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Clamp a (1,3,H,W) tensor ([-1,1] or [0,1]) to a PIL RGB image."""
    import numpy as np
    if t.min() < -0.5:
        t = (t + 1) / 2
    t = t.clamp(0, 1)
    arr = (t[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)


def _save_image(t: torch.Tensor, path: str) -> None:
    """Save a (1,3,H,W) tensor in [0,1] (or [-1,1]) to ``path`` as PNG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _tensor_to_pil(t).save(path)


@torch.no_grad()
def _safe_reflect_pad(x: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    """Reflect-pad the right/bottom of ``x`` by (pad_h, pad_w).

    torch's ``reflect`` mode requires pad < dim, which breaks for inputs
    smaller than the pad amount.  Reflect as much as possible (dim-1), then
    constant-pad the remainder.
    """
    _, _, H, W = x.shape
    rh = min(pad_h, max(H - 1, 0))
    rw = min(pad_w, max(W - 1, 0))
    ch = pad_h - rh
    cw = pad_w - rw
    if rh or rw:
        x = F.pad(x, (0, rw, 0, rh), mode="reflect")
    if ch or cw:
        x = F.pad(x, (0, cw, 0, ch), mode="constant", value=0)
    return x


def upscale(
    lr_img: torch.Tensor, unet, refiner, vae, ddim, device,
    scale: int = 4, steps: int = 30, crop: int = 0,
) -> torch.Tensor:
    """Run the full SR pipeline on (1,3,h,w) in [-1,1] → (1,3,H,W) in [-1,1].

    ``crop > 0`` tiles large images (hann-window blend, 25% overlap) to bound VRAM.
    """
    _, _, h, w = lr_img.shape
    if crop and (h > crop or w > crop):
        return _tile_upscale(lr_img, unet, refiner, vae, ddim, device,
                             scale, steps, crop)
    lr_img = lr_img.to(device)
    # pad H/W to multiples of (scale*8) so the VAE + DDIM are exact.  Reflect
    # padding mirrors the boundary (no seam); but torch reflect requires
    # pad < dim, so for inputs smaller than scale*8 we reflect up to dim-1
    # and constant-pad the remainder.  Inputs below scale*8 are uncommon in
    # practice (real photos are larger) but the CLI must not crash on them.
    Hl_factor = scale * 8
    pad_h = (Hl_factor - h % Hl_factor) % Hl_factor
    pad_w = (Hl_factor - w % Hl_factor) % Hl_factor
    padded = _safe_reflect_pad(lr_img, pad_h, pad_w)

    lr01 = (padded + 1) / 2
    lr_lat = vae.encode(lr01)
    H, W = padded.shape[-2] * scale, padded.shape[-1] * scale
    Hl, Wl = H // 8, W // 8
    z = torch.randn(1, 4, Hl, Wl, device=device)
    ddim.set_timesteps(steps, device=device)
    for t in ddim.timesteps:
        eps = unet(sr_cond_input(lr_lat, z, up=unet.lr_up), t.expand(1))
        z = ddim.step(eps, t, z)
    img = vae.decode(z)            # (1,3,H,W) [-1,1]
    refined = refiner(img)
    # remove padding
    refined = refined[..., :h * scale, :w * scale]
    return refined


def _tile_upscale(lr_img, unet, refiner, vae, ddim, device, scale, steps, crop):
    """Tile-based upscale for large images (overlap-blend)."""
    _, _, H, W = lr_img.shape
    stride = crop - crop // 4                       # 75% overlap → 25% blend
    out_H, out_W = H * scale, W * scale
    acc = torch.zeros(1, 3, out_H, out_W, device=device)
    wsum = torch.zeros(1, 1, out_H, out_W, device=device)
    win1d = torch.hann_window(crop, device=device)
    win2d = win1d[:, None] * win1d[None, :]
    win2d = win2d.unsqueeze(0).unsqueeze(0)
    # interpolate once — same args for acc and wsum
    win2d_up = F.interpolate(win2d, scale_factor=scale, mode="bilinear",
                             align_corners=False)
    for y in range(0, H, stride):
        for x in range(0, W, stride):
            y1 = min(y, H - crop); x1 = min(x, W - crop)
            tile = lr_img[:, :, y1:y1 + crop, x1:x1 + crop]
            sr_tile = upscale(tile, unet, refiner, vae, ddim, device,
                               scale, steps, crop=0)
            sy, sx = y1 * scale, x1 * scale
            acc[:, :, sy:sy + crop * scale, sx:sx + crop * scale] += sr_tile * win2d_up
            wsum[:, :, sy:sy + crop * scale, sx:sx + crop * scale] += win2d_up
    return acc / wsum.clamp(min=1e-6)


def main():
    ap = argparse.ArgumentParser(description="Upscale-SR — 4× photo super-resolution")
    ap.add_argument("--in", dest="inp", required=True, help="input LR image")
    ap.add_argument("--out", required=True, help="output HR image path")
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--steps", type=int, default=None, help="DDIM/DPM++ steps (default from cfg: 30)")
    ap.add_argument("--fast", action="store_true", help="10-step fast mode")
    ap.add_argument("--sampler", choices=["ddim", "dpm++"], default="dpm++",
                    help="inference ODE sampler (dpm++ [2x faster] or ddim)")
    ap.add_argument("--ckpt", default=None, help="safetensors ckpt (default from cfg)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--crop", type=int, default=512,
                    help="tile size for large images (0 = whole image)")
    ap.add_argument("--device", default=None, help="cuda / cpu (auto)")
    ap.add_argument("--stub-vae", action="store_true",
                    help="use a no-op VAE (local test without SD weights)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    mcfg = cfg["model"]
    icfg = cfg.get("infer", {})

    ckpt_path = args.ckpt or icfg.get("ckpt", "sr_x4_final.safetensors")
    print(f"[upscale-sr] device={device} | ckpt={ckpt_path} | sampler={args.sampler}")
    if args.stub_vae:
        from training.train import _StubVAE
        vae = _StubVAE().to(device)
    else:
        vae = load_frozen_vae(mcfg, device)
    unet = build_sr_unet(mcfg).to(device).eval()
    refiner = build_ssm_refiner(mcfg).to(device).eval()
    if device.type == "cuda":
        apply_channels_last(unet, refiner, device=device)
    # Shape-filter load tolerates channel-count drift in test ckpts.
    unet_sd, refiner_sd = load_upsr_state(ckpt_path, unet=unet, refiner=refiner, device=str(device))
    unet.load_state_dict(unet_sd, strict=False)
    refiner.load_state_dict(refiner_sd, strict=False)

    if args.sampler == "dpm++":
        sampler = build_dpm_solver(mcfg).to(device)
    else:
        sampler = build_ddim(mcfg).to(device)

    steps = args.steps or (icfg.get("fast_steps", 10) if args.fast
                           else mcfg["scheduler"].get("infer_steps", 30))

    lr = load_image_pil(Image.open(args.inp))
    print(f"[upscale-sr] LR {tuple(lr.shape)} → {args.scale}× | {steps} steps ({args.sampler}) | tiling crop={args.crop}")
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext():
            hr = upscale(lr, unet, refiner, vae, sampler, device,
                         scale=args.scale, steps=steps, crop=args.crop)
    dt = time.perf_counter() - t0
    _save_image(hr, args.out)
    print(f"[upscale-sr] HR {tuple(hr.shape)} written → {args.out} | {dt:.2f}s")


if __name__ == "__main__":
    main()