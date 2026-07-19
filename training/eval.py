"""Evaluation — PSNR / LPIPS on DIV2K-val (×4, Real-ESRGAN degradation, fixed seed).

The eval pipeline (DESIGN §6 / EXECUTION-PLAN acceptance):
  1. load val (LR, HR) pairs with a **fixed degradation seed** (reproducible);
  2. run DDIM (30 steps) on the LR latent with **EMA** weights;
  3. run the **refiner** (EMA) on the diffusion output image;
  4. compute PSNR + LPIPS between the refined HR and the clean HR target.

Used both as an in-loop eval (called from ``train.py`` every ``eval_every``
iters) and as a standalone script (``python -m training.eval``) for the
final reported number.
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils.config import load_config
from utils.logging import is_main_process, setup_logger
from utils.losses import build_lpips, lpips_loss_fn
from data.sr_dataset import build_val_dataset
from models import build_ddim, sr_cond_input


def _psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> float:
    """PSNR (dB) between x, y in [0,1]. Higher = better."""
    mse = F.mse_loss(x.clamp(0, 1), y.clamp(0, 1)).item()
    if mse <= 1e-12:
        return 100.0
    return 10.0 * math.log10(data_range ** 2 / mse)


@torch.no_grad()
def _sr_inference(model, ddim, lr, device, steps=30):
    """Run the full SR pipeline on one LR batch → HR image (B,3,H,W) in [0,1].

    DDIM in latent space conditioned on the LR latent, then the refiner in
    pixel space.
    """
    vae = model["vae"]
    unet = model["unet_compiled"]
    refiner = model["refiner_compiled"]
    lr01 = (lr + 1) / 2                                       # [0,1]
    lr_lat = vae.encode(lr01)                                 # (B,4,h/8,w/8)
    H, W = lr.shape[-2] * 4, lr.shape[-1] * 4                  # 4× upscale
    Hl, Wl = H // 8, W // 8

    # start from noise at the HR latent size
    z = torch.randn(lr.shape[0], 4, Hl, Wl, device=device)
    ddim.set_timesteps(steps, device=device)
    for t in ddim.timesteps:
        unet_in = sr_cond_input(lr_lat, z)
        eps = unet(unet_in, t.expand(lr.shape[0]))
        z = ddim.step(eps, t, z)

    img = vae.decode(z)                                       # (B,3,H,W) [-1,1]
    refined = refiner(img)                                    # (B,3,H,W) [-1,1]
    return (refined + 1) / 2                                  # [0,1]


@torch.no_grad()
def evaluate(model, cfg, device, ema_unet=None, ema_refiner=None,
             max_imgs=20, steps=None):
    """Run DDIM+refiner on the val set, return mean PSNR / LPIPS."""
    ddim = build_ddim(cfg["model"]).to(device)
    steps = steps or cfg["model"]["scheduler"].get("infer_steps", 30)
    val_ds = build_val_dataset(cfg)
    if len(val_ds) == 0:
        return {"psnr": 0.0, "lpips": 0.0, "n": 0}
    lpips_fn = build_lpips(device)

    # swap in EMA weights for inference
    unet_raw = model["ddp_unet"] if hasattr(model["ddp_unet"], "module") else model["unet_compiled"]
    backup_u = ema_unet.apply_shadow(unet_raw) if ema_unet is not None else {}
    backup_r = ema_refiner.apply_shadow(model["refiner_compiled"]) if ema_refiner is not None else {}
    try:
        psnrs, lpips_vals = [], []
        n = min(max_imgs, len(val_ds))
        for i in range(n):
            s = val_ds[i]
            lr = s["lr"].unsqueeze(0).to(device)
            hr01 = ((s["hr"] + 1) / 2).unsqueeze(0).to(device)
            out = _sr_inference(model, ddim, lr, device, steps=steps)
            # crop to common size (decode may differ by ±1 px)
            H = min(out.shape[-2], hr01.shape[-2])
            W = min(out.shape[-1], hr01.shape[-1])
            psnrs.append(_psnr(out[..., :H, :W], hr01[..., :H, :W]))
            lpips_vals.append(float(lpips_loss_fn(lpips_fn, out[..., :H, :W],
                                                   hr01[..., :H, :W]).detach()))
        return {"psnr": sum(psnrs) / len(psnrs),
                "lpips": sum(lpips_vals) / len(lpips_vals), "n": n}
    finally:
        if ema_unet is not None:
            ema_unet.restore(unet_raw, backup_u)
        if ema_refiner is not None:
            ema_refiner.restore(model["refiner_compiled"], backup_r)


def _main():
    import torch.nn as nn
    ap = argparse.ArgumentParser(description="Standalone Upscale-SR eval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ckpt", required=True, help="safetensors checkpoint (sr_step_*.safetensors)")
    ap.add_argument("--max-imgs", type=int, default=100)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()

    from models import build_sr_unet, build_ssm_refiner
    from models.vae_frozen import load_frozen_vae
    from utils.checkpoint import CheckpointManager
    mcfg = cfg["model"]
    vae = load_frozen_vae(mcfg, device)
    unet = build_sr_unet(mcfg).to(device).eval()
    refiner = build_ssm_refiner(mcfg).to(device).eval()
    if torch.cuda.is_available():
        unet = unet.to(memory_format=torch.channels_last)
        refiner = refiner.to(memory_format=torch.channels_last)

    # load weights
    from safetensors.torch import load_file
    sd = load_file(args.ckpt, device=str(device))
    unet_sd = {k.removeprefix("unet."): v for k, v in sd.items() if k.startswith("unet.")}
    ref_sd = {k.removeprefix("refiner."): v for k, v in sd.items() if k.startswith("refiner.")}
    unet.load_state_dict(unet_sd, strict=False)
    refiner.load_state_dict(ref_sd, strict=False)

    model = {"vae": vae, "unet_compiled": unet, "refiner_compiled": refiner,
             "ddp_unet": unet}
    steps = args.steps or (cfg["model"]["scheduler"].get("fast_steps", 10) if args.fast
                          else cfg["model"]["scheduler"].get("infer_steps", 30))
    metrics = evaluate(model, cfg, device, ema_unet=None, ema_refiner=None,
                       max_imgs=args.max_imgs, steps=steps)
    logger.info("FINAL EVAL: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    _main()