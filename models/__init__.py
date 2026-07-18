"""Upscale-SR models package — build helpers wired to the YAML config."""
from __future__ import annotations

import torch.nn as nn

from .vae_frozen import FrozenVAE, load_frozen_vae
from .sr_unet import SRUNet, build_sr_unet, fa2_available
from .ssm_refiner import SSMRefiner, build_ssm_refiner, mamba_available
from .scheduler import DDPMScheduler, DDIMScheduler, build_ddpm, build_ddim

__all__ = [
    "FrozenVAE", "load_frozen_vae",
    "SRUNet", "build_sr_unet", "fa2_available",
    "SSMRefiner", "build_ssm_refiner", "mamba_available",
    "DDPMScheduler", "DDIMScheduler", "build_ddpm", "build_ddim",
    "build_model",
]


def build_model(cfg: dict, device) -> dict:
    """Build all model components from a config dict and move to device.

    Returns a dict with ``vae`` (frozen), ``unet`` (trainable),
    ``refiner`` (trainable), ``ddpm`` (train scheduler), ``ddim`` (infer sampler).
    """
    model_cfg = cfg.get("model", cfg)
    vae = load_frozen_vae(model_cfg, device)
    unet = build_sr_unet(model_cfg).to(device)
    refiner = build_ssm_refiner(model_cfg).to(device)
    ddpm = build_ddpm(model_cfg).to(device)
    ddim = build_ddim(model_cfg).to(device)
    return {"vae": vae, "unet": unet, "refiner": refiner, "ddpm": ddpm, "ddim": ddim}