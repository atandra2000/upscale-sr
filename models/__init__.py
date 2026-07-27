"""Upscale-SR models package — build helpers wired to the YAML config."""
from __future__ import annotations

from .vae_frozen import FrozenVAE, load_frozen_vae
from .sr_unet import SRUNet, build_sr_unet, fa2_available
from .ssm_refiner import SSMRefiner, build_ssm_refiner, mamba_available
from .scheduler import (DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler,
                        build_ddpm, build_ddim, build_dpm_solver)
from .cond import sr_cond_input

__all__ = [
    "FrozenVAE", "load_frozen_vae",
    "SRUNet", "build_sr_unet", "fa2_available",
    "SSMRefiner", "build_ssm_refiner", "mamba_available",
    "DDPMScheduler", "DDIMScheduler", "DPMSolverMultistepScheduler",
    "build_ddpm", "build_ddim", "build_dpm_solver",
    "sr_cond_input",
]
