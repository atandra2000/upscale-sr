"""Frozen SD1.5-class VAE — encode LR → latent, decode SR latent → HR image.

Permanently frozen; reused from ``stabilityai/sd-vae-ft-mse``.  The scale
factor (0.18215) normalises latents to unit variance for stable diffusion.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenVAE(nn.Module):
    """Thin frozen wrapper around a HuggingFace ``AutoencoderKL``.

    Uses ``.mean`` for deterministic encode.  SCALE_FACTOR normalises latents.
    """

    SCALE_FACTOR = 0.18215

    def __init__(self, model_id: str = "stabilityai/sd-vae-ft-mse",
                 dtype: torch.dtype = torch.bfloat16,
                 local_dir: str | None = None):
        super().__init__()
        from diffusers import AutoencoderKL

        src = local_dir or model_id
        self.vae = AutoencoderKL.from_pretrained(src, torch_dtype=dtype)
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.dtype = dtype
        self.scale_factor = self.SCALE_FACTOR

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Images in [-1, 1] → scaled latents (B, 4, H/8, W/8)."""
        x = x.to(dtype=self.dtype)
        posterior = self.vae.encode(x).latent_dist
        latents = posterior.mean  # deterministic → faster + reproducible
        return latents * self.scale_factor

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Scaled latents → images in [-1, 1] (B, 3, H, W)."""
        z = (z.to(dtype=self.dtype) / self.scale_factor)
        return self.vae.decode(z).sample


def load_frozen_vae(cfg: dict, device) -> FrozenVAE:
    """Build the VAE from a config dict and move it to device (frozen)."""
    vae_cfg = cfg.get("vae", cfg)
    vae = FrozenVAE(
        model_id=vae_cfg.get("model_id", "stabilityai/sd-vae-ft-mse"),
        dtype=getattr(torch, vae_cfg.get("dtype", "bfloat16")),
        local_dir=vae_cfg.get("local_dir"),
    ).to(device)
    vae.scale_factor = vae_cfg.get("scale_factor", FrozenVAE.SCALE_FACTOR)
    return vae