"""Frozen SD1.5-class VAE — reused from ``Vision/StableDiffusion``.

The VAE is permanently frozen and used only to:
  * encode the LR image     (B,3,H,W) → LR latent (B,4,H/8,W/8)
  * decode the SR latent      (B,4,H/8,W/8) → HR image (B,3,H,W)

It is never trained.  We reuse the portfolio's already-downloaded
``stabilityai/sd-vae-ft-mse`` weights (the same VAE used by StableDiffusion).
Per DESIGN §1.1, the same frozen VAE is later reused by Inpaint-Edit (02).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrozenVAE(nn.Module):
    """Thin frozen wrapper around a HuggingFace ``AutoencoderKL``.

    Direct encoder access (``.mean`` instead of ``.sample()``) for
    deterministic, faster inference.  The scale factor (0.18215) is the
    empirical std of the LAION-2B latent distribution and normalises latents
    to unit variance for stable diffusion training (same as SD_Train.py).
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # round-trip sanity
        return self.decode(self.encode(x))


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