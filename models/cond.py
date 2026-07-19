"""Shared SR conditioning helper used by train / infer / eval."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sr_cond_input(lr_lat: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
    """Build the 9-channel U-Net input from LR latent and noisy latent.

    lr_lat : (B,4,h/8,w/8)  — frozen VAE encode of the LR image
    z_t    : (B,4,H/8,W/8) — noisy HR latent at the target resolution
    Returns (B,9,H/8,W/8) = [lr_lat_up(4), z_t(4), mask(1)].
    """
    lr_up = F.interpolate(lr_lat, size=z_t.shape[-2:], mode="bilinear",
                          align_corners=False)
    mask = lr_up.mean(dim=1, keepdim=True)            # 1-channel structure hint
    return torch.cat([lr_up, z_t, mask], dim=1)
