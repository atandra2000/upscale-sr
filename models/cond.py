"""Shared SR conditioning helper used by train / infer / eval."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LRUpConv(nn.Module):
    """Learned LR-latent upsample: bilinear to HR size, then 3×3 conv.

    Identity-initialised so it starts as pure bilinear (StableSR-style) and
    only deviates as training pushes it.  Owner (U-Net, not a module-level
    singleton) holds the conv so DDP doesn't race on device moves.
    """

    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        # eye-init the centre tap so a zero-initialized conv becomes identity
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        with torch.no_grad():
            for c in range(channels):
                self.conv.weight[c, c, 1, 1] = 1.0

    def forward(self, lr: torch.Tensor, size) -> torch.Tensor:
        up = F.interpolate(lr, size=size, mode="bilinear", align_corners=False)
        return self.conv(up)


def sr_cond_input(lr_lat: torch.Tensor, z_t: torch.Tensor,
                  up: LRUpConv | None = None) -> torch.Tensor:
    """Build the 9-channel U-Net input from LR latent and noisy latent.

    lr_lat : (B,4,h/8,w/8)  — frozen VAE encode of the LR image
    z_t    : (B,4,H/8,W/8) — noisy HR latent at the target resolution
    up     : optional ``LRUpConv`` (passed in by the U-Net owner so the conv
             lives on the right device and is a normal PyTorch submodule).
             When None, falls back to a no-op bilinear-only upsample.
    Returns (B,9,H/8,W/8) = [lr_lat_up(4), z_t(4), mask(1)].
    """
    if up is not None:
        lr_up = up(lr_lat, z_t.shape[-2:])
    else:
        lr_up = F.interpolate(lr_lat, size=z_t.shape[-2:], mode="bilinear",
                              align_corners=False)
    mask = lr_up.mean(dim=1, keepdim=True)            # 1-channel structure hint
    return torch.cat([lr_up, z_t, mask], dim=1)
