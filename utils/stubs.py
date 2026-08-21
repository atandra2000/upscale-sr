"""Shared stub components for smoke tests / local runs without heavy deps.

``StubVAE`` and ``StubDataset`` stand in for the frozen SD VAE and the real
data pipeline so the training loop, inference CLI, and demo can be exercised
on a machine without diffusers or GPU.  NOT for real training.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StubVAE(nn.Module):
    """A no-op VAE stand-in: 8× bilinear downsample / upsample, fixed scale.

    Used only by ``--stub-vae`` so the training loop can be smoke-tested on a
    machine without the SD VAE weights. NOT for real training."""

    SCALE_FACTOR = 0.18215

    def __init__(self):
        super().__init__()
        self.scale_factor = self.SCALE_FACTOR
        self.dtype = torch.float32

    @torch.no_grad()
    def encode(self, x):
        # (B,3,H,W) → (B,4,H/8,W/8) pseudo-latent (channel-split+pad)
        d = F.avg_pool2d(x, 8)
        pad = 4 - d.shape[1]
        if pad > 0:
            d = F.pad(d, (0, 0, 0, 0, 0, pad))
        return d[:, :4] * self.scale_factor

    @torch.no_grad()
    def decode(self, z):
        z = z / self.scale_factor
        return F.interpolate(z[:, :3], scale_factor=8, mode="bilinear",
                             align_corners=False)


class StubDataset(torch.utils.data.Dataset):
    """Tiny synthetic (LR, HR) dataset for smoke runs (no real HR images)."""

    def __init__(self, n=16, patch_hr=128, scale=4):
        self.n, self.patch_hr, self.scale = n, patch_hr, scale

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        hr = torch.rand(3, self.patch_hr, self.patch_hr) * 2 - 1     # [-1,1]
        lr = F.interpolate(hr.unsqueeze(0), scale_factor=1 / self.scale,
                            mode="bicubic", align_corners=False).squeeze(0)
        return {"lr": lr.clamp(-1, 1), "hr": hr, "seed": idx}
