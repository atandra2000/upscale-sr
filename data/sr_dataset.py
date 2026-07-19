"""Streaming SR dataset — HR image → random-crop 256² → on-the-fly degradation.

Two modes:
  * ``train``: random crop of ``patch_hr`` (256²), stochastic Real-ESRGAN
    degradation (fresh seed per sample per epoch), returns (LR, HR) tensors
    in [-1, 1] for both the latent path (HR encoded by VAE) and the refiner.
  * ``val``  : full image (or centred crop to a multiple of ``scale``), fixed
    degradation seed (reproducible PSNR/LPIPS — DESIGN §6 / EXECUTION-PLAN
    acceptance criteria).

HR images are read from a flat directory of ``*.png/.jpg/.webp`` and decoded
lazily with ``PIL``. The heavy Real-ESRGAN pipeline runs on CPU in the worker.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler

from .realesrgan_degrade import RealESRGANDegrader, make_lr_pair


def _pil_to_tensor(img) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0  # (H,W,3)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()        # (3,H,W) [0,1]


def _to_minus1_to1(x: torch.Tensor) -> torch.Tensor:
    return x * 2.0 - 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Flat-directory HR source
# ─────────────────────────────────────────────────────────────────────────────
class _FlatDirHR(Dataset):
    """Lazily lists ``*.png/.jpg/.jpeg/.webp/.bmp`` under ``root`` (recursive)."""

    _EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(self, root: str):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"HR image dir not found: {root}")
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in self._EXTS)
        if not self.files:
            raise FileNotFoundError(f"No HR images under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        from PIL import Image
        return _pil_to_tensor(Image.open(self.files[idx]))


# ─────────────────────────────────────────────────────────────────────────────
# SR pair dataset (wraps an HR source + degrader)
# ─────────────────────────────────────────────────────────────────────────────
class SRDataset(Dataset):
    """Returns (LR, HR) tensors in [-1, 1] at the configured patch size.

    Args:
        hr_root:  directory of HR images.
        degrader: ``RealESRGANDegrader`` instance.
        patch_hr: HR crop size (train). 256² default.
        scale:    SR scale factor (4).
        mode:     ``"train"`` (random crop, stochastic deg) or ``"val"``
                  (centred crop to multiple of scale, fixed-seed deg).
        base_seed: master seed — per-sample seed = base_seed + idx (+ epoch
                  offset applied via ``set_epoch`` for train freshness).
    """

    def __init__(self, hr_root: str, degrader: RealESRGANDegrader,
                 patch_hr: int = 256, scale: int = 4, mode: str = "train",
                 base_seed: int = 42):
        self.degrader = degrader
        self.patch_hr = patch_hr
        self.scale = scale
        self.mode = mode
        self.base_seed = base_seed
        self.epoch = 0
        self._flat = _FlatDirHR(hr_root)
        self._len = len(self._flat)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return self._len

    def _crop(self, hr: torch.Tensor) -> torch.Tensor:
        if hr.dim() == 3:
            hr = hr.unsqueeze(0)
        _, _, H, W = hr.shape
        if self.mode == "val":
            # centred crop to the largest multiple of scale ≤ min(H,W)
            s = self.scale
            ch = (H // s) * s if H >= s else s
            cw = (W // s) * s if W >= s else s
            ch = min(ch, self.patch_hr) if self.patch_hr > 0 else ch
            cw = min(cw, self.patch_hr) if self.patch_hr > 0 else cw
            top = (H - ch) // 2
            left = (W - cw) // 2
            return hr[:, :, top:top + ch, left:left + cw]
        # train: random crop
        ph = min(self.patch_hr, H)
        pw = min(self.patch_hr, W)
        top = random.randint(0, H - ph)
        left = random.randint(0, W - pw)
        return hr[:, :, top:top + ph, left:left + pw]

    def __getitem__(self, idx):
        hr = self._flat[idx]                       # (3,H,W) [0,1]
        hr_crop = self._crop(hr.unsqueeze(0))      # (1,3,ph,pw)
        # per-sample, per-epoch seed → fresh degradation each epoch (train)
        seed = self.base_seed + idx + self.epoch * 10_000_000
        lr, hr_pair = make_lr_pair(hr_crop, self.degrader, self.scale, seed)
        lr_m1, hr_m1 = _to_minus1_to1(lr), _to_minus1_to1(hr_pair)
        return {"lr": lr_m1.squeeze(0), "hr": hr_m1.squeeze(0), "seed": seed}


# ─────────────────────────────────────────────────────────────────────────────
# Distributed sampler
# ─────────────────────────────────────────────────────────────────────────────
class SRDistributedSampler(DistributedSampler):
    """Shards the SR dataset across DDP ranks, epoch-shuffled."""

    def __init__(self, dataset: SRDataset, num_replicas=None, rank=None,
                 seed: int = 42):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=True, seed=seed, drop_last=True)

    def __iter__(self):
        self.dataset.set_epoch(self.epoch)
        return super().__iter__()


def build_train_dataset(cfg: dict) -> SRDataset:
    d = cfg.get("data", cfg)
    deg = RealESRGANDegrader(d.get("degradation", {}))
    return SRDataset(
        hr_root=d.get("hr_shard_dir", d.get("hr_root")),
        degrader=deg,
        patch_hr=int(d.get("patch_hr", 256)),
        scale=int(d.get("scale", 4)),
        mode="train",
        base_seed=int(d.get("seed", 42)),
    )


def build_val_dataset(cfg: dict) -> SRDataset:
    d = cfg.get("data", cfg)
    deg = RealESRGANDegrader(d.get("degradation", {}))
    return SRDataset(
        hr_root=d.get("val_dir", d.get("hr_root")),
        degrader=deg,
        patch_hr=int(d.get("patch_hr", 256)),
        scale=int(d.get("scale", 4)),
        mode="val",
        base_seed=int(d.get("seed", 42)),
    )