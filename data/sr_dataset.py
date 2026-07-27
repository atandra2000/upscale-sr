"""Streaming SR dataset — HR image → random-crop 256² → on-the-fly degradation.

Two modes: ``train`` (random crop, stochastic degradation, fresh seed per
sample per epoch) and ``val`` (centred crop to a multiple of ``scale``,
fixed-seed degradation → reproducible PSNR/LPIPS).  HR images are read from a
flat directory of ``*.png/.jpg/.webp`` and decoded lazily via PIL.
"""
from __future__ import annotations

import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DistributedSampler
from PIL import Image
from torchvision.transforms.functional import to_tensor

from .realesrgan_degrade import RealESRGANDegrader, make_lr_pair


class _FlatDirHR(Dataset):
    """Lazily lists ``*.png/.jpg/.jpeg/.webp/.bmp`` under ``root`` (recursive)."""

    _EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

    def __init__(self, root: "str | Path"):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"HR image dir not found: {root}")
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in self._EXTS)
        if not self.files:
            raise FileNotFoundError(f"No HR images under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        return to_tensor(Image.open(self.files[idx]).convert("RGB"))


class SRDataset(Dataset):
    """Returns (LR, HR) tensors in [-1, 1] at the configured patch size.

    ``mode="val"`` uses fixed-seed degradation for reproducible PSNR/LPIPS;
    ``mode="train"`` uses a fresh per-sample, per-epoch seed.
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
            s = self.scale
            ch = (H // s) * s if H >= s else s
            cw = (W // s) * s if W >= s else s
            top = (H - ch) // 2
            left = (W - cw) // 2
            return hr[:, :, top:top + ch, left:left + cw]
        ph = min(self.patch_hr, H)
        pw = min(self.patch_hr, W)
        top = random.randint(0, H - ph)
        left = random.randint(0, W - pw)
        return hr[:, :, top:top + ph, left:left + pw]

    def __getitem__(self, idx):
        hr = self._flat[idx]
        hr_crop = self._crop(hr.unsqueeze(0))
        seed = self.base_seed + idx + self.epoch * 10_000_000
        lr, hr_pair = make_lr_pair(hr_crop, self.degrader, self.scale, seed)
        lr_m1, hr_m1 = (lr * 2.0 - 1.0), (hr_pair * 2.0 - 1.0)
        return {"lr": lr_m1.squeeze(0), "hr": hr_m1.squeeze(0), "seed": seed}


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