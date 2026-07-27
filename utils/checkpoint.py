"""Atomic checkpoint manager — safetensors weights + torch state + full RNG.

Atomic write: ``.tmp`` → ``os.replace``; a crash mid-save leaves a ``.tmp``
file ignored on resume, never a corrupt main ckpt.  Weights ship as
safetensors (no pickle); aux state (optim/sched/ema/rng) uses ``torch.save``.
"""
from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def rng_state_dict() -> dict:
    """Capture the full RNG state for perfect resumption."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": (torch.cuda.get_rng_state_all()
                       if torch.cuda.is_available() else None),
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class CheckpointManager:
    """Atomic save/load.  Per iter ``i``: ``sr_step_{i:07d}.safetensors`` +
    ``.state.pt``; ``sr_latest.*`` is a convenience copy for fast resume."""

    def __init__(self, save_dir: str, keep_last: int = 6):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    def save(
        self,
        unet: torch.nn.Module,
        refiner: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema,
        step: int,
        best_loss: float,
        extra_meta: Optional[dict] = None,
    ) -> None:
        sd = {}
        for k, v in unet.state_dict().items():
            sd[f"unet.{k}"] = v.detach().contiguous()
        for k, v in refiner.state_dict().items():
            sd[f"refiner.{k}"] = v.detach().contiguous()

        weights_tmp = self.save_dir / f".sr_step_{step:07d}.safetensors.tmp"
        weights_out = self.save_dir / f"sr_step_{step:07d}.safetensors"
        save_file(sd, str(weights_tmp))
        os.replace(weights_tmp, weights_out)

        state = {
            "step": step,
            "best_loss": best_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "ema": ema.state_dict(),
            "rng": rng_state_dict(),
            "meta": extra_meta or {},
        }
        state_tmp = self.save_dir / f".sr_step_{step:07d}.state.pt.tmp"
        state_out = self.save_dir / f"sr_step_{step:07d}.state.pt"
        torch.save(state, state_tmp)
        os.replace(state_tmp, state_out)

        self._copy(weights_out, self.save_dir / "sr_latest.safetensors")
        self._copy(state_out, self.save_dir / "sr_latest.state.pt")
        self._prune()

    def _copy(self, src: Path, dst: Path) -> None:
        # Atomic copy: stream to a temp file then rename.  shutil.copyfile
        # streams (no double-buffering) — src is already a committed file.
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)

    def _prune(self) -> None:
        steps = sorted(
            int(p.stem.removeprefix("sr_step_"))
            for p in self.save_dir.glob("sr_step_*.safetensors")
        )
        if len(steps) <= self.keep_last:
            return
        for s in steps[:-self.keep_last]:
            for p in self.save_dir.glob(f"sr_step_{s:07d}.*"):
                p.unlink(missing_ok=True)

    def latest_step(self) -> Optional[int]:
        steps = sorted(
            int(p.stem.removeprefix("sr_step_"))
            for p in self.save_dir.glob("sr_step_*.safetensors")
        )
        return steps[-1] if steps else None

    def load(
        self,
        unet: torch.nn.Module,
        refiner: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        ema,
        step: Optional[int] = None,
        device: str = "cuda",
    ) -> dict:
        step = step if step is not None else self.latest_step()
        if step is None:
            return {"step": 0, "best_loss": float("inf")}
        weights = load_file(str(self.save_dir / f"sr_step_{step:07d}.safetensors"),
                            device=device)
        unet_sd, refiner_sd = {}, {}
        for k, v in weights.items():
            if k.startswith("unet."):
                unet_sd[k.removeprefix("unet.")] = v
            elif k.startswith("refiner."):
                refiner_sd[k.removeprefix("refiner.")] = v
        unet.load_state_dict(unet_sd, strict=False)
        refiner.load_state_dict(refiner_sd, strict=False)

        state_path = self.save_dir / f"sr_step_{step:07d}.state.pt"
        if state_path.exists():
            state = torch.load(str(state_path), map_location=device, weights_only=False)
            try:
                optimizer.load_state_dict(state["optimizer"])
            except Exception:
                pass
            try:
                scheduler.load_state_dict(state["scheduler"])
            except Exception:
                pass
            try:
                ema.load_state_dict(state["ema"], device=device)
            except Exception:
                pass
            restore_rng_state(state["rng"])
            return {"step": state["step"], "best_loss": state["best_loss"],
                    "meta": state.get("meta", {})}
        return {"step": step, "best_loss": float("inf")}


def load_sd_epoch42_weights(unet: torch.nn.Module, ckpt_path: str, device: str = "cpu") -> tuple[int, int]:
    """Load pre-trained SD 1.x UNet weights into SRUNet with 9-channel conv_in
    adaptation: channels [4:8] ← SD 4-ch weights, [0:4] + channel 8 ← zero."""
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"SD checkpoint not found: {ckpt_path}")

    if path.suffix == ".safetensors":
        raw_sd = load_file(str(path), device=str(device))
    else:
        loaded = torch.load(str(path), map_location=device, weights_only=False)
        raw_sd = loaded.get("state_dict", loaded.get("unet", loaded.get("model", loaded)))

    sd = {}
    for k, v in raw_sd.items():
        k_clean = (k.removeprefix("model.diffusion_model.")
                   .removeprefix("unet.")
                   .removeprefix("model."))
        sd[k_clean] = v

    target_sd = unet.state_dict()
    matched_sd = {}
    loaded_count = 0

    for k, v in sd.items():
        if k in target_sd:
            t_shape = target_sd[k].shape
            if v.shape == t_shape:
                matched_sd[k] = v.to(device=device, dtype=target_sd[k].dtype)
                loaded_count += 1
            elif k == "conv_in.weight" and v.dim() == 4 and target_sd[k].dim() == 4:
                adapted = torch.zeros_like(target_sd[k], device=device)
                ch_in_src = min(v.shape[1], 4)
                adapted[:, 4:4 + ch_in_src] = v[:, :ch_in_src].to(device=device)
                matched_sd[k] = adapted
                loaded_count += 1

    unet.load_state_dict(matched_sd, strict=False)
    return loaded_count, len(target_sd)


def load_upsr_state(path: str, unet: torch.nn.Module | None = None,
                    refiner: torch.nn.Module | None = None,
                    device: str = "cpu", *,
                    strict: bool = False) -> tuple[dict, dict]:
    """Load a shipped Upscale-SR safetensors ckpt → (unet_sd, refiner_sd).

    Strips the `unet.` / `refiner.` prefixes.  When both target modules are
    provided, keys whose shape does not match the target are dropped so the
    loader tolerates channel-count drift in test ckpts (used by ``infer.py``).
    Without the targets, all keys are returned (used by ``to_safetensors.py`` /
    ``to_onnx.py`` which load the *full* saved tensor set).
    """
    sd = load_file(path, device=str(device))
    unet_target = unet.state_dict() if unet is not None else None
    refiner_target = refiner.state_dict() if refiner is not None else None
    unet_sd, refiner_sd = {}, {}
    for k, v in sd.items():
        if k.startswith("unet."):
            clean = k.removeprefix("unet.")
            if unet_target is not None:
                tgt = unet_target.get(clean)
                if tgt is None or tuple(v.shape) != tuple(tgt.shape):
                    continue
            unet_sd[clean] = v
        elif k.startswith("refiner."):
            clean = k.removeprefix("refiner.")
            if refiner_target is not None:
                tgt = refiner_target.get(clean)
                if tgt is None or tuple(v.shape) != tuple(tgt.shape):
                    continue
            refiner_sd[clean] = v
    return unet_sd, refiner_sd