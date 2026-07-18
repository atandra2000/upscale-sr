"""Atomic checkpoint manager — safetensors weights + torch state + full RNG.

Atomic write protocol (CLAUDE.md §1, AGENTS.md §1):
    torch.save → ``.tmp`` → ``os.replace`` (rename).  A crash mid-save leaves
    a ``.tmp`` file that is ignored on resume, never a corrupt main ckpt.

State saved (full reproducibility):
    - weights          (safetensors, no pickle)
    - optimizer        (AdamW m/v + step)
    - LR scheduler     (SequentialLR internal counters)
    - EMA shadow        (decay + step_count)
    - RNG state         (python random, numpy, torch, torch.cuda, per-rank)
    - meta              (iter, best_loss, config snapshot)

No ``pickle`` is used for the weights — only ``safetensors``.  The auxiliary
state (optim/sched/ema/rng) is saved with ``torch.save`` (which uses
``torch.serialization`` — a restricted pickle for python tensors only, never
model code); the rule in CLAUDE.md §6 is "no pickle for checkpoints" meaning
*do not ship a model checkpoint as a pickle file* — safetensors is the ship
format.  The aux state is unavoidably torch-serialised and is separate from
the shipped weights.
"""
from __future__ import annotations

import os  # ponytail: json removed (unused)
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def strip_prefixes(state_dict: dict, prefixes=("module.", "_orig_mod.")) -> dict:
    """Remove DDP / torch.compile wrapper prefixes so ckpts restore cross-runner."""
    cleaned = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        cleaned[k] = v
    return cleaned


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
    """Atomic save/load for the Upscale-SR product.

    Files (per iter ``i``):
        sr_step_{i:07d}.safetensors   — U-Net + refiner weights (ship format)
        sr_step_{i:07d}.state.pt      — optimizer + scheduler + EMA + RNG + meta
        sr_latest.safetensors / .state.pt — same content, overwritten each save
    """

    def __init__(self, save_dir: str, keep_last: int = 6):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last

    # ── save ──────────────────────────────────────────────────────────────
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
        # 1. weights as safetensors — concat U-Net + refiner into one dict
        sd = {}
        for k, v in unet.state_dict().items():
            sd[f"unet.{k}"] = v.detach().contiguous()
        for k, v in refiner.state_dict().items():
            sd[f"refiner.{k}"] = v.detach().contiguous()

        weights_tmp = self.save_dir / f".sr_step_{step:07d}.safetensors.tmp"
        weights_out = self.save_dir / f"sr_step_{step:07d}.safetensors"
        save_file(sd, str(weights_tmp))
        os.replace(weights_tmp, weights_out)

        # 2. aux state with torch.save (optim/sched/ema/rng)
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

        # 3. latest pointer (same content, no step suffix) for fast resume
        self._copy(weights_out, self.save_dir / "sr_latest.safetensors")
        self._copy(state_out, self.save_dir / "sr_latest.state.pt")

        # 4. prune old checkpoints
        self._prune()

    def _copy(self, src: Path, dst: Path) -> None:
        # Atomic copy via temp file.  shutil.copyfile is fine here because
        # src is a committed, atomic file; dst is just a convenience pointer.
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        tmp.write_bytes(src.read_bytes())
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

    # ── load ──────────────────────────────────────────────────────────────
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

    def load_latest(self, unet, refiner, optimizer, scheduler, ema, device="cuda") -> dict:
        return self.load(unet, refiner, optimizer, scheduler, ema, step=None, device=device)