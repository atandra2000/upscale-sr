"""Ship the final Upscale-SR checkpoint as a single safetensors file.

Takes a training ckpt (U-Net + refiner weights), optionally swaps in EMA
shadow weights, and writes ``sr_x4_final.safetensors`` (the file ``infer.py`` loads).
"""
from __future__ import annotations

import argparse

import torch
from safetensors.torch import save_file

from utils.config import load_config
from utils.checkpoint import load_upsr_state
from models import build_sr_unet, build_ssm_refiner
from utils.stability import EMA


def _swap_ema(unet, refiner, state_path, device):
    """Apply EMA shadow weights from the training state.pt into the modules."""
    if state_path is None or not state_path.exists():
        return False
    state = torch.load(str(state_path), map_location=device, weights_only=False)
    ema_state = state.get("ema")
    if ema_state is None:
        return False
    ema = EMA(unet, decay=0.9999, device=device)
    ema.step_count = ema_state.get("step_count", 0)
    ema.shadow = {k: v.to(device) for k, v in ema_state["shadow"].items()}
    ema.apply_shadow(unet)
    # refiner EMA is stored under meta["refiner_ema"] by the training loop
    ref_ema = state.get("meta", {}).get("refiner_ema")
    if ref_ema is not None:
        ema_r = EMA(refiner, decay=0.9999, device=device)
        ema_r.shadow = {k: v.to(device) for k, v in ref_ema["shadow"].items()}
        ema_r.apply_shadow(refiner)
    return True


def main():
    ap = argparse.ArgumentParser(description="Ship Upscale-SR weights as safetensors")
    ap.add_argument("--in", dest="inp", required=True, help="training ckpt .safetensors")
    ap.add_argument("--state", default=None, help="matching .state.pt (for EMA swap)")
    ap.add_argument("--out", default="sr_x4_final.safetensors")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-ema", action="store_true", help="ship live weights, not EMA")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg["model"]
    unet = build_sr_unet(mcfg).to(device).eval()
    refiner = build_ssm_refiner(mcfg).to(device).eval()

    unet_sd, refiner_sd = load_upsr_state(args.inp, device=str(device))
    unet.load_state_dict(unet_sd, strict=False)
    refiner.load_state_dict(refiner_sd, strict=False)

    from pathlib import Path
    if not args.no_ema:
        swapped = _swap_ema(unet, refiner,
                            Path(args.state) if args.state else
                            Path(args.inp).with_suffix(".state.pt"), device)
        print(f"EMA swap: {'applied' if swapped else 'skipped (no state.pt)'}")

    out = {}
    for k, v in unet.state_dict().items():
        out[f"unet.{k}"] = v.detach().contiguous()
    for k, v in refiner.state_dict().items():
        out[f"refiner.{k}"] = v.detach().contiguous()
    save_file(out, args.out)
    n = sum(v.numel() for v in out.values())
    print(f"shipped {len(out)} tensors | {n/1e6:.1f}M params → {args.out}")


if __name__ == "__main__":
    main()