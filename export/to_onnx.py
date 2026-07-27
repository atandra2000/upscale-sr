"""Export the SSM refiner to ONNX for fast deployment inference.

The refiner is the deployment-friendly component (fixed (3,H,W)→(3,H,W)
shape, pixel space).  The U-Net stays in PyTorch (dynamic-shape DDIM loop).
A smoke round-trip is verified at the end.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from utils.config import load_config
from utils.checkpoint import load_upsr_state
from models import build_ssm_refiner


def main():
    ap = argparse.ArgumentParser(description="Export the Upscale-SR refiner to ONNX")
    ap.add_argument("--ckpt", default="sr_x4_final.safetensors")
    ap.add_argument("--out", default="refiner_x4.onnx")
    ap.add_argument("--config", default=None)
    ap.add_argument("--size", type=int, default=256,
                    help="export image size (H=W); pick the deployment tile size")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    cfg = load_config(args.config)
    export_cfg = cfg.get("export", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    refiner = build_ssm_refiner(cfg["model"]).to(device).eval()

    _, refiner_sd = load_upsr_state(args.ckpt, device=str(device))
    refiner.load_state_dict(refiner_sd, strict=False)

    dummy = torch.randn(1, 3, args.size, args.size, device=device)
    opset = args.opset or export_cfg.get("onnx_opset", 17)
    torch.onnx.export(
        refiner, (dummy,), args.out,
        input_names=["image"], output_names=["refined"],
        dynamic_axes={"image": {0: "batch", 2: "h", 3: "w"},
                      "refined": {0: "batch", 2: "h", 3: "w"}},
        opset_version=opset,
    )
    print(f"exported refiner → {args.out} (opset {opset}, size {args.size})")

    # ── round-trip smoke check (torch vs onnxruntime) ──────────────────────
    try:
        import onnxruntime as ort
        with torch.no_grad():
            ref_out = refiner(dummy).cpu().numpy()
        sess = ort.InferenceSession(args.out)
        onnx_out = sess.run(None, {"image": dummy.cpu().numpy()})[0]
        max_diff = float(np.abs(ref_out - onnx_out).max())
        print(f"ONNX round-trip max-abs diff vs torch: {max_diff:.2e} "
              f"({'PASS' if max_diff < 1e-3 else 'CHECK'})")
    except ImportError:
        print("onnxruntime not installed — install to verify the round-trip")


if __name__ == "__main__":
    main()