"""Upscale-SR — Gradio before/after demo.

    python demo_gradio.py [--ckpt sr_x4_final.safetensors] [--steps 30]

Drops a browser UI: upload an image → see bicubic-baseline vs Upscale-SR
side-by-side.  A recruiter can click and see it work (DESIGN §5 / candidate §1).
"""
from __future__ import annotations

import argparse
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from PIL import Image

from utils.config import load_config
from models import build_sr_unet, build_ssm_refiner, build_ddim
from models.vae_frozen import load_frozen_vae
from infer import upscale, _load_image, _save_image, _tensor_to_pil, load_image_pil


def _bicubic_baseline(lr_img: torch.Tensor, scale: int) -> torch.Tensor:
    """Bicubic upsample — the naive baseline shown alongside Upscale-SR."""
    return F.interpolate(lr_img, scale_factor=scale, mode="bicubic",
                        align_corners=False)


def build_pipeline(cfg, ckpt_path, device, stub_vae=False):
    from safetensors.torch import load_file
    sd = load_file(ckpt_path, device=str(device))
    mcfg = cfg["model"]
    if stub_vae:
        from training.train import _StubVAE
        vae = _StubVAE().to(device)
    else:
        vae = load_frozen_vae(mcfg, device)
    unet = build_sr_unet(mcfg).to(device).eval()
    refiner = build_ssm_refiner(mcfg).to(device).eval()
    if device.type == "cuda":
        unet = unet.to(memory_format=torch.channels_last)
        refiner = refiner.to(memory_format=torch.channels_last)
    unet.load_state_dict({k.removeprefix("unet."): v for k, v in sd.items()
                          if k.startswith("unet.")}, strict=False)
    refiner.load_state_dict({k.removeprefix("refiner."): v for k, v in sd.items()
                             if k.startswith("refiner.")}, strict=False)
    ddim = build_ddim(mcfg).to(device)
    return {"vae": vae, "unet": unet, "refiner": refiner, "ddim": ddim, "device": device}


def run(pipe, pil_img, scale, steps, crop):
    lr = load_image_pil(pil_img).to(pipe["device"])
    bic = _bicubic_baseline(lr, scale).clamp(-1, 1)
    t0 = time.perf_counter()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16) if pipe["device"].type == "cuda" else nullcontext():
            sr = upscale(lr, pipe["unet"], pipe["refiner"], pipe["vae"],
                         pipe["ddim"], pipe["device"], scale, steps, crop)
    dt = time.perf_counter() - t0
    bic_img = _tensor_to_pil(bic)
    sr_img = _tensor_to_pil(sr)
    return bic_img, sr_img, f"{dt:.2f}s | LR {tuple(lr.shape)} → SR {tuple(sr.shape)}"


def main():
    ap = argparse.ArgumentParser(description="Upscale-SR Gradio demo")
    ap.add_argument("--ckpt", default="sr_x4_final.safetensors")
    ap.add_argument("--config", default=None)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--share", action="store_true", help="public Gradio link")
    ap.add_argument("--stub-vae", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = build_pipeline(cfg, args.ckpt, device, stub_vae=args.stub_vae)

    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("gradio not installed: pip install gradio")

    def _fn(image, steps, crop):
        if image is None:
            return None, None, "no image"
        bic, sr, info = run(pipe, image, args.scale, int(steps), int(crop))
        return bic, sr, info

    demo = gr.Interface(
        fn=_fn,
        inputs=[gr.Image(type="pil", label="LR input"),
                gr.Slider(1, 50, value=args.steps, step=1, label="DDIM steps"),
                gr.Slider(0, 1024, value=args.crop, step=64, label="tile crop (0=whole)")],
        outputs=[gr.Image(label="bicubic baseline"),
                 gr.Image(label="Upscale-SR"),
                 gr.Textbox(label="info")],
        title="Upscale-SR — 4× real-world photo super-resolution",
        description="Latent-diffusion SR (FA2 U-Net) + mamba-ssm refiner. "
                    "Drop an image; the left pane is the bicubic baseline, "
                    "the right is the model output.",
    )
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()