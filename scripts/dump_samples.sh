#!/usr/bin/env bash
# scripts/dump_samples.sh — 8 LR→SR before/after PNGs for the README.
#
# Usage:
#   bash scripts/dump_samples.sh /path/to/val_or_real_photos [sr_x4_final.safetensors]
#
# Reads every image in the input dir, downsamples 4× to make the LR input,
# runs Upscale-SR, and writes a 3-panel side-by-side PNG
# (LR | bicubic | Upscale-SR) per image into samples/.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

IN_DIR="${1:?usage: dump_samples.sh <input_image_dir> [ckpt]}"
CKPT="${2:-sr_x4_final.safetensors}"
OUT_DIR="${OUT_DIR:-samples}"
mkdir -p "$OUT_DIR"

"$PYTHON" - <<PY
import glob, os, torch, torch.nn.functional as F, time
from PIL import Image
import numpy as np
from utils.config import load_config
from models import build_sr_unet, build_ssm_refiner, build_ddim
from models.vae_frozen import load_frozen_vae
from infer import upscale

cfg = load_config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from safetensors.torch import load_file
sd = load_file("$CKPT", device=str(device))
vae = load_frozen_vae(cfg["model"], device)
unet = build_sr_unet(cfg["model"]).to(device).eval()
refiner = build_ssm_refiner(cfg["model"]).to(device).eval()
if device.type == "cuda":
    unet = unet.to(memory_format=torch.channels_last)
    refiner = refiner.to(memory_format=torch.channels_last)
unet.load_state_dict({k.removeprefix('unet.'): v for k,v in sd.items() if k.startswith('unet.')}, strict=False)
refiner.load_state_dict({k.removeprefix('refiner.'): v for k,v in sd.items() if k.startswith('refiner.')}, strict=False)
ddim = build_ddim(cfg["model"]).to(device)

def load_pil(p):
    arr = torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1.0
    return arr.to(device)

def to_pil(t):
    t = ((t+1)/2).clamp(0,1)
    arr = (t[0].permute(1,2,0).cpu().numpy()*255).round().astype('uint8')
    return Image.fromarray(arr)

files = sorted(glob.glob(os.path.join("$IN_DIR", "*")))
files = [f for f in files if f.lower().endswith(('.png','.jpg','.jpeg','.webp','.bmp'))][:8]
assert files, f"no images in $IN_DIR"
for f in files:
    hr = load_pil(f)
    # make the LR by 4× bicubic downsample (the "real photo" the model sees)
    lr = F.interpolate(hr, scale_factor=1/4, mode='bicubic', align_corners=False).clamp(-1,1)
    bic = F.interpolate(lr, scale_factor=4, mode='bicubic', align_corners=False).clamp(-1,1)
    with torch.no_grad(), (torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else torch.no_grad()):
        sr = upscale(lr, unet, refiner, vae, ddim, device, scale=4, steps=30, crop=512)
    # 3-panel montage: LR (upsampled for display) | bicubic | Upscale-SR
    lr_disp = F.interpolate(lr, scale_factor=4, mode='nearest').expand(-1,-1,sr.shape[-2],sr.shape[-1])
    panels = [to_pil(x) for x in [lr_disp, bic, sr]]
    w = sum(p.width for p in panels); h = panels[0].height
    canvas = Image.new('RGB', (w, h)); x = 0
    for p in panels:
        canvas.paste(p, (x, 0)); x += p.width
    out = os.path.join("$OUT_DIR", os.path.basename(f).split('.')[0] + '_sr.png')
    canvas.save(out)
    print(f"  {os.path.basename(f)} → {out}")
print(f"dumped {len(files)} before/after PNGs → $OUT_DIR")
PY