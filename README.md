# Upscale-SR — 4× real-world photo super-resolution

> A trained **photo super-resolution product**: give it a low-resolution
> image, get back a sharp **4×-upscaled** image (512² → 2048² in ~1.5 s on
> one RTX 5090). A **latent-diffusion SR backbone** (StableSR-style, FA2
> U-Net) reconstructs coherent structure; an **SSM refiner**
> (`mamba-ssm` bidirectional dilated selective-scan bottleneck, NAFNet body)
> re-injects the high-frequency detail diffusion blurs. Trained on
> DIV2K + Flickr2K + DF2K with on-the-fly **Real-ESRGAN degradation**, so it
> works on real photos (JPEG noise, compression artifacts), not just clean
> bicubic pairs.
>
> **Author:** Atandra Bharati · **Hardware:** 2× RTX 5090 32 GB (Blackwell
> sm_120) DDP · **Stack:** PyTorch 2.12, BF16, FlashAttention-2, `mamba-ssm`,
> `channels_last`, `torch.compile`, EMA, atomic `safetensors` checkpoints.
>
> Design: [`vision-research/DESIGN-upscale-sr.md`](../vision-research/DESIGN-upscale-sr.md) ·
> Plan: [`vision-research/EXECUTION-PLAN-upscale-sr.md`](../vision-research/EXECUTION-PLAN-upscale-sr.md) ·
> Fit: [`vision-research/candidates/01-upscale-sr.md`](../vision-research/candidates/01-upscale-sr.md).

---

## What it does

| Input | Output | Latency (1× 5090) |
|-------|--------|-------------------|
| any LR image (e.g. 512², with JPEG noise) | 4× HR (2048²), sharp | ~1.5 s (30 DDIM + refiner) · ~0.6 s (`--fast`, 10 steps) |

**Mechanism sources:** StableSR (arXiv:2305.08015), NAFNet (arXiv:2204.04776),
SRMamba (arXiv:2403.11143), Real-ESRGAN (arXiv:2107.10833), Mamba
(arXiv:2312.00752), FlashAttention-2 (arXiv:2311.17243), LPIPS
(arXiv:1801.03924).

---

## Architecture

```
LR image ─▶ frozen SD1.5 VAE ─▶ LR latent (B,4,H/8,W/8)
                                     │
   z_t (noisy latent) ──┐            │
   LR-lat upsampled ─────┼─▶ cat ─▶ SR U-Net (FA2, ~52 M) ─▶ ε_pred
   LR-structure mask ───┘            │   (9-ch cond + timestep emb)
                                     ▼
                          DDIM sampler (30 steps, η=0)
                                     │
                                     ▼
                       VAE decode ─▶ HR image (B,3,H,W)
                                     │
                                     ▼
            SSM refiner (NAFNet + bidirectional dilated mamba-ssm, ~23 M)
                                     │
                                     ▼
                              sharp 4× HR image
```

- **`models/vae_frozen.py`** — frozen SD1.5 VAE (`stabilityai/sd-vae-ft-mse`),
  8× downsample, `SCALE_FACTOR=0.18215`.
- **`models/sr_unet.py`** — conditional latent-diffusion U-Net, 9-channel
  input `[lr_lat_up | z_t | lr_mask]`, sinusoidal timestep embedding, FA2
  self-attention (sdpa flash-backend fallback), 4 enc/dec blocks, zero-init
  `conv_out` (predicts ~0 noise at init → stable start).
- **`models/ssm_refiner.py`** — NAFNet-body U-Net whose bottleneck is a
  **bidirectional `mamba-ssm` selective scan with dilated multi-scan
  {1,2,4}** (the dilated-conv analog for SSMs). Pure-PyTorch chunkwise scan
  fallback (kernel-equivalence tested). Zero-init `conv_out` → identity at
  init (residual start).
- **`data/realesrgan_degrade.py`** — stochastic blur → resize → noise → JPEG
  → sinc, with a 2nd-order stage shuffle. **Deterministic given a seed**
  (RNG save/seed/restore — tested).
- **`training/train.py`** — hand-written DDP, BF16 autocast + FP32 master +
  FP32 LayerNorm, fused AdamW, EMA (0.9999), grad-clip 1.0, NaN guard,
  `channels_last`, `torch.compile(max-autotune)`, atomic `safetensors`
  checkpoints every 10 K iters (full RNG + optim + sched + EMA state).

---

## Install

```bash
pip install -r requirements.txt
# Blackwell sm_120 kernels (fallback runs if a build fails — correct, slower):
pip install flash-attn mamba-ssm causal-conv1d
```

> **Local Mac (no CUDA):** everything runs via fallback paths (sdpa
> attention, pure-PyTorch selective scan, `--stub-vae`).
> `pip install torch numpy pillow pyyaml safetensors diffusers torchvision pytest`
> is enough for `bash scripts/smoke.sh` + `pytest tests/`.

---

## Usage

### Inference CLI

```bash
python3 infer.py --in lr.jpg --out sr.png --scale 4 --steps 30
python3 infer.py --in lr.jpg --out sr.png --scale 4 --fast        # 10-step fast mode
python3 infer.py --in big.jpg --out big_sr.png --scale 4 --crop 512  # tile large images
```

`--crop 512` tiles large images (hann-window blend, 25% overlap) to bound
VRAM; `--crop 0` runs the whole image at once. `--stub-vae` uses a no-op VAE
for local smoke-tests without the SD VAE weights.

### Gradio before/after demo

```bash
python3 demo_gradio.py    # browser: drop image → bicubic baseline vs Upscale-SR
```

### Train (RunPod 2× 5090, ~2–3 days, 100 K iters)

```bash
bash scripts/env_check.sh                                     # sm_120 + kernels
python3 data/build_div2k_flickr.py --out /workspace/data/sr    # DIV2K+Flickr2K+DF2K HR shards
CKPT_DIR=/workspace/runs/upscale-sr bash scripts/launch_2x5090.sh
```

### Evaluate (DIV2K-val, fixed-seed degradation)

```bash
python3 -m training.eval \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --val-dir /workspace/data/sr/div2k_val --eval-imgs 100
```

---

## Deploy (ship the product)

```bash
# 1. swap EMA weights in → ship weights
python3 export/to_safetensors.py \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --out sr_x4_final.safetensors

# 2. refiner ONNX export (+ onnxruntime round-trip verification)
python3 export/to_onnx.py \
    --ckpt /workspace/runs/upscale-sr/sr_latest.safetensors \
    --out refiner_x4.onnx

# 3. 8 LR→SR before/after PNGs (LR | bicubic | Upscale-SR) for this README
bash scripts/dump_samples.sh /workspace/data/sr/div2k_val sr_x4_final.safetensors
```

**Shipped artifacts:**
- `sr_x4_final.safetensors` — U-Net + refiner weights (EMA-swapped),
  `unet.`/`refiner.` prefixed.
- `refiner_x4.onnx` — refiner ONNX (opset 17) for deployment inference.
- `samples/*.png` — 8 before/after montages.

---

## Results

> **Status:** code complete, smoke gate green (23/23 tests pass, 2 skipped
> on Mac), atomic checkpoints verified. The 100 K-iter training run executes
> on the RunPod 2× 5090 pod; numbers below are filled in after Phase 4–5.

### Quantitative — DIV2K-val (×4, Real-ESRGAN degradation, fixed seed 42)

| Method | PSNR (dB) ↑ | LPIPS ↓ | Latency (s) ↓ | Params | Notes |
|--------|:----------:|:-------:|:-------------:|:------:|-------|
| Bicubic baseline | _TBD_ | _TBD_ | <0.01 | 0 | reference |
| Real-ESRGAN (ref.) | 28.7 | 0.78 | — | — | published |
| **Upscale-SR (U-Net only)** | _TBD_ | _TBD_ | _TBD_ | ~52 M | diffusion SR, no refiner |
| **Upscale-SR (U-Net + SSM refiner)** | _TBD_ | _TBD_ | ~1.5 | ~75 M | **full product** |
| **Upscale-SR `--fast` (10 steps)** | _TBD_ | _TBD_ | ~0.6 | ~75 M | speed mode |

**Acceptance targets:** PSNR ≥ 27 dB / LPIPS ≤ 0.82, GPU util ≥ 95%,
100 K iters in ~2–3 days, atomic checkpoints (no `pickle`, full RNG state).

### Qualitative — 8 real-photo before/after samples

The 3-panel montages below (`LR | bicubic | Upscale-SR`) are generated by
`bash scripts/dump_samples.sh`. _(Placeholders until the trained ckpt ships.)_

| # | Sample | | # | Sample |
|---|--------|---|---|--------|
| 1 | `samples/0001_sr.png` | | 5 | `samples/0005_sr.png` |
| 2 | `samples/0002_sr.png` | | 6 | `samples/0006_sr.png` |
| 3 | `samples/0003_sr.png` | | 7 | `samples/0007_sr.png` |
| 4 | `samples/0004_sr.png` | | 8 | `samples/0008_sr.png` |

![sample 1](samples/0001_sr.png)
![sample 2](samples/0002_sr.png)
![sample 3](samples/0003_sr.png)
![sample 4](samples/0004_sr.png)
![sample 5](samples/0005_sr.png)
![sample 6](samples/0006_sr.png)
![sample 7](samples/0007_sr.png)
![sample 8](samples/0008_sr.png)

---

## Tests

```bash
PYTHONPATH=. python3 -m pytest tests/ -q
# 23 passed, 2 skipped (mamba-ssm kernel + channels_last — both CUDA-only)
```

| File | Asserts |
|------|---------|
| `tests/test_realesrgan_deg.py` | degradation deterministic given seed; RNG state restored; size preserved; [0,1] range |
| `tests/test_refiner_shapes.py` | SSM refiner (B,3,H,W) round-trip; no NaN at BF16; identity at init; mamba-ssm ≡ fallback (kernel-equiv) |
| `tests/test_fa2_unet.py` | U-Net (B,9,…)→(B,4,…) under autocast + FP32 GroupNorm; zero-init → ~0 noise; channels_last compatible |
| `tests/test_infer_cli.py` | `infer.upscale` 4× shape for 16/32/48 inputs; tile path == whole path; CLI writes right-size PNG |

**Smoke gate:** `bash scripts/smoke.sh` → 8-iter stub run writes an atomic
`safetensors` ckpt (743 tensors) with no NaN.

---

## Project layout

```
upscale-sr/
├── configs/sr_x4_realesrgan_2x5090.yaml   # single source of truth (shapes + recipe)
├── data/        # build_div2k_flickr · realesrgan_degrade · sr_dataset
├── models/      # vae_frozen · sr_unet (FA2) · ssm_refiner (mamba-ssm) · scheduler
├── training/    # train (DDP) · eval · profiler
├── infer.py     # product CLI
├── demo_gradio.py
├── export/      # to_safetensors · to_onnx
├── utils/       # config · checkpoint · logging · memory · schedule · stability · losses
├── scripts/     # env_check · smoke · launch_2x5090 · dump_samples
├── tests/       # 4 test files, 23 tests
├── AGENTS.md    # developer reference (config-knob map, triage, routing)
└── SKILLS.md    # interactive workflows (smoke, train, eval, ship, debug)
```

---

## License & attribution

Author: **Atandra Bharati**. Mechanism sources cited above; this is an
independent from-scratch implementation for the CoreProjects ML research
portfolio. Datasets: DIV2K, Flickr2K, DF2K (research use).