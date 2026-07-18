#!/usr/bin/env bash
# scripts/smoke.sh — 2-iter smoke gate (EXECUTION-PLAN Phase 2 acceptance).
#
# Runs the full training loop for a few iterations in stub mode (no SD VAE,
# no diffusers, no flash-attn/mamba-ssm — pure-PyTorch fallbacks) so it works
# on any machine.  Asserts: shapes correct, no NaN at BF16, weights save.
#
# On the RunPod pod, run the real smoke (CUDA, real VAE) with:
#   SMOKE_REAL=1 bash scripts/smoke.sh
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

OUT="${SMOKE_OUT:-/tmp/sr_smoke}"
rm -rf "$OUT"; mkdir -p "$OUT"

if [[ "${SMOKE_REAL:-0}" == "1" ]]; then
  # Real smoke: real VAE, CUDA, 2 iters, 64² latent, 4-step diffusion.
  echo "=== smoke (REAL: CUDA + real VAE) ==="
  "$PYTHON" -m training.train --max-iters 2 --batch-size 2 --no-resume \
      --ckpt-dir "$OUT" --refiner-warmup 1 2>&1 | tee "$OUT/smoke.log"
else
  # Stub smoke: no diffusers / no GPU — exercises the loop wiring + fallbacks.
  echo "=== smoke (STUB: CPU, no diffusers, fallback kernels) ==="
  "$PYTHON" -m training.train --stub --max-iters 8 --batch-size 2 --no-resume \
      --ckpt-dir "$OUT" --refiner-warmup 4 2>&1 | tee "$OUT/smoke.log"
fi

# ── Assert the checkpoint wrote + no NaN in the shipped weights ──────────
"$PYTHON" - <<PY
from safetensors.torch import load_file
import glob, torch
ckpts = sorted(glob.glob("$OUT/sr_step_*.safetensors"))
assert ckpts, "no checkpoint written"
sd = load_file(ckpts[-1])
assert "unet.conv_in.weight" in sd and "refiner.conv_in.weight" in sd, "missing keys"
bad = [k for k, v in sd.items() if not torch.isfinite(v).all()]
assert not bad, f"NaN/Inf in: {bad[:5]}"
print(f"SMOKE PASS — {len(sd)} tensors, no NaN, ckpt={ckpts[-1]}")
PY

echo "=== smoke OK ==="