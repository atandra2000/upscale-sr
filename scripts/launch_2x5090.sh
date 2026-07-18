#!/usr/bin/env bash
# scripts/launch_2x5090.sh — full 100 K-iter training on 2× RTX 5090 32 GB.
#
# Usage:
#   bash scripts/launch_2x5090.sh                 # 100 K iters, default config
#   CKPT_DIR=/workspace/runs/upscale-sr/v2 bash scripts/launch_2x5090.sh
#
# Assumes the data shards are already built at /workspace/data/sr (run
# ``python data/build_div2k_flickr.py --out /workspace/data/sr`` first).
#
# Launches DDP via torchrun (2 GPUs), BF16, FA2, channels_last, torch.compile,
# EMA, atomic ckpts every 10 K iters.  A background profiler on rank 0 polls
# GPU-0 utilisation for the ≥ 95% acceptance check.
set -euo pipefail
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

CKPT_DIR="${CKPT_DIR:-/workspace/runs/upscale-sr}"
CONFIG="${CONFIG:-configs/sr_x4_realesrgan_2x5090.yaml}"
WORLD="${WORLD:-2}"

mkdir -p "$CKPT_DIR"
echo "=== Upscale-SR — 2× RTX 5090 DDP | 100 K iters | ckpt → $CKPT_DIR ==="

# Background ≥ 95% utilisation probe (rank 0, GPU 0).  Writes to $CKPT_DIR/util.log.
( "$PYTHON" -m training.profiler --gpu 0 --duration 999999 --interval 5 \
      --target-util 95.0 > "$CKPT_DIR/util.log" 2>&1 ) &
PROFILER_PID=$!
trap 'kill $PROFILER_PID 2>/dev/null || true' EXIT

# ── the run ────────────────────────────────────────────────────────────────
torchrun --nproc-per-node="$WORLD" --master-port=29555 \
    -m training.train --config "$CONFIG" --ckpt-dir "$CKPT_DIR" "$@"

echo "=== training complete — ship with: python export/to_safetensors.py ==="