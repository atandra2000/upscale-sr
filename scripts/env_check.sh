#!/usr/bin/env bash
# scripts/env_check.sh — verify the pod environment before training (EXECUTION-PLAN Phase 0).
set -uo pipefail
PYTHON="${PYTHON:-python3}"
cd "$(dirname "$0")/.."

echo "=== Upscale-SR env check ==="
"$PYTHON" - <<'PY'
import torch, importlib, sys
ok = True
def chk(name, fn=lambda m: True):
    global ok
    try:
        m = importlib.import_module(name)
        fn(m); print(f"  [OK] {name}")
    except Exception as e:
        ok = False
        print(f"  [MISSING] {name} — {e}")

print(f"python {sys.version.split()[0]} | torch {torch.__version__}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name} | sm_{p.major}{p.minor} | {p.total_memory/1e9:.1f} GB")
    cc = torch.cuda.get_device_capability(0)
    print(f"  capability: sm_{cc[0]}{cc[1]} (Blackwell target = sm_120)")
else:
    print("  CUDA not available — smoke/stub mode only")

for mod in ["numpy", "PIL", "yaml", "safetensors", "diffusers", "torchvision"]:
    chk(mod)
for mod in ["flash_attn", "mamba_ssm", "causal_conv1d"]:
    try:
        importlib.import_module(mod); print(f"  [OK] {mod} (kernel)")
    except Exception:
        print(f"  [fallback] {mod} — pure-PyTorch/sdpa fallback will be used")
for mod in ["webdataset", "lpips", "gradio", "onnxruntime"]:
    try:
        importlib.import_module(mod); print(f"  [OK] {mod}")
    except Exception:
        print(f"  [optional-missing] {mod}")

print("ENV CHECK:", "PASS" if ok else "FAIL (install missing deps)")
PY