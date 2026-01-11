#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ts="$(date +%Y%m%d_%H%M%S)"
log="logs/continual_bg_${ts}.log"
pidf="logs/continual_bg_${ts}.pid"

mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../../env/bin/python}"

nohup "$PYTHON_BIN" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_crl_bg \
  seed=0 \
  "$@" \
  >"$log" 2>&1 &

echo $! | tee "$pidf"
echo "log=$log"
