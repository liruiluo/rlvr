#!/usr/bin/env bash
# Continual RL (task-sequential) on Reasoning Gym with verl (single GPU).
# Preset: Algorithms domain (5 tasks) × 100 steps per task.
# Usage:
#   bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

VISIBLE_GPUS="$("${RLVR_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${VISIBLE_GPUS}" -lt 1 ]]; then
  echo "Expected >=1 visible CUDA GPU but got ${VISIBLE_GPUS}. Check CUDA_VISIBLE_DEVICES / driver setup." >&2
  exit 1
fi

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/continual_algorithmsdomain_100steps_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_crl_single_gpu_perf \
  -m seed=0,1,2 \
  crl.seq=AlgorithmsDomain \
  crl.steps_per_phase=100 \
  'trainer.experiment_name=continual_1gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf_seed${seed}' \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
