#!/usr/bin/env bash
# Continual RL (task-sequential) on Reasoning Gym with verl (single GPU).
# Usage:
#   bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_moe_lora_sphere_hf.sh [Hydra overrides...]
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/continual_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_crl_single_gpu_perf \
  -m seed=0,1,2 \
  'trainer.experiment_name=continual_single_gpu_moe_lora_sphere_hf_seed${seed}' \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
