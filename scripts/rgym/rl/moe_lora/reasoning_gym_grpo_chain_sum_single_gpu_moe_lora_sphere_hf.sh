#!/usr/bin/env bash
# Single-task RL (GRPO) on Reasoning Gym with verl (single GPU).
# Usage:
#   bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh [Hydra overrides...]
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/run_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf \
  -m seed=0,1,2 \
  reasoning_gym.dataset_size=256 \
  trainer.total_training_steps=500 \
  trainer.test_freq=8 \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
