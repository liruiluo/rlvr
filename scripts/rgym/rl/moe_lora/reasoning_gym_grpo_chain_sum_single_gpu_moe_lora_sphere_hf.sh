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
  reasoning_gym.dataset_size=64 \
  data.train_batch_size=8 \
  data.val_batch_size=8 \
  trainer.total_epochs=1 \
  trainer.test_freq=8 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
