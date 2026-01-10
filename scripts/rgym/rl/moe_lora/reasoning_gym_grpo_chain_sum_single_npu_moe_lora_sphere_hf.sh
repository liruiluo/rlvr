#!/usr/bin/env bash
# GRPO on Reasoning Gym with verl (single NPU, HF rollout, MoE-LoRA + SPHERE).
# Usage:
#   bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_npu_moe_lora_sphere_hf.sh [Hydra overrides...]
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
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_npu \
  -m seed=0,1,2 \
  /task=chain_sum \
  'trainer.experiment_name=chain_sum_single_npu_moe_lora_sphere_hf_seed${seed}' \
  reasoning_gym.dataset_size=256 \
  actor_rollout_ref.rollout.max_model_len=384 \
  data.max_response_length=64 \
  trainer.test_freq=8 \
  trainer.save_freq=8 \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
