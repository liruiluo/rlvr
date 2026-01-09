#!/usr/bin/env bash
# GRPO on Reasoning Gym with verl (Ray cluster, NPU, HF rollout, MoE-LoRA + SPHERE).
# Usage:
#   bash scripts/rgym/cluster/rl/moe_lora/reasoning_gym_grpo_chain_sum_moe_lora_sphere_hf_npu.sh [Hydra overrides...]
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/cluster_run_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_npu \
  /task=chain_sum \
  ray.address=auto \
  trainer.experiment_name=chain_sum_cluster_npu_moe_lora_sphere_hf \
  trainer.n_gpus_per_node=4 \
  "$@" \
  2>&1 | tee "${LOG_PATH}"

