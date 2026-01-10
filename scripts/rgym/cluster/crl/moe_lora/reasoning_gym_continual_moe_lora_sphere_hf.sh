#!/usr/bin/env bash
# Continual RL (task-sequential) on Reasoning Gym with verl (connect to an existing Ray cluster).
# Usage:
#   bash scripts/rgym/cluster/crl/moe_lora/reasoning_gym_continual_moe_lora_sphere_hf.sh [Hydra overrides...]
#
# Notes:
# - This script does NOT call `ray stop -f`.
# - Pass `ray.address=auto` (or a concrete address) to connect to your cluster.
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/cluster_continual_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf \
  -m seed=0,1,2 \
  crl.enabled=true \
  'trainer.experiment_name=continual_cluster_moe_lora_sphere_hf_seed${seed}' \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
