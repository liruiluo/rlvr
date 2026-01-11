#!/usr/bin/env bash
# Continual RL (task-sequential) on Reasoning Gym with verl (4 GPUs on a single node).
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_moe_lora_sphere_hf.sh [Hydra overrides...]
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/continual_4gpu_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_crl_single_gpu_perf \
  seed=0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.experiment_name="continual_4gpu_moe_lora_sphere_hf_seed0" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"

