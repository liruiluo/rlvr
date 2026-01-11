#!/usr/bin/env bash
# Single-task RL (GRPO) on Reasoning Gym with verl (4 GPUs on a single node).
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_4gpu_moe_lora_sphere_hf.sh [Hydra overrides...]
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/run_4gpu_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_single_gpu_perf \
  seed=0 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  reasoning_gym.dataset_size=256 \
  trainer.total_training_steps=500 \
  trainer.test_freq=8 \
  trainer.experiment_name="grpo_chain_sum_4gpu_moe_lora_sphere_hf_seed0" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"

