#!/usr/bin/env bash
# Plain LoRA variant (disables MoE-LoRA + SPHERE).
# Preset: run 5 Algorithms-domain tasks as 5 independent runs, 100 steps each (4 GPUs single node).
# Not CRL: each task is a separate training run/checkpoint dir.
# Usage:
#   bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_4gpu_lora_hf.sh
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VISIBLE_GPUS="$("${RLVR_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${VISIBLE_GPUS}" -lt 4 ]]; then
  echo "Expected >=4 visible CUDA GPUs but got ${VISIBLE_GPUS}. Set CUDA_VISIBLE_DEVICES to 4 GPUs." >&2
  exit 1
fi

TASKS=(chain_sum gcd lcm base_conversion spell_backward)

for task in "${TASKS[@]}"; do
  ray stop -f

  mkdir -p logs
  DATE_TIME="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="logs/rl_4gpu_algorithmsdomain_${task}_100steps_lora_${DATE_TIME}.log"
  echo "Task=${task} Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

  "${RLVR_PYTHON}" grpo_train_local.py \
    --config-path configs \
    --config-name algo/rgym/grpo_moe_lora_sphere_hf \
    -m seed=0,1,2 \
    "task=${task}" \
    reasoning_gym.dataset_size=256 \
    trainer.total_training_steps=100 \
    trainer.test_freq=20 \
    trainer.n_gpus_per_node=4 \
    actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
    "trainer.experiment_name=AlgorithmsDomain_rl_4gpu_${task}_100steps_lora_hf_seed\${seed}" \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
done
