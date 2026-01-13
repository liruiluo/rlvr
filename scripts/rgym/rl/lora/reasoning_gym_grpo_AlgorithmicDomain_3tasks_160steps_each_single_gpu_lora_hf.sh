#!/usr/bin/env bash
# Plain LoRA variant (disables MoE-LoRA + SPHERE).
# Preset: ReasoningGym Algorithmic-domain tasks (3 tasks) as independent runs, 160 steps each (single GPU).
# Mirrors upstream ReasoningGym intra-domain Algorithmic composite: spell_backward + letter_jumble + word_sorting.
# Usage:
#   bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_single_gpu_lora_hf.sh
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

VISIBLE_GPUS="$("${RLVR_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${VISIBLE_GPUS}" -lt 1 ]]; then
  echo "Expected >=1 visible CUDA GPU but got ${VISIBLE_GPUS}. Check CUDA_VISIBLE_DEVICES / driver setup." >&2
  exit 1
fi

TASKS=(spell_backward letter_jumble word_sorting)

for task in "${TASKS[@]}"; do
  ray stop -f

  mkdir -p logs
  DATE_TIME="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="logs/rl_algorithmicdomain_${task}_160steps_lora_${DATE_TIME}.log"
  echo "Task=${task} Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

  "${RLVR_PYTHON}" grpo_train_local.py \
    --config-path configs \
    --config-name algo/rgym/grpo_moe_lora_sphere_hf \
    -m seed=0,1,2 \
    "task=${task}" \
    reasoning_gym.dataset_size=256 \
    trainer.total_training_steps=160 \
    trainer.test_freq=20 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.custom.verl_ext.hf_max_batch_size=4 \
    actor_rollout_ref.rollout.custom.verl_ext.hf_batch_wait_ms=2.0 \
    actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
    "trainer.experiment_name=AlgorithmicDomain_rl_${task}_160steps_lora_hf_seed\${seed}" \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
done

