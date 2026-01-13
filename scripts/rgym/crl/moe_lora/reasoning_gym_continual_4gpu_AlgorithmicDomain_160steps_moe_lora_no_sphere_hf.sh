#!/usr/bin/env bash
# Continual RL (task-sequential) on Reasoning Gym with verl (4 GPUs on a single node).
# Preset: Algorithmic domain (3 tasks) × 160 steps per task.
# Mirrors upstream ReasoningGym intra-domain Algorithmic composite: spell_backward + letter_jumble + word_sorting.
# Usage:
#   bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_AlgorithmicDomain_160steps_moe_lora_no_sphere_hf.sh
set -euo pipefail

source scripts/common/setup_env.sh

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

ray stop -f
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

VISIBLE_GPUS="$("${RLVR_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "${VISIBLE_GPUS}" -lt 4 ]]; then
  echo "Expected >=4 visible CUDA GPUs but got ${VISIBLE_GPUS}. Set CUDA_VISIBLE_DEVICES to 4 GPUs." >&2
  exit 1
fi

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/continual_4gpu_algorithmicdomain_160steps_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_crl \
  -m seed=0,1,2 \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  crl.seq=AlgorithmicDomain \
  crl.steps_per_phase=160 \
  actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=true \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
  'trainer.experiment_name=continual_4gpu_AlgorithmicDomain_160steps_moe_lora_no_sphere_hf_seed${seed}' \
  "$@" \
  2>&1 | tee "${LOG_PATH}"

