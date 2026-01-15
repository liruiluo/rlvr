#!/usr/bin/env bash
set -euo pipefail

# Background-friendly single-task RL run (4 NPUs, single node).
# Usage:
#   nohup bash scripts/rgym/rl/lora/run_spell_backward_4npu_160steps_bg.sh &

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLVR_REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
export RLVR_REPO_ROOT

# Use system-installed dependencies (verl + reasoning_gym) and enable Ascend NPU runtime.
export LC_ALL="${LC_ALL:-C}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

# Ascend env scripts are not errexit/nounset-safe.
set +e
set +u
source /usr/local/Ascend/ascend-toolkit/latest/aarch64-linux/bin/setenv.bash
set -u
set -e

export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}"
export PATH="/usr/local/python3.11.13/bin:${PATH}"
export RLVR_PYTHON="${RLVR_PYTHON:-/usr/local/python3.11.13/bin/python3}"

# veRL Hydra searchpath uses VERL_REPO_ROOT/verl/trainer/config.
export VERL_REPO_ROOT="${VERL_REPO_ROOT:-/verl}"

# Ensure Ray workers can import veRL when it's not in site-packages.
export PYTHONPATH="${RLVR_REPO_ROOT}/experiments/verl_rgym:${VERL_REPO_ROOT}:${PYTHONPATH:-}"

# Prefer offline to avoid interactive auth issues in background; user can export WANDB_MODE=online if desired.
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

cd "${RLVR_REPO_ROOT}/experiments/verl_rgym"

REQUIRED_NPUS=4
if [[ -z "${ASCEND_VISIBLE_DEVICES:-}" ]] && command -v npu-smi >/dev/null 2>&1; then
  DETECTED_NPU_IDS="$(
    npu-smi info -l 2>/dev/null \
      | awk -F: '/NPU ID/ {gsub(/ /,"",$2); print $2}' \
      | head -n "${REQUIRED_NPUS}" \
      | paste -sd, - \
      || true
  )"
  if [[ -n "${DETECTED_NPU_IDS}" ]]; then
    export ASCEND_VISIBLE_DEVICES="${DETECTED_NPU_IDS}"
  fi
fi

echo "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}" | tee /dev/stderr

task="spell_backward"
seed="${SEED:-0}"

mkdir -p logs
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="logs/rl_4npu_${task}_160steps_lora_seed${seed}_${DATE_TIME}.log"
echo "Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

ray stop -f >/dev/null 2>&1 || true

"${RLVR_PYTHON}" grpo_train_local.py \
  --config-path configs \
  --config-name algo/rgym/grpo_moe_lora_sphere_hf_npu \
  "seed=${seed}" \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  "task=${task}" \
  reasoning_gym.dataset_size=256 \
  trainer.total_training_steps=160 \
  trainer.test_freq=20 \
  actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
  "trainer.experiment_name=AlgorithmicDomain_rl_4npu_${task}_160steps_lora_hf_seed${seed}" \
  "$@" \
  2>&1 | tee "${LOG_PATH}"
