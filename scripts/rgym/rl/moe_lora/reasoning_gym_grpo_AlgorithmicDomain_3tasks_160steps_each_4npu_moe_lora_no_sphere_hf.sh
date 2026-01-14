#!/usr/bin/env bash
# MoE-LoRA without SPHERE variant.
# Preset: ReasoningGym Algorithmic-domain tasks (3 tasks) as independent runs, 160 steps each (4 NPUs single node).
# Mirrors upstream ReasoningGym intra-domain Algorithmic composite: spell_backward + letter_jumble + word_sorting.
# Usage:
#   bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_4npu_moe_lora_no_sphere_hf.sh
set -euo pipefail

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
echo "ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-<unset>}"

VISIBLE_NPUS="$(${RLVR_PYTHON} - <<PY
try:
    import torch_npu  # noqa: F401
except Exception:
    pass
import torch
count = 0
if hasattr(torch, "npu"):
    count = int(torch.npu.device_count())
print(count)
PY
)"
if [[ "${VISIBLE_NPUS}" -lt 4 ]]; then
  echo "Expected >=4 visible NPUs but got ${VISIBLE_NPUS}. Set ASCEND_VISIBLE_DEVICES to 4 NPUs." >&2
  exit 1
fi

TASKS=(spell_backward letter_jumble word_sorting)

for task in "${TASKS[@]}"; do
  ray stop -f || true

  mkdir -p logs
  DATE_TIME="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="logs/rl_4npu_algorithmicdomain_${task}_160steps_moe_lora_no_sphere_${DATE_TIME}.log"
  echo "Task=${task} Log: ${RLVR_REPO_ROOT}/experiments/verl_rgym/${LOG_PATH}"

  EXTRA_OVERRIDES=()
  if [[ "${task}" == "letter_jumble" ]]; then
    EXTRA_OVERRIDES+=(data.max_prompt_length=512 data.max_response_length=1024 actor_rollout_ref.rollout.max_model_len=1536)
  fi

  "${RLVR_PYTHON}" grpo_train_local.py \
    --config-path configs \
    --config-name algo/rgym/grpo_moe_lora_sphere_hf_npu \
    -m seed=0,1,2 \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    "task=${task}" \
    reasoning_gym.dataset_size=256 \
    trainer.total_training_steps=160 \
    trainer.test_freq=20 \
    actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=true \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
    actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
    "trainer.experiment_name=AlgorithmicDomain_rl_4npu_${task}_160steps_moe_lora_no_sphere_hf_seed\${seed}" \
    "${EXTRA_OVERRIDES[@]}" \
    "$@" \
    2>&1 | tee "${LOG_PATH}"
done
