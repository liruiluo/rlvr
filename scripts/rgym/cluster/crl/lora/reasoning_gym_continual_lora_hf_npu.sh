#!/usr/bin/env bash
# Plain LoRA variant of `.../moe_lora/...` (disables MoE-LoRA + SPHERE).
# Usage:
#   bash scripts/rgym/cluster/crl/lora/reasoning_gym_continual_lora_hf_npu.sh [Hydra overrides...]
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/../moe_lora/reasoning_gym_continual_moe_lora_sphere_hf_npu.sh" \
  trainer.experiment_name=continual_cluster_npu_lora_hf \
  actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
  "$@"

