#!/usr/bin/env bash
# Plain LoRA wrapper (disables MoE-LoRA + SPHERE).
# Preset: Algorithms domain (5 tasks) × 100 steps per task (single GPU).
# Usage:
#   bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_lora_hf.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/../moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh" \
  'trainer.experiment_name=continual_1gpu_AlgorithmsDomain_100steps_lora_hf_seed${seed}' \
  actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
  "$@"

