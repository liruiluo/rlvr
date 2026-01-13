#!/usr/bin/env bash
# LoRA wrapper for the MoE-LoRA CRL script (disables MoE-LoRA + SPHERE).
# Preset: Algorithmic domain (3 tasks) × 160 steps per task (4 GPUs).
# Usage:
#   bash scripts/rgym/crl/lora/reasoning_gym_continual_4gpu_AlgorithmicDomain_160steps_lora_hf.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/../moe_lora/reasoning_gym_continual_4gpu_AlgorithmicDomain_160steps_moe_lora_sphere_hf.sh" \
  actor_rollout_ref.rollout.custom.verl_ext.use_moe_lora=false \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_feature_ratio=0.0 \
  actor_rollout_ref.rollout.custom.verl_ext.sphere_gating_ratio=0.0 \
  'trainer.experiment_name=continual_4gpu_AlgorithmicDomain_160steps_lora_hf_seed${seed}' \
  "$@"

