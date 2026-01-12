#!/usr/bin/env bash
# One-command continual RL run (single GPU): 5 tasks, 100 steps each.
# Usage:
#   bash scripts/run_crl_1gpu_5tasks_100steps.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_moe_lora_sphere_hf.sh \
  crl.seq=default5 \
  crl.steps_per_phase=100 \
  'trainer.experiment_name=continual_1gpu_5tasks_100steps_moe_lora_sphere_hf_seed${seed}' \
  "$@"

