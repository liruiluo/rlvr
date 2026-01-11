#!/usr/bin/env bash
# One-command local run (single GPU).
# Usage:
#   bash scripts/run_1gpu.sh
set -euo pipefail

# If the user didn't pin GPUs, default to the first visible CUDA device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh "$@"
