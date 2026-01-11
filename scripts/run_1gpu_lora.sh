#!/usr/bin/env bash
# One-command local run (single GPU, plain LoRA).
# Usage:
#   bash scripts/run_1gpu_lora.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash scripts/rgym/rl/lora/reasoning_gym_grpo_chain_sum_single_gpu_lora_hf.sh "$@"

