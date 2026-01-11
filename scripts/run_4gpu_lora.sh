#!/usr/bin/env bash
# One-command local run (4 GPUs on a single node, plain LoRA).
# Usage:
#   bash scripts/run_4gpu_lora.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

bash scripts/rgym/rl/lora/reasoning_gym_grpo_chain_sum_4gpu_lora_hf.sh "$@"

