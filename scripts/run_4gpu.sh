#!/usr/bin/env bash
# One-command local run (4 GPUs on a single node).
# Usage:
#   bash scripts/run_4gpu.sh
set -euo pipefail

bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_4gpu_moe_lora_sphere_hf.sh "$@"

