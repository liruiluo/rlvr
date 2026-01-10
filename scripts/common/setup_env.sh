#!/usr/bin/env bash
set -euo pipefail

# This script bootstraps the repo-local Python environment (a self-contained conda env under `env/`).
# It is meant to be sourced by other run scripts so users don't need to manually activate anything.

SETUP_ENV_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RLVR_REPO_ROOT="$(cd -- "${SETUP_ENV_DIR}/../.." && pwd)"
export RLVR_REPO_ROOT

RLVR_PYTHON="${RLVR_REPO_ROOT}/env/bin/python"
export RLVR_PYTHON

if [[ ! -x "${RLVR_PYTHON}" ]]; then
  echo "Missing repo environment python at: ${RLVR_PYTHON}" >&2
  echo "Expected layout: <repo>/env/bin/python" >&2
  return 1
fi

export PATH="${RLVR_REPO_ROOT}/env/bin:${PATH}"
export PYTHONNOUSERSITE=1

# External dependency roots
VERL_REPO_ROOT="${VERL_REPO_ROOT:-${RLVR_REPO_ROOT}/external/verl}"
export VERL_REPO_ROOT

# Make repo-local examples importable in Ray workers.
export PYTHONPATH="${RLVR_REPO_ROOT}/experiments/verl_rgym:${VERL_REPO_ROOT}:${PYTHONPATH:-}"

# Cache everything under the repo so models are persisted locally.
RLVR_CACHE_ROOT="${RLVR_CACHE_ROOT:-${RLVR_REPO_ROOT}/.cache}"
export RLVR_CACHE_ROOT
mkdir -p "${RLVR_CACHE_ROOT}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${RLVR_CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${RLVR_CACHE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${RLVR_CACHE_ROOT}/vllm}"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${VLLM_CACHE_ROOT}"

# Keep W&B artifacts local by default.
export WANDB_DIR="${WANDB_DIR:-${RLVR_REPO_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

# Reduce noisy tokenizer parallelism warnings for most local runs.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
