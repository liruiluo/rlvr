# verl + reasoning_gym (single-GPU)

This directory contains a small runner (`grpo_train_local.py`) that trains an LM with `verl` on `reasoning_gym` procedural datasets.

## Dependencies in this repo

- `verl`: external dependency under `external/verl` (added to `PYTHONPATH` by `scripts/common/setup_env.sh`)
- `reasoning_gym`: installed as a Python package in `env/`

## Quickstart (single GPU, local HF rollout)

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh
```

Token-level MoE-LoRA + SPHERE requires HF rollout (we keep `external/verl` unmodified).

Single Ascend NPU (910/910B):

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_npu_moe_lora_sphere_hf.sh
```

Plain LoRA (no MoE, no SPHERE):

```bash
bash scripts/rgym/rl/lora/reasoning_gym_grpo_chain_sum_single_gpu_lora_hf.sh
```

## Continual learning (task-sequential)

Recommended sequential task order for a fast “works → forgets → relearns” demo:

1) `spell_backward` → 2) `chain_sum` → 3) `gcd` → 4) `base_conversion`

Run:

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_moe_lora_sphere_hf.sh crl.seq=default
```

Task-specific dataset settings live in `experiments/verl_rgym/configs/task/` (used by continual runs to swap datasets per phase).
Task order lives in `experiments/verl_rgym/configs/seq/` (selected via `crl.seq=<name>`), or override with `crl.tasks=[a,b,c]`.

Single NPU:

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_npu_moe_lora_sphere_hf.sh crl.seq=default
```

Plain LoRA (no MoE, no SPHERE):

```bash
bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_lora_hf.sh crl.seq=default
```

Enable a small rehearsal stream (optional):

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_moe_lora_sphere_hf.sh crl.seq=default crl.replay_weight=0.05
```

## W&B logging

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh trainer.logger=[console,wandb]
```

Models/datasets are cached under `./.cache/` by default via `scripts/common/setup_env.sh` (override with `RLVR_CACHE_ROOT=/path`).

## Timing

Wall-clock timings are appended to `checkpoints/<project>/<experiment>/timing.jsonl`.

## 4-GPU / cluster

Single node with 4 GPUs:

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh trainer.n_gpus_per_node=4
```

Existing Ray cluster:

```bash
bash scripts/rgym/cluster/rl/moe_lora/reasoning_gym_grpo_chain_sum_moe_lora_sphere_hf_npu.sh
```
