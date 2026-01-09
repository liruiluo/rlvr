# rlvr workspace

This repo is organized as a small experiment workspace around `verl` (external dependency) + `reasoning_gym` (installed as a Python package).

## Where to run

- Main experiment entrypoint: `experiments/verl_rgym/grpo_train_local.py`
- Configs: `experiments/verl_rgym/configs/`
- Non-invasive extensions (MoE-LoRA + SPHERE + local HF rollout): `experiments/verl_rgym/verl_ext/`
- One-command run scripts (auto env + local cache): `scripts/`
- External dependencies: `external/` (e.g. `external/verl`)

## Quickstart

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh
```

Plain LoRA:

```bash
bash scripts/rgym/rl/lora/reasoning_gym_grpo_chain_sum_single_gpu_lora_hf.sh
```

## Continual learning (task-sequential)

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_moe_lora_sphere_hf.sh crl.seq=default
```

Task configs: `experiments/verl_rgym/configs/task/` (override via `task=<name>`).
Task order: `experiments/verl_rgym/configs/seq/` (selected via `crl.seq=<name>`), or override with `crl.tasks=[a,b,c]`.

Plain LoRA:

```bash
bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_lora_hf.sh crl.seq=default
```

## W&B logging

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_chain_sum_single_gpu_moe_lora_sphere_hf.sh trainer.logger=[console,wandb]
```

See `experiments/verl_rgym/README.md` for more runnable examples and curriculum scripts.
