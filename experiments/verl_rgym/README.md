# verl + reasoning_gym

This directory contains a small runner (`grpo_train_local.py`) that trains an LM with `verl` on `reasoning_gym` procedural datasets.

## Dependencies in this repo

- `verl`: external dependency under `external/verl` (added to `PYTHONPATH` by `scripts/common/setup_env.sh`)
- `reasoning_gym`: installed as a Python package in `env/`

## Quickstart

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

Task order is defined in `experiments/verl_rgym/configs/seq/AlgorithmsDomain.yaml`.

Other variants (all runnable with no args):

```bash
# CRL, MoE-LoRA (no SPHERE)
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_no_sphere_hf.sh

# CRL, plain LoRA
bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_lora_hf.sh

# Non-CRL: 5 independent GRPO runs (one per task)
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_moe_lora_sphere_hf.sh
```

## W&B logging

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

To disable W&B:

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh trainer.logger=[console]
```

Models/datasets are cached under `./.cache/` by default via `scripts/common/setup_env.sh` (override with `RLVR_CACHE_ROOT=/path`).

## Timing

Wall-clock timings are appended to `checkpoints/<project>/<experiment>/timing.jsonl`.

## Resume

- Single-task: re-run the same script with the same `trainer.project_name` + `trainer.experiment_name` (default `trainer.resume_mode=auto`).
- CRL: re-run with the same experiment name; phase boundaries are tracked in `checkpoints/<project>/<experiment>/crl_state.json`.
  - To change task order or `crl.steps_per_phase`, use a new `trainer.experiment_name`.

## 4-GPU

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```
