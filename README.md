# rlvr workspace

This repo is organized as a small experiment workspace around `verl` (external dependency) + `reasoning_gym` (installed as a Python package).

## Where to run

- Main experiment entrypoint: `experiments/verl_rgym/grpo_train_local.py`
- Configs: `experiments/verl_rgym/configs/`
- Non-invasive extensions (MoE-LoRA + SPHERE + local HF rollout): `experiments/verl_rgym/verl_ext/`
- One-command run scripts (auto env + local cache): `scripts/rgym/`
- External dependencies: `external/` (e.g. `external/verl`)

## Quickstart

Algorithms-domain, 5 tasks × 100 steps (CRL):

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

MoE-LoRA (no SPHERE), same schedule:

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_no_sphere_hf.sh
```

Algorithms-domain, 5 tasks × 100 steps each (non-CRL: 5 independent runs):

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_moe_lora_sphere_hf.sh
```

MoE-LoRA (no SPHERE), same schedule:

```bash
bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_moe_lora_no_sphere_hf.sh
```

Plain LoRA variants:

```bash
bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_lora_hf.sh
bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_lora_hf.sh
```

## Continual learning (task-sequential)

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

Task order is defined in `experiments/verl_rgym/configs/seq/AlgorithmsDomain.yaml`.

## W&B logging

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

To disable W&B:

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh trainer.logger=[console]
```

## Timing

Wall-clock timings are appended to `checkpoints/<project>/<experiment>/timing.jsonl`.

## Resume

- Single-task: re-run the same script with the same `trainer.project_name` + `trainer.experiment_name` (default `trainer.resume_mode=auto`).
- CRL: re-run with the same experiment name; phase boundaries are tracked in `checkpoints/<project>/<experiment>/crl_state.json`.
  - To change task order or `crl.steps_per_phase`, use a new `trainer.experiment_name`.

## 4-GPU / cluster

If you run on a single node with 4 GPUs (local Ray on that node):

```bash
bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh
```

See `experiments/verl_rgym/README.md` for more runnable examples and curriculum scripts.

For a map of what each script is for, see `scripts/README.md`.
