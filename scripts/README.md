# Scripts index

This repo keeps only scripts that match these constraints:

- Strict single-domain curriculum: `crl.seq=AlgorithmsDomain` (see `experiments/verl_rgym/configs/seq/AlgorithmsDomain.yaml`)
- 5 tasks
- 100 steps per task
- Runnable without passing any args (optional Hydra overrides still accepted)

## CRL (one run, task-sequential)

**MoE-LoRA + SPHERE**

- 1 GPU: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh`
- 4 GPUs: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_AlgorithmsDomain_100steps_moe_lora_sphere_hf.sh`

**MoE-LoRA (no SPHERE)**

- 1 GPU: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_moe_lora_no_sphere_hf.sh`
- 4 GPUs: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4gpu_AlgorithmsDomain_100steps_moe_lora_no_sphere_hf.sh`

**Plain LoRA**

- 1 GPU: `bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_AlgorithmsDomain_100steps_lora_hf.sh`
- 4 GPUs: `bash scripts/rgym/crl/lora/reasoning_gym_continual_4gpu_AlgorithmsDomain_100steps_lora_hf.sh`

## Non-CRL (five independent runs)

Each script runs 5 separate GRPO runs (one per task), each with `trainer.total_training_steps=100`.

**MoE-LoRA + SPHERE**

- 1 GPU: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_moe_lora_sphere_hf.sh`
- 4 GPUs: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_4gpu_moe_lora_sphere_hf.sh`

**MoE-LoRA (no SPHERE)**

- 1 GPU: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_moe_lora_no_sphere_hf.sh`
- 4 GPUs: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_4gpu_moe_lora_no_sphere_hf.sh`

**Plain LoRA**

- 1 GPU: `bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_single_gpu_lora_hf.sh`
- 4 GPUs: `bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmsDomain_5tasks_100steps_each_4gpu_lora_hf.sh`
