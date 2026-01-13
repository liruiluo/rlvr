# Scripts index

This repo keeps only scripts that match these constraints:

- Strict single-domain curriculum: `crl.seq=AlgorithmicDomain` (see `experiments/verl_rgym/configs/seq/AlgorithmicDomain.yaml`)
- 3 tasks
- 160 steps per task
- Runnable without passing any args (optional Hydra overrides still accepted)

W&B is enabled by default (disable with `trainer.logger=[console]`).

## CRL (one run, task-sequential)

**MoE-LoRA + SPHERE**

- 1 GPU: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmicDomain_160steps_moe_lora_sphere_hf.sh`
- 4 NPUs: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4npu_AlgorithmicDomain_160steps_moe_lora_sphere_hf.sh`

**MoE-LoRA (no SPHERE)**

- 1 GPU: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_single_gpu_AlgorithmicDomain_160steps_moe_lora_no_sphere_hf.sh`
- 4 NPUs: `bash scripts/rgym/crl/moe_lora/reasoning_gym_continual_4npu_AlgorithmicDomain_160steps_moe_lora_no_sphere_hf.sh`

**Plain LoRA**

- 1 GPU: `bash scripts/rgym/crl/lora/reasoning_gym_continual_single_gpu_AlgorithmicDomain_160steps_lora_hf.sh`
- 4 NPUs: `bash scripts/rgym/crl/lora/reasoning_gym_continual_4npu_AlgorithmicDomain_160steps_lora_hf.sh`

## Non-CRL (three independent runs)

Each script runs 3 separate GRPO runs (one per task), each with `trainer.total_training_steps=160`.

**MoE-LoRA + SPHERE**

- 1 GPU: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_single_gpu_moe_lora_sphere_hf.sh`
- 4 NPUs: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_4npu_moe_lora_sphere_hf.sh`

**MoE-LoRA (no SPHERE)**

- 1 GPU: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_single_gpu_moe_lora_no_sphere_hf.sh`
- 4 NPUs: `bash scripts/rgym/rl/moe_lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_4npu_moe_lora_no_sphere_hf.sh`

**Plain LoRA**

- 1 GPU: `bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_single_gpu_lora_hf.sh`
- 4 NPUs: `bash scripts/rgym/rl/lora/reasoning_gym_grpo_AlgorithmicDomain_3tasks_160steps_each_4npu_lora_hf.sh`
