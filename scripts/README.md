# Scripts index

This folder intentionally has **two layers**:

1) **One-command entrypoints** (recommended): `scripts/run_*.sh`  
2) **Composable presets** under `scripts/rgym/**` (RL/CRL, GPU/NPU, local/cluster)

The “redundancy” you see is mostly thin wrapper scripts (especially `lora/`) that reuse the `moe_lora/` scripts and only override a few Hydra flags. This is on purpose so we don’t duplicate full training commands.

## Start here (no args needed)

**Default = MoE-LoRA + SPHERE (still LoRA training, just with extra extensions enabled)**

- 1 GPU: `bash scripts/run_1gpu.sh`
- 4 GPUs (single node): `bash scripts/run_4gpu.sh`

**Plain LoRA = disables MoE-LoRA + SPHERE**

- 1 GPU: `bash scripts/run_1gpu_lora.sh`
- 4 GPUs (single node): `bash scripts/run_4gpu_lora.sh`

All four accept optional Hydra overrides, but you can run them with zero parameters.

## How to read the tree

- `scripts/common/setup_env.sh`: repo-local env bootstrap (used by all run scripts)
- `scripts/rgym/rl/**`: single-task RL (GRPO)
- `scripts/rgym/crl/**`: continual RL (task-sequential)
- `scripts/rgym/**/moe_lora/**`: MoE-LoRA + SPHERE presets
- `scripts/rgym/**/lora/**`: *wrappers* that turn off MoE-LoRA + SPHERE
- `scripts/rgym/cluster/**`: connect to an existing Ray cluster (does not call `ray stop -f`)

