# This example is an adapted version of Bytedance's code:
# https://github.com/volcengine/verl/blob/a65c9157bc0b85b64cd753de19f94e80a11bd871/verl/trainer/main_ppo.py
import os
import random
import re
import sys
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

VE_RL_DIR = Path(__file__).resolve().parent
REPO_ROOT = VE_RL_DIR.parents[1]
VERL_ROOT = REPO_ROOT / "external" / "verl"

# Keep the whole project runnable without manual env activation/export.
os.environ.setdefault("VERL_REPO_ROOT", str(VERL_ROOT))
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache" / "huggingface"))

# Ray worker processes rely on `PYTHONPATH` (runtime_env) rather than this process's `sys.path`.
py_path_parts = [str(VE_RL_DIR), str(VERL_ROOT)]
existing_py_path = os.environ.get("PYTHONPATH", "")
if existing_py_path:
    py_path_parts.append(existing_py_path)
os.environ["PYTHONPATH"] = ":".join(py_path_parts)

sys.path.insert(0, str(VE_RL_DIR))
sys.path.insert(0, str(VERL_ROOT))

import hydra
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer
from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.utils.dataset.rl_dataset import collate_fn as verl_collate_fn

import reasoning_gym
from reasoning_gym import utils as rg_utils
from reasoning_gym.coaching.experiment import Experiment
from reasoning_gym.composite import DatasetSpec
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.utils import extract_answer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    return str(obj)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=json_default) + "\n")

def _maybe_enable_wandb_resume(*, ckpt_root: Path, experiment_name: str) -> str | None:
    """Make CRL phase-start eval metrics land in the same W&B run as training.

    veRL creates the W&B run inside `RayPPOTrainer.fit()`, but CRL phase-start eval happens
    before `fit()`. We start+finish a W&B session here and have `fit()` resume the same run ID.
    """
    if not experiment_name:
        return None

    ckpt_root.mkdir(parents=True, exist_ok=True)
    run_id_path = ckpt_root / "wandb_run_id.txt"
    if run_id_path.exists():
        run_id = run_id_path.read_text(encoding="utf-8").strip()
    else:
        run_id = uuid.uuid4().hex[:8]
        run_id_path.write_text(run_id + "\n", encoding="utf-8")

    os.environ.setdefault("WANDB_RUN_ID", run_id)
    os.environ.setdefault("WANDB_RESUME", "allow")
    return run_id


def now_ts() -> tuple[float, str]:
    ts = time.time()
    iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return ts, iso


class ReasoningGymDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        procedural_dataset: Optional[ProceduralDataset] = None,
        experiment: Optional[Experiment] = None,
        developer_prompt: Optional[str] = None,
        developer_role: str = "system",
        max_prompt_length: int = 2048,
        truncation: str = "error",  ##  ['left', 'right', 'error']
    ):
        assert procedural_dataset or experiment, "One of `procedural_dataset` or `experiment` must be provided"
        assert (
            procedural_dataset is None or experiment is None
        ), "Only one of `procedural_dataset` or `experiment` may be provided"

        self.tokenizer = tokenizer
        self.data = procedural_dataset or experiment.composite
        self.experiment = experiment
        self.developer_prompt = developer_prompt
        self.developer_role = developer_role
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        row_dict = self.data[index].copy()
        q = row_dict["question"]
        a = row_dict["answer"]
        metadata = row_dict.get("metadata", {})

        chat = []
        if self.developer_prompt is not None:
            chat.append({"role": self.developer_role, "content": self.developer_prompt})
        chat.append({"role": "user", "content": q})

        item = {}
        item["index"] = index

        # `verl` (current async rollout) expects chat messages under `raw_prompt`.
        item["raw_prompt"] = chat

        # Minimal dummy tensor to keep DataProto.batch non-empty.
        item["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        item["uid"] = str(index)
        item["data_source"] = f"reasoning_gym/{metadata.get('source_dataset', 'unknown')}"
        item["reward_model"] = {"ground_truth": str(a), "style": "rule"}
        item["extra_info"] = {"index": index, "metadata": metadata}

        return item


def make_dataset(
    tokenizer,
    data_source: Experiment | ProceduralDataset,
    developer_prompt: str,
    max_prompt_length: int = 2048,
) -> ReasoningGymDataset:
    """
    Create ReasoningGymDataset object using either a ProceduralDataset or Experiment as the underlying data source.
    """
    if isinstance(data_source, Experiment):
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            experiment=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )
    else:
        return ReasoningGymDataset(
            tokenizer=tokenizer,
            procedural_dataset=data_source,
            developer_prompt=developer_prompt,
            developer_role="system",
            max_prompt_length=max_prompt_length,
            truncation="error",
        )


def prepare_datasets(config, tokenizer) -> tuple[ReasoningGymDataset, ReasoningGymDataset]:
    """Prepare training and validation datasets."""
    dataset_size = config.reasoning_gym.dataset_size
    developer_prompt_setting = config.reasoning_gym.developer_prompt
    developer_prompt = rg_utils.SYSTEM_PROMPTS[developer_prompt_setting]
    seed = int(OmegaConf.select(config, "seed") or 0)
    dataset_specs = []
    for name, ds in config.reasoning_gym.datasets.items():
        if ds.weight <= 0:
            continue
        dataset_specs.append(
            DatasetSpec(
                name=name,
                weight=ds.weight,
                config=OmegaConf.to_container(ds.config, resolve=True) if "config" in ds else {},
            )
        )
    # Paper-style: keep evaluation sets identical across independent runs.
    train_data_source = reasoning_gym.create_dataset(
        "composite", seed=1 + seed, size=dataset_size, datasets=dataset_specs
    )
    val_data_source = reasoning_gym.create_dataset("composite", seed=2, size=dataset_size, datasets=dataset_specs)
    train_dataset = make_dataset(
        tokenizer, train_data_source, developer_prompt, max_prompt_length=config.data.max_prompt_length
    )
    val_dataset = make_dataset(
        tokenizer, val_data_source, developer_prompt, max_prompt_length=config.data.max_prompt_length
    )
    return train_dataset, val_dataset


def load_rgym_seq_tasks(seq_name: str) -> list[str]:
    seq_path = VE_RL_DIR / "configs" / "seq" / f"{seq_name}.yaml"
    seq_cfg = OmegaConf.load(seq_path)
    tasks = list(seq_cfg.rgym_seq.tasks)
    if not tasks:
        raise ValueError(f"Empty rgym_seq.tasks in {seq_path}")
    return tasks


def load_rgym_task_override(task_name: str):
    task_path = VE_RL_DIR / "configs" / "task" / f"{task_name}.yaml"
    if not task_path.exists():
        raise FileNotFoundError(f"Missing task config: {task_path}")
    return OmegaConf.load(task_path)


def resolve_ckpt_root(config) -> Path:
    ckpt_root = Path(config.trainer.default_local_dir).expanduser()
    if not ckpt_root.is_absolute():
        ckpt_root = (Path.cwd() / ckpt_root).resolve()
    return ckpt_root


def read_latest_checkpointed_iteration(ckpt_root: Path) -> int:
    path = ckpt_root / "latest_checkpointed_iteration.txt"
    if not path.exists():
        return 0
    return int(path.read_text().strip())


def remove_latest_dataloader_state(ckpt_root: Path) -> None:
    candidates = sorted(ckpt_root.glob("global_step_*"), reverse=True)
    if not candidates:
        return
    data_pt = candidates[0] / "data.pt"
    if data_pt.exists():
        data_pt.unlink()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def load_checkpoint(trainer) -> None:
    getattr(trainer, "_load_checkpoint")()


def validate(trainer):
    return getattr(trainer, "_validate")()


def save_checkpoint(trainer) -> None:
    getattr(trainer, "_save_checkpoint")()


class RayPPOTrainerCustom(RayPPOTrainer):
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager,
        ray_worker_group_cls,
        train_dataset: ReasoningGymDataset,
        val_dataset: ReasoningGymDataset,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        def make_reward_fn(num_examine: int):
            dataset = self.train_dataset.data if num_examine == 0 else self.val_dataset.data

            def reward_fn(data: DataProto, return_dict: bool = False, **unused_kwargs):
                tensor = self.score_output(data, dataset=dataset, num_examine=num_examine)
                if return_dict:
                    # wrap it so trainer can pull out extras
                    return {"reward_tensor": tensor, "reward_extra_info": {}}
                return tensor

            return reward_fn

        train_reward_fn = make_reward_fn(num_examine=0)
        val_reward_fn = make_reward_fn(num_examine=1)

        super().__init__(
            config=config,
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=train_reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_sampler=None,
        )

    def score_output(self, data: DataProto, dataset: ProceduralDataset, num_examine: int = 0) -> torch.Tensor:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        num_printed = 0
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]  # tokenized prompts
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            sequences_str = prompt_str + response_str

            index = data_item.non_tensor_batch["extra_info"]["index"]
            score = self.compute_score(
                solution_str=response_str,
                dataset=dataset,
                index=index,
            )

            reward_tensor[i, valid_response_length - 1] = score

            if num_printed < num_examine:
                print(f"reward={score}, seq={sequences_str}")
                num_printed += 1

        return reward_tensor

    def compute_score(self, solution_str: str, dataset: ProceduralDataset, index: int) -> float:
        found_answer = extract_answer(solution_str, tag_name="answer")
        if found_answer is None:
            matches = re.findall(r"[-+]?\d+(?:\.\d+)?", solution_str.replace(",", ""))
            found_answer = matches[-1] if matches else None
        if found_answer is None:
            tokens = re.findall(r"[A-Za-z0-9]+", solution_str)
            found_answer = tokens[-1] if tokens else None
        entry = dataset[index]
        reward = dataset.score_answer(found_answer, entry=entry)
        return reward

    def create_dataloader(self, train_dataset, val_dataset, collate_fn=None, sampler=None):

        if collate_fn is None:
            collate_fn = verl_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        self.val_dataloader = StatefulDataLoader(
            dataset=val_dataset,
            batch_size=self.config.data.val_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps


setattr(RayPPOTrainerCustom, "_create_dataloader", RayPPOTrainerCustom.create_dataloader)


@ray.remote
def main_task(config):
    # print initial config
    from pprint import pprint

    from verl.utils import hf_tokenizer
    from verl.utils.fs import copy_local_path_from_hdfs
    from verl.trainer.ppo.utils import need_critic

    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    seed = int(OmegaConf.select(config, "seed") or 0)
    seed_offset = int(OmegaConf.select(config, "seed_offset") or 0)
    set_global_seed(seed_offset + seed)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    tokenizer = hf_tokenizer(local_path)
    train_dataset, val_dataset = prepare_datasets(config, tokenizer)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import AsyncActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    use_ref_policy = bool(config.algorithm.use_kl_in_reward) or bool(config.actor_rollout_ref.actor.use_kl_loss)
    use_critic_worker = need_critic(config)

    verl_ext_cfg = OmegaConf.select(config, "actor_rollout_ref.rollout.custom.verl_ext")
    use_verl_ext = bool(verl_ext_cfg) or str(config.actor_rollout_ref.rollout.name) == "hf"

    actor_worker_cls = AsyncActorRolloutRefWorker
    if use_verl_ext and config.actor_rollout_ref.actor.strategy == "fsdp":
        if str(config.actor_rollout_ref.rollout.name) == "hf":
            from verl_ext.rollout_registry import ensure_hf_rollout_registered

            ensure_hf_rollout_registered()
        from verl_ext.fsdp_worker import ExtAsyncActorRolloutRefWorker

        actor_worker_cls = ExtAsyncActorRolloutRefWorker

    role_worker_mapping = {Role.ActorRollout: ray.remote(actor_worker_cls)}
    if use_critic_worker:
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
    if use_ref_policy:
        role_worker_mapping[Role.RefPolicy] = ray.remote(AsyncActorRolloutRefWorker)

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {role: global_pool_id for role in role_worker_mapping.keys()}

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainerCustom(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.init_workers()
    trainer.global_steps = int(getattr(trainer, "global_steps", 0))

    ckpt_root = resolve_ckpt_root(config)
    timing_path = ckpt_root / "timing.jsonl"
    phase_task = OmegaConf.select(config, "crl.phase_task")
    phase_index = OmegaConf.select(config, "crl.phase_index")

    if bool(OmegaConf.select(config, "crl.enabled")) and bool(OmegaConf.select(config, "crl.eval_at_phase_start")):
        load_checkpoint(trainer)
        with open_dict(config):
            # Keep `resume_mode` enabled: `RayPPOTrainer.fit()` resets `global_steps` to 0 and
            # relies on `_load_checkpoint()` to restore the correct step counter. Disabling resume
            # here would cause training to restart from step 0 and potentially overwrite existing
            # `global_step_*` checkpoints.
            config.trainer.val_before_train = False

        start_task = str(OmegaConf.select(config, "crl.phase_task") or "unknown")
        start_step = int(getattr(trainer, "global_steps", 0))
        start_eval_dir = ckpt_root / "crl_eval_start" / start_task / f"global_step_{start_step}"
        start_eval_dir.mkdir(parents=True, exist_ok=True)

        start_dump_samples = bool(OmegaConf.select(config, "crl.dump_eval_samples_at_phase_start"))
        original_val_dir = OmegaConf.select(config, "trainer.validation_data_dir")
        if start_dump_samples:
            with open_dict(config):
                config.trainer.validation_data_dir = str(start_eval_dir / "generations")

        eval_t0 = time.perf_counter()
        metrics = validate(trainer)
        eval_end_ts, eval_end_iso = now_ts()
        (start_eval_dir / "metrics.json").write_text(
            json.dumps(
                {"task": start_task, "global_step": start_step, "when": "start", "metrics": metrics},
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )
        append_jsonl(
            timing_path,
            {
                "event": "eval_start_end",
                "ts": eval_end_ts,
                "iso_utc": eval_end_iso,
                "elapsed_sec": float(time.perf_counter() - eval_t0),
                "global_step": start_step,
                "phase_task": start_task,
                "phase_index": int(phase_index) if phase_index is not None else None,
            },
        )

        # Also log phase-start eval metrics to W&B so they show up in the same run as training.
        if "wandb" in list(config.trainer.logger or []):
            from verl.utils.tracking import Tracking

            _maybe_enable_wandb_resume(ckpt_root=ckpt_root, experiment_name=str(config.trainer.experiment_name))
            eval_logger = Tracking(
                project_name=config.trainer.project_name,
                experiment_name=config.trainer.experiment_name,
                default_backend=["wandb"],
                config=OmegaConf.to_container(config, resolve=True),
            )
            eval_logger.log(data=metrics, step=start_step)
            eval_logger.logger["wandb"].finish(exit_code=0)
            del eval_logger

        with open_dict(config):
            config.trainer.validation_data_dir = original_val_dir

    start_ts, start_iso = now_ts()
    fit_t0 = time.perf_counter()
    append_jsonl(
        timing_path,
        {
            "event": "fit_start",
            "ts": start_ts,
            "iso_utc": start_iso,
            "phase_task": str(phase_task) if phase_task is not None else None,
            "phase_index": int(phase_index) if phase_index is not None else None,
            "target_total_training_steps": int(OmegaConf.select(config, "trainer.total_training_steps") or -1),
        },
    )
    trainer.fit()
    end_ts, end_iso = now_ts()
    append_jsonl(
        timing_path,
        {
            "event": "fit_end",
            "ts": end_ts,
            "iso_utc": end_iso,
            "elapsed_sec": float(time.perf_counter() - fit_t0),
            "global_step": int(trainer.global_steps),
            "phase_task": str(phase_task) if phase_task is not None else None,
            "phase_index": int(phase_index) if phase_index is not None else None,
        },
    )

    if bool(OmegaConf.select(config, "crl.enabled")) and bool(OmegaConf.select(config, "crl.eval_at_phase_end")):
        phase_task = str(OmegaConf.select(config, "crl.phase_task") or "unknown")
        print(f"[CRL] final eval task={phase_task} global_step={trainer.global_steps}")

        dump_samples = bool(OmegaConf.select(config, "crl.dump_eval_samples_at_phase_end"))
        original_val_dir = OmegaConf.select(config, "trainer.validation_data_dir")
        eval_dir = ckpt_root / "crl_eval" / phase_task / f"global_step_{trainer.global_steps}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        if dump_samples:
            with open_dict(config):
                config.trainer.validation_data_dir = str(eval_dir / "generations")

        eval_t0 = time.perf_counter()
        metrics = validate(trainer)
        print(f"[CRL] final eval metrics: {metrics}")
        eval_end_ts, eval_end_iso = now_ts()
        append_jsonl(
            timing_path,
            {
                "event": "eval_end",
                "ts": eval_end_ts,
                "iso_utc": eval_end_iso,
                "elapsed_sec": float(time.perf_counter() - eval_t0),
                "global_step": int(trainer.global_steps),
                "phase_task": phase_task,
            },
        )

        (eval_dir / "metrics.json").write_text(
            json.dumps(
                {"task": phase_task, "global_step": int(trainer.global_steps), "metrics": metrics},
                ensure_ascii=False,
                indent=2,
                default=json_default,
            )
            + "\n",
            encoding="utf-8",
        )

        with open_dict(config):
            config.trainer.validation_data_dir = original_val_dir

    if bool(OmegaConf.select(config, "crl.enabled")) and bool(OmegaConf.select(config, "crl.save_ckpt_at_phase_end")):
        print(f"[CRL] saving checkpoint global_step={trainer.global_steps}")
        ckpt_t0 = time.perf_counter()
        save_checkpoint(trainer)
        ckpt_end_ts, ckpt_end_iso = now_ts()
        append_jsonl(
            timing_path,
            {
                "event": "ckpt_end",
                "ts": ckpt_end_ts,
                "iso_utc": ckpt_end_iso,
                "elapsed_sec": float(time.perf_counter() - ckpt_t0),
                "global_step": int(trainer.global_steps),
                "phase_task": str(OmegaConf.select(config, "crl.phase_task") or "unknown"),
            },
        )


def init_ray(address: str, include_dashboard: bool) -> None:
    ray.init(
        address=address,
        include_dashboard=include_dashboard,
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
                "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
                "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            }
        },
    )


def run_continual_learning(config) -> None:
    tasks = OmegaConf.select(config, "crl.tasks") or []
    tasks = list(tasks)
    if not tasks:
        seq = str(OmegaConf.select(config, "crl.seq") or "safe")
        tasks = load_rgym_seq_tasks(seq)

    steps_per_phase = int(OmegaConf.select(config, "crl.steps_per_phase") or 0)
    if steps_per_phase <= 0:
        raise ValueError("crl.steps_per_phase must be > 0")

    replay_weight = float(OmegaConf.select(config, "crl.replay_weight") or 0.0)
    crl_save_freq = int(OmegaConf.select(config, "crl.save_freq") or -1)
    ray_address = str(OmegaConf.select(config, "ray.address") or "local")
    ray_dashboard = bool(OmegaConf.select(config, "ray.include_dashboard") or False)

    ckpt_root = resolve_ckpt_root(config)
    current_step = read_latest_checkpointed_iteration(ckpt_root)
    state_path = ckpt_root / "crl_state.json"
    state = load_json(state_path)
    if state:
        if state.get("tasks") != tasks:
            raise ValueError(
                f"CRL resume expects the same task list as {state_path}. "
                f"Got tasks={tasks}, state.tasks={state.get('tasks')}. "
                "Use a new trainer.experiment_name for a new schedule."
            )
        if int(state.get("steps_per_phase", -1)) != steps_per_phase:
            raise ValueError(
                f"CRL resume expects the same steps_per_phase as {state_path}. "
                f"Got steps_per_phase={steps_per_phase}, state.steps_per_phase={state.get('steps_per_phase')}. "
                "Use a new trainer.experiment_name for a new schedule."
            )
    else:
        phases = []
        if current_step > 0:
            for idx, task in enumerate(tasks):
                start_step = idx * steps_per_phase
                end_step = (idx + 1) * steps_per_phase
                phases.append({"index": idx, "task": task, "start_step": start_step, "end_step": end_step})
        write_json(
            state_path,
            {
                "tasks": tasks,
                "steps_per_phase": steps_per_phase,
                "phases": phases,
            },
        )
        state = load_json(state_path)

    for idx, task in enumerate(tasks):
        current_step = read_latest_checkpointed_iteration(ckpt_root)

        phases = list(state.get("phases") or [])
        phase = next((p for p in phases if int(p.get("index", -1)) == idx), None)
        if phase is None:
            phase = {
                "index": idx,
                "task": task,
                "start_step": current_step,
                "end_step": current_step + steps_per_phase,
            }
            phases.append(phase)
            state["phases"] = phases
            write_json(state_path, state)
        else:
            if str(phase.get("task")) != str(task):
                raise ValueError(
                    f"CRL resume mismatch at phase index {idx}: state has task={phase.get('task')} but got {task}. "
                    "Use a new trainer.experiment_name for a new schedule."
                )

        target_total_steps = int(phase["end_step"])
        if current_step >= target_total_steps:
            print(
                f"[CRL] phase={idx + 1}/{len(tasks)} task={task} already done "
                f"(steps={current_step}>=end_step={target_total_steps}), skipping."
            )
            remove_latest_dataloader_state(ckpt_root)
            continue

        if ray.is_initialized():
            ray.shutdown()
        init_ray(address=ray_address, include_dashboard=ray_dashboard)

        phase_cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        with open_dict(phase_cfg):
            for ds_name in phase_cfg.reasoning_gym.datasets.keys():
                phase_cfg.reasoning_gym.datasets[ds_name].weight = 0
        task_override = load_rgym_task_override(task)
        phase_cfg = OmegaConf.merge(phase_cfg, task_override)
        with open_dict(phase_cfg):
            phase_cfg.crl.phase_task = str(task)
            phase_cfg.crl.phase_index = int(idx)
            phase_cfg.trainer.resume_mode = "auto"
            if crl_save_freq > 0:
                phase_cfg.trainer.save_freq = int(crl_save_freq)
        if replay_weight > 0 and idx > 0:
            with open_dict(phase_cfg):
                for prev_task in tasks[:idx]:
                    if prev_task in phase_cfg.reasoning_gym.datasets:
                        phase_cfg.reasoning_gym.datasets[prev_task].weight = replay_weight

        with open_dict(phase_cfg):
            phase_cfg.trainer.total_epochs = 999999
            phase_cfg.trainer.total_training_steps = target_total_steps

        print(
            f"[CRL] phase={idx + 1}/{len(tasks)} task={task} "
            f"steps={current_step}->{target_total_steps} ckpt_root={ckpt_root}"
        )
        ray.get(main_task.remote(phase_cfg))
        remove_latest_dataloader_state(ckpt_root)
    if ray.is_initialized():
        ray.shutdown()


@hydra.main(config_path="configs", config_name="algo/rgym/grpo_moe_lora_sphere_hf", version_base=None)
def main(config):
    if bool(OmegaConf.select(config, "crl.enabled")):
        run_continual_learning(config)
    else:
        if not ray.is_initialized():
            ray_address = str(OmegaConf.select(config, "ray.address") or "local")
            ray_dashboard = bool(OmegaConf.select(config, "ray.include_dashboard") or False)
            init_ray(address=ray_address, include_dashboard=ray_dashboard)
        ray.get(main_task.remote(config))
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
