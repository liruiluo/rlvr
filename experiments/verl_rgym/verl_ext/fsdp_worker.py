from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.workers.config import FSDPEngineConfig
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker

from verl_ext.actor import SphereDataParallelPPOActor
from verl_ext.hf_local_rollout import HFLocalAsyncRollout
from verl_ext.peft_patch import patch_verl_get_peft_model_for_moe_lora


class ExtAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.import_utils import import_external_libs

        if self.role == "ref":
            return super().init_model()

        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        verl_ext_cfg = OmegaConf.select(self.config, "rollout.custom.verl_ext")

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)

        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = omega_conf_to_dataclass(self.config.actor.fsdp_config)
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()

            with patch_verl_get_peft_model_for_moe_lora(verl_ext_cfg or {}):
                (
                    self.actor_module_fsdp,
                    self.actor_optimizer,
                    self.actor_lr_scheduler,
                    self.actor_model_config,
                ) = self._build_model_optimizer(
                    model_path=local_path,
                    fsdp_config=fsdp_config,
                    optim_config=optim_config,
                    override_model_config=override_model_config,
                    use_remove_padding=use_remove_padding,
                    use_fused_kernels=use_fused_kernels,
                    enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                    trust_remote_code=self.config.model.get("trust_remote_code", False),
                    use_liger=self.config.model.get("use_liger", False),
                    role="actor",
                    enable_activation_offload=self.config.model.get("enable_activation_offload", False),
                    use_prefix_grouper=self.config.actor.get("use_prefix_grouper", False),
                    use_tiled_mlp=False,
                    tiled_mlp_shards=4,
                )

            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

        if self._is_actor:
            actor_cfg = omega_conf_to_dataclass(self.config.actor)
            self.actor = SphereDataParallelPPOActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
                verl_ext_config=verl_ext_cfg or {},
            )
            from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
            from verl.utils.flops_counter import FlopsCounter

            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        if self._is_rollout:
            if str(self.config.rollout.name) == "hf":
                self.rollout = HFLocalAsyncRollout(
                    model=self.actor_module_fsdp,
                    tokenizer=self.tokenizer,
                    rollout_config=self.config.rollout,
                )
            else:
                self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

    async def rollout_mode(self):
        if str(self.config.rollout.name) == "hf":
            if self._is_offload_param:
                load_fsdp_model_to_gpu(self.actor_module_fsdp)
            self.actor_module_fsdp.eval()
            return
        return await super().rollout_mode()

    async def trainer_mode(self):
        if str(self.config.rollout.name) == "hf":
            if getattr(self, "rollout", None) is not None and hasattr(self.rollout, "drain"):
                await self.rollout.drain()
            self.actor_module_fsdp.train()
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            return
        return await super().trainer_mode()

    @register(dispatch_mode=Dispatch.DIRECT_ROLLOUT_METHOD, blocking=False)
    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data=None,
        video_data=None,
    ):
        del video_data
        return await self.rollout.generate(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
        )
