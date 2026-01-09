from typing import Any, Optional

import torch
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.workers.actor.dp_actor import DataParallelPPOActor

from verl_ext.moe_lora import (
    clear_moe_lora_sphere_losses,
    clear_moe_lora_sphere_token_mask,
    enable_moe_lora_sphere,
    pop_moe_lora_sphere_losses,
    set_moe_lora_sphere_token_mask_flat,
)


class SphereDataParallelPPOActor(DataParallelPPOActor):
    def __init__(self, *args, verl_ext_config: Optional[dict[str, Any]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._verl_ext_config = verl_ext_config or {}

    def update_policy(self, data: DataProto):
        self.actor_module.train()

        actor_param = next(self.actor_module.parameters())
        actor_device = actor_param.device

        temperature = data.meta_info["temperature"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        sphere_feature_ratio = float(self._verl_ext_config.get("sphere_feature_ratio", 0.0))
        sphere_gating_ratio = float(self._verl_ext_config.get("sphere_gating_ratio", 0.0))
        sphere_eps = float(self._verl_ext_config.get("sphere_eps", 1e-8))
        sphere_mode = str(self._verl_ext_config.get("sphere_mode", "loss_ratio"))
        sphere_enabled = (sphere_feature_ratio > 0.0) or (sphere_gating_ratio > 0.0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for mini_batch in mini_batches:
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(actor_device)
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    if sphere_enabled:
                        enable_moe_lora_sphere(self.actor_module, True)
                        clear_moe_lora_sphere_losses(self.actor_module)
                        input_ids = model_inputs["input_ids"]
                        batch_size, seqlen = input_ids.shape
                        response_length = model_inputs["responses"].size(-1)
                        start = seqlen - response_length - 1
                        end = seqlen - 1
                        token_mask = input_ids.new_zeros((batch_size, seqlen), dtype=torch.bool)
                        token_mask[:, start:end] = response_mask.to(torch.bool)
                        if self.use_remove_padding:
                            attn = model_inputs["attention_mask"]
                            indices = torch.nonzero(attn.reshape(-1), as_tuple=False).squeeze(-1)
                            token_mask_flat = token_mask.reshape(-1).index_select(0, indices)
                        else:
                            token_mask_flat = token_mask.reshape(-1)
                        set_moe_lora_sphere_token_mask_flat(self.actor_module, token_mask_flat)

                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    base_policy_loss = policy_loss
                    if sphere_enabled:
                        sphere_feature_losses, sphere_gating_losses = pop_moe_lora_sphere_losses(self.actor_module)
                        if sphere_feature_ratio > 0.0:
                            sphere_feat_loss = sphere_feature_losses[-1]
                            if sphere_mode == "fixed":
                                sphere_feat_scale = base_policy_loss.new_tensor(sphere_feature_ratio)
                            else:
                                sphere_feat_scale = (
                                    sphere_feature_ratio
                                    * base_policy_loss.detach().abs()
                                    / (sphere_feat_loss.detach() + sphere_eps)
                                ).detach()
                            policy_loss = policy_loss + sphere_feat_scale * sphere_feat_loss
                            micro_batch_metrics["actor/sphere_feature_loss"] = sphere_feat_loss.detach().item()
                            micro_batch_metrics["actor/sphere_feature_scale"] = sphere_feat_scale.item()

                        if sphere_gating_ratio > 0.0:
                            sphere_gate_loss = sphere_gating_losses[-1]
                            if sphere_mode == "fixed":
                                sphere_gate_scale = base_policy_loss.new_tensor(sphere_gating_ratio)
                            else:
                                sphere_gate_scale = (
                                    sphere_gating_ratio
                                    * base_policy_loss.detach().abs()
                                    / (sphere_gate_loss.detach() + sphere_eps)
                                ).detach()
                            policy_loss = policy_loss + sphere_gate_scale * sphere_gate_loss
                            micro_batch_metrics["actor/sphere_gating_loss"] = sphere_gate_loss.detach().item()
                            micro_batch_metrics["actor/sphere_gating_scale"] = sphere_gate_scale.item()

                    loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if sphere_enabled:
                        enable_moe_lora_sphere(self.actor_module, False)
                        clear_moe_lora_sphere_losses(self.actor_module)
                        clear_moe_lora_sphere_token_mask(self.actor_module)

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)

        self.actor_optimizer.zero_grad()
        return metrics
