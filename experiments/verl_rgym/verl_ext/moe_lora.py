"""
MoE-LoRA implementation adapted from VLA-Adapter (token-level router over LoRA experts),
plus SPHERE/Parseval-style regularizers adapted from EmergeMOE.

This module is designed to be used without modifying `verl` source code:
- Use PEFT LoRA for injection (broad HF model support).
- Convert injected PEFT LoRA Linear layers into token-level MoE-LoRA.
- Collect SPHERE losses inside the MoE-LoRA modules (module attributes) to remain checkpoint-safe.
"""

import copy
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import peft
from peft.tuners.lora.layer import Linear as PeftLoraLinear


@dataclass(frozen=True)
class MoELoRAConfig:
    num_experts: int
    top_k: int


def enable_moe_lora_sphere(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, MoELoRALinear):
            module.sphere_enabled = bool(enabled)


def set_moe_lora_sphere_token_mask_flat(model: nn.Module, token_mask_flat: Optional[torch.Tensor]) -> None:
    for module in model.modules():
        if isinstance(module, MoELoRALinear):
            module.sphere_token_mask_flat = token_mask_flat


def clear_moe_lora_sphere_token_mask(model: nn.Module) -> None:
    set_moe_lora_sphere_token_mask_flat(model, None)


def clear_moe_lora_sphere_losses(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MoELoRALinear):
            module.sphere_feature_loss = None
            module.sphere_gating_loss = None


def pop_moe_lora_sphere_losses(model: nn.Module) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    feature_losses: list[torch.Tensor] = []
    gating_losses: list[torch.Tensor] = []
    for module in model.modules():
        if not isinstance(module, MoELoRALinear):
            continue
        if module.sphere_feature_loss is not None:
            feature_losses.append(module.sphere_feature_loss)
            module.sphere_feature_loss = None
        if module.sphere_gating_loss is not None:
            gating_losses.append(module.sphere_gating_loss)
            module.sphere_gating_loss = None
    return feature_losses, gating_losses


def parseval_feature_loss_sphere(phi: torch.Tensor) -> torch.Tensor:
    phi = phi.float()
    tokens, dim = phi.shape
    t = phi.new_tensor(tokens)

    if dim <= tokens:
        gram = (phi.T @ phi) / t
        fro2 = (gram * gram).sum()
        tr = torch.diagonal(gram, 0).sum()
    else:
        gram = (phi @ phi.T) / t
        fro2 = (gram * gram).sum()
        tr = torch.diagonal(gram, 0).sum()

    d = phi.new_tensor(dim)
    return fro2 - (tr * tr) / d


def parseval_output_loss(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.float()
    tokens, experts = probs.shape
    t = probs.new_tensor(tokens)
    g_bar = (probs.T @ probs) / t
    fro2 = (g_bar * g_bar).sum()
    tr = torch.diagonal(g_bar, 0).sum()
    e = probs.new_tensor(experts)
    loss = fro2 - (tr * tr) / e
    return loss / e


class MoELoRALinear(PeftLoraLinear):
    moe: Optional[MoELoRAConfig] = None
    sphere_target: bool = False
    sphere_enabled: bool = False
    sphere_token_mask_flat: Optional[torch.Tensor] = None
    sphere_feature_loss: Optional[torch.Tensor] = None
    sphere_gating_loss: Optional[torch.Tensor] = None

    def set_moe(self, num_experts: int, top_k: int) -> None:
        if self.merged:
            raise NotImplementedError("MoE-LoRA layers should not be merged; unmerge before converting.")
        if self.disable_adapters:
            raise RuntimeError("Cannot configure MoE-LoRA while adapters are disabled.")

        num_experts = int(num_experts)
        if num_experts <= 0:
            raise ValueError(f"MoE-LoRA requires `num_experts > 0`, got {num_experts}")

        top_k = int(top_k)
        if not (0 < top_k < num_experts):
            raise ValueError(f"MoE-LoRA requires `0 < top_k < num_experts`, got top_k={top_k} num_experts={num_experts}")

        if "default" not in self.lora_A or "default" not in self.lora_B:
            raise RuntimeError("Expected a PEFT-initialized LoRA layer with adapter_name='default'.")

        if getattr(self, "use_dora", {}).get("default", False):
            raise NotImplementedError("MoE-LoRA does not support DoRA adapters.")

        base_A: nn.Linear = self.lora_A["default"]
        base_B: nn.Linear = self.lora_B["default"]
        base_dropout: nn.Module = self.lora_dropout["default"]
        base_scaling = self.scaling["default"]

        expert_names = [f"expert_{i}" for i in range(num_experts)]
        self.lora_A = nn.ModuleDict({name: copy.deepcopy(base_A) for name in expert_names})
        self.lora_B = nn.ModuleDict({name: copy.deepcopy(base_B) for name in expert_names})
        self.lora_dropout = nn.ModuleDict({name: copy.deepcopy(base_dropout) for name in expert_names})
        self.scaling = {name: base_scaling for name in expert_names}
        self.use_dora = {name: False for name in expert_names}
        self.set_adapter(expert_names)

        in_features = base_A.weight.shape[1]
        dtype = base_A.weight.dtype
        device = base_A.weight.device
        self.gate = nn.Linear(in_features, num_experts, bias=False, dtype=dtype, device=device)
        nn.init.zeros_(self.gate.weight)

        for name in expert_names:
            nn.init.kaiming_uniform_(self.lora_A[name].weight, a=5**0.5)
            nn.init.zeros_(self.lora_B[name].weight)

        self.moe = MoELoRAConfig(num_experts=num_experts, top_k=top_k)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.moe is None:
            return super().forward(x, *args, **kwargs)

        adapter_names = kwargs.pop("adapter_names", None)
        if adapter_names is not None:
            raise NotImplementedError("MoE-LoRA does not support PEFT mixed-batch forwarding.")
        if self.merged:
            raise NotImplementedError("MoE-LoRA layers should not be merged.")
        if self.disable_adapters:
            return self.base_layer(x, *args, **kwargs)

        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype

        gate_logits = self.gate(x.to(self.gate.weight.dtype)).to(result_dtype)
        topk_vals, topk_idx = torch.topk(gate_logits, self.moe.top_k, dim=-1)

        masked = gate_logits.new_full(gate_logits.shape, float("-inf"))
        masked.scatter_(-1, topk_idx, topk_vals)
        gate_probs = F.softmax(masked, dim=-1)

        if self.sphere_enabled:
            self.sphere_feature_loss = None
            self.sphere_gating_loss = None

        collect_sphere = self.sphere_enabled and self.sphere_target
        sphere_features: Optional[list[torch.Tensor]] = [] if collect_sphere else None

        for expert_id, expert_name in enumerate(self.active_adapters):
            if expert_name not in self.lora_A:
                continue
            lora_A = self.lora_A[expert_name]
            lora_B = self.lora_B[expert_name]
            dropout = self.lora_dropout[expert_name]
            scaling = self.scaling[expert_name]

            x_cast = x.to(lora_A.weight.dtype)
            a_out = lora_A(dropout(x_cast))
            delta = lora_B(a_out) * scaling
            weight = gate_probs[..., expert_id].unsqueeze(-1)
            if sphere_features is not None:
                sphere_features.append((weight * a_out.to(result_dtype)).to(result_dtype))
            result = result + weight * delta.to(result_dtype)

        if collect_sphere:
            probs_2d = gate_probs.reshape(-1, gate_probs.shape[-1])
            assert sphere_features is not None
            phi = torch.cat(sphere_features, dim=-1)
            phi_2d = phi.reshape(-1, phi.shape[-1])
            token_mask = self.sphere_token_mask_flat
            if token_mask is not None:
                token_mask = token_mask.to(device=phi_2d.device, dtype=torch.bool)
                probs_2d = probs_2d[token_mask]
                phi_2d = phi_2d[token_mask]

            if probs_2d.numel() == 0 or phi_2d.numel() == 0:
                self.sphere_gating_loss = gate_probs.new_tensor(0.0)
                self.sphere_feature_loss = gate_probs.new_tensor(0.0)
            else:
                self.sphere_gating_loss = parseval_output_loss(probs_2d)
                self.sphere_feature_loss = parseval_feature_loss_sphere(phi_2d)

        return result.to(result_dtype)


def convert_peft_lora_to_moe(
    peft_model: nn.Module,
    *,
    num_experts: int,
    top_k: int,
    sphere_target: str = "last",
) -> nn.Module:
    sphere_target = str(sphere_target)
    if sphere_target not in {"none", "last", "all"}:
        raise ValueError("convert_peft_lora_to_moe: `sphere_target` must be one of {'none','last','all'}.")

    converted_modules: list[MoELoRALinear] = []
    for _, module in peft_model.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer) and isinstance(module, PeftLoraLinear):
            module.__class__ = MoELoRALinear
            module.set_moe(num_experts=num_experts, top_k=top_k)
            converted_modules.append(module)  # type: ignore[arg-type]

    if not converted_modules:
        raise ValueError(
            "convert_peft_lora_to_moe did not convert any PEFT LoRA Linear layers. "
            "Check `target_modules`/`exclude_modules` and ensure LoRA injection succeeded."
        )

    if sphere_target == "all":
        for module in converted_modules:
            module.sphere_target = True
    elif sphere_target == "last":
        for module in converted_modules:
            module.sphere_target = False
        converted_modules[-1].sphere_target = True
    else:
        for module in converted_modules:
            module.sphere_target = False

    print(f"[MoE-LoRA/PEFT] Converted {len(converted_modules)} LoRA Linear layers to `MoELoRALinear`.")
    return peft_model
