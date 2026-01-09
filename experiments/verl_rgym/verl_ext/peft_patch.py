from contextlib import contextmanager
from typing import Any, Callable


@contextmanager
def patch_verl_get_peft_model_for_moe_lora(verl_ext_cfg: dict[str, Any]):
    use_moe_lora = bool(verl_ext_cfg.get("use_moe_lora"))
    if not use_moe_lora:
        yield
        return

    num_experts = int(verl_ext_cfg["moe_num_experts"])
    top_k = int(verl_ext_cfg.get("moe_top_k", 2))
    sphere_target = str(verl_ext_cfg.get("sphere_target", "last"))

    from verl_ext.moe_lora import convert_peft_lora_to_moe

    import verl.workers.fsdp_workers as fsdp_workers

    original: Callable[..., Any] = fsdp_workers.get_peft_model

    def wrapped_get_peft_model(model, peft_config, *args, **kwargs):
        peft_model = original(model, peft_config, *args, **kwargs)
        convert_peft_lora_to_moe(
            peft_model,
            num_experts=num_experts,
            top_k=top_k,
            sphere_target=sphere_target,
        )
        return peft_model

    fsdp_workers.get_peft_model = wrapped_get_peft_model  # type: ignore[assignment]

    import verl.workers.engine.fsdp.transformer_impl as transformer_impl_mod  # type: ignore[assignment]

    transformer_impl_original = getattr(transformer_impl_mod, "get_peft_model")
    setattr(transformer_impl_mod, "get_peft_model", wrapped_get_peft_model)

    try:
        yield
    finally:
        fsdp_workers.get_peft_model = original  # type: ignore[assignment]
        setattr(transformer_impl_mod, "get_peft_model", transformer_impl_original)
