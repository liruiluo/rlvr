from __future__ import annotations

import contextlib
from typing import Any, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig

from verl.utils.model import compute_position_id_with_mask
from verl.workers.rollout.replica import TokenOutput


class HFLocalAsyncRollout:
    def __init__(self, *, model, tokenizer, rollout_config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = rollout_config

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, image_data, video_data

        param = next(self.model.parameters())
        device = param.device
        autocast_dtype = param.dtype
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        position_ids = compute_position_id_with_mask(attention_mask)

        max_new_tokens = sampling_params.get("max_new_tokens", None)
        if max_new_tokens is None:
            max_new_tokens = sampling_params.get("max_tokens", int(getattr(self.config, "response_length", 128)))

        temperature = float(sampling_params.get("temperature", float(getattr(self.config, "temperature", 1.0))))
        top_p = float(sampling_params.get("top_p", float(getattr(self.config, "top_p", 1.0))))
        repetition_penalty = float(sampling_params.get("repetition_penalty", float(getattr(self.config, "repetition_penalty", 1.0))))
        do_sample = temperature > 0.0

        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
            max_new_tokens=int(max_new_tokens),
            num_beams=1,
            repetition_penalty=repetition_penalty,
        )

        eos_token_id = None if bool(getattr(self.config, "ignore_eos", False)) else self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id

        self.model.eval()
        param_ctx = (
            FSDP.summon_full_params(self.model, writeback=False, recurse=False)
            if isinstance(self.model, FSDP)
            else contextlib.nullcontext()
        )
        with param_ctx, torch.no_grad(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
            seq = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                generation_config=generation_config,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                output_scores=False,
                return_dict_in_generate=False,
                use_cache=True,
            )
        response = seq[0, input_ids.size(1) :].tolist()
        return TokenOutput(token_ids=response, log_probs=None, routed_experts=None, stop_reason="completed")
